"""Background movement jobs — keeps a first UMAP refit off the request path.

`backend/jobs.py` exists because a full analysis build can outlast the ~100s
response timeout a proxy such as Cloudflare enforces. A movement inherits a
smaller version of the same problem: the analysis serializes coordinates and
discards its fitted reducers, so the FIRST movement on a UMAP node has to refit
the projection before it can answer anything. That fit is the same UMAP fit the
build paid for, and on a large node it is minutes, not seconds.

So the movement route follows the analysis route's shape: `POST /api/movement`
returns a job envelope when work is needed, `GET /api/movement/jobs/{job_id}`
polls it. Two deliberate differences:

* **PCA never comes here, and neither does a UMAP request whose reducer is
  already cached.** A pinv solve against a warm artifact must not regress into a
  polling round trip — that is the interactive path behind the strength slider.
* **The result lives here, not in a cache.** Movement results are never
  persisted: nothing is written to `backend/run_cache.py` or the tree cache,
  because a counterfactual preview is not an analysis run. A finished job holds
  its response in memory just long enough for a poll to collect it, in a bounded
  store — same reasoning as `_MAX_JOBS` in `backend/jobs.py`.

Jobs are keyed by the movement request's own identity, so a debounced slider or
a retried POST re-attaches to the run already in flight instead of starting a
second UMAP fit of the same node.

The wire shape is `MovementJob` in `frontend/src/types.ts`::

    {"status": "running", "job_id": …}
    {"status": "error", "job_id": …, "detail": …}
    <the MovementResponse itself, whose own "status" is "ok">

Note the finished variant is the response, not a `{"status": "done", …}` wrapper
as on the analysis route: a movement response already carries a `status` field,
and discriminating on that one field keeps the inline and the polled result
literally the same object.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.movement import MovementRequestData

# Smaller than `backend/jobs.py::_MAX_JOBS` (64) because a finished movement job
# carries its OWN result — up to `MAX_MOVEMENT_PREVIEW_POINTS` ghost points —
# where an analysis job only names a run-cache key. Finished jobs are kept so a
# repeated or delayed poll still finds the answer.
_MAX_JOBS = 16


@dataclass
class MovementJob:
    id: str
    key: str  # request identity; a duplicate POST re-attaches instead of refitting
    status: str = "running"  # running | done | error
    detail: str = ""
    response: dict[str, Any] | None = None  # in memory only, never persisted


_lock = threading.Lock()
_jobs: dict[str, MovementJob] = {}
_by_key: dict[str, str] = {}  # request key -> job id


def job_key(req: MovementRequestData) -> str:
    """Identity of a movement request: same run, node, source, target, strength.

    `dataset` and `feature_cols` are not part of it — `analysis_id` already
    digests them, and the request has been validated against that run before a
    job is ever submitted.
    """
    return json.dumps(
        {
            "analysis_id": req.analysis_id,
            "node_id": req.node_id,
            "source_child_index": req.source_child_index,
            "target": req.target,
            "strength": req.strength,
        },
        sort_keys=True,
        default=str,
    )


def submit(key: str, work: Callable[[], dict[str, Any]]) -> MovementJob:
    """Run `work()` in a thread, or return the job already running for `key`."""
    with _lock:
        running = _jobs.get(_by_key.get(key, ""))
        if running is not None and running.status == "running":
            return running
        job = MovementJob(id=uuid.uuid4().hex, key=key)
        _jobs[job.id] = job
        _by_key[key] = job.id
        _prune()

    def run() -> None:
        try:
            response = work()
        except Exception as exc:  # noqa: BLE001 - reaches the client as the poll's `detail`
            _finish(job, "error", detail=str(exc))
        else:
            _finish(job, "done", response=response)

    threading.Thread(target=run, name=f"movement-{job.id[:8]}", daemon=True).start()
    return job


def get(job_id: str) -> MovementJob | None:
    with _lock:
        return _jobs.get(job_id)


def envelope(job: MovementJob) -> dict[str, Any]:
    """The `MovementJob` union of `frontend/src/types.ts` for one job."""
    if job.status == "running":
        return {"status": "running", "job_id": job.id}
    if job.status == "error":
        return {"status": "error", "job_id": job.id, "detail": job.detail}
    if job.response is None:  # only if the job was pruned between finishing and polling
        return {
            "status": "error",
            "job_id": job.id,
            "detail": "This movement's result is no longer available — request it again.",
        }
    return job.response


def clear() -> None:
    """Drop every job (tests)."""
    with _lock:
        _jobs.clear()
        _by_key.clear()


def _finish(
    job: MovementJob,
    status: str,
    *,
    detail: str = "",
    response: dict[str, Any] | None = None,
) -> None:
    with _lock:
        job.status = status
        job.detail = detail
        job.response = response


def _prune() -> None:
    """Drop the oldest finished jobs. Caller holds the lock; dicts keep insertion order."""
    excess = len(_jobs) - _MAX_JOBS
    if excess <= 0:
        return
    for job in [j for j in _jobs.values() if j.status != "running"][:excess]:
        _jobs.pop(job.id, None)
        if _by_key.get(job.key) == job.id:
            _by_key.pop(job.key, None)

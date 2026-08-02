"""Background analysis jobs — keeps long tree builds off the request path.

A full build (MDS/UMAP + HDBSCAN + ZADU) can run for many minutes, well past the
100s response timeout a proxy such as Cloudflare enforces; a request that sits
open that long is cut off with a 524 even though the work is fine. So
`/api/analysis` starts the run here, returns a job id immediately, and the
frontend polls until the job reports done.

Jobs are keyed by the same (dataset, feature_cols, config) signature the run
cache uses, so a retry or a page reload re-attaches to the run already in flight
instead of launching a second copy of it.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

# Finished jobs are kept only so a late poll can still read the outcome.
_MAX_JOBS = 64


@dataclass
class Job:
    id: str
    key: str  # run-cache key; the result payload lives in the caller's cache, not here
    status: str = "running"  # running | done | error
    detail: str = ""


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_by_key: dict[str, str] = {}  # cache key -> job id


def submit(key: str, work: Callable[[], None]) -> Job:
    """Run `work()` in a thread, or return the job already running for `key`."""
    with _lock:
        running = _jobs.get(_by_key.get(key, ""))
        if running is not None and running.status == "running":
            return running
        job = Job(id=uuid.uuid4().hex, key=key)
        _jobs[job.id] = job
        _by_key[key] = job.id
        _prune()

    def run() -> None:
        try:
            work()
        except Exception as exc:  # the failure reaches the client through the poll
            _finish(job, "error", str(exc))
        else:
            _finish(job, "done", "")

    threading.Thread(target=run, name=f"analysis-{job.id[:8]}", daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _finish(job: Job, status: str, detail: str) -> None:
    with _lock:
        job.status = status
        job.detail = detail


def _prune() -> None:
    """Drop the oldest finished jobs. Caller holds the lock; dicts keep insertion order."""
    excess = len(_jobs) - _MAX_JOBS
    if excess <= 0:
        return
    for job in [j for j in _jobs.values() if j.status != "running"][:excess]:
        _jobs.pop(job.id, None)
        if _by_key.get(job.key) == job.id:
            _by_key.pop(job.key, None)

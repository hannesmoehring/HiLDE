"""Cache-only mode: stored runs are served, nothing is built.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_cache_only
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_cache_only.py

The routes are called as functions, as in `test_counterfactual`. The Iris payloads
are shared with `test_movement` (memoized).
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend import app as backend_app
from backend import jobs, movement_jobs, run_cache, run_listing
from backend.movement import clear_artifact_cache
from backend.tests import test_movement as tm

_ENV = ("HILDE_HOSTING", "HILDE_CACHE_ONLY", "HILDE_CACHE_DIR")


@contextlib.contextmanager
def _cache_only(*payloads: dict[str, Any]) -> Iterator[Path]:
    """A cache-only server whose cache directory holds exactly `payloads`."""
    before = {k: os.environ.get(k) for k in _ENV}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.pop("HILDE_HOSTING", None)  # cache-only must imply it
        os.environ["HILDE_CACHE_ONLY"] = "1"
        os.environ["HILDE_CACHE_DIR"] = tmp
        backend_app._tree_cache.clear()  # every hit below has to come off the disk
        try:
            for payload in payloads:
                run_cache.store(_key(payload), payload)
            yield Path(tmp)
        finally:
            backend_app._tree_cache.clear()
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _key(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    return backend_app._cache_key(meta["dataset"], meta["feature_cols"], meta["config"])


def _analysis_request(
    payload: dict[str, Any], **overrides: Any
) -> backend_app.AnalysisRequest:
    meta = payload["meta"]
    fields = {
        "dataset": meta["dataset"],
        "feature_cols": meta["feature_cols"],
        "config": meta["config"],
    }
    fields.update(overrides)
    return backend_app.AnalysisRequest(**fields)


def _movement_fields(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload["meta"]
    return {
        "analysis_id": meta["analysis_id"],
        "dataset": meta["dataset"],
        "feature_cols": meta["feature_cols"],
        "config": meta["config"],
        "node_id": "root",
        "source_child_index": 0,
        "target": {"kind": "point", "x": 0.5, "y": -0.25},
    }


def _expect_409(fn, *args: Any) -> HTTPException:
    try:
        fn(*args)
    except HTTPException as exc:
        assert exc.status_code == 409, f"expected 409, got {exc.status_code}"
        assert "precomputed runs" in exc.detail
        return exc
    raise AssertionError("expected a 409, the request was answered")


def test_cache_only_implies_hosting_and_is_reported():
    with _cache_only() as tmp:
        assert run_cache.is_hosting()
        assert backend_app.mode() == {
            "hosting": True,
            "cache_dir": str(tmp),
            "cache_only": True,
            "maintenance": False,
        }
    assert backend_app.mode()["cache_only"] is False


def test_maintenance_is_reported_and_off_by_default():
    before = os.environ.get("HILDE_MAINTENANCE")
    try:
        os.environ.pop("HILDE_MAINTENANCE", None)
        assert backend_app.mode()["maintenance"] is False
        os.environ["HILDE_MAINTENANCE"] = "1"
        assert backend_app.mode()["maintenance"] is True
    finally:
        if before is None:
            os.environ.pop("HILDE_MAINTENANCE", None)
        else:
            os.environ["HILDE_MAINTENANCE"] = before


def test_listing_names_exactly_the_runs_a_request_can_reach():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    with _cache_only(pca) as tmp:
        # Present on disk, but under a name no request hashes to.
        run_cache.store("not the signature of this payload", umap)
        (tmp / f"{'0' * 32}.json.gz").write_bytes(b"truncated upload")
        assert len(list(tmp.iterdir())) == 3

        listed = backend_app.cached_runs()
        assert listed == [
            {
                "dataset": pca["meta"]["dataset"],
                "feature_cols": pca["meta"]["feature_cols"],
                "n_total": pca["meta"]["n_total"],
                "configs": [pca["meta"]["config"]],
            }
        ]
        # The listing is what the client sends back, and that has to be a hit.
        answer = backend_app.analysis(
            backend_app.AnalysisRequest(
                dataset=listed[0]["dataset"],
                feature_cols=listed[0]["feature_cols"],
                config=listed[0]["configs"][0],
            )
        )
        assert answer["status"] == "done" and answer["cached"] is True
        assert len(list(tmp.iterdir())) == 3, "cache-only mode deleted an entry"


def test_listing_reads_the_head_of_an_entry_and_follows_the_directory():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    with _cache_only(pca) as tmp:
        # Only the head is parsed: a tree that is not JSON at all goes unnoticed.
        path = run_cache._path_for(_key(pca))
        head = json.dumps({"meta": pca["meta"]})[:-1]
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(head + ', "tree": <not parsed by the listing>')
        assert [g["configs"] for g in backend_app.cached_runs()] == [
            [pca["meta"]["config"]]
        ]
        # A meta that outgrows the head falls back to parsing the whole entry.
        head_chars = run_listing._HEAD_CHARS
        run_listing._HEAD_CHARS = 40
        try:
            run_cache.store(_key(pca), pca)
            assert len(backend_app.cached_runs()) == 1
        finally:
            run_listing._HEAD_CHARS = head_chars
        # Same dataset and columns: one group, and it picks up the new entry.
        run_cache.store(_key(umap), umap)
        (group,) = backend_app.cached_runs()
        assert len(group["configs"]) == 2
        assert len(list(tmp.iterdir())) == 2


def test_a_stored_run_is_served_even_when_a_recompute_is_asked_for():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca):
        jobs_before = dict(jobs._jobs)
        for use_cache in (True, False):
            answer = backend_app.analysis(_analysis_request(pca, use_cache=use_cache))
            assert answer["status"] == "done" and answer["cached"] is True
            assert answer["tree"] == pca["tree"]
        assert jobs._jobs == jobs_before, "a build was started"


def test_an_unstored_run_is_refused_without_starting_a_build():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca) as tmp:
        jobs_before = dict(jobs._jobs)
        other = {**pca["meta"]["config"], "hclust_min_cluster_size": 7}
        _expect_409(backend_app.analysis, _analysis_request(pca, config=other))
        assert jobs._jobs == jobs_before, "a build was started"
        assert len(list(tmp.iterdir())) == 1


def test_a_corrupt_entry_is_a_miss_and_is_left_on_disk():
    pca, _ = tm._payload("PCA")
    with _cache_only() as tmp:
        path = run_cache._path_for(_key(pca))
        path.write_bytes(b"truncated upload")
        _expect_409(backend_app.analysis, _analysis_request(pca))
        assert path.is_file()
        assert list(tmp.iterdir()) == [path]


def test_pca_movement_previews_but_apply_and_umap_refits_are_refused():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    with _cache_only(pca, umap):
        clear_artifact_cache()
        jobs_before = dict(movement_jobs._jobs)

        moved = backend_app.movement(
            backend_app.MovementRequest(**_movement_fields(pca))
        )
        assert moved["status"] == "ok"

        _expect_409(
            backend_app.movement, backend_app.MovementRequest(**_movement_fields(umap))
        )
        for payload in (pca, umap):
            _expect_409(
                backend_app.counterfactual_apply,
                backend_app.CounterfactualApplyRequest(
                    **_movement_fields(payload), strength=0.5
                ),
            )
        assert movement_jobs._jobs == jobs_before, "a refit job was started"


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"OK — {_name}")
    print("OK — cache-only mode passed")

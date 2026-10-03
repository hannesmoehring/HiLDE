"""Cache-only mode: stored runs are served, nothing is built.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_cache_only
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_cache_only.py

The routes are called as functions, as in `test_counterfactual`. The Iris payloads
are shared with `test_movement` (memoized).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import gzip
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend import app as backend_app
from backend import (
    jobs,
    movement_jobs,
    run_cache,
    run_index,
    run_listing,
    tree_cache,
)
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
        tree_cache.clear()  # every hit below has to come off the disk
        try:
            for payload in payloads:
                run_cache.store(_key(payload), payload)
            yield Path(tmp)
        finally:
            tree_cache.clear()
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


def _listed(dataset: str) -> list[tuple[list[str], dict[str, Any]]]:
    """`/api/cached-runs/{dataset}` decoded the way `frontend/src/api.ts` does."""
    try:
        response = backend_app.cached_runs_of(dataset, "gzip")
    except HTTPException as exc:
        assert exc.status_code == 404
        return []
    assert response.headers["content-encoding"] == "gzip"
    doc = json.loads(gzip.decompress(response.body))
    runs = []
    for g, *idx in doc["runs"]:
        config = dict(doc["fixed"])
        for k, i in enumerate(idx):
            if i >= 0:
                config[doc["knobs"][k]] = doc["values"][k][i]
        runs.append((doc["groups"][g]["feature_cols"], config))
    return runs


def _entries(tmp: Path) -> list[Path]:
    return sorted(tmp.glob("*.json.gz"))


def test_listing_names_exactly_the_runs_a_request_can_reach():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    with _cache_only(pca) as tmp:
        # Present on disk, but under a name no request hashes to.
        run_cache.store("not the signature of this payload", umap)
        (tmp / f"{'0' * 32}.json.gz").write_bytes(b"truncated upload")
        assert len(_entries(tmp)) == 3

        dataset = pca["meta"]["dataset"]
        assert backend_app.cached_runs() == [{"dataset": dataset, "n_runs": 1}]
        ((cols, config),) = _listed(dataset)
        assert (cols, config) == (pca["meta"]["feature_cols"], pca["meta"]["config"])
        other = next(
            d["key"] for d in backend_app.list_datasets() if d["key"] != dataset
        )
        assert _listed(other) == [], "a dataset without stored runs is a 404"
        # The listing is what the client sends back, and that has to be a hit.
        answer = backend_app.analysis(
            backend_app.AnalysisRequest(
                dataset=dataset, feature_cols=cols, config=config
            )
        )
        assert _served(answer)[0]["meta"] == pca["meta"]
        assert len(_entries(tmp)) == 3, "cache-only mode deleted an entry"


def test_listing_reads_the_head_of_an_entry_and_follows_the_directory():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    dataset = pca["meta"]["dataset"]
    with _cache_only(pca) as tmp:
        # Only the head is parsed: a tree that is not JSON at all goes unnoticed.
        path = run_cache._path_for(_key(pca))
        head = json.dumps({"meta": pca["meta"]})[:-1]
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(head + ', "tree": <not parsed by the listing>')
        assert [c for _, c in _listed(dataset)] == [pca["meta"]["config"]]
        # A meta longer than the first read is read on to its end.
        chunk = run_listing._CHUNK
        run_listing._CHUNK = 16
        try:
            run_cache.store(_key(pca), pca)
            assert len(_listed(dataset)) == 1
        finally:
            run_listing._CHUNK = chunk
        # Same dataset and columns: one group, and it picks up the new entry.
        run_cache.store(_key(umap), umap)
        assert sorted(c["method"] for _, c in _listed(dataset)) == ["PCA", "UMAP"]
        assert len(_entries(tmp)) == 2


def test_the_listing_sends_each_run_with_its_exact_config():
    pca, umap = (copy.deepcopy(tm._payload(m)[0]) for m in ("PCA", "UMAP"))
    # A knob one run was requested without, and one stored as 1 vs 1.0: either
    # would change the cache key if the listing filled it in or normalized it.
    umap["meta"]["config"].pop("normalize", None)
    umap["meta"]["config"]["umap_min_dist"] = 1.0
    pca["meta"]["config"]["umap_min_dist"] = 1
    dataset = pca["meta"]["dataset"]
    with _cache_only(pca, umap):
        listed = [backend_app._cache_key(dataset, *run) for run in _listed(dataset)]
        assert sorted(listed) == sorted([_key(pca), _key(umap)])
        plain = backend_app.cached_runs_of(dataset, "identity")
        assert "content-encoding" not in plain.headers
        assert len(json.loads(plain.body)["runs"]) == 2


def test_listing_is_kept_next_to_the_cache_and_survives_a_restart():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    dataset = pca["meta"]["dataset"]
    with _cache_only(pca, umap) as tmp:
        assert len(_listed(dataset)) == 2
        index = tmp / run_index.INDEX_DIR / run_index.INDEX_FILE
        assert index.is_file()
        # A restart reads the index, not the entries: hide one entry's meta
        # behind an unchanged mtime and directory, and it is still listed.
        run_listing._index = None
        path = run_cache._path_for(_key(umap))
        stat = path.stat()
        doc = json.loads(gzip.decompress(index.read_bytes()))
        doc["stamp"] = tmp.stat().st_mtime_ns  # as if indexed long after the change
        index.write_bytes(gzip.compress(json.dumps(doc).encode()))
        path.write_bytes(gzip.compress(b"not an entry"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.utime(tmp, ns=(tmp.stat().st_atime_ns, doc["stamp"]))
        assert len(_listed(dataset)) == 2
        # Any change to the directory is noticed, and the changed entry re-read.
        path.unlink()
        assert len(_listed(dataset)) == 1


def test_listing_works_without_a_writable_index():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca) as tmp:
        run_listing._index = None
        (tmp / run_index.INDEX_DIR).write_text("not a directory")  # like a ro mount
        assert len(_listed(pca["meta"]["dataset"])) == 1
        assert (tmp / run_index.INDEX_DIR).read_text() == "not a directory"


def _served(response: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """The JSON a client decodes from an `/api/analysis` hit, and the headers."""

    async def drain() -> bytes:
        return b"".join([chunk async for chunk in response.body_iterator])

    if isinstance(response, FileResponse):
        body = Path(response.path).read_bytes()
    else:
        body = asyncio.run(drain())
    if response.headers.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body), dict(response.headers)


def test_a_stored_run_is_served_even_when_a_recompute_is_asked_for():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca):
        jobs_before = dict(jobs._jobs)
        for use_cache in (True, False):
            answer = backend_app.analysis(_analysis_request(pca, use_cache=use_cache))
            served, _ = _served(answer)
            assert served["tree"] == pca["tree"] and served["meta"] == pca["meta"]
        assert jobs._jobs == jobs_before, "a build was started"
        assert not tree_cache.keys(), "a stored run was parsed to serve it"


def test_a_stored_run_is_sent_as_its_gzip_bytes_only_when_accepted():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca):
        req = _analysis_request(pca)
        path = run_cache._path_for(_key(pca))
        for accept, gz in [
            ("gzip, deflate, br", True),
            ("br;q=1.0, gzip;q=0.8", True),
            ("*", True),
            ("", False),
            ("identity", False),
            ("gzip;q=0", False),
        ]:
            answer = backend_app.analysis(req, accept)
            served, headers = _served(answer)
            assert served["tree"] == pca["tree"], accept
            assert (headers.get("content-encoding") == "gzip") is gz, accept
            assert headers["vary"] == "Accept-Encoding"
            if gz:
                assert Path(answer.path) == path


def test_the_parsed_tree_cache_is_bounded_by_bytes_in_cache_only_mode():
    pca, _ = tm._payload("PCA")
    umap, _ = tm._payload("UMAP")
    with _cache_only(pca, umap):
        sizes = [run_cache._path_for(_key(p)).stat().st_size for p in (pca, umap)]
        budget = tree_cache.MAX_BYTES
        try:  # room for one parsed entry, not two
            tree_cache.MAX_BYTES = max(sizes) * tree_cache.PARSED_PER_GZ_BYTE
            assert backend_app._cached_payload(_key(pca)) is not None
            assert backend_app._cached_payload(_key(umap)) is not None
            assert tree_cache.keys() == [_key(umap)]
            tree_cache.MAX_BYTES = 1  # the newest stays even over budget
            assert backend_app._cached_payload(_key(pca)) is not None
            assert tree_cache.keys() == [_key(pca)]
        finally:
            tree_cache.MAX_BYTES = budget


def test_an_unstored_run_is_refused_without_starting_a_build():
    pca, _ = tm._payload("PCA")
    with _cache_only(pca) as tmp:
        jobs_before = dict(jobs._jobs)
        other = {**pca["meta"]["config"], "hclust_min_cluster_size": 7}
        _expect_409(backend_app.analysis, _analysis_request(pca, config=other))
        assert jobs._jobs == jobs_before, "a build was started"
        assert len(_entries(tmp)) == 1


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

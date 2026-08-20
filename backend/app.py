"""FastAPI service for the D3 frontend.

Wraps the unchanged calc layer (`src/analysis`, `src/evaluation`) and serves the
JSON data contract emitted by `backend/serialize.py`. The tree is built on demand
and cached by (dataset, feature_cols, config); navigation/drill-down happens
client-side.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Make the repo root importable so `src.*` resolves when uvicorn is started from
# elsewhere; PYTHONPATH=. does the same thing for the documented dev command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import datasets as ds
from backend import images as ds_images
from backend import jobs, run_cache
from backend.characteristics import compute_selection_characteristics
from backend.predicate import compute_predicate
from backend.serialize import serialize_tree
from backend.targets import compute_targets
from src.config_defaults import default_config
from src.evaluation.evaluate import start_evaluation

app = FastAPI(title="HiLDE API", version="0.1.0")

# Dev: Vite dev server (5173) calls the API cross-origin. Tightened in prod (single container).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache serialized trees by request signature; tree build (UMAP+HDBSCAN+ZADU) is expensive.
# Bounded LRU: a payload is ~8-10 MB on the smallest realistic dataset, and a
# hyperparameter sweep visits a new key every build, so an unbounded dict grows
# monotonically into the container's memory limit.
_TREE_CACHE_MAX = 8
_tree_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    _tree_cache[key] = payload
    _tree_cache.move_to_end(key)
    while len(_tree_cache) > _TREE_CACHE_MAX:
        _tree_cache.popitem(last=False)


def _cache_get(key: str) -> dict[str, Any] | None:
    payload = _tree_cache.get(key)
    if payload is not None:
        _tree_cache.move_to_end(key)
    return payload


def _merge_config(partial: dict[str, Any]) -> dict[str, Any]:
    """Overlay the frontend's config knobs on the full default Config."""
    config = default_config()
    config.update(partial)  # type: ignore[typeddict-item]
    return config  # type: ignore[return-value]


def _cache_key(dataset: str, feature_cols: list[str], config: dict[str, Any]) -> str:
    return json.dumps(
        {"dataset": dataset, "feature_cols": feature_cols, "config": config},
        sort_keys=True,
        default=str,
    )


class AnalysisRequest(BaseModel):
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any] = {}
    use_cache: bool = True


class PredicateRequest(BaseModel):
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any] = {}
    row_indices: list[int]
    selected_local_indices: list[int]
    scope: str = "local"


class CharacteristicsRequest(BaseModel):
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any] = {}
    row_indices: list[int]
    selected_local_indices: list[int]


class TargetsRequest(BaseModel):
    dataset: str
    target_cols: list[str]
    row_indices: list[int]
    selected_local_indices: list[int]


class RowsRequest(BaseModel):
    dataset: str
    ids: list[int]
    columns: list[str] | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/mode")
def mode() -> dict[str, Any]:
    """Whether the server runs in hosting mode (persistent run cache + UI banner)."""
    hosting = run_cache.is_hosting()
    return {
        "hosting": hosting,
        "cache_dir": str(run_cache.cache_dir()) if hosting else None,
    }


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, str]]:
    """Cheap listing — does not load any dataset."""
    return [{"key": k, "label": k} for k in ds.dataset_keys()]


@app.get("/api/datasets/{key}/columns")
def dataset_columns(key: str) -> dict[str, Any]:
    """Loads (and caches) the dataset to report its columns + default feature selection."""
    try:
        df = ds.load(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {key}") from exc
    return {
        "key": key,
        "n_rows": len(df),
        "columns": [str(c) for c in df.columns],
        "default_feature_cols": ds.default_feature_cols(df),
        "image": ds_images.spec(key),  # non-null = rows can be rendered as images
    }


@app.get("/api/datasets/{key}/image/{row_id}")
def dataset_image(key: str, row_id: int) -> dict[str, Any]:
    """Greyscale pixels of a single row, for the image-valued datasets."""
    if ds_images.spec(key) is None:
        raise HTTPException(status_code=404, detail=f"Dataset has no image form: {key}")
    try:
        df = ds.load(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {key}") from exc
    if not 0 <= row_id < len(df):
        raise HTTPException(status_code=404, detail=f"Row out of range: {row_id}")
    return ds_images.pixels(df, key, row_id)


def _cached_payload(key: str) -> dict[str, Any] | None:
    payload = _cache_get(key)
    if payload is None and run_cache.is_hosting():
        payload = run_cache.load(key)
        if payload is not None:
            _cache_put(key, payload)
    return payload


def _build(req: AnalysisRequest, df: Any, key: str) -> None:
    """The expensive part, run on a job thread (see backend/jobs.py)."""
    config = _merge_config(req.config)
    try:
        tree = start_evaluation(df, req.feature_cols, config)  # type: ignore[arg-type]
    except Exception as exc:  # reaches the client as the job's error detail
        raise RuntimeError(f"Analysis failed: {exc}") from exc
    payload = {
        "meta": {
            "dataset": req.dataset,
            "feature_cols": req.feature_cols,
            "config": req.config,
            "n_total": len(df),
        },
        "tree": serialize_tree(tree),  # type: ignore[arg-type]
    }
    _cache_put(key, payload)
    if run_cache.is_hosting():
        run_cache.store(key, payload)  # a forced rerun replaces the stored entry


def _job_payload(job: jobs.Job) -> dict[str, Any]:
    if job.status == "running":
        return {"status": "running", "job_id": job.id}
    if job.status == "error":
        return {"status": "error", "job_id": job.id, "detail": job.detail}
    payload = _cache_get(job.key)
    if payload is None:  # only if the entry was evicted between finishing and polling
        return {
            "status": "error",
            "job_id": job.id,
            "detail": "Result no longer available — rerun.",
        }
    return {"status": "done", "job_id": job.id, **payload, "cached": False}


@app.post("/api/analysis")
def analysis(req: AnalysisRequest) -> dict[str, Any]:
    """Starts a build and returns a job id; the client polls /api/analysis/jobs/{id}.

    A cache hit still answers inline — only the runs that would outlast a proxy's
    response timeout go through the job path.
    """
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown dataset: {req.dataset}"
        ) from exc

    key = _cache_key(req.dataset, req.feature_cols, req.config)

    # `use_cache=False` bypasses both tiers, so the toggle really does recompute.
    if req.use_cache:
        payload = _cached_payload(key)
        if payload is not None:
            return {"status": "done", **payload, "cached": True}

    return _job_payload(jobs.submit(key, lambda: _build(req, df, key)))


@app.get("/api/analysis/jobs/{job_id}")
def analysis_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _job_payload(job)


@app.post("/api/predicate")
def predicate(req: PredicateRequest) -> dict[str, Any]:
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown dataset: {req.dataset}"
        ) from exc
    normalize = bool(req.config.get("normalize", True))
    try:
        return compute_predicate(
            df,
            req.feature_cols,
            normalize,
            req.row_indices,
            req.selected_local_indices,
            req.scope,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Predicate failed: {exc}") from exc


@app.post("/api/characteristics")
def characteristics(req: CharacteristicsRequest) -> dict[str, Any]:
    """A selection's characteristics, on the tree's own z-score baseline."""
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown dataset: {req.dataset}"
        ) from exc
    normalize = bool(req.config.get("normalize", True))
    try:
        return {
            "characteristics": compute_selection_characteristics(
                df,
                req.feature_cols,
                req.row_indices,
                req.selected_local_indices,
                normalize,
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Characteristics failed: {exc}"
        ) from exc


@app.post("/api/targets")
def targets(req: TargetsRequest) -> dict[str, Any]:
    """Label values for a selection — reported alongside, never inside, the predicate."""
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown dataset: {req.dataset}"
        ) from exc
    try:
        return compute_targets(
            df, req.target_cols, req.row_indices, req.selected_local_indices
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Targets failed: {exc}") from exc


@app.post("/api/rows")
def rows(req: RowsRequest) -> dict[str, Any]:
    """On-demand raw feature values for a set of row ids (selected-points table)."""
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown dataset: {req.dataset}"
        ) from exc
    # Ids are positions into *this* dataset's frame. A client holding a tree built on
    # another dataset sends ids that are perfectly valid integers and simply too large,
    # which pandas raises on — a bad request, not a server fault. Answer 400, as
    # every sibling endpoint does, rather than letting it surface as a 500 traceback.
    if req.ids and (max(req.ids) >= len(df) or min(req.ids) < 0):
        raise HTTPException(
            status_code=400,
            detail=f"Row ids out of range for {req.dataset} ({len(df)} rows)",
        )
    # `None` = every column; `[]` is a request for no columns, not for all of them.
    if req.columns is None:
        cols = [str(c) for c in df.columns]
    else:
        known = {str(c) for c in df.columns}
        unknown = [c for c in req.columns if c not in known]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown columns for {req.dataset}: {unknown[:5]}",
            )
        cols = req.columns
    sub = df.iloc[req.ids][cols]
    records = json.loads(
        sub.to_json(orient="records")
    )  # to_json coerces NaN->null, np types->native
    return {"columns": cols, "rows": records}


# Serve the built frontend (production single-container). Mounted last so /api/*
# routes above take precedence; html=True serves index.html at "/". No-op in dev
# (no dist/ yet) — the Vite dev server serves the frontend there.
_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")

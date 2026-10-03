"""FastAPI service for the D3 frontend.

Wraps the unchanged calc layer (`src/analysis`, `src/evaluation`) and serves the
JSON data contract emitted by `backend/serialize.py`. The tree is built on demand
and cached by (dataset, feature_cols, config); navigation/drill-down happens
client-side.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

# Make the repo root importable so `src.*` resolves when uvicorn is started from
# elsewhere; PYTHONPATH=. does the same thing for the documented dev command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import counterfactual as cf
from backend import datasets as ds
from backend import images as ds_images
from backend import jobs, movement_jobs, run_cache, run_listing, stored_run, tree_cache
from backend.characteristics import compute_selection_characteristics
from backend.counterfactual_apply import (
    ApplyRequestData,
    apply_validated,
    validate_apply,
)
from backend.movement import (
    MovementError,
    MovementRequestData,
    compute_validated_movement,
    needs_background,
    validate_request,
)
from backend.predicate import compute_predicate
from backend.serialize import (
    SCHEMA_VERSION,
    analysis_id,
    effective_config,
    serialize_tree,
)
from backend.targets import compute_targets
from src.config_defaults import default_config
from src.evaluation.evaluate import start_evaluation


@contextlib.asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    if run_cache.is_cache_only():  # index the stored runs before the first visitor asks
        threading.Thread(target=cached_runs, daemon=True).start()
    yield


app = FastAPI(title="HiLDE API", version="0.1.0", lifespan=_lifespan)

# Dev: Vite dev server (5173) calls the API cross-origin. Tightened in prod (single container).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class MovementRequest(BaseModel):
    analysis_id: str
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any] = {}
    node_id: str
    source_child_index: int
    target: dict[str, Any]
    strength: float | None = None


class CounterfactualApplyRequest(BaseModel):
    """`MovementRequest` with a required, positive strength (backend/counterfactual_apply.py)."""

    analysis_id: str
    dataset: str  # the base, or an existing counterfactual key to stack on
    feature_cols: list[str]
    config: dict[str, Any] = {}
    node_id: str
    source_child_index: int
    target: dict[str, Any]
    strength: float | None = None


class RowsRequest(BaseModel):
    dataset: str
    ids: list[int]
    columns: list[str] | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _load_dataset(key: str) -> Any:
    """`ds.load`, with the two ways a key can be unknown told apart: a base that
    does not exist, and a counterfactual id that was never registered or has
    expired (see backend/counterfactual.py)."""
    try:
        return ds.load(key)
    except cf.CounterfactualUnknown as exc:
        raise HTTPException(
            status_code=404, detail=f"Counterfactual expired or unknown: {key}"
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {key}") from exc


@app.get("/api/mode")
def mode() -> dict[str, Any]:
    """Whether the server runs in hosting mode (persistent run cache + UI banner),
    and whether it is restricted to the runs already in that cache."""
    hosting = run_cache.is_hosting()
    return {
        "hosting": hosting,
        "cache_dir": str(run_cache.cache_dir()) if hosting else None,
        "cache_only": run_cache.is_cache_only(),
        "maintenance": run_cache.is_maintenance(),
    }


_CACHE_ONLY_DETAIL = (
    "This server only serves precomputed runs — computing is disabled here."
)


@app.get("/api/cached-runs")
def cached_runs() -> list[dict[str, Any]]:
    """The datasets `/api/analysis` has stored runs of, with how many each."""
    return run_listing.summary(_cache_key, set(ds.dataset_keys()))


@app.get("/api/cached-runs/{dataset}")
def cached_runs_of(
    dataset: str, accept_encoding: Annotated[str, Header()] = ""
) -> Response:
    """The stored runs of one dataset, compact (`run_index.encode`); each config,
    sent back with its group's two, is a cache hit. Gzipped once, sent as is."""
    body = None
    if dataset in ds.dataset_keys():
        body = run_listing.encoded(_cache_key, dataset)
    if body is None:
        raise HTTPException(status_code=404, detail=f"No stored runs of {dataset}")
    if stored_run.accepts_gzip(accept_encoding):
        headers = {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"}
        return Response(body, media_type="application/json", headers=headers)
    return Response(gzip.decompress(body), media_type="application/json")


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, str]]:
    """Cheap listing — does not load any dataset."""
    return [{"key": k, "label": k} for k in ds.dataset_keys()]


@app.get("/api/datasets/{key}/columns")
def dataset_columns(key: str) -> dict[str, Any]:
    """Loads (and caches) the dataset to report its columns + default feature selection."""
    df = _load_dataset(key)
    return {
        "key": key,
        "n_rows": len(df),
        "columns": [str(c) for c in df.columns],
        "default_feature_cols": ds.default_feature_cols(df),
        # Non-null = rows can be rendered as images. A counterfactual key keeps
        # its base's image form.
        "image": ds_images.spec(ds.base_key(key)),
    }


@app.get("/api/datasets/{key}/image/{row_id}")
def dataset_image(key: str, row_id: int) -> dict[str, Any]:
    """Greyscale pixels of a single row, for the image-valued datasets."""
    base = ds.base_key(key)
    if ds_images.spec(base) is None:
        raise HTTPException(status_code=404, detail=f"Dataset has no image form: {key}")
    df = _load_dataset(key)
    if not 0 <= row_id < len(df):
        raise HTTPException(status_code=404, detail=f"Row out of range: {row_id}")
    return ds_images.pixels(df, base, row_id)


def _cached_payload(key: str) -> dict[str, Any] | None:
    payload = tree_cache.get(key)
    if payload is not None or not run_cache.is_hosting():
        return payload
    if run_cache.is_cache_only():  # byte-bounded, one parse at a time
        path = run_cache._path_for(key)
        size = path.stat().st_size if path.is_file() else 0
        return tree_cache.load(key, size, run_cache.load)
    payload = run_cache.load(key)
    if payload is not None:
        tree_cache.put(key, payload)
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
            # Movement prerequisites (backend/movement.py). `config` above is
            # what the client sent; `effective_config` is what the tree was
            # built with after `compute_analysis_tree` clamped it in place —
            # `start_evaluation` has returned, so `config` now holds the mutated
            # values, and that is the only config a refit may read.
            "analysis_id": analysis_id(req.dataset, req.feature_cols, req.config),
            "effective_config": effective_config(config),  # type: ignore[arg-type]
            "schema_version": SCHEMA_VERSION,
        },
        "tree": serialize_tree(tree),  # type: ignore[arg-type]
    }
    tree_cache.put(key, payload)
    # A counterfactual run is session state: hosting mode must never write it
    # to disk, where it would outlive the history that gives its key meaning.
    if run_cache.is_hosting() and not cf.is_counterfactual_key(req.dataset):
        run_cache.store(key, payload)  # a forced rerun replaces the stored entry


def _job_payload(job: jobs.Job) -> dict[str, Any]:
    if job.status == "running":
        return {"status": "running", "job_id": job.id}
    if job.status == "error":
        return {"status": "error", "job_id": job.id, "detail": job.detail}
    payload = tree_cache.get(job.key)
    if payload is None:  # only if the entry was evicted between finishing and polling
        return {
            "status": "error",
            "job_id": job.id,
            "detail": "Result no longer available — rerun.",
        }
    return {"status": "done", "job_id": job.id, **payload, "cached": False}


@app.post("/api/analysis", response_model=None)
def analysis(
    req: AnalysisRequest, accept_encoding: Annotated[str, Header()] = ""
) -> dict[str, Any] | Response:
    """Starts a build and returns a job id; the client polls /api/analysis/jobs/{id}.

    A cache hit still answers inline — only the runs that would outlast a proxy's
    response timeout go through the job path.
    """
    df = _load_dataset(req.dataset)

    key = _cache_key(req.dataset, req.feature_cols, req.config)

    # `use_cache=False` bypasses both tiers, so the toggle really does recompute.
    # Cache-only mode has nothing to recompute with: a hit is the only answer.
    cache_only = run_cache.is_cache_only()
    if cache_only:  # the stored bytes, never parsed (backend/stored_run.py)
        stored = stored_run.response(key, accept_encoding)
        if stored is not None:
            return stored
    elif req.use_cache:
        payload = _cached_payload(key)
        if payload is not None:
            return {"status": "done", **payload, "cached": True}
    if cache_only:
        raise HTTPException(
            status_code=409,
            detail=f"{_CACHE_ONLY_DETAIL} This dataset and configuration are not among them.",
        )

    return _job_payload(jobs.submit(key, lambda: _build(req, df, key)))


@app.get("/api/analysis/jobs/{job_id}")
def analysis_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return _job_payload(job)


@app.post("/api/predicate")
def predicate(req: PredicateRequest) -> dict[str, Any]:
    df = _load_dataset(req.dataset)
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


@app.post("/api/movement")
def movement(req: MovementRequest) -> dict[str, Any]:
    """A counterfactual preview: move one child cluster toward a point or a sibling.

    PCA answers inline, always — the displacement is one pinv solve against an
    already cached projection artifact, and it is what the strength slider calls;
    routing it through a poll would turn an instant interaction into a round
    trip. So does a UMAP request whose reducer is already in the artifact LRU.
    Only a UMAP request that still has to refit the node's projection becomes a
    job: the analysis discards its fitted reducers, so that first movement pays
    for a full UMAP fit, which can outlast a proxy's response timeout.

    Validation runs here either way, before any job is submitted, so a bad
    request is an HTTP 400/409 rather than a job that fails one poll later.

    Nothing is written back: the dataframe, the tree and the cached payload are
    all read-only here, and the movement result itself is never persisted (see
    backend/movement_jobs.py).

    The movement is answered against the run the client is looking at, which is
    the payload under this request's own cache key. `validate_request` compares
    the stored `analysis_id` with the requested one and answers 409 when they
    disagree, when the payload is gone, or when it predates `effective_config`.
    """
    df = _load_dataset(req.dataset)
    payload = _cached_payload(_cache_key(req.dataset, req.feature_cols, req.config))
    data = MovementRequestData(**req.model_dump())
    try:
        validated = validate_request(data, payload)
    except MovementError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if needs_background(validated):
        if run_cache.is_cache_only():  # the job would be a full UMAP fit
            raise HTTPException(
                status_code=409,
                detail=f"{_CACHE_ONLY_DETAIL} Moving a cluster of a UMAP run needs a refit.",
            )
        return movement_jobs.envelope(
            movement_jobs.submit(
                movement_jobs.job_key(data),
                lambda: compute_validated_movement(validated, df),
            )
        )
    try:
        return compute_validated_movement(validated, df)
    except MovementError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.get("/api/movement/jobs/{job_id}")
def movement_job(job_id: str) -> dict[str, Any]:
    """Poll a movement job. Terminal success returns the `MovementResponse` itself."""
    job = movement_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown movement job")
    return movement_jobs.envelope(job)


@app.post("/api/counterfactual/apply")
def counterfactual_apply(req: CounterfactualApplyRequest) -> dict[str, Any]:
    """Write the previewed movement into a counterfactual copy of the dataset.

    Same validation and the same job rule as `/api/movement`: the server
    recomputes the movement at the requested strength (a cold UMAP node goes
    through the job path — poll `/api/counterfactual/jobs/{job_id}`), then
    records the edit on top of `req.dataset` and answers with the new dataset
    key. Nothing about the original frame or any existing payload changes; the
    client rebuilds the analysis on the new key with `/api/analysis`.

    Refused in cache-only mode: that rebuild is exactly what the server cannot do.
    """
    if run_cache.is_cache_only():
        raise HTTPException(
            status_code=409,
            detail=f"{_CACHE_ONLY_DETAIL} Applying a movement needs a rebuild.",
        )
    df = _load_dataset(req.dataset)
    payload = _cached_payload(_cache_key(req.dataset, req.feature_cols, req.config))
    data = ApplyRequestData(**req.model_dump())
    try:
        validated = validate_apply(data, payload)
    except MovementError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if needs_background(validated):
        return movement_jobs.envelope(
            movement_jobs.submit(
                "counterfactual:" + movement_jobs.job_key(data.as_movement()),
                lambda: apply_validated(validated, df, req.dataset),
            )
        )
    try:
        return apply_validated(validated, df, req.dataset)
    except MovementError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.get("/api/counterfactual/jobs/{job_id}")
def counterfactual_job(job_id: str) -> dict[str, Any]:
    """Poll an Apply that needed a refit. Terminal success is the apply response itself."""
    job = movement_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown counterfactual job")
    return movement_jobs.envelope(job)


@app.get("/api/counterfactual/{cf_id}")
def counterfactual_chain(cf_id: str) -> dict[str, Any]:
    """The edit chain behind a counterfactual key, base first."""
    try:
        edits = cf.chain(cf_id)
    except cf.CounterfactualUnknown as exc:
        raise HTTPException(
            status_code=404, detail=f"Counterfactual expired or unknown: {cf_id}"
        ) from exc
    return {
        "cf_id": cf_id,
        "dataset_key": edits[-1].dataset_key,
        "base_dataset": edits[-1].base_dataset,
        "edits": [e.summary() for e in edits],
    }


@app.get("/api/counterfactual/{cf_id}/rows.csv")
def counterfactual_rows_csv(cf_id: str) -> PlainTextResponse:
    """The changed rows: row_id, then original and counterfactual values of
    every feature any edit in the chain touched."""
    try:
        base = cf.get(cf_id).base_dataset
        columns, records = cf.changed_rows(base, cf_id, ds.load)
    except cf.CounterfactualUnknown as exc:
        raise HTTPException(
            status_code=404, detail=f"Counterfactual expired or unknown: {cf_id}"
        ) from exc
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)
    return PlainTextResponse(
        out.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="counterfactual_{cf_id}.csv"'
        },
    )


@app.post("/api/characteristics")
def characteristics(req: CharacteristicsRequest) -> dict[str, Any]:
    """A selection's characteristics, on the tree's own z-score baseline."""
    df = _load_dataset(req.dataset)
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
    df = _load_dataset(req.dataset)
    try:
        return compute_targets(
            df, req.target_cols, req.row_indices, req.selected_local_indices
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Targets failed: {exc}") from exc


@app.post("/api/rows")
def rows(req: RowsRequest) -> dict[str, Any]:
    """On-demand raw feature values for a set of row ids (selected-points table)."""
    df = _load_dataset(req.dataset)
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

"""FastAPI service for the D3 frontend.

Wraps the unchanged calc layer (`src/analysis`, `src/evaluation`) and serves the
JSON data contract in PLAN.md. The tree is built on demand and cached by
(dataset, feature_cols, config); navigation/drill-down happens client-side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Make the repo root importable (mirrors src/ui/app.py) so `src.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import datasets as ds
from backend.predicate import compute_predicate
from backend.serialize import serialize_tree
from src.evaluation.evaluate import start_evaluation
from src.ui.state import init_state

app = FastAPI(title="SHD API", version="0.1.0")

# Dev: Vite dev server (5173) calls the API cross-origin. Tightened in prod (single container).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache serialized trees by request signature; tree build (UMAP+HDBSCAN+ZADU) is expensive.
_tree_cache: dict[str, dict[str, Any]] = {}


def _merge_config(partial: dict[str, Any]) -> dict[str, Any]:
    """Overlay the frontend's config knobs on the full default Config."""
    config = init_state(init_streamlit=False)
    config.update(partial)
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


class PredicateRequest(BaseModel):
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any] = {}
    row_indices: list[int]
    selected_local_indices: list[int]
    scope: str = "local"


class RowsRequest(BaseModel):
    dataset: str
    ids: list[int]
    columns: list[str] | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "default_feature_cols": ds.default_feature_cols(df),
    }


@app.post("/api/analysis")
def analysis(req: AnalysisRequest) -> dict[str, Any]:
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {req.dataset}") from exc

    key = _cache_key(req.dataset, req.feature_cols, req.config)
    if key not in _tree_cache:
        config = _merge_config(req.config)
        try:
            tree = start_evaluation(df, req.feature_cols, config)  # type: ignore[arg-type]
        except Exception as exc:  # surface calc-layer failures as 400s
            raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc
        _tree_cache[key] = {
            "meta": {
                "dataset": req.dataset,
                "feature_cols": req.feature_cols,
                "config": req.config,
                "n_total": int(len(df)),
            },
            "tree": serialize_tree(tree),  # type: ignore[arg-type]
        }
    return _tree_cache[key]


@app.post("/api/predicate")
def predicate(req: PredicateRequest) -> dict[str, Any]:
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {req.dataset}") from exc
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


@app.post("/api/rows")
def rows(req: RowsRequest) -> dict[str, Any]:
    """On-demand raw feature values for a set of row ids (selected-points table)."""
    try:
        df = ds.load(req.dataset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {req.dataset}") from exc
    cols = req.columns or [str(c) for c in df.columns]
    sub = df.iloc[req.ids][cols]
    records = json.loads(sub.to_json(orient="records"))  # to_json coerces NaN->null, np types->native
    return {"columns": cols, "rows": records}

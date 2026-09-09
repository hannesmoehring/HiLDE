"""Serialize the in-memory analysis tree into the JSON data contract.

The calc layer (`src/analysis`, `src/evaluation`) returns a nested tree of
TypedDicts carrying numpy arrays and pandas DataFrames — not JSON-serializable.
This module walks that tree and emits the `Node` schema; its TypeScript counterpart
is `frontend/src/types.ts`.

Nothing here mutates the calc layer; it only reads from it.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    """Coerce a scalar to a JSON-safe float; non-finite (NaN/Inf) -> None.

    Starlette's JSONResponse encodes with allow_nan=False, so NaN/Inf must be
    stripped before they reach the wire.
    """
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _float_list(arr: np.ndarray | None) -> list[float | None] | None:
    if arr is None:
        return None
    return [_finite(x) for x in np.asarray(arr).ravel()]


def _int_list(arr: np.ndarray | None) -> list[int] | None:
    if arr is None:
        return None
    return [int(x) for x in np.asarray(arr).ravel()]


def _xy_list(arr: np.ndarray | None) -> list[list[float | None]] | None:
    """Nx2 embedding -> [[x, y], ...]."""
    if arr is None:
        return None
    a = np.asarray(arr)
    return [[_finite(row[0]), _finite(row[1])] for row in a]


def _pair(pos: tuple[float, float] | None) -> list[float | None] | None:
    if pos is None:
        return None
    return [_finite(pos[0]), _finite(pos[1])]


def characteristic_records(rc: pd.DataFrame | list[Any] | None) -> list[dict[str, Any]]:
    """rel_characteristics DataFrame (index = feature, cols z_mean/z_std/raw_mean)
    -> list of records. Empty/None -> [].
    """
    if rc is None:
        return []
    if isinstance(rc, list):
        return rc
    if isinstance(rc, pd.DataFrame):
        if rc.empty:
            return []
        return [
            {
                "feature": str(feature),
                "z_mean": _finite(row.get("z_mean")),
                "z_std": _finite(row.get("z_std")),
                "raw_mean": _finite(row.get("raw_mean")),
                "is_feature": bool(row.get("is_feature", True)),
            }
            for feature, row in rc.iterrows()
        ]
    return []


def _scores(scores: dict[str, Any] | None) -> dict[str, Any] | None:
    """NodeScores is already native float/int/None, but guard non-finite floats."""
    if scores is None:
        return None
    return {
        "n_points": int(scores["n_points"]),
        "k": None if scores.get("k") is None else int(scores["k"]),
        "trustworthiness": _finite(scores.get("trustworthiness")),
        "continuity": _finite(scores.get("continuity")),
        "mrre_false": _finite(scores.get("mrre_false")),
        "mrre_missing": _finite(scores.get("mrre_missing")),
        "stress": _finite(scores.get("stress")),
        "cadi": _finite(scores.get("cadi")),
    }


def serialize_node(node: dict[str, Any], node_id: str, depth: int) -> dict[str, Any]:
    """Convert one tree node (HierarchyObject or ExplorationObject) to the Node schema.

    Internal nodes do not store `depth`; it is threaded through the recursion.
    Leaf nodes store their own `depth`, which we trust when present.
    """
    is_leaf = "is_leaf" in node
    out: dict[str, Any] = {
        "id": node_id,
        "is_leaf": is_leaf,
        "depth": int(node["depth"]) if is_leaf else depth,
        "n_points": len(node["row_indices"]),
        "row_indices": _int_list(node["row_indices"]),
        "embedding_original": _xy_list(node["embedding_original"]),
        "embedding_original_variance": _float_list(
            node.get("embedding_original_variance")
        ),
        "rel_position": _pair(node["rel_position"]),
        "rel_characteristics": characteristic_records(node["rel_characteristics"]),
        "scores": _scores(node.get("scores")),
    }
    if is_leaf:
        out["outlier_scores"] = None
        out["children"] = None
    else:
        out["outlier_scores"] = _float_list(node.get("outlier_scores"))
        children = node.get("next_object_layer") or []
        out["children"] = [
            serialize_node(child, f"{node_id}/{i}", depth + 1)
            for i, child in enumerate(children)
        ]
    return out


def serialize_tree(root: dict[str, Any]) -> dict[str, Any]:
    """Serialize the whole tree from the root. The root `scaler` is dropped (server-only)."""
    return serialize_node(root, "root", 0)


# ── Analysis identity, schema version, effective config ─────────────────────
# Prerequisites of the cluster-movement feature (see backend/movement.py). All
# three live on `meta`, never on a node: no existing serialized field changes
# shape or meaning.

SCHEMA_VERSION = 2
"""Version of the serialized analysis payload.

Version 1 is the implicit version of everything written before cluster
movement: those payloads carry no `analysis_id`, no `effective_config` and no
`schema_version`. That matters because `backend/run_cache.py` keys its on-disk
entries on (dataset, feature_cols, config) — *not* on the code — so a payload
written by an older build is still a cache hit today. `/api/movement` checks
this field and answers 409 (rebuild required) instead of refitting a reducer
from the raw request config.
"""

# The knobs a movement refit reads back. Everything else in `Config` is dropped
# deliberately:
#   * `selected_df` / `plot_df` are DataFrames, `cluster_labels` an ndarray,
#     `global_scaler` a fitted sklearn object and `analysis_tree` the tree
#     itself — none of them JSON-serializable, and none of them consulted by a
#     reducer fit.
#   * the clustering knobs (`hclust_*`, `hierarchical_layers`, `dbscan_eps`)
#     shape which rows land in which node, and the resulting tree is already
#     serialized in full; a refit projects a node's rows, it does not re-cluster.
#     `hclust_umap_n_components` is the other value `compute_analysis_tree`
#     clamps in place, but it only feeds the pre-clustering reduction, which no
#     node projection uses.
#   * `pca_components` is plumbed through the whole config but read nowhere —
#     node embeddings are hardcoded to 2D in `_embed_original`.
# `umap_n_neighbors` is the clamped knob that *does* reach a node reducer, which
# is the reason this dict exists at all.
_EFFECTIVE_CONFIG_KEYS = (
    "normalize",  # decides whether a root StandardScaler was fit at all
    "method",
    "umap_n_neighbors",  # clamped against the root size before any node was fit
    "umap_min_dist",
    "umap_random_state",
    "tsne_perplexity",
    "tsne_learning_rate",
    "tsne_random_state",
    "mds_metric",
    "mds_n_init",
    "mds_max_iter",
    "mds_random_state",
)


def _json_scalar(value: Any) -> Any:
    """Config values are bools/ints/floats/strings; coerce numpy scalars and drop
    anything else (a knob that is not a scalar cannot describe a reducer fit)."""
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _finite(value)
    return None


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """The refit-relevant knobs of the config the tree was ACTUALLY built with.

    Call this with the config *after* `compute_analysis_tree` returned:
    it clamps `hclust_umap_n_components` and `umap_n_neighbors` against the root
    size **in place**, before any per-node reducer is fit, so every visible
    embedding was produced with the clamped values. A movement refit that reads
    the client's raw config instead builds a different reducer whenever the clamp
    fired, and the alignment gate then rejects the movement for an invisible
    reason.
    """
    out: dict[str, Any] = {}
    for key in _EFFECTIVE_CONFIG_KEYS:
        if key in config:
            out[key] = _json_scalar(config[key])
    return out


def run_signature(dataset: str, feature_cols: list[str], config: dict[str, Any]) -> str:
    """Canonical signature string of a run.

    Deliberately the same shape as `backend/app.py::_cache_key`, which keys both
    the in-memory tree cache and the on-disk run cache — one identity for a run,
    not two. `backend/tests/test_movement.py` pins the two together.
    """
    return json.dumps(
        {"dataset": dataset, "feature_cols": feature_cols, "config": config},
        sort_keys=True,
        default=str,
    )


def analysis_id(dataset: str, feature_cols: list[str], config: dict[str, Any]) -> str:
    """Stable digest of (dataset, feature columns, requested config, schema version).

    The *requested* config is hashed, not the effective one, so the id is
    reproducible from a movement request alone — that is what lets
    `/api/movement` refuse a request aimed at a different run than the one the
    server holds. The schema version is part of the digest so a payload written
    before the movement fields existed can never collide with a current one.
    """
    signature = (
        f"{run_signature(dataset, feature_cols, config)}|schema={SCHEMA_VERSION}"
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()

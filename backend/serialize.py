"""Serialize the in-memory analysis tree into the JSON data contract.

The calc layer (`src/analysis`, `src/evaluation`) returns a nested tree of
TypedDicts carrying numpy arrays and pandas DataFrames — not JSON-serializable.
This module walks that tree and emits the `Node` schema documented in PLAN.md.

Nothing here mutates the calc layer; it only reads from it.
"""

from __future__ import annotations

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


def _characteristics(rc: pd.DataFrame | list[Any] | None) -> list[dict[str, Any]]:
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
        "n_points": int(len(node["row_indices"])),
        "row_indices": _int_list(node["row_indices"]),
        "embedding_original": _xy_list(node["embedding_original"]),
        "embedding_original_variance": _float_list(node.get("embedding_original_variance")),
        "rel_position": _pair(node["rel_position"]),
        "rel_characteristics": _characteristics(node["rel_characteristics"]),
        "scores": _scores(node.get("scores")),
    }
    if is_leaf:
        out["outlier_scores"] = None
        out["children"] = None
    else:
        out["outlier_scores"] = _float_list(node.get("outlier_scores"))
        children = node.get("next_object_layer") or []
        out["children"] = [
            serialize_node(child, f"{node_id}/{i}", depth + 1) for i, child in enumerate(children)
        ]
    return out


def serialize_tree(root: dict[str, Any]) -> dict[str, Any]:
    """Serialize the whole tree from the root. The root `scaler` is dropped (server-only)."""
    return serialize_node(root, "root", 0)

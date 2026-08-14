"""Target-column values for a selection.

The predicate is induced from the feature columns only — the `target_*` label
columns must stay out of it (see `backend/datasets.py::default_feature_cols`).
This module answers the separate question "what are the labels of the points I
just selected?", reported against the whole-dataset range so the selection has a
scale to sit in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.serialize import _finite


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as float. Booleans become 0/1; anything non-numeric becomes NaN."""
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _stats(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_min": None, f"{prefix}_max": None, f"{prefix}_mean": None}
    return {
        f"{prefix}_min": _finite(finite.min()),
        f"{prefix}_max": _finite(finite.max()),
        f"{prefix}_mean": _finite(finite.mean()),
    }


def compute_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    row_indices: list[int],
    selected_local_indices: list[int],
) -> dict[str, Any]:
    """Return {n_selected, targets} for the target-value bands.

    - `row_indices`: the node's rows into the source df.
    - `selected_local_indices`: indices into `row_indices` (0..N-1) from the lasso/box.

    `is_boolean` flags the one-hot `target_<class>` columns, whose mean is a class
    share rather than a magnitude — the frontend renders those two cases differently.
    """
    cols = [c for c in target_cols if c in df.columns]
    row_idx = np.asarray(row_indices, dtype=int)
    sel_idx = row_idx[np.asarray(selected_local_indices, dtype=int)]

    targets: list[dict[str, Any]] = []
    for col in cols:
        values = _column(df, col)
        finite = values[np.isfinite(values)]
        targets.append(
            {
                "feature": col,
                "is_boolean": bool(
                    finite.size > 0 and np.isin(finite, (0.0, 1.0)).all()
                ),
                **_stats(values[sel_idx], "sel"),
                **_stats(values, "global"),
            }
        )
    return {"n_selected": int(sel_idx.size), "targets": targets}

"""Characteristics of a selection, read against the space it was made in.

The tree stores one `rel_characteristics` frame per cluster, contrasting that
cluster with its parent's space. Exploration needs the same contrast for an
arbitrary lasso selection inside a node, so this reuses the unchanged calc layer
(`compute_cluster_characteristics`) with a two-label split — selected vs. the rest
of the node — instead of a cluster id.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.serialize import characteristic_records
from src.analysis.characteristics import compute_cluster_characteristics

_SELECTED = 1


def nonfeature_cols(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Numeric columns the user did not pick as features.

    Same rule as analysis_routine, so the exploration panel reports the same set of
    extra columns the tree's characteristics do.
    """
    return [
        str(c)
        for c in df.columns
        if c not in feature_cols and c != "row_id" and pd.api.types.is_numeric_dtype(df[c])
    ]


def compute_selection_characteristics(
    df: pd.DataFrame,
    feature_cols: list[str],
    row_indices: list[int],
    selected_local_indices: list[int],
) -> list[dict[str, Any]]:
    """Return characteristic records for the selection, z-scored within the node.

    - `row_indices`: the explored node's rows into the source df — the space.
    - `selected_local_indices`: indices into `row_indices` (0..N-1) from the lasso.

    The scaler is fit on the node's rows only, so a z-score reads "relative to this
    space", matching the local-scope predicate rather than the whole dataset.
    """
    row_idx = np.asarray(row_indices, dtype=int)
    sel = np.asarray(selected_local_indices, dtype=int)
    if sel.size == 0:
        return []

    sub = df.iloc[row_idx]
    extra_cols = nonfeature_cols(df, feature_cols)

    labels = np.zeros(len(row_idx), dtype=int)
    labels[sel] = _SELECTED

    # Both frames carry a plain RangeIndex so the in-cluster mask lines up across them.
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(sub[feature_cols].to_numpy()),
        columns=feature_cols,
    )
    scaled["cluster"] = labels

    raw = sub[feature_cols + extra_cols].reset_index(drop=True)
    raw["cluster"] = labels

    rows = compute_cluster_characteristics(
        cluster_id=_SELECTED,
        df=raw,
        X_scaled_df=scaled,
        feature_cols=feature_cols,
        extra_cols=extra_cols,
    )
    return characteristic_records(rows)

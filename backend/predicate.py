"""Selection-time predicate induction, wrapping the unchanged calc layer.

Ported from the former `src/ui/components/exploration.py::render_range_analysis`
+ `_global_predicate_inputs`, but pure (no Streamlit): given a node's row indices
and a within-node selection, reproduce the local/global scaling and run
`generate_predicate("db", ...)` at RCM 1.0 (full) and 0.9 (trimmed core).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.serialize import _finite
from src.analysis.predicate_generator import generate_predicate


def _sanitized(row: dict[str, Any]) -> dict[str, Any]:
    """NaN/Inf -> None, as every other payload module does.

    Starlette encodes with allow_nan=False, and the failure happens while the
    response is serialized — outside this module's caller's try — so an all-NaN
    feature answered 500 instead of a payload.
    """
    return {k: (_finite(v) if isinstance(v, float) else v) for k, v in row.items()}


def compute_predicate(
    df: pd.DataFrame,
    feature_cols: list[str],
    normalize: bool,
    row_indices: list[int],
    selected_local_indices: list[int],
    scope: str,
) -> dict[str, Any]:
    """Return {full, trimmed, summary} for the feature-range band chart.

    - `row_indices`: the node's rows into the source df (frontend already holds these).
    - `selected_local_indices`: indices into `row_indices` (0..N-1) from the lasso/box.
    - `scope`: "local" (scale within the node) or "global" (whole-dataset scaler).
    - `normalize`: config["normalize"] — drives whether the global scaler is applied.
    """
    row_idx = np.asarray(row_indices, dtype=int)
    sel = np.asarray(selected_local_indices, dtype=int)

    sub_X = df.iloc[row_idx][feature_cols].to_numpy()
    # Leaf-local scaler is always fit fresh (parity with compute_data_layer).
    X_scaled_local = StandardScaler().fit_transform(sub_X)

    if scope == "global":
        all_features = df[feature_cols].to_numpy()
        global_scaler = StandardScaler().fit(all_features) if normalize else None
        background = global_scaler.transform(all_features) if global_scaler is not None else all_features
        global_sel_rows = row_idx[sel].tolist()
        sel_scaled = background[global_sel_rows]
        sel_idx: list[int] = global_sel_rows
    else:
        background = X_scaled_local
        sel_scaled = X_scaled_local[sel]
        sel_idx = sel.tolist()

    if sel_scaled.shape[0] == 0:
        return {"full": [], "trimmed": [], "summary": None}

    selected_scaled_df = pd.DataFrame(sel_scaled, columns=feature_cols)
    full = generate_predicate("db", selected_scaled_df, background, threshold=1.0, selected_indices=sel_idx)
    trimmed = generate_predicate("db", selected_scaled_df, background, threshold=0.9, selected_indices=sel_idx)

    full_rows: list[dict[str, Any]] = [_sanitized(r) for r in full]  # type: ignore[union-attr]
    trimmed_rows: list[dict[str, Any]] = [_sanitized(r) for r in trimmed]  # type: ignore[union-attr]

    # Not sanitized: `_f1` returns 0.0 for every degenerate denominator, so this is
    # always finite — and the client formats it with .toFixed, which a null breaks.
    predicate_f1 = float(full_rows[0]["predicate_f1"]) if full_rows else 0.0
    n_clauses = sum(1 for r in full_rows if r.get("in_predicate"))
    summary = {
        "predicate_f1": predicate_f1,
        "n_features_used": n_clauses,
        "n_features_total": len(full_rows),
        "n_selected": int(sel.shape[0]),
    }
    return {"full": full_rows, "trimmed": trimmed_rows, "summary": summary}

import numpy as np
import pandas as pd


def generate_predicate(
    method: str,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    threshold: float = 0.9,
    selected_indices: list[int] | None = None,
    tail_split: str = "severity",
) -> object:
    match method:
        case "hm":
            return _predicate_hm(df, X_scaled, threshold=threshold, tail_split=tail_split)
        case "threshold":
            return _predicate_threshold(df, X_scaled, threshold=threshold, tail_split=tail_split)
        case "db":
            return _predicate_db(df, X_scaled, threshold=threshold, selected_indices=selected_indices, tail_split=tail_split)
        case _:
            raise ValueError(f"Unknown predicate generation method: {method}")


def _validate_threshold(threshold: float) -> None:
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")


def _tail_removal_shares(values: np.ndarray, median: float) -> tuple[float, float]:
    left_tail = values[values < median]
    right_tail = values[values > median]

    left_severity = median - np.min(left_tail) if left_tail.size else 0.0
    right_severity = np.max(right_tail) - median if right_tail.size else 0.0
    total_severity = left_severity + right_severity

    if total_severity <= 0:
        return 0.5, 0.5

    return left_severity / total_severity, right_severity / total_severity


def _build_range_row(
    values: np.ndarray,
    full_values: np.ndarray,
    feature: str,
    threshold: float,
    tail_split: str = "severity",
) -> dict[str, object]:
    median = float(np.median(values))
    total_trim = 1.0 - threshold
    if tail_split == "severity":
        left_share, right_share = _tail_removal_shares(values, median)
    elif tail_split == "symmetric":
        left_share, right_share = 0.5, 0.5
    else:
        raise ValueError(f"Unknown tail split: {tail_split}")
    left_trim = total_trim * left_share
    right_trim = total_trim * right_share

    sel_min = float(np.quantile(values, left_trim))
    sel_max = float(np.quantile(values, 1.0 - right_trim))
    global_min = float(np.min(full_values))
    global_max = float(np.max(full_values))

    return {
        "sel_min": sel_min,
        "sel_max": sel_max,
        "sel_range": sel_max - sel_min,
        "feature": feature,
        "global_min": global_min,
        "global_max": global_max,
    }


def _f1(pred_mask: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """F1, precision and recall between a predicate's membership and the selection labels."""
    true_positive = int(np.count_nonzero(pred_mask & y))
    predicted_positive = int(np.count_nonzero(pred_mask))
    actual_positive = int(np.count_nonzero(y))

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def _predicate_db(
    df: pd.DataFrame,
    X_scaled_full: np.ndarray,
    threshold: float = 0.9,
    selected_indices: list[int] | None = None,
    tail_split: str = "severity",
) -> list[dict[str, object]]:
    """DimBridge-style predicate induction.

    Explains the selection (pattern points ``P``) against the full dataset
    (background points ``B``) by greedily building a conjunction of per-feature
    interval clauses, adding the clause that most improves the F1 between predicate
    membership and the selection labels (Recursive Predicate Induction, paper §5.2).
    """
    _validate_threshold(threshold)
    df = df.copy()
    selected = df.to_numpy()
    if selected.size == 0:
        return []

    features = [str(c) for c in df.columns]
    rows = [
        _build_range_row(
            values=selected[:, j],
            full_values=X_scaled_full[:, j],
            feature=features[j],
            threshold=threshold,
            tail_split=tail_split,
        )
        for j in range(selected.shape[1])
    ]

    # Per-feature clause membership over the full dataset.
    clause_masks = [
        (X_scaled_full[:, j] >= rows[j]["sel_min"]) & (X_scaled_full[:, j] <= rows[j]["sel_max"])
        for j in range(len(rows))
    ]

    # Without selection labels we can only report the marginal ranges (no scoring).
    if selected_indices is None:
        for row in rows:
            row.update(
                clause_f1=0.0,
                clause_precision=0.0,
                clause_recall=0.0,
                in_predicate=False,
                predicate_step=None,
                predicate_f1=0.0,
            )
        return rows

    y = np.zeros(X_scaled_full.shape[0], dtype=bool)
    y[np.asarray(selected_indices, dtype=int)] = True

    for j, row in enumerate(rows):
        clause_f1, clause_precision, clause_recall = _f1(clause_masks[j], y)
        row.update(
            clause_f1=clause_f1,
            clause_precision=clause_precision,
            clause_recall=clause_recall,
            in_predicate=False,
            predicate_step=None,
        )

    # Greedy conjunction: keep adding the clause that most improves F1.
    current_mask = np.ones(X_scaled_full.shape[0], dtype=bool)
    best_f1 = _f1(current_mask, y)[0]
    remaining = set(range(len(rows)))
    step = 0

    while remaining:
        scored = [(_f1(current_mask & clause_masks[j], y)[0], j) for j in remaining]
        candidate_f1, best_j = max(scored, key=lambda s: s[0])
        if candidate_f1 <= best_f1 + 1e-6:
            break
        current_mask &= clause_masks[best_j]
        best_f1 = candidate_f1
        rows[best_j]["in_predicate"] = True
        rows[best_j]["predicate_step"] = step
        remaining.discard(best_j)
        step += 1

    for row in rows:
        row["predicate_f1"] = best_f1

    return rows


def _predicate_hm(df: pd.DataFrame, X_scaled_full: np.ndarray, threshold: float = 0.9, tail_split: str = "severity") -> list[dict[str, object]]:
    return _predicate_threshold(df, X_scaled_full, threshold=threshold, tail_split=tail_split)


def _predicate_threshold(
    df: pd.DataFrame, X_scaled_full: np.ndarray, threshold: float = 0.9, tail_split: str = "severity"
) -> list[dict[str, object]]:
    _validate_threshold(threshold)
    df = df.copy()
    X_scaled = df.to_numpy()
    if X_scaled.size == 0:
        return []

    return [
        _build_range_row(
            values=X_scaled[:, i],
            full_values=X_scaled_full[:, i],
            feature=str(df.columns[i]),
            threshold=threshold,
            tail_split=tail_split,
        )
        for i in range(X_scaled.shape[1])
    ]

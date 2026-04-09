import numpy as np
import pandas as pd


def generate_predicate(
    method: str,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    threshold: float = 0.9,
) -> object:
    match method:
        case "hm":
            return _predicate_hm(df, X_scaled, threshold=threshold)
        case "threshold":
            return _predicate_threshold(df, X_scaled, threshold=threshold)
        case "db":
            ...
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
) -> dict[str, object]:
    median = float(np.median(values))
    total_trim = 1.0 - threshold
    left_share, right_share = _tail_removal_shares(values, median)
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


def _predicate_hm(df: pd.DataFrame, X_scaled_full: np.ndarray, threshold: float = 0.9) -> list[dict[str, object]]:
    return _predicate_threshold(df, X_scaled_full, threshold=threshold)


def _predicate_threshold(df: pd.DataFrame, X_scaled_full: np.ndarray, threshold: float = 0.9) -> list[dict[str, object]]:
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
        )
        for i in range(X_scaled.shape[1])
    ]

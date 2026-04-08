import numpy as np
import pandas as pd


def generate_predicate(method: str, df: pd.DataFrame, X_scaled: np.ndarray) -> object:
    match method:
        case "hm":
            return _predicate_hm(df, X_scaled)
        case "db":
            ...
        case _:
            raise ValueError(f"Unknown predicate generation method: {method}")


def _predicate_hm(df: pd.DataFrame, X_scaled_full: np.ndarray) -> list[dict[str, object]]:
    df = df.copy()
    X_scaled = df.to_numpy()
    ranges: list = []
    for i in range(X_scaled.shape[1]):
        col_min = np.min(X_scaled[:, i])
        col_max = np.max(X_scaled[:, i])
        global_min = np.min(X_scaled_full[:, i])
        global_max = np.max(X_scaled_full[:, i])
        ranges.append(
            {
                "sel_min": col_min,
                "sel_max": col_max,
                "sel_range": col_max - col_min,
                "feature": df.columns[i],
                "global_min": global_min,
                "global_max": global_max,
            },
        )

    return ranges

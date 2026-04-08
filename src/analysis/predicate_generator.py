import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_predicate(method: str, df: pd.DataFrame):
    match method:
        case "hm":
            ...
        case "db":
            ...
        case _:
            raise ValueError(f"Unknown predicate generation method: {method}")


def _predicate_hm(df: pd.DataFrame):
    scaler = StandardScaler()

    df: pd.DataFrame = df.copy()[:-3]
    X = df.to_numpy()
    X_scaled = scaler.fit_transform(X)
    ranges: list = []
    for i in range(X_scaled.shape[1]):
        col_min = np.min(X_scaled[:, i])
        col_max = np.max(X_scaled[:, i])
        ranges.append((col_min, col_max, col_max - col_min))

    return ranges

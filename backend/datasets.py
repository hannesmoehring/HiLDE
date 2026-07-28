"""Thin access to the (Streamlit-free) dataset registry.

Reuses the loaders in `src/datasets.py`. Those are `functools.cache`-memoized, so
each loader runs at most once per process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.datasets import DATASETS

if TYPE_CHECKING:
    import pandas as pd


def dataset_keys() -> list[str]:
    return list(DATASETS)


def load(key: str) -> "pd.DataFrame":
    if key not in DATASETS:
        raise KeyError(key)
    return DATASETS[key]()


def default_feature_cols(df: "pd.DataFrame") -> list[str]:
    """Match the Streamlit default: every column except `row_id`."""
    return [c for c in df.columns if c != "row_id"]

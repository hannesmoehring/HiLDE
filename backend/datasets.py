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


def load(key: str) -> pd.DataFrame:
    if key not in DATASETS:
        raise KeyError(key)
    return DATASETS[key]()


def default_feature_cols(df: pd.DataFrame) -> list[str]:
    """Every column except `row_id` and the `target_*` label columns.

    Labels must stay out of the feature space: clustering, the per-node projection,
    the DR quality scores and the induced predicates would otherwise all run on a
    space containing the ground truth. They remain in the frame and surface as
    non-feature characteristics.
    """
    return [c for c in df.columns if c != "row_id" and not str(c).startswith("target_")]

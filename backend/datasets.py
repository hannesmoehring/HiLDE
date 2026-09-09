"""Thin access to the (Streamlit-free) dataset registry.

Reuses the loaders in `src/datasets.py`. Those are `functools.cache`-memoized, so
each loader runs at most once per process.

A key may also name a counterfactual copy — `"{base}@cf:{cf_id}"`, see
`backend/counterfactual.py` — which resolves through the same function, so every
endpoint that loads a dataset works on a counterfactual run unchanged. Only the
plain keys are listed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend import counterfactual as cf
from src.datasets import DATASETS

if TYPE_CHECKING:
    import pandas as pd


def dataset_keys() -> list[str]:
    return list(DATASETS)


def base_key(key: str) -> str:
    """The plain dataset a (possibly counterfactual) key is built on."""
    return cf.split_key(key)[0]


def _load_base(base: str) -> pd.DataFrame:
    if base not in DATASETS:
        raise KeyError(base)
    return DATASETS[base]()


def load(key: str) -> pd.DataFrame:
    """The frame for a plain key, or the materialized frame for a counterfactual
    key. `KeyError` for an unknown base; `cf.CounterfactualUnknown` (a KeyError)
    for a counterfactual id that was never registered or has expired."""
    base, cf_id = cf.split_key(key)
    if cf_id is None:
        return _load_base(base)
    if base not in DATASETS:
        raise KeyError(key)
    return cf.resolve_frame(base, cf_id, _load_base)


def default_feature_cols(df: pd.DataFrame) -> list[str]:
    """Every column except `row_id` and the `target_*` label columns.

    Labels must stay out of the feature space: clustering, the per-node projection,
    the DR quality scores and the induced predicates would otherwise all run on a
    space containing the ground truth. They remain in the frame and surface as
    non-feature characteristics.
    """
    return [c for c in df.columns if c != "row_id" and not str(c).startswith("target_")]

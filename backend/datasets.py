"""Thin, framework-free access to the dataset registry.

Reuses the loaders in `src/ui/data.py` (do not duplicate loader logic). Those are
`@st.cache_data`-decorated; calling them outside a Streamlit runtime works but is
noisy, so we memoize results here and call each loader at most once per process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ui.data import DATASETS

if TYPE_CHECKING:
    import pandas as pd

# Display label -> difficulty marker embedded in the registry key, e.g. "(Low)".
_loaded: dict[str, "pd.DataFrame"] = {}


def dataset_keys() -> list[str]:
    return list(DATASETS)


def load(key: str) -> "pd.DataFrame":
    if key not in DATASETS:
        raise KeyError(key)
    if key not in _loaded:
        _loaded[key] = DATASETS[key]()
    return _loaded[key]


def default_feature_cols(df: "pd.DataFrame") -> list[str]:
    """Match the Streamlit default: every column except `row_id`."""
    return [c for c in df.columns if c != "row_id"]

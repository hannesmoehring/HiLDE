"""Streamlit-free analysis config defaults.

Extracted from the former `src/ui/state.py::init_state` so the backend can build
a full `Config` without importing Streamlit. Values are identical to the UI
defaults; the calc layer overlays request-supplied knobs on top of these.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.types import Config


def default_config() -> Config:
    """A fresh, complete default Config (the calc layer mutates config in place,
    so callers must not share a single instance)."""
    defaults: dict[str, object] = {
        "dataset_choice": "Wine quality (Low)",
        "characteristics_non_feature_only": False,
        "selected_indices": [],
        "selected_df": pd.DataFrame(),
        "latest_selection_config": None,
        "plot_df": pd.DataFrame(),
        "interactive_ranges_mode": False,
        "interactive_features": [],
        "predicate_scope": "local",
        "clusters_in_original_space": False,
        "cluster_method": "HDBSCAN",
        "cluster_n_clusters": 5,
        "cluster_labels": np.array([], dtype=int),
        "hierarchical_mode": True,
        "hierarchical_layers": 1,
        "hclust_normalize": True,
        "hclust_umap_n_components": 2,
        "hclust_min_samples": 5,
        "hclust_min_cluster_size": 25,
        "dbscan_eps": 0.5,
        "analysis_tree": None,
        "tree_path": [],
        "global_scaler": None,
        "cluster_pca_x_component": 0,
        "cluster_pca_y_component": 1,
        "method": "UMAP",
        "normalize": True,
        "pca_components": 4,
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_random_state": 42,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_random_state": 42,
        "mds_metric": True,
        "mds_n_init": 2,
        "mds_max_iter": 100,
        "mds_random_state": 42,
    }
    return cast("Config", defaults)


def init_state(init_streamlit: bool = False) -> Config:  # noqa: ARG001 — kept for back-compat
    """Back-compat shim for callers of the former `src.ui.state.init_state`
    (e.g. src_research scripts). The Streamlit-session behaviour is gone; this
    just returns the default config."""
    return default_config()

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import pandas as pd
import streamlit as st
from src.types import Config


def init_state(init_streamlit: bool = True) -> Config:
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
        # hierarchical config panel
        "hclust_normalize": True,
        "hclust_umap_n_components": 2,
        "hclust_min_samples": 5,
        "hclust_min_cluster_size": 15,
        "dbscan_eps": 0.5,
        # pre-computed analysis tree + navigation path
        "analysis_tree": None,
        "tree_path": [],
        # scaler fit once during tree build (for the global predicate scope)
        "global_scaler": None,
        # cluster-level exploration PCA axis picker
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
    }
    if init_streamlit:
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    return cast("Config", dict(defaults))


def current_config() -> Config:
    # Return a copy: downstream analysis mutates the config in place, and writing
    # widget-backed keys (e.g. hclust_umap_n_components) back into session_state raises.
    config: Config = cast("Config", dict(st.session_state))
    return config


def get_selected_indices(event: object) -> list[int]:
    points: list[dict[str, object]] = []

    if event is None:
        return []

    if hasattr(event, "selection") and isinstance(event.selection, dict):
        selection = cast("dict[str, object]", event.selection)
        points_obj = selection.get("points")
        if isinstance(points_obj, list):
            points = []
            for point in points_obj:
                if isinstance(point, dict):
                    points.append(cast("dict[str, object]", point))
    elif isinstance(event, dict):
        event_dict = cast("dict[str, object]", event)
        selection_obj = event_dict.get("selection")
        if isinstance(selection_obj, dict):
            selection_dict = cast("dict[str, object]", selection_obj)
            points_obj = selection_dict.get("points")
            if isinstance(points_obj, list):
                points = []
                for point in points_obj:
                    if isinstance(point, dict):
                        points.append(cast("dict[str, object]", point))

    indices: list[int] = []
    for point in points:
        point_index = point.get("point_index")
        point_number = point.get("pointNumber")
        if isinstance(point_index, int):
            indices.append(point_index)
        elif isinstance(point_number, int):
            indices.append(point_number)

    return indices

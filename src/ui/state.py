from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import pandas as pd
import streamlit as st


class ReductionConfig(TypedDict):
    method: str
    normalize: bool
    pca_components: int
    pca_x_component: int | None
    pca_y_component: int | None
    tsne_perplexity: float
    tsne_learning_rate: float
    tsne_random_state: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_random_state: int


def init_state() -> None:
    defaults: dict[str, object] = {
        "selected_indices": [],
        "selected_df": pd.DataFrame(),
        "latest_selection_config": None,
        "plot_df": pd.DataFrame(),
        "embedding_full": np.empty((0, 0), dtype=float),
        "explained_variance_ratio": np.array([], dtype=float),
        "computed_method": None,
        "interactive_ranges_mode": False,
        "interactive_features": [],
        "clusters_in_original_space": False,
        "cluster_method": "KMeans",
        "cluster_n_clusters": 5,
        "cluster_labels": np.array([], dtype=int),
        "hierarchical_mode": True,
        "hierarchical_layers": 1,
        # hierarchical config panel
        "hclust_normalize": True,
        "hclust_umap_n_components": 2,
        "hclust_min_samples": 5,
        "hclust_min_cluster_size": 15,
        # pre-computed analysis tree + navigation path
        "analysis_tree": None,
        "tree_path": [],
        "cluster_path_for_embed": (),
        # cluster-level exploration embedding
        "cluster_embedding_full": np.empty((0, 0), dtype=float),
        "cluster_explained_variance": np.array([], dtype=float),
        "cluster_pca_x_component": 0,
        "cluster_pca_y_component": 1,
        "method": "UMAP",
        "normalize": True,
        "pca_components": 4,
        "pca_x_component": 0,
        "pca_y_component": 1,
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_random_state": 42,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_random_state": 42,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_config() -> ReductionConfig:
    return {
        "method": st.session_state["method"],
        "normalize": st.session_state["normalize"],
        "pca_components": st.session_state["pca_components"],
        "pca_x_component": st.session_state.get("pca_x_component"),
        "pca_y_component": st.session_state.get("pca_y_component"),
        "tsne_perplexity": st.session_state["tsne_perplexity"],
        "tsne_learning_rate": st.session_state["tsne_learning_rate"],
        "tsne_random_state": st.session_state["tsne_random_state"],
        "umap_n_neighbors": st.session_state["umap_n_neighbors"],
        "umap_min_dist": st.session_state["umap_min_dist"],
        "umap_random_state": st.session_state["umap_random_state"],
    }


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

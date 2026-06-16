from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.analysis_routine import compute_analysis_tree
from src.ui.data import compute_embedding
from src.ui.state import current_config
from src.ui.tree_nav import get_node_at_path

_MIN_COMPONENTS_FOR_2D = 2


def _build_tree_config() -> dict:
    method = st.session_state["method"]
    config: dict = {
        "normalize": st.session_state["hclust_normalize"],
        "hierarchical_layers": int(st.session_state["hierarchical_layers"]),
        "min_cluster_size": int(st.session_state["hclust_min_cluster_size"]),
        "min_samples": int(st.session_state["hclust_min_samples"]),
        "kde_dr_method": method,
    }
    if method == "t-SNE":
        config["perplexity"] = float(st.session_state["tsne_perplexity"])
        config["learning_rate"] = float(st.session_state["tsne_learning_rate"])
    elif method == "UMAP":
        config["n_neighbors"] = int(st.session_state["umap_n_neighbors"])
        config["min_dist"] = float(st.session_state["umap_min_dist"])
    return config


def handle_hierarchical_save(df: pd.DataFrame, feature_columns: list[str]) -> None:
    config = _build_tree_config()
    with st.spinner("Computing analysis tree…"):
        tree = compute_analysis_tree(df, feature_columns, config)

    st.session_state["analysis_tree"] = tree
    st.session_state["tree_path"] = []
    st.session_state["cluster_embedding_full"] = np.empty((0, 0), dtype=float)
    st.session_state["cluster_explained_variance"] = np.array([], dtype=float)
    st.session_state["cluster_path_for_embed"] = ()
    st.session_state["selected_indices"] = []
    st.session_state["selected_df"] = pd.DataFrame()


def handle_exploration_save() -> None:
    root = st.session_state.get("analysis_tree")
    if root is None:
        return
    tree_path: list[int] = st.session_state.get("tree_path", [])
    n_layers = int(st.session_state["hierarchical_layers"])
    path = tree_path[:n_layers]
    if not path:
        return

    leaf = get_node_at_path(root, path)
    sub_X = leaf.get("exploration_points") if "is_leaf" in leaf else leaf.get("cluster_points")  # type: ignore[union-attr]
    if sub_X is None or len(sub_X) == 0:
        return

    with st.spinner("Computing cluster embedding…"):
        result = compute_embedding(method=st.session_state.method, X=sub_X, config=current_config())

    st.session_state["cluster_embedding_full"] = result.embedding
    st.session_state["cluster_explained_variance"] = (
        result.explained_variance_ratio if result.explained_variance_ratio is not None else np.array([], dtype=float)
    )

    if st.session_state.method == "PCA" and st.session_state["cluster_explained_variance"].size >= _MIN_COMPONENTS_FOR_2D:
        ordered = np.argsort(st.session_state["cluster_explained_variance"])[::-1]
        st.session_state["cluster_pca_x_component"] = int(ordered[0])
        st.session_state["cluster_pca_y_component"] = int(ordered[1])

    st.session_state["cluster_path_for_embed"] = tuple(path)

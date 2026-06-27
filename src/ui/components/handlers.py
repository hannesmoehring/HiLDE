from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.evaluate import start_evaluation
from src.types import Config
from src.ui.data import compute_embedding
from src.ui.state import current_config
from src.ui.tree_nav import get_node_at_path

_MIN_COMPONENTS_FOR_2D = 2


def handle_hierarchical_save(df: pd.DataFrame, feature_columns: list[str]) -> None:
    config: Config = current_config()
    with st.spinner("Computing analysis tree & quality scores…"):
        tree = start_evaluation(df, feature_columns, config)

    st.session_state["analysis_tree"] = tree
    st.session_state["tree_path"] = []
    st.session_state["cluster_embedding_full"] = np.empty((0, 0), dtype=float)
    st.session_state["cluster_explained_variance"] = np.array([], dtype=float)
    st.session_state["cluster_path_for_embed"] = ()
    st.session_state["selected_indices"] = []
    st.session_state["selected_df"] = pd.DataFrame()


def handle_exploration_save(df: pd.DataFrame, feature_columns: list[str]) -> None:
    root = st.session_state.get("analysis_tree")
    if root is None:
        return
    tree_path: list[int] = st.session_state.get("tree_path", [])
    n_layers = int(st.session_state["hierarchical_layers"])
    path = tree_path[:n_layers]
    if not path:
        return

    leaf = get_node_at_path(root, path)
    row_indices = leaf["row_indices"]  # type: ignore[index]
    sub_X = df.iloc[row_indices][feature_columns].to_numpy()
    if len(sub_X) == 0:
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

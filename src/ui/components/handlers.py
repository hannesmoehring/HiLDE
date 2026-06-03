from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.ui.data import compute_embedding, get_path_subset, run_hierarchical_clustering
from src.ui.state import current_config
from src.ui.visualization import cluster_characteristics, cluster_gauss_kde

_MIN_COMPONENTS_FOR_2D = 2


def handle_hierarchical_save(df: pd.DataFrame, _X: np.ndarray, feature_columns: list[str]) -> None:

    X_hc = df[feature_columns].to_numpy()
    if st.session_state["normalize"]:
        X_hc = StandardScaler().fit_transform(X_hc)

    with st.spinner("Computing hierarchical clusters…"):
        layout_df, h_labels, n_outliers, glosh_scores = run_hierarchical_clustering(
            df,
            X_hc,
            feature_columns,
            n_components=st.session_state["hclust_umap_n_components"],
            min_samples=st.session_state["hclust_min_samples"],
            min_cluster_size=st.session_state["hclust_min_cluster_size"],
            min_dist=st.session_state["umap_min_dist"],
            n_neighbors=st.session_state["umap_n_neighbors"],
        )

    st.session_state["hierarchical_layout_df"] = layout_df
    st.session_state["hierarchical_labels"] = h_labels
    st.session_state["hierarchical_n_outliers"] = n_outliers
    st.session_state["hierarchical_glosh_scores"] = glosh_scores
    st.session_state["selected_cluster_id"] = None
    st.session_state["cluster_selected_id_for_embed"] = None
    st.session_state["cluster_embedding_full"] = np.empty((0, 0), dtype=float)
    st.session_state["hierarchical_selection_stack"] = []
    st.session_state["hierarchical_sublevel_cache"] = {}
    st.session_state["cluster_path_for_embed"] = ()

    # Invalidate stale cached visuals
    st.session_state["hclust_topo_fig"] = None
    st.session_state["hclust_characteristics"] = {}

    # Build shared inputs for KDE + characteristics (always StandardScaler for vis)
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = h_labels
    X_scaled_hc = pd.DataFrame(
        StandardScaler().fit_transform(df[feature_columns].to_numpy()),
        columns=feature_columns,
    )

    with st.spinner("Rendering cluster topography…"):
        topo_fig = cluster_gauss_kde(
            df_with_clusters,
            X_scaled_hc,
            layout_df,
            kde_dr_method=st.session_state["method"],
            perplexity=st.session_state["tsne_perplexity"] if st.session_state["method"] == "t-SNE" else None,
            learning_rate=st.session_state["tsne_learning_rate"] if st.session_state["method"] == "t-SNE" else None,
            n_neighbors=st.session_state["umap_n_neighbors"] if st.session_state["method"] == "UMAP" else None,
            min_dist=st.session_state["umap_min_dist"] if st.session_state["method"] == "UMAP" else None,
        )
    st.session_state["hclust_topo_fig"] = topo_fig

    if st.session_state["hclust_precompute"]:
        valid_ids = [int(cid) for cid in layout_df["cluster"].unique() if int(cid) != -1]
        extra_cols = [
            c
            for c in df_with_clusters.columns
            if c not in feature_columns and c not in {"row_id", "cluster"} and pd.api.types.is_numeric_dtype(df_with_clusters[c])
        ]
        chars: dict[int, tuple] = {}
        with st.spinner(f"Precomputing characteristics for {len(valid_ids)} clusters…"):
            for cid in valid_ids:
                chars[cid] = cluster_characteristics(
                    cid,
                    df_with_clusters,
                    X_scaled_hc[feature_columns],
                    feature_columns,
                    extra_cols=extra_cols,
                )
        st.session_state["hclust_characteristics"] = chars


def handle_exploration_save(df: pd.DataFrame, X: np.ndarray, _feature_columns: list[str]) -> None:
    stack = st.session_state["hierarchical_selection_stack"]
    n_layers = st.session_state["hierarchical_layers"]

    current_path = tuple(stack[:n_layers])

    _sub_df, sub_X = get_path_subset(df, X, current_path)

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

    st.session_state["cluster_path_for_embed"] = current_path

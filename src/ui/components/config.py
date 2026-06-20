from __future__ import annotations

import streamlit as st


def render_hierarchical_config(max_dims: int) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.checkbox("Normalize (StandardScaler)", key="hclust_normalize")
        st.number_input(
            "Hierarchical levels deep",
            min_value=1,
            max_value=5,
            step=1,
            key="hierarchical_layers",
        )
    with c2:
        st.slider(
            "UMAP dims before HDBSCAN",
            min_value=2,
            max_value=max_dims,
            key="hclust_umap_n_components",
        )
        c2a, c2b = st.columns(2)
        with c2a:
            st.number_input(
                "HDBSCAN min_samples",
                min_value=1,
                max_value=100,
                step=1,
                key="hclust_min_samples",
            )
        with c2b:
            st.number_input(
                "HDBSCAN min_cluster_size",
                min_value=2,
                max_value=500,
                step=1,
                key="hclust_min_cluster_size",
            )


def render_exploration_config(max_components: int, n_rows: int) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.checkbox("Normalize (StandardScaler)", key="normalize")
        st.selectbox("Method", options=["PCA", "t-SNE", "UMAP"], key="method")
    with c2:
        match st.session_state.method:
            case "PCA":
                st.slider(
                    "PCA fitted components",
                    min_value=2,
                    max_value=max_components,
                    key="pca_components",
                )
            case "t-SNE":
                max_perplexity = float(max(5.0, min(50.0, n_rows - 1.0)))
                st.slider("Perplexity", min_value=5.0, max_value=max_perplexity, key="tsne_perplexity")
                st.number_input("Learning rate", min_value=10.0, max_value=2000.0, key="tsne_learning_rate")
                st.number_input("Random state (t-SNE)", min_value=0, max_value=9999, key="tsne_random_state")
            case "UMAP":
                st.slider("n_neighbors", min_value=2, max_value=200, key="umap_n_neighbors")
                st.slider("min_dist", min_value=0.0, max_value=0.99, key="umap_min_dist")
                st.number_input("Random state (UMAP)", min_value=0, max_value=9999, key="umap_random_state")

    c4, c5 = st.columns([1, 2])
    with c4:
        st.checkbox("Interactive ranges mode", key="interactive_ranges_mode")
    with c5:
        st.checkbox("Clusters in original space", key="clusters_in_original_space")
        if st.session_state["clusters_in_original_space"]:
            st.selectbox("Cluster method", options=["KMeans", "GMM"], key="cluster_method")
            st.slider("Number of clusters", min_value=2, max_value=10, key="cluster_n_clusters")

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.analysis.clustering import compute_clusters
from src.analysis.predicate_generator import generate_predicate
from src.ui.util.data import (
    build_plot_df,
    compute_embedding,
    compute_interactive_mask,
    export_selection,
    get_cluster_subset,
    run_hierarchical_clustering,
)
from src.ui.util.state import current_config, get_selected_indices
from src.ui.util.visualization import (
    cluster_characteristics,
    cluster_gauss_kde,
    make_feature_range_fig,
    make_pca_variance_fig,
    make_scatter_fig,
)

MIN_COMPONENTS_FOR_2D = 2


def _get_path_subset(
    df: pd.DataFrame,
    X: np.ndarray,
    path: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not path:
        return df, X
    h_labels = st.session_state["hierarchical_labels"]
    sub_df, sub_X = get_cluster_subset(df, X, h_labels, path[0])
    for i, cluster_id in enumerate(path[1:], 1):
        level_labels = st.session_state["hierarchical_sublevel_cache"][str(path[:i])]["h_labels"]
        sub_df, sub_X = get_cluster_subset(sub_df, sub_X, level_labels, cluster_id)
    return sub_df, sub_X


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 1 — Hierarchical clustering
# ──────────────────────────────────────────────────────────────────────────────


def render_hierarchical_config(max_dims: int) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.checkbox("Normalize (StandardScaler)", key="hclust_normalize")
        st.checkbox("Precompute characteristics", key="hclust_precompute")
        st.number_input(
            "Hierarchical levels deep",
            min_value=1,
            max_value=5,
            step=1,
            key="hierarchical_layers",
        )
    with c2:
        st.slider(
            "UMAP precompute dims (before HDBSCAN)",
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


def _hclust_snapshot() -> dict:
    return {
        "normalize": bool(st.session_state["hclust_normalize"]),
        "umap_n_components": int(st.session_state["hclust_umap_n_components"]),
        "min_samples": int(st.session_state["hclust_min_samples"]),
        "min_cluster_size": int(st.session_state["hclust_min_cluster_size"]),
        "explore_method": str(st.session_state["method"]),
        "layers": int(st.session_state["hierarchical_layers"]),
        "precompute": bool(st.session_state["hclust_precompute"]),
    }


def handle_hierarchical_save(df: pd.DataFrame, _X: np.ndarray, feature_columns: list[str]) -> None:
    snapshot = _hclust_snapshot()
    # if snapshot == st.session_state.get("hclust_saved_config"):
    #     st.info("No changes — hierarchical config unchanged.")
    #     return

    X_hc = df[feature_columns].to_numpy()
    if snapshot["normalize"]:
        X_hc = StandardScaler().fit_transform(X_hc)

    with st.spinner("Computing hierarchical clusters…"):
        layout_df, h_labels, n_outliers, glosh_scores = run_hierarchical_clustering(
            df,
            X_hc,
            feature_columns,
            n_components=snapshot["umap_n_components"],
            # n_neighbors=st.session_state["umap_n_neighbors"] if snapshot["explore_method"] == "UMAP" else 15,
            # min_dist=st.session_state["umap_min_dist"] if snapshot["explore_method"] == "UMAP" else 0.0,
            min_samples=snapshot["min_samples"],
            min_cluster_size=snapshot["min_cluster_size"],
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
    pure_feature_cols = [c for c in df.columns if c not in ["row_id"]]  # quality was here

    with st.spinner("Rendering cluster topography…"):
        topo_fig = cluster_gauss_kde(
            df_with_clusters,
            X_scaled_hc,
            layout_df,
            kde_dr_method=snapshot["explore_method"],
            perplexity=st.session_state["tsne_perplexity"] if snapshot["explore_method"] == "t-SNE" else None,
            learning_rate=st.session_state["tsne_learning_rate"] if snapshot["explore_method"] == "t-SNE" else None,
            n_neighbors=st.session_state["umap_n_neighbors"] if snapshot["explore_method"] == "UMAP" else None,
            min_dist=st.session_state["umap_min_dist"] if snapshot["explore_method"] == "UMAP" else None,
        )
    st.session_state["hclust_topo_fig"] = topo_fig

    if snapshot["precompute"]:
        valid_ids = [int(cid) for cid in layout_df["cluster"].unique() if int(cid) != -1]
        chars: dict[int, tuple] = {}
        with st.spinner(f"Precomputing characteristics for {len(valid_ids)} clusters…"):
            for cid in valid_ids:
                chars[cid] = cluster_characteristics(
                    cid,
                    df_with_clusters,
                    X_scaled_hc[pure_feature_cols],
                    pure_feature_cols,
                )
        st.session_state["hclust_characteristics"] = chars

    st.session_state["hclust_saved_config"] = snapshot


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG 2 — Low-level exploration
# ──────────────────────────────────────────────────────────────────────────────


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


def _explore_snapshot() -> dict:
    snap: dict = {
        "method": str(st.session_state["method"]),
        "normalize": bool(st.session_state["normalize"]),
    }
    match snap["method"]:
        case "PCA":
            snap["pca_components"] = int(st.session_state["pca_components"])
        case "t-SNE":
            snap["tsne_perplexity"] = float(st.session_state["tsne_perplexity"])
            snap["tsne_learning_rate"] = float(st.session_state["tsne_learning_rate"])
            snap["tsne_random_state"] = int(st.session_state["tsne_random_state"])
        case "UMAP":
            snap["umap_n_neighbors"] = int(st.session_state["umap_n_neighbors"])
            snap["umap_min_dist"] = float(st.session_state["umap_min_dist"])
            snap["umap_random_state"] = int(st.session_state["umap_random_state"])
    return snap


def handle_exploration_save(df: pd.DataFrame, X: np.ndarray, _feature_columns: list[str]) -> None:
    snapshot = _explore_snapshot()
    stack = st.session_state.get("hierarchical_selection_stack", [])
    n_layers = int(st.session_state.get("hierarchical_layers", 1))

    if len(stack) < n_layers:
        st.session_state["explore_saved_config"] = snapshot
        return

    current_path = tuple(stack[:n_layers])
    path_changed = st.session_state.get("cluster_path_for_embed") != current_path
    config_changed = snapshot != st.session_state.get("explore_saved_config")

    if not path_changed and not config_changed:
        st.info("No changes — exploration config unchanged.")
        return

    _sub_df, sub_X = _get_path_subset(df, X, current_path)

    with st.spinner("Computing cluster embedding…"):
        result = compute_embedding(method=st.session_state.method, X=sub_X, config=current_config())

    st.session_state["cluster_embedding_full"] = result.embedding
    st.session_state["cluster_explained_variance"] = (
        result.explained_variance_ratio if result.explained_variance_ratio is not None else np.array([], dtype=float)
    )

    if st.session_state.method == "PCA" and st.session_state["cluster_explained_variance"].size >= MIN_COMPONENTS_FOR_2D:
        ordered = np.argsort(st.session_state["cluster_explained_variance"])[::-1]
        st.session_state["cluster_pca_x_component"] = int(ordered[0])
        st.session_state["cluster_pca_y_component"] = int(ordered[1])

    st.session_state["cluster_path_for_embed"] = current_path
    st.session_state["cluster_selected_id_for_embed"] = stack[0]  # backward compat
    st.session_state["explore_saved_config"] = snapshot


# ──────────────────────────────────────────────────────────────────────────────
# Embedding resolution for the cluster scatter plot
# ──────────────────────────────────────────────────────────────────────────────


def resolve_cluster_embedding_2d() -> tuple[np.ndarray, np.ndarray, int, list[str]] | None:
    if st.session_state.method != "PCA":
        return st.session_state["cluster_embedding_full"], np.array([], dtype=float), 0, []

    explained_ratio = st.session_state["cluster_explained_variance"]
    if explained_ratio.size < MIN_COMPONENTS_FOR_2D:
        st.error("PCA metadata is unavailable. Save the exploration config again.")
        return None

    component_count = st.session_state["cluster_embedding_full"].shape[1]
    component_labels = [f"PC {i + 1}" for i in range(component_count)]
    st.session_state["cluster_pca_x_component"] = min(
        max(0, int(st.session_state["cluster_pca_x_component"])),
        component_count - 1,
    )
    st.session_state["cluster_pca_y_component"] = min(
        max(0, int(st.session_state["cluster_pca_y_component"])),
        component_count - 1,
    )

    if st.session_state["cluster_pca_x_component"] == st.session_state["cluster_pca_y_component"]:
        st.warning("Choose two different PCA components for x and y.")
        return None

    xi = st.session_state["cluster_pca_x_component"]
    yi = st.session_state["cluster_pca_y_component"]
    embedding_2d = st.session_state["cluster_embedding_full"][:, [xi, yi]]
    return embedding_2d, explained_ratio, component_count, component_labels


# ──────────────────────────────────────────────────────────────────────────────
# PCA component controls (rendered above the scatter plot)
# ──────────────────────────────────────────────────────────────────────────────


def render_pca_controls(
    explained_ratio: np.ndarray,
    component_count: int,
    component_labels: list[str],
) -> None:
    left, right = st.columns(2)
    with left:
        st.selectbox(
            "PCA x-axis",
            options=list(range(component_count)),
            format_func=lambda i: component_labels[i],
            key="cluster_pca_x_component",
        )
    with right:
        st.selectbox(
            "PCA y-axis",
            options=list(range(component_count)),
            format_func=lambda i: component_labels[i],
            key="cluster_pca_y_component",
        )
    pc_labels = [f"PC{i + 1}" for i in range(len(explained_ratio))]
    explained_pct = explained_ratio * 100.0
    preview_count = min(6, len(pc_labels))
    preview = ", ".join(f"{pc_labels[i]}: {explained_pct[i]:.1f}%" for i in range(preview_count))
    suffix = " …" if len(pc_labels) > preview_count else ""
    st.caption(f"Explained variance by component: {preview}{suffix}")
    st.plotly_chart(make_pca_variance_fig(explained_ratio), width="stretch")
    selected_total = explained_ratio[st.session_state["cluster_pca_x_component"]] + explained_ratio[st.session_state["cluster_pca_y_component"]]
    st.info(f"Total variance explained by selected components: {selected_total:.2%}")


def _render_glosh_column(h_labels: np.ndarray) -> None:
    st.markdown("**Outliers GLOSH**")
    glosh_scores = st.session_state.get("hierarchical_glosh_scores")
    if glosh_scores is None:
        return
    outlier_mask = h_labels == -1
    if not outlier_mask.any():
        st.caption("No outlier points detected.")
        return
    outlier_scores = glosh_scores[outlier_mask]
    st.caption(
        f"min: {outlier_scores.min():.3f}  median: {float(np.median(outlier_scores)):.3f}  max: {outlier_scores.max():.3f}",
    )
    top_n = (
        pd.DataFrame({"row": np.where(outlier_mask)[0], "glosh": glosh_scores[outlier_mask]})
        .sort_values("glosh", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(top_n, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical section (topography + cluster characteristics)
# ──────────────────────────────────────────────────────────────────────────────


def render_hierarchical_section(df: pd.DataFrame, feature_columns: list[str]) -> None:
    st.subheader("Hierarchical Cluster Topography (HDBSCAN)")

    h_layout_df = st.session_state.get("hierarchical_layout_df")
    h_labels = st.session_state.get("hierarchical_labels")

    if h_layout_df is None or h_labels is None:
        st.info("Save the hierarchical config above to compute clusters.")
        return

    n_outliers = st.session_state["hierarchical_n_outliers"]

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("**Information**")
        st.text(f"Outliers (noise points): {n_outliers}, ratio: {n_outliers / len(df):.2%}")
        st.text(f"Clusters found: {len(h_layout_df):,}, number of points in cluster: {h_layout_df[['cluster', 'size']]['size'].sum():,}")

    with c2:
        st.markdown("**Cluster size distribution**")
        st.dataframe(
            h_layout_df[["cluster", "size"]].sort_values("size", ascending=False).reset_index(drop=True),
            hide_index=True,
        )

    with c3:
        _render_glosh_column(h_labels)

    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = h_labels
    X_scaled_hc = pd.DataFrame(
        StandardScaler().fit_transform(df[feature_columns].to_numpy()),
        columns=feature_columns,
    )

    topo_fig = st.session_state.get("hclust_topo_fig")
    if topo_fig is None:
        with st.spinner("Rendering cluster topography…"):
            topo_fig = cluster_gauss_kde(
                df_with_clusters,
                X_scaled_hc,
                h_layout_df,
                kde_dr_method=str(st.session_state["method"]),
            )
        st.session_state["hclust_topo_fig"] = topo_fig
    topo_fig.update_layout(height=650)

    topo_col, char_col = st.columns([1, 1])
    with topo_col:
        topo_event = st.plotly_chart(
            topo_fig,
            key="hierarchical_topo_plot",
            width="stretch",
            on_select="rerun",
            selection_mode="points",
        )

    clicked = get_selected_indices(topo_event)
    if clicked:
        st.session_state["selected_cluster_id"] = int(h_layout_df["cluster"].iloc[clicked[0]])

    selected_cluster = st.session_state.get("selected_cluster_id")
    with char_col:
        if selected_cluster is None:
            st.info("Click a cluster on the topography map to explore its characteristics.")
            return
        st.markdown(f"**Cluster {selected_cluster}** — n={int((h_labels == selected_cluster).sum())} points")
        pure_feature_cols = [c for c in df.columns if c not in ["row_id"]]  # quality was here
        precomputed = st.session_state.get("hclust_characteristics", {})
        if selected_cluster in precomputed:
            char_fig, rules = precomputed[selected_cluster]
        else:
            char_fig, rules = cluster_characteristics(
                selected_cluster,
                df_with_clusters,
                X_scaled_hc[pure_feature_cols],
                pure_feature_cols,
            )
        char_fig.update_layout(height=620)
        st.plotly_chart(char_fig, width="stretch")
        with st.expander("Decision tree rules"):
            st.code(rules)


# ──────────────────────────────────────────────────────────────────────────────
# Intermediate hierarchical sublevel panel
# ──────────────────────────────────────────────────────────────────────────────


def render_hierarchical_sublevel(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_columns: list[str],
    parent_path: tuple[int, ...],
    layer: int,
) -> int | None:
    cache_key = str(parent_path)
    cache: dict = st.session_state.setdefault("hierarchical_sublevel_cache", {})

    if cache_key not in cache:
        sub_df, sub_X = _get_path_subset(df, X, parent_path)
        snapshot = st.session_state.get("hclust_saved_config", {})
        min_cluster_size = int(snapshot.get("min_cluster_size", 15))

        if len(sub_df) < min_cluster_size * 2:
            return None

        X_hc = sub_X.copy()
        if snapshot.get("normalize", True):
            X_hc = StandardScaler().fit_transform(X_hc)

        n_comp = min(int(snapshot.get("umap_n_components", 2)), sub_X.shape[1], len(sub_df) - 1)
        n_nbrs = min(30, len(sub_df) - 1)

        with st.spinner(f"Computing sub-clusters for layer {layer}…"):
            sub_layout_df, sub_h_labels, sub_n_outliers, _sub_glosh = run_hierarchical_clustering(
                sub_df,
                X_hc,
                feature_columns,
                n_components=n_comp,
                n_neighbors=n_nbrs,
                min_samples=int(snapshot.get("min_samples", 5)),
                min_cluster_size=min_cluster_size,
            )

        if sub_layout_df.empty:
            return None

        df_with_sub = sub_df.copy()
        df_with_sub["cluster"] = sub_h_labels
        X_scaled_vis = pd.DataFrame(
            StandardScaler().fit_transform(sub_df[feature_columns].to_numpy()),
            columns=feature_columns,
        )
        pure_cols = [c for c in sub_df.columns if c not in ["row_id"]]
        explore_method = str(snapshot.get("explore_method", "UMAP"))

        with st.spinner(f"Rendering layer {layer} topography…"):
            sub_topo_fig = cluster_gauss_kde(
                df_with_sub,
                X_scaled_vis,
                sub_layout_df,
                kde_dr_method=explore_method,
                perplexity=st.session_state.get("tsne_perplexity") if explore_method == "t-SNE" else None,
                learning_rate=st.session_state.get("tsne_learning_rate") if explore_method == "t-SNE" else None,
                n_neighbors=st.session_state.get("umap_n_neighbors") if explore_method == "UMAP" else None,
                min_dist=st.session_state.get("umap_min_dist") if explore_method == "UMAP" else None,
            )

        sub_chars: dict[int, tuple] = {}
        if snapshot.get("precompute", True):
            valid_ids = [int(cid) for cid in sub_layout_df["cluster"].unique() if int(cid) != -1]
            with st.spinner(f"Precomputing layer {layer} characteristics…"):
                for cid in valid_ids:
                    sub_chars[cid] = cluster_characteristics(cid, df_with_sub, X_scaled_vis[pure_cols], pure_cols)

        cache[cache_key] = {
            "layout_df": sub_layout_df,
            "h_labels": sub_h_labels,
            "n_outliers": sub_n_outliers,
            "topo_fig": sub_topo_fig,
            "characteristics": sub_chars,
        }

    cached = cache[cache_key]
    sub_layout_df = cached["layout_df"]
    sub_h_labels = cached["h_labels"]
    sub_n_outliers = cached["n_outliers"]
    sub_topo_fig = cached["topo_fig"]
    sub_chars = cached["characteristics"]

    sub_df_full, _ = _get_path_subset(df, X, parent_path)
    n_parent = len(sub_df_full)

    st.subheader(f"Layer {layer} Sub-Cluster Topography — parent path {parent_path}")
    st.text(f"Points in parent cluster: {n_parent:,}")
    st.text(f"Outliers (noise points): {sub_n_outliers}, ratio: {sub_n_outliers / n_parent:.2%}")
    st.text(
        f"Sub-clusters found: {len(sub_layout_df):,}, points in sub-clusters: {sub_layout_df['size'].sum():,}",
    )
    st.dataframe(
        sub_layout_df[["cluster", "size"]].sort_values("size", ascending=False).reset_index(drop=True),
        hide_index=True,
    )

    sub_topo_fig.update_layout(height=650)
    topo_col, char_col = st.columns([1, 1])
    chart_key = f"hierarchical_topo_plot_layer_{layer}_{parent_path}"

    with topo_col:
        topo_event = st.plotly_chart(
            sub_topo_fig,
            key=chart_key,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
        )

    clicked = get_selected_indices(topo_event)
    if clicked:
        new_id = int(sub_layout_df["cluster"].iloc[clicked[0]])
        full_idx = layer - 1
        stack = list(st.session_state.get("hierarchical_selection_stack", []))
        stack = stack[:full_idx]
        stack.append(new_id)
        st.session_state["hierarchical_selection_stack"] = stack

    stack = st.session_state.get("hierarchical_selection_stack", [])
    full_idx = layer - 1
    current_selection: int | None = stack[full_idx] if len(stack) > full_idx else None

    with char_col:
        if current_selection is None:
            st.info(f"Click a sub-cluster in layer {layer} to continue drilling down.")
        else:
            st.markdown(f"**Sub-cluster {current_selection}** — n={int((sub_h_labels == current_selection).sum())} points")
            pure_cols = [c for c in sub_df_full.columns if c not in ["row_id"]]
            if current_selection in sub_chars:
                char_fig, rules = sub_chars[current_selection]
            else:
                df_with_sub = sub_df_full.copy()
                df_with_sub["cluster"] = sub_h_labels
                X_scaled_vis = pd.DataFrame(
                    StandardScaler().fit_transform(sub_df_full[feature_columns].to_numpy()),
                    columns=feature_columns,
                )
                char_fig, rules = cluster_characteristics(
                    current_selection,
                    df_with_sub,
                    X_scaled_vis[pure_cols],
                    pure_cols,
                )
            char_fig.update_layout(height=620)
            st.plotly_chart(char_fig, width="stretch")
            with st.expander("Decision tree rules"):
                st.code(rules)

    return current_selection


# ──────────────────────────────────────────────────────────────────────────────
# Data layer for the exploration scatter plot
# ──────────────────────────────────────────────────────────────────────────────


def compute_data_layer(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_columns: list[str],
    embedding_2d: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, bool, bool, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    cluster_mode = bool(st.session_state["clusters_in_original_space"])
    cluster_labels = np.array([], dtype=int)
    if cluster_mode:
        try:
            with st.spinner("Computing clusters in original space…"):  # TODO here, original space or space after init UMAP?
                cluster_labels = compute_clusters(
                    X_scaled=X_scaled,
                    method=str(st.session_state["cluster_method"]),
                    n_clusters=int(st.session_state["cluster_n_clusters"]),
                )
        except ValueError as exc:
            st.error(f"Clustering failed: {exc}")
            cluster_mode = False
            st.session_state.cluster_labels = np.array([], dtype=int)
        else:
            st.session_state.cluster_labels = cluster_labels
    else:
        st.session_state.cluster_labels = np.array([], dtype=int)

    interactive_mode = bool(st.session_state["interactive_ranges_mode"])
    interactive_mask = compute_interactive_mask(X_scaled_df, feature_columns) if interactive_mode else None
    assert not isinstance(cluster_labels, tuple)  # yeah yeah i know
    plot_df = build_plot_df(df, embedding_2d, cluster_labels, interactive_mask)
    st.session_state.plot_df = plot_df

    return X_scaled, X_scaled_df, cluster_labels, cluster_mode, interactive_mode, plot_df


# ──────────────────────────────────────────────────────────────────────────────
# Analysis column (range analysis or interactive filters)
# ──────────────────────────────────────────────────────────────────────────────


def render_interactive_filters(X_scaled_df: pd.DataFrame, feature_columns: list[str]) -> None:
    st.caption("Configure feature-wise standardized ranges. Matching points are highlighted in blue in the projection.")
    selected_feature_defaults = [f for f in st.session_state["interactive_features"] if f in feature_columns]
    st.multiselect(
        "Features to filter",
        options=feature_columns,
        default=selected_feature_defaults,
        key="interactive_features",
    )
    for feature in st.session_state["interactive_features"]:
        full_min = float(X_scaled_df[feature].min())
        full_max = float(X_scaled_df[feature].max())
        slider_key = f"interactive_range_{feature}"
        current_range = st.session_state.get(slider_key, (full_min, full_max))
        current_min = max(full_min, float(current_range[0]))
        current_max = min(full_max, float(current_range[1]))
        if current_min > current_max:
            current_min, current_max = full_min, full_max
        st.slider(
            f"{feature} (standardized)",
            min_value=full_min,
            max_value=full_max,
            value=(current_min, current_max),
            key=slider_key,
        )
    if st.session_state.selected_df.empty:
        st.info("No points match the active feature ranges. Adjust the sliders to widen the filter.")


def render_range_analysis(X_scaled: np.ndarray, feature_columns: list[str]) -> None:
    selected_scaled = X_scaled.take(st.session_state.selected_indices, axis=0)
    selected_scaled_df = pd.DataFrame(selected_scaled, columns=feature_columns)
    range_df_full = pd.DataFrame(generate_predicate("hm", selected_scaled_df, X_scaled, threshold=1.0))
    range_df_trimmed = pd.DataFrame(generate_predicate("hm", selected_scaled_df, X_scaled, threshold=0.9))
    if not range_df_full.empty:
        st.plotly_chart(make_feature_range_fig(range_df_full, range_df_trimmed), width="stretch")


def render_analysis_column(
    X_scaled: np.ndarray,
    X_scaled_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    interactive_mode: bool,
) -> None:
    st.subheader("Analysis of selected Datapoints")
    st.text(
        "RCM = Retained Center Mass. RCM=1.0 means the full range of selected points, "
        "RCM=0.9 trims the tails to exclude outliers and show the core range.",
    )
    if interactive_mode:
        render_interactive_filters(X_scaled_df, feature_columns)
    elif st.session_state.selected_df.empty:
        st.info("Use lasso or box selection in the plot to capture points.")
    else:
        render_range_analysis(X_scaled, feature_columns)


# ──────────────────────────────────────────────────────────────────────────────
# Selection update
# ──────────────────────────────────────────────────────────────────────────────


def update_selection(event: object, plot_df: pd.DataFrame, *, interactive_mode: bool) -> None:
    selected_indices = np.flatnonzero(plot_df["interactive_group"] == "Matches filters").tolist() if interactive_mode else get_selected_indices(event)
    st.session_state.selected_indices = selected_indices
    st.session_state.selected_df = plot_df.iloc[selected_indices].copy() if selected_indices else pd.DataFrame()


def render_cluster_exploration(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_columns: list[str],
    selection_path: tuple[int, ...],
) -> None:
    path_label = " → ".join(f"C{c}" for c in selection_path)
    st.subheader(f"Exploration — {path_label}")

    # Auto-compute when the selection path changes
    path_switched = st.session_state.get("cluster_path_for_embed") != selection_path
    if st.session_state.get("explore_saved_config") is not None and path_switched:
        handle_exploration_save(df, X, feature_columns)

    if st.session_state["cluster_embedding_full"].size == 0:
        st.info("Save the exploration config above to compute the embedding.")
        return

    sub_df, sub_X = _get_path_subset(df, X, selection_path)

    resolved = resolve_cluster_embedding_2d()
    if resolved is None:
        return
    embedding_2d, explained_ratio, component_count, component_labels = resolved

    X_scaled, X_scaled_df, _cluster_labels, cluster_mode, interactive_mode, plot_df = compute_data_layer(
        sub_df,
        sub_X,
        feature_columns,
        embedding_2d,
    )

    if st.session_state.method == "PCA":
        render_pca_controls(explained_ratio, component_count, component_labels)

    col_ranges, col_plot = st.columns([1, 1.4])
    scatter_fig = make_scatter_fig(
        plot_df,
        st.session_state.method,
        cluster_mode=cluster_mode,
        interactive_mode=interactive_mode,
    )
    with col_plot:
        event = st.plotly_chart(
            scatter_fig,
            key="reduction_plot",
            width="stretch",
            on_select="ignore" if interactive_mode else "rerun",
            selection_mode=("lasso", "box"),
        )

    update_selection(event, plot_df, interactive_mode=interactive_mode)

    with col_ranges:
        render_analysis_column(X_scaled, X_scaled_df, feature_columns, interactive_mode=interactive_mode)

    st.divider()
    st.subheader("Selected Datapoints")
    st.write(f"Selected points: {len(st.session_state.selected_indices)}")

    if st.session_state.selected_df.empty:
        st.info("Use lasso or box selection in the plot to capture points.")
        return

    st.dataframe(st.session_state.selected_df.head(50), width="stretch")
    if st.button("Export selected points to CSV"):
        file_path = export_selection(
            st.session_state.selected_df,
            st.session_state.latest_selection_config or current_config(),
        )
        st.success(f"Selection exported to {file_path}")

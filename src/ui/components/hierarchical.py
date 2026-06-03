from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.ui.data import get_path_subset, run_hierarchical_clustering
from src.ui.state import get_selected_indices
from src.ui.visualization import cluster_characteristics, cluster_gauss_kde


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
        precomputed = st.session_state.get("hclust_characteristics", {})
        if selected_cluster in precomputed:
            char_fig, rules = precomputed[selected_cluster]
        else:
            extra_cols = [
                c for c in df_with_clusters.columns
                if c not in feature_columns and c not in {"row_id", "cluster"}
                and pd.api.types.is_numeric_dtype(df_with_clusters[c])
            ]
            char_fig, rules = cluster_characteristics(
                selected_cluster,
                df_with_clusters,
                X_scaled_hc[feature_columns],
                feature_columns,
                extra_cols=extra_cols,
            )
        char_fig.update_layout(height=620)
        st.plotly_chart(char_fig, width="stretch")
        with st.expander("Decision tree rules"):
            st.code(rules)


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
        sub_df, sub_X = get_path_subset(df, X, parent_path)
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
            extra_cols = [
                c for c in df_with_sub.columns
                if c not in feature_columns and c not in {"row_id", "cluster"}
                and pd.api.types.is_numeric_dtype(df_with_sub[c])
            ]
            with st.spinner(f"Precomputing layer {layer} characteristics…"):
                for cid in valid_ids:
                    sub_chars[cid] = cluster_characteristics(cid, df_with_sub, X_scaled_vis[feature_columns], feature_columns, extra_cols=extra_cols)

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

    sub_df_full, _ = get_path_subset(df, X, parent_path)
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
            if current_selection in sub_chars:
                char_fig, rules = sub_chars[current_selection]
            else:
                df_with_sub = sub_df_full.copy()
                df_with_sub["cluster"] = sub_h_labels
                X_scaled_vis = pd.DataFrame(
                    StandardScaler().fit_transform(sub_df_full[feature_columns].to_numpy()),
                    columns=feature_columns,
                )
                extra_cols = [
                    c for c in df_with_sub.columns
                    if c not in feature_columns and c not in {"row_id", "cluster"}
                    and pd.api.types.is_numeric_dtype(df_with_sub[c])
                ]
                char_fig, rules = cluster_characteristics(
                    current_selection,
                    df_with_sub,
                    X_scaled_vis[feature_columns],
                    feature_columns,
                    extra_cols=extra_cols,
                )
            char_fig.update_layout(height=620)
            st.plotly_chart(char_fig, width="stretch")
            with st.expander("Decision tree rules"):
                st.code(rules)

    return current_selection

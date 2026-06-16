from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.analysis_routine import HierarchyObject
from src.ui.state import get_selected_indices
from src.ui.tree_nav import child_size, get_node_at_path
from src.ui.visualization import cluster_characteristics_fig, cluster_gauss_kde


def _render_glosh_column(node: HierarchyObject) -> None:
    st.markdown("**Outliers GLOSH**")
    scores = node.get("outlier_scores")
    if scores is None:
        st.caption("No outlier scores available.")
        return

    children = node.get("next_object_layer") or []
    n_in_clusters = sum(child_size(c) for c in children)
    n_outliers = node["cluster_points"].shape[0] - n_in_clusters

    if n_outliers <= 0:
        st.caption("No outlier points detected.")
        return

    st.caption(f"Outlier points: {n_outliers}  |  score min: {scores.min():.3f}  median: {float(np.median(scores)):.3f}  max: {scores.max():.3f}")
    top_n = (
        pd.DataFrame({"row": np.arange(len(scores)), "glosh": scores})
        .sort_values("glosh", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(top_n, hide_index=True)


def render_hierarchical_section(df: pd.DataFrame, feature_columns: list[str]) -> None:
    st.subheader("Hierarchical Cluster Topography (HDBSCAN)")

    root = st.session_state.get("analysis_tree")
    if root is None or "is_leaf" in root:
        st.info("Save the hierarchical config above to compute clusters.")
        return

    children = root["next_object_layer"] or []
    sizes = [child_size(c) for c in children]
    n_in_clusters = sum(sizes)
    n_outliers = root["cluster_points"].shape[0] - n_in_clusters

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown("**Information**")
        st.text(f"Outliers (noise points): {n_outliers}, ratio: {n_outliers / max(root['cluster_points'].shape[0], 1):.2%}")
        st.text(f"Clusters found: {len(children):,}, points in clusters: {n_in_clusters:,}")

    with c2:
        st.markdown("**Cluster size distribution**")
        size_df = pd.DataFrame({"cluster": range(len(children)), "size": sizes}).sort_values("size", ascending=False).reset_index(drop=True)
        st.dataframe(size_df, hide_index=True)

    with c3:
        _render_glosh_column(root)

    topo_fig = cluster_gauss_kde(root)
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
    if clicked and clicked[0] < len(children):
        st.session_state["tree_path"] = [clicked[0]]

    tree_path: list[int] = st.session_state.get("tree_path", [])
    selected_idx = tree_path[0] if tree_path else None

    with char_col:
        if selected_idx is None:
            st.info("Click a cluster on the topography map to explore its characteristics.")
            return
        child = children[selected_idx]
        n_pts = child_size(child)
        st.markdown(f"**Cluster {selected_idx}** — n={n_pts} points")
        char_fig, rules = cluster_characteristics_fig(
            child["rel_characteristics"],
            n_pts,
            df,
            child["row_indices"],
            feature_columns,
        )
        char_fig.update_layout(height=620)
        st.plotly_chart(char_fig, width="stretch")
        with st.expander("Decision tree rules"):
            st.code(rules)


def render_hierarchical_sublevel(
    df: pd.DataFrame,
    feature_columns: list[str],
    layer: int,
) -> int | None:
    root = st.session_state.get("analysis_tree")
    if root is None:
        return None

    tree_path: list[int] = st.session_state.get("tree_path", [])
    parent_path = tree_path[: layer - 1]
    parent_node = get_node_at_path(root, parent_path)

    if "is_leaf" in parent_node:
        return None

    children = parent_node["next_object_layer"] or []
    if not children:
        return None

    sizes = [child_size(c) for c in children]
    n_in_clusters = sum(sizes)
    n_parent = parent_node["cluster_points"].shape[0]
    n_outliers = n_parent - n_in_clusters

    st.subheader(f"Layer {layer} Sub-Cluster Topography — parent path {tuple(parent_path)}")
    st.text(f"Points in parent cluster: {n_parent:,}")
    st.text(f"Outliers (noise points): {n_outliers}, ratio: {n_outliers / max(n_parent, 1):.2%}")
    st.text(f"Sub-clusters found: {len(children):,}, points in sub-clusters: {n_in_clusters:,}")
    size_df = pd.DataFrame({"cluster": range(len(children)), "size": sizes}).sort_values("size", ascending=False).reset_index(drop=True)
    st.dataframe(size_df, hide_index=True)

    topo_fig = cluster_gauss_kde(parent_node)
    topo_fig.update_layout(height=650)

    topo_col, char_col = st.columns([1, 1])
    chart_key = f"hierarchical_topo_plot_layer_{layer}_{tuple(parent_path)}"

    with topo_col:
        topo_event = st.plotly_chart(
            topo_fig,
            key=chart_key,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
        )

    clicked = get_selected_indices(topo_event)
    if clicked and clicked[0] < len(children):
        new_idx = clicked[0]
        stack = list(tree_path)
        stack = stack[: layer - 1]
        stack.append(new_idx)
        st.session_state["tree_path"] = stack
        tree_path = stack

    current_idx = tree_path[layer - 1] if len(tree_path) >= layer else None

    with char_col:
        if current_idx is None:
            st.info(f"Click a sub-cluster in layer {layer} to continue drilling down.")
        else:
            child = children[current_idx]
            n_pts = child_size(child)
            st.markdown(f"**Sub-cluster {current_idx}** — n={n_pts} points")
            char_fig, rules = cluster_characteristics_fig(
                child["rel_characteristics"],
                n_pts,
                df,
                child["row_indices"],
                feature_columns,
            )
            char_fig.update_layout(height=620)
            st.plotly_chart(char_fig, width="stretch")
            with st.expander("Decision tree rules"):
                st.code(rules)

    return current_idx

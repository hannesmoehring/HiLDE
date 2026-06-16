from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.ui.components.config import render_exploration_config, render_hierarchical_config
from src.ui.components.exploration import render_cluster_exploration
from src.ui.components.handlers import handle_hierarchical_save
from src.ui.components.hierarchical import render_hierarchical_section, render_hierarchical_sublevel
from src.ui.data import load_dataset
from src.ui.state import init_state
from src.ui.tree_nav import get_node_at_path


def main() -> None:
    st.set_page_config(page_title="SHD - Dimensionality Reduction Explorer", layout="wide")
    init_state()
    st.title("SHD Dimensionality Reduction Explorer")
    st.caption("Dataset is hardcoded to winequality-red.csv")

    df = load_dataset()

    # -- GENERAL CONFIG --
    st.subheader("General Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Select dataset")
        st.text_input("Path to dataset CSV", value="datasets/wine_quality/wine+quality/winequality-red.csv")

    with c2:
        st.markdown("### Select feature columns")
        feature_columns = st.multiselect("Feature columns", options=df.columns, default=df.columns.drop("row_id").tolist())

    X = df[feature_columns].to_numpy()
    max_dims = int(min(X.shape[0], X.shape[1]))

    # ── CONFIG 1: Hierarchical clustering ────────────────────────────────────
    st.subheader("Hierarchical Clustering Configuration")
    render_hierarchical_config(max_dims)

    st.divider()

    # ── CONFIG 2: Exploration ─────────────────────────────────────────────────
    st.subheader("Exploration Configuration")
    render_exploration_config(max_dims, X.shape[0])

    # ── SINGLE SAVE BUTTON ────────────────────────────────────────────────────
    _, center_col, _ = st.columns([2, 1, 2])
    with center_col:
        save_clicked = st.button("Save & Apply", key="save_all_btn", type="primary", use_container_width=True)
    if save_clicked:
        handle_hierarchical_save(df, feature_columns)

    st.divider()
    st.expander("Full Dataframe", expanded=False).write(df)
    st.divider()

    # ── HIERARCHICAL TOPOGRAPHY ───────────────────────────────────────────────
    if st.session_state.get("analysis_tree") is None:
        st.info("Configure and save the hierarchical settings above to begin.")
        return

    render_hierarchical_section(df, feature_columns)

    st.divider()

    n_layers = int(st.session_state.get("hierarchical_layers", 1))
    tree_path: list[int] = st.session_state.get("tree_path", [])

    if not tree_path:
        st.info("Click a cluster in the topography above to explore it.")
        return

    # Intermediate layers (empty when n_layers == 1)
    for layer in range(2, n_layers + 1):
        st.divider()
        sub_selection = render_hierarchical_sublevel(df, feature_columns, layer)

        if sub_selection is None:
            # Node at this level is a leaf or had too few points — show exploration at current depth
            current_path = tuple(tree_path[: layer - 1])
            root = st.session_state.get("analysis_tree")
            node = get_node_at_path(root, list(current_path))
            if "is_leaf" in node or (node.get("next_object_layer") is not None and len(node["next_object_layer"]) == 0):
                st.info(
                    f"Sub-clustering at layer {layer} is not possible for this cluster "
                    f"(too few points or all noise). Showing exploration panel for current path.",
                )
            render_cluster_exploration(df, feature_columns, current_path)
            return

        # Sync the new selection into tree_path
        tree_path = st.session_state.get("tree_path", [])
        if len(tree_path) < layer:
            return  # user hasn't selected at this layer yet

    # ── FINAL EXPLORATION PANEL ───────────────────────────────────────────────
    st.divider()
    render_cluster_exploration(df, feature_columns, tuple(tree_path[:n_layers]))


if __name__ == "__main__":
    main()

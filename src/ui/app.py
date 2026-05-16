from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st

from src.ui.util.data import load_dataset
from src.ui.util.state import init_state
from src.ui.util.ui_controls import (
    handle_exploration_save,
    handle_hierarchical_save,
    render_cluster_exploration,
    render_exploration_config,
    render_hierarchical_config,
    render_hierarchical_section,
    render_hierarchical_sublevel,
)


def main() -> None:
    st.set_page_config(page_title="SHD - Dimensionality Reduction Explorer", layout="wide")
    init_state()
    st.title("SHD Dimensionality Reduction Explorer")
    st.caption("Dataset is hardcoded to winequality-red.csv")

    df = load_dataset()
    feature_columns = [c for c in df.columns if c not in ["row_id"]]
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
        handle_hierarchical_save(df, X, feature_columns)
        handle_exploration_save(df, X, feature_columns)

    st.divider()
    st.expander("Full Dataframe", expanded=False).write(df)
    st.divider()
    # ── HIERARCHICAL TOPOGRAPHY ───────────────────────────────────────────────
    if st.session_state["hierarchical_layout_df"] is None:
        st.info("Configure and save the hierarchical settings above to begin.")
        return

    render_hierarchical_section(df, feature_columns)

    st.divider()

    n_layers = int(st.session_state.get("hierarchical_layers", 1))
    selected_l1 = st.session_state.get("selected_cluster_id")

    if selected_l1 is None:
        st.info("Click a cluster in the topography above to explore it.")
        return

    # Sync L1 into stack; reset deeper state if L1 changed
    stack = list(st.session_state.get("hierarchical_selection_stack", []))
    if not stack or stack[0] != selected_l1:
        stack = [selected_l1]
        st.session_state["hierarchical_selection_stack"] = stack
        st.session_state["hierarchical_sublevel_cache"] = {}
        st.session_state["cluster_path_for_embed"] = ()
        st.session_state["cluster_embedding_full"] = np.empty((0, 0), dtype=float)

    # Intermediate layers — loop is empty when n_layers == 1 (backward compat)
    for layer in range(2, n_layers + 1):
        st.divider()
        parent_path = tuple(stack[: layer - 1])
        sub_selection = render_hierarchical_sublevel(df, X, feature_columns, parent_path, layer)

        if sub_selection is None:
            # Check if the sublevel cache entry is absent (degenerate: too few points / all noise)
            # vs. simply no click yet (cache entry present but nothing selected)
            cache_key = str(parent_path)
            if cache_key not in st.session_state.get("hierarchical_sublevel_cache", {}):
                st.info(
                    f"Sub-clustering at layer {layer} is not possible for this cluster "
                    f"(too few points or all noise). Showing exploration panel for current path.",
                )
                render_cluster_exploration(df, X, feature_columns, tuple(stack[: layer - 1]))
            return

        # Update stack with this layer's selection (truncate deeper entries if changed)
        if len(stack) <= layer - 1:
            stack.append(sub_selection)
        elif stack[layer - 1] != sub_selection:
            stack = [*stack[: layer - 1], sub_selection]
            st.session_state["hierarchical_selection_stack"] = stack

    # ── FINAL EXPLORATION PANEL ───────────────────────────────────────────────
    st.divider()
    render_cluster_exploration(df, X, feature_columns, tuple(stack[:n_layers]))


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

    # ── HIERARCHICAL TOPOGRAPHY ───────────────────────────────────────────────
    if st.session_state["hierarchical_layout_df"] is None:
        st.info("Configure and save the hierarchical settings above to begin.")
        return

    render_hierarchical_section(df, feature_columns)

    st.divider()

    # ── CLUSTER EXPLORATION (only when cluster selected) ──────────────────────
    selected_cluster = st.session_state.get("selected_cluster_id")
    if selected_cluster is None:
        st.info("Click a cluster in the topography above to explore it.")
        return

    render_cluster_exploration(df, X, feature_columns, selected_cluster)


if __name__ == "__main__":
    main()

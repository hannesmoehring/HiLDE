from __future__ import annotations

import numpy as np
import streamlit as st

from src.ui.visualization import make_pca_variance_fig

_MIN_COMPONENTS_FOR_2D = 2


def resolve_cluster_embedding_2d(
    embedding_full: np.ndarray,
    explained_variance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, list[str]] | None:
    if st.session_state.method != "PCA":
        return embedding_full, np.array([], dtype=float), 0, []

    explained_ratio = explained_variance if explained_variance is not None else np.array([], dtype=float)
    if explained_ratio.size < _MIN_COMPONENTS_FOR_2D:
        st.error("PCA metadata is unavailable. Re-run Save & Apply.")
        return None

    component_count = embedding_full.shape[1]
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
    embedding_2d = embedding_full[:, [xi, yi]]
    return embedding_2d, explained_ratio, component_count, component_labels


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

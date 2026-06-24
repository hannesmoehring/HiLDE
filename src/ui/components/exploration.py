from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.analysis.clustering import compute_clusters
from src.analysis.predicate_generator import generate_predicate
from src.ui.components.handlers import handle_exploration_save
from src.ui.components.pca import render_pca_controls, resolve_cluster_embedding_2d
from src.ui.data import build_plot_df, compute_interactive_mask, export_selection
from src.ui.state import current_config, get_selected_indices
from src.ui.tree_nav import get_node_at_path
from src.ui.visualization import make_feature_range_fig, make_scatter_fig


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
            with st.spinner("Computing clusters in original space…"):
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
    assert not isinstance(cluster_labels, tuple)
    plot_df = build_plot_df(df, embedding_2d, cluster_labels, interactive_mask)
    st.session_state.plot_df = plot_df

    return X_scaled, X_scaled_df, cluster_labels, cluster_mode, interactive_mode, plot_df


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


def _update_selection(event: object, plot_df: pd.DataFrame, *, interactive_mode: bool) -> None:
    selected_indices = np.flatnonzero(plot_df["interactive_group"] == "Matches filters").tolist() if interactive_mode else get_selected_indices(event)
    st.session_state.selected_indices = selected_indices
    st.session_state.selected_df = plot_df.iloc[selected_indices].copy() if selected_indices else pd.DataFrame()


def render_cluster_exploration(
    df: pd.DataFrame,
    feature_columns: list[str],
    selection_path: tuple[int, ...],
) -> None:
    path_label = " → ".join(f"C{c}" for c in selection_path)
    st.subheader(f"Exploration — {path_label}")

    path_switched = st.session_state["cluster_path_for_embed"] != selection_path
    if path_switched:
        handle_exploration_save(df, feature_columns)

    if st.session_state["cluster_embedding_full"].size == 0:
        st.info("Save the exploration config above to compute the embedding.")
        return

    root = st.session_state.get("analysis_tree")
    if root is None:
        return
    leaf = get_node_at_path(root, list(selection_path))
    row_indices = leaf["row_indices"]  # type: ignore[index]
    sub_df = df.iloc[row_indices].reset_index(drop=True)
    sub_X = sub_df[feature_columns].to_numpy()

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

    _update_selection(event, plot_df, interactive_mode=interactive_mode)

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

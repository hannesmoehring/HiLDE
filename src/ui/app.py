from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import UTC, datetime
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import fit_dimensionality_reducer
from src.analysis.predicate_generator import generate_predicate

DATASET_PATH = Path("datasets/wine_quality/wine+quality/winequality-red.csv")
EXPORT_DIR = Path("outputs/selections")
MIN_COMPONENTS_FOR_2D = 2
PCA_VARIANCE_LABEL_THRESHOLD = 4.0


class ReductionConfig(TypedDict):
    method: str
    normalize: bool
    pca_components: int
    pca_x_component: int | None
    pca_y_component: int | None
    tsne_perplexity: float
    tsne_learning_rate: float
    tsne_random_state: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_random_state: int


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, sep=";")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


@st.cache_data
def compute_embedding(
    *,
    method: str,
    X: np.ndarray,
    config: ReductionConfig,
):
    normalize = config["normalize"]

    if method == "PCA":
        return fit_dimensionality_reducer(
            method=method,
            X=X,
            n_components=config["pca_components"],
            normalize=normalize,
            random_state=config["tsne_random_state"],
        )

    if method == "t-SNE":
        return fit_dimensionality_reducer(
            method=method,
            X=X,
            n_components=2,
            normalize=normalize,
            perplexity=config["tsne_perplexity"],
            learning_rate=config["tsne_learning_rate"],
            random_state=config["tsne_random_state"],
            init="pca",
        )

    return fit_dimensionality_reducer(
        method=method,
        X=X,
        n_components=2,
        normalize=normalize,
        n_neighbors=config["umap_n_neighbors"],
        min_dist=config["umap_min_dist"],
        random_state=config["umap_random_state"],
    )


def get_selected_indices(event: object) -> list[int]:
    points: list[dict[str, object]] = []

    if event is None:
        return []

    if hasattr(event, "selection") and isinstance(event.selection, dict):
        selection = cast("dict[str, object]", event.selection)
        points_obj = selection.get("points")
        if isinstance(points_obj, list):
            points = []
            for point in points_obj:
                if isinstance(point, dict):
                    points.append(cast("dict[str, object]", point))
    elif isinstance(event, dict):
        event_dict = cast("dict[str, object]", event)
        selection_obj = event_dict.get("selection")
        if isinstance(selection_obj, dict):
            selection_dict = cast("dict[str, object]", selection_obj)
            points_obj = selection_dict.get("points")
            if isinstance(points_obj, list):
                points = []
                for point in points_obj:
                    if isinstance(point, dict):
                        points.append(cast("dict[str, object]", point))

    indices: list[int] = []
    for point in points:
        point_index = point.get("point_index")
        point_number = point.get("pointNumber")
        if isinstance(point_index, int):
            indices.append(point_index)
        elif isinstance(point_number, int):
            indices.append(point_number)

    return indices


def current_config() -> ReductionConfig:
    return {
        "method": st.session_state["method"],
        "normalize": st.session_state["normalize"],
        "pca_components": st.session_state["pca_components"],
        "pca_x_component": st.session_state.get("pca_x_component"),
        "pca_y_component": st.session_state.get("pca_y_component"),
        "tsne_perplexity": st.session_state["tsne_perplexity"],
        "tsne_learning_rate": st.session_state["tsne_learning_rate"],
        "tsne_random_state": st.session_state["tsne_random_state"],
        "umap_n_neighbors": st.session_state["umap_n_neighbors"],
        "umap_min_dist": st.session_state["umap_min_dist"],
        "umap_random_state": st.session_state["umap_random_state"],
    }


def export_selection(selected_df: pd.DataFrame, config: ReductionConfig) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = EXPORT_DIR / f"selected_points_{timestamp}.csv"

    export_df = selected_df.copy()
    export_df["reduction_method"] = config["method"]
    export_df["reduction_config"] = str(config)
    export_df.to_csv(output_file, index=False)
    return output_file


def init_state() -> None:
    defaults: dict[str, object] = {
        "method": "PCA",
        "normalize": True,
        "pca_components": 4,
        "pca_x_component": 0,
        "pca_y_component": 1,
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_random_state": 42,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_random_state": 42,
        "selected_indices": [],
        "selected_df": pd.DataFrame(),
        "latest_selection_config": None,
        "plot_df": pd.DataFrame(),
        "embedding_full": np.empty((0, 0), dtype=float),
        "explained_variance_ratio": np.array([], dtype=float),
        "computed_method": None,
        "interactive_ranges_mode": False,
        "interactive_features": [],
        "clusters_in_original_space": False,
        "cluster_method": "KMeans",
        "cluster_n_clusters": 5,
        "cluster_labels": np.array([], dtype=int),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_pca_variance_summary(explained_ratio: np.ndarray) -> None:
    pc_labels = [f"PC{i + 1}" for i in range(len(explained_ratio))]
    explained_pct = explained_ratio * 100.0
    cumulative_pct = np.cumsum(explained_pct)

    preview_count = min(6, len(pc_labels))
    preview = ", ".join(f"{pc_labels[i]}: {explained_pct[i]:.1f}%" for i in range(preview_count))
    suffix = " ..." if len(pc_labels) > preview_count else ""
    st.caption(f"Explained variance by component: {preview}{suffix}")

    fig = go.Figure()
    for label, pct, cumulative in zip(pc_labels, explained_pct, cumulative_pct, strict=True):
        fig.add_trace(
            go.Bar(
                y=["Variance"],
                x=[pct],
                orientation="h",
                name=label,
                text=[f"{pct:.1f}%"] if pct >= PCA_VARIANCE_LABEL_THRESHOLD else None,
                textposition="inside",
                customdata=[[cumulative]],
                hovertemplate=(f"{label}<br>Variance: %{{x:.2f}}%<br>Cumulative: %{{customdata[0]:.2f}}%<extra></extra>"),
            ),
        )

    fig.update_layout(
        barmode="stack",
        height=150,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis_title="Share of total variance (%)",
        yaxis_title=None,
        yaxis={"showticklabels": False},
        legend={"orientation": "h", "y": 1.5, "x": 0},
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, width="stretch")


def compute_interactive_mask(X_scaled_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    selected_features = [feature for feature in cast("list[str]", st.session_state.get("interactive_features", [])) if feature in feature_columns]
    if not selected_features:
        return np.ones(len(X_scaled_df), dtype=bool)

    mask = np.ones(len(X_scaled_df), dtype=bool)
    for feature in selected_features:
        full_min = float(X_scaled_df[feature].min())
        full_max = float(X_scaled_df[feature].max())
        slider_key = f"interactive_range_{feature}"
        lower, upper = cast("tuple[float, float]", st.session_state.get(slider_key, (full_min, full_max)))
        lower = max(full_min, float(lower))
        upper = min(full_max, float(upper))
        mask &= (X_scaled_df[feature] >= lower) & (X_scaled_df[feature] <= upper)

    return mask


def main() -> None:
    st.set_page_config(page_title="SHD - Dimensionality Reduction Explorer", layout="wide")
    init_state()

    st.title("SHD Dimensionality Reduction Explorer")
    st.caption("Dataset is hardcoded to winequality-red.csv")

    df = load_dataset()
    feature_columns = [c for c in df.columns if c not in ["row_id"]]  # "quality",
    X = df[feature_columns].to_numpy()

    # ── Top configuration ───────────────────────────────────────────────────────
    max_components = int(min(X.shape[0], X.shape[1]))

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.checkbox("Normalize (StandardScaler)", key="normalize")
        st.selectbox("Method", options=["PCA", "t-SNE", "UMAP"], key="method")
    with c2:
        match st.session_state.method:
            case "PCA":
                st.slider("PCA fitted components", min_value=2, max_value=max_components, key="pca_components")
            case "t-SNE":
                max_perplexity = float(max(5.0, min(50.0, X.shape[0] - 1.0)))
                st.slider("Perplexity", min_value=5.0, max_value=max_perplexity, key="tsne_perplexity")
                st.number_input("Learning rate", min_value=10.0, max_value=2000.0, key="tsne_learning_rate")
                st.number_input("Random state (t-SNE)", min_value=0, max_value=9999, key="tsne_random_state")
            case "UMAP":
                st.slider("n_neighbors", min_value=2, max_value=200, key="umap_n_neighbors")
                st.slider("min_dist", min_value=0.0, max_value=0.99, key="umap_min_dist")
                st.number_input("Random state (UMAP)", min_value=0, max_value=9999, key="umap_random_state")
    with c3:
        run_clicked = st.button("Run analysis", type="primary")

    c4, c5 = st.columns([1, 2])
    with c4:
        st.checkbox("Interactive ranges mode", key="interactive_ranges_mode")
    with c5:
        st.checkbox("Clusters in original space", key="clusters_in_original_space")
        if st.session_state["clusters_in_original_space"]:
            st.selectbox("Cluster method", options=["KMeans", "GMM"], key="cluster_method")
            st.slider("Number of clusters", min_value=2, max_value=10, key="cluster_n_clusters")

    st.divider()

    # ── Handle run click ────────────────────────────────────────────────────────
    if run_clicked:
        config = current_config()

        with st.spinner("Computing embedding..."):
            result = compute_embedding(
                method=st.session_state.method,
                X=X,
                config=config,
            )

        st.session_state.embedding_full = result.embedding
        st.session_state.computed_method = st.session_state.method
        st.session_state.explained_variance_ratio = (
            result.explained_variance_ratio if result.explained_variance_ratio is not None else np.array([], dtype=float)
        )

        if st.session_state.method == "PCA" and st.session_state.explained_variance_ratio.size >= MIN_COMPONENTS_FOR_2D:
            ordered_components = np.argsort(st.session_state.explained_variance_ratio)[::-1]
            st.session_state.pca_x_component = int(ordered_components[0])
            st.session_state.pca_y_component = int(ordered_components[1])

        st.session_state.selected_indices = []
        st.session_state.selected_df = pd.DataFrame()
        st.session_state.latest_selection_config = config

    if st.session_state.embedding_full.size == 0:
        st.info("Click 'Run analysis' above to compute and display the embedding.")
        return

    if st.session_state.method != st.session_state.computed_method:
        st.warning("Method changed. Click 'Run analysis' to refresh the embedding.")
        return

    # ── Resolve embedding_2d (PCA component selection handled in plot column) ──
    if st.session_state.method == "PCA":
        explained_ratio = st.session_state.explained_variance_ratio
        if explained_ratio.size < MIN_COMPONENTS_FOR_2D:
            st.error("PCA metadata is unavailable. Run reduction again.")
            return

        component_count = st.session_state.embedding_full.shape[1]
        component_labels = [f"PC {i + 1}" for i in range(component_count)]
        st.session_state.pca_x_component = min(max(0, int(st.session_state.pca_x_component)), component_count - 1)
        st.session_state.pca_y_component = min(max(0, int(st.session_state.pca_y_component)), component_count - 1)

        if st.session_state.pca_x_component == st.session_state.pca_y_component:
            st.warning("Choose two different PCA components for x and y.")
            return

        embedding_2d = st.session_state.embedding_full[:, [st.session_state.pca_x_component, st.session_state.pca_y_component]]
    else:
        embedding_2d = st.session_state.embedding_full
        explained_ratio = np.array([], dtype=float)
        component_count = 0
        component_labels: list[str] = []

    # ── Build plot_df, scaler, clusters, interactive mask ──────────────────────
    plot_df = df.copy()
    plot_df["x"] = embedding_2d[:, 0]
    plot_df["y"] = embedding_2d[:, 1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    cluster_mode = bool(st.session_state["clusters_in_original_space"])
    cluster_labels = np.array([], dtype=int)
    if cluster_mode:
        try:
            with st.spinner("Computing clusters in original space..."):
                cluster_labels = compute_clusters(
                    X_scaled=X_scaled,
                    method=cast("str", st.session_state["cluster_method"]),
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
    if interactive_mode:
        interactive_mask = compute_interactive_mask(X_scaled_df, feature_columns)
        plot_df["interactive_group"] = np.where(interactive_mask, "Matches filters", "Other")

    if cluster_mode and cluster_labels.shape[0] == len(plot_df):
        plot_df["cluster_label"] = pd.Series(cluster_labels, index=plot_df.index).astype(str)

    st.session_state.plot_df = plot_df

    # ── Build scatter figure ────────────────────────────────────────────────────
    if cluster_mode and interactive_mode and "cluster_label" in st.session_state.plot_df.columns:
        fig = px.scatter(
            st.session_state.plot_df,
            x="x",
            y="y",
            color="interactive_group",
            color_discrete_map={"Matches filters": "#1f77b4", "Other": "#bdbdbd"},
            symbol="cluster_label",
            hover_data=["row_id", "quality"],
            title=f"{st.session_state.method} projection - interactive filters + clusters in original space",
            height=700,
        )
    elif cluster_mode and "cluster_label" in st.session_state.plot_df.columns:
        fig = px.scatter(
            st.session_state.plot_df,
            x="x",
            y="y",
            color="cluster_label",
            hover_data=["row_id", "quality"],
            title=f"{st.session_state.method} projection - clusters in original space",
            height=700,
        )
    elif interactive_mode:
        fig = px.scatter(
            st.session_state.plot_df,
            x="x",
            y="y",
            color="interactive_group",
            color_discrete_map={"Matches filters": "#1f77b4", "Other": "#bdbdbd"},
            hover_data=["row_id", "quality"],
            title=f"{st.session_state.method} projection - interactive feature range filtering",
            height=700,
        )
    else:
        fig = px.scatter(
            st.session_state.plot_df,
            x="x",
            y="y",
            hover_data=["row_id", "quality"],
            title=f"{st.session_state.method} projection",
            height=700,
        )

    fig.update_traces(marker={"size": 8, "opacity": 0.85})
    fig.update_layout(dragmode="lasso")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if st.session_state.method == "PCA":
        left, right = st.columns(2)
        with left:
            st.selectbox(
                "PCA x-axis",
                options=list(range(component_count)),
                format_func=lambda i: component_labels[i],
                key="pca_x_component",
            )
        with right:
            st.selectbox(
                "PCA y-axis",
                options=list(range(component_count)),
                format_func=lambda i: component_labels[i],
                key="pca_y_component",
            )
        render_pca_variance_summary(explained_ratio)
        selected_total_variance = explained_ratio[st.session_state.pca_x_component] + explained_ratio[st.session_state.pca_y_component]
        st.info(f"Total variance explained by selected components: {selected_total_variance:.2%}")

    st.divider()
    # ── Two-column results layout ───────────────────────────────────────────────
    col_ranges, col_plot = st.columns([1, 1.4])

    with col_plot:
        event = st.plotly_chart(
            fig,
            key="reduction_plot",
            width="stretch",
            on_select="ignore" if interactive_mode else "rerun",
            selection_mode=("lasso", "box"),
        )

    # ── Handle selection (uses event from col_plot) ─────────────────────────────
    if interactive_mode:
        selected_indices = np.flatnonzero(st.session_state.plot_df["interactive_group"] == "Matches filters").tolist()
        st.session_state.selected_indices = selected_indices
        if selected_indices:
            st.session_state.selected_df = st.session_state.plot_df.iloc[selected_indices].copy()
        else:
            st.session_state.selected_df = pd.DataFrame()
    else:
        selected_indices = get_selected_indices(event)
        st.session_state.selected_indices = selected_indices

        if selected_indices:
            st.session_state.selected_df = st.session_state.plot_df.iloc[selected_indices].copy()
        else:
            st.session_state.selected_df = pd.DataFrame()

    # ── Left column: analysis & ranges ─────────────────────────────────────────
    with col_ranges:
        st.subheader("Analysis of selected Datapoints")
        st.text(
            "RCM = Retained Center Mass. RCM=1.0 means the full range of selected points, "
            "RCM=0.9 trims the tails to exclude outliers and show the core range.",
        )
        if interactive_mode:
            st.caption("Configure feature-wise standardized ranges. Matching points are highlighted in blue in the projection.")
            selected_feature_defaults = [feature for feature in st.session_state["interactive_features"] if feature in feature_columns]
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
                current_range = cast("tuple[float, float]", st.session_state.get(slider_key, (full_min, full_max)))
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
        elif st.session_state.selected_df.empty:
            st.info("Use lasso or box selection in the plot to capture points.")
        else:
            selected_scaled = X_scaled.take(st.session_state.selected_indices, axis=0)
            selected_scaled_df = pd.DataFrame(selected_scaled, columns=feature_columns)

            range_data_full = generate_predicate("hm", selected_scaled_df, X_scaled, threshold=1.0)
            range_data_trimmed = generate_predicate("hm", selected_scaled_df, X_scaled, threshold=0.9)
            range_df_full = pd.DataFrame(range_data_full)
            range_df_trimmed = pd.DataFrame(range_data_trimmed)

            if not range_df_full.empty:
                g_span = range_df_full["global_max"] - range_df_full["global_min"]

                def norm(series: pd.Series) -> pd.Series:
                    return 2.0 * (series - range_df_full["global_min"]) / g_span - 1.0

                full_norm_min = norm(range_df_full["sel_min"])
                full_norm_max = norm(range_df_full["sel_max"])

                g_span_t = range_df_trimmed["global_max"] - range_df_trimmed["global_min"]
                trim_norm_min = 2.0 * (range_df_trimmed["sel_min"] - range_df_trimmed["global_min"]) / g_span_t - 1.0
                trim_norm_max = 2.0 * (range_df_trimmed["sel_max"] - range_df_trimmed["global_min"]) / g_span_t - 1.0

                feature_range_fig = go.Figure()
                feature_range_fig.add_trace(
                    go.Bar(
                        name="Global range",
                        y=range_df_full["feature"],
                        x=[2.0] * len(range_df_full),
                        base=[-1.0] * len(range_df_full),
                        orientation="h",
                        marker={"color": "rgba(120, 120, 120, 0.35)"},
                        hoverinfo="skip",
                    ),
                )
                feature_range_fig.add_trace(
                    go.Bar(
                        name="Selected range (threshold=1.0)",
                        y=range_df_full["feature"],
                        x=full_norm_max - full_norm_min,
                        base=full_norm_min,
                        customdata=range_df_full[["sel_min", "sel_max"]],
                        orientation="h",
                        marker={"color": "rgba(40, 130, 255, 0.75)"},
                        hovertemplate=(
                            "<b>%{y}</b><br>Selected min: %{customdata[0]:.2f}<br>Selected max: %{customdata[1]:.2f}"
                            "<br>Selected range: %{x:.2f}<extra></extra>"
                        ),
                    ),
                )
                feature_range_fig.add_trace(
                    go.Bar(
                        name="Selected range (threshold=0.9)",
                        y=range_df_trimmed["feature"],
                        x=trim_norm_max - trim_norm_min,
                        base=trim_norm_min,
                        customdata=range_df_trimmed[["sel_min", "sel_max"]],
                        orientation="h",
                        marker={"color": "rgba(255, 140, 60, 0.8)"},
                        hovertemplate=(
                            "<b>%{y}</b><br>Trimmed min: %{customdata[0]:.2f}<br>Trimmed max: %{customdata[1]:.2f}"
                            "<br>Trimmed range: %{x:.2f}<extra></extra>"
                        ),
                    ),
                )
                feature_range_fig.update_layout(
                    title="Selected subset standardized ranges (RCM=1.0 and 0.9) inside global feature range",
                    xaxis_title="Standardized feature value",
                    yaxis_title="Feature",
                    barmode="overlay",
                    bargap=0.45,
                    height=max(320, 28 * len(range_df_full) + 120),
                    legend={"orientation": "h", "y": 1.1, "x": 0},
                )
                feature_range_fig.update_xaxes(range=[-1, 1])
                st.plotly_chart(feature_range_fig, width="stretch")

    # ── Full-width: selected datapoints table ───────────────────────────────────
    st.subheader("Selected Datapoints")
    st.write(f"Selected points: {len(st.session_state.selected_indices)}")

    if st.session_state.selected_df.empty:
        st.info("Use lasso or box selection in the plot to capture points.")
        return

    st.dataframe(st.session_state.selected_df.head(50), width="stretch")

    if st.button("Export selected points to CSV"):
        file_path = export_selection(st.session_state.selected_df, st.session_state.latest_selection_config or current_config())
        st.success(f"Selection exported to {file_path}")


if __name__ == "__main__":
    main()

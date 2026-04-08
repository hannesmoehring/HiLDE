from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.dim_reducer import fit_dimensionality_reducer

DATASET_PATH = Path("datasets/wine_quality/wine+quality/winequality-red.csv")
EXPORT_DIR = Path("outputs/selections")
MIN_COMPONENTS_FOR_2D = 2


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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="SHD - Dimensionality Reduction Explorer", layout="wide")
    init_state()

    st.title("SHD Dimensionality Reduction Explorer")
    st.caption("Dataset is hardcoded to winequality-red.csv")

    df = load_dataset()
    feature_columns = [c for c in df.columns if c not in ["quality", "row_id"]]
    X = df[feature_columns].to_numpy()

    with st.sidebar:
        st.header("Controls")

        st.selectbox("Method", options=["PCA", "t-SNE", "UMAP"], key="method")
        st.checkbox("Normalize (StandardScaler)", key="normalize")

        max_components = int(min(X.shape[0], X.shape[1]))
        st.slider("PCA fitted components", min_value=2, max_value=max_components, key="pca_components")

        st.subheader("t-SNE")
        max_perplexity = float(max(5.0, min(50.0, X.shape[0] - 1.0)))
        st.slider("Perplexity", min_value=5.0, max_value=max_perplexity, key="tsne_perplexity")
        st.number_input("Learning rate", min_value=10.0, max_value=2000.0, key="tsne_learning_rate")
        st.number_input("Random state (t-SNE)", min_value=0, max_value=9999, key="tsne_random_state")

        st.subheader("UMAP")
        st.slider("n_neighbors", min_value=2, max_value=200, key="umap_n_neighbors")
        st.slider("min_dist", min_value=0.0, max_value=0.99, key="umap_min_dist")
        st.number_input("Random state (UMAP)", min_value=0, max_value=9999, key="umap_random_state")

        run_clicked = st.button("Run reduction", type="primary")

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
        st.info("Click 'Run reduction' in the sidebar to compute and display the embedding.")
        return

    if st.session_state.method != st.session_state.computed_method:
        st.warning("Method changed. Click 'Run reduction' to refresh the embedding.")
        return

    if st.session_state.method == "PCA":
        explained_ratio = st.session_state.explained_variance_ratio
        if explained_ratio.size < MIN_COMPONENTS_FOR_2D:
            st.error("PCA metadata is unavailable. Run reduction again.")
            return

        component_count = st.session_state.embedding_full.shape[1]
        component_labels = [f"PC {i + 1}" for i in range(component_count)]
        st.session_state.pca_x_component = min(max(0, int(st.session_state.pca_x_component)), component_count - 1)
        st.session_state.pca_y_component = min(max(0, int(st.session_state.pca_y_component)), component_count - 1)

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

        if st.session_state.pca_x_component == st.session_state.pca_y_component:
            st.warning("Choose two different PCA components for x and y.")
            return

        embedding_2d = st.session_state.embedding_full[:, [st.session_state.pca_x_component, st.session_state.pca_y_component]]
        st.bar_chart(pd.DataFrame({"explained_variance_ratio": explained_ratio}))
    else:
        embedding_2d = st.session_state.embedding_full

    plot_df = df.copy()
    plot_df["x"] = embedding_2d[:, 0]
    plot_df["y"] = embedding_2d[:, 1]
    st.session_state.plot_df = plot_df

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

    event = st.plotly_chart(
        fig,
        key="reduction_plot",
        use_container_width=True,
        on_select="rerun",
        selection_mode=("lasso", "box"),
    )

    selected_indices = get_selected_indices(event)
    st.session_state.selected_indices = selected_indices

    if selected_indices:
        st.session_state.selected_df = st.session_state.plot_df.iloc[selected_indices].copy()
    else:
        st.session_state.selected_df = pd.DataFrame()

    st.subheader("Selection State")
    st.write(f"Selected points: {len(st.session_state.selected_indices)}")

    if st.session_state.selected_df.empty:
        st.info("Use lasso or box selection in the plot to capture points.")
        return

    st.dataframe(st.session_state.selected_df.head(50), use_container_width=True)

    if st.button("Export selected points to CSV"):
        file_path = export_selection(st.session_state.selected_df, st.session_state.latest_selection_config or current_config())
        st.success(f"Selection exported to {file_path}")


if __name__ == "__main__":
    main()

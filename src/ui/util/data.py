from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.analysis.clustering import hierarchical_clustering
from src.analysis.dim_reducer import fit_dimensionality_reducer
from src.ui.util.state import ReductionConfig

DATASET_PATH_RED = Path("datasets/wine_quality/wine+quality/winequality-red.csv")
DATASET_PATH_WHITE = Path("datasets/wine_quality/wine+quality/winequality-white.csv")
EXPORT_DIR = Path("outputs/selections")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df_red = pd.read_csv(DATASET_PATH_RED, sep=";")
    df_white = pd.read_csv(DATASET_PATH_WHITE, sep=";")
    df_red["is_red"] = True
    df_white["is_red"] = False
    df = pd.concat([df_red, df_white], ignore_index=True)

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
    return fit_dimensionality_reducer(
        method=method,
        X=X,
        n_components=config["pca_components"] if method == "PCA" else 2,
        normalize=normalize,
        perplexity=config["tsne_perplexity"] if method == "t-SNE" else None,
        learning_rate=config["tsne_learning_rate"] if method == "t-SNE" else None,
        n_neighbors=config["umap_n_neighbors"] if method == "UMAP" else None,
        min_dist=config["umap_min_dist"] if method == "UMAP" else None,
    )


@st.cache_data
def run_hierarchical_clustering(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    *,
    n_components: int = 10,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    min_samples: int = 5,
    min_cluster_size: int = 15,
) -> tuple[pd.DataFrame, np.ndarray, int, np.ndarray]:
    return hierarchical_clustering(
        df,
        X_scaled,
        feature_cols,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )


def get_cluster_subset(
    df: pd.DataFrame,
    X: np.ndarray,
    h_labels: np.ndarray,
    cluster_id: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = h_labels == cluster_id
    return df[mask].reset_index(drop=True), X[mask]


def export_selection(selected_df: pd.DataFrame, config: ReductionConfig) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = EXPORT_DIR / f"selected_points_{timestamp}.csv"

    export_df = selected_df.copy()
    export_df["reduction_method"] = config["method"]
    export_df["reduction_config"] = str(config)
    export_df.to_csv(output_file, index=False)
    return output_file


def build_plot_df(
    df: pd.DataFrame,
    embedding_2d: np.ndarray,
    cluster_labels: np.ndarray,
    interactive_mask: np.ndarray | None,
) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["x"] = embedding_2d[:, 0]
    plot_df["y"] = embedding_2d[:, 1]

    if interactive_mask is not None:
        plot_df["interactive_group"] = np.where(interactive_mask, "Matches filters", "Other")

    if cluster_labels.shape[0] == len(plot_df):
        plot_df["cluster_label"] = pd.Series(cluster_labels, index=plot_df.index).astype(str)

    return plot_df


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

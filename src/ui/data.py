from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    make_swiss_roll,
)

from src.types import Config
from src.util.datasets import load_mnist_images, load_mnist_labels

if TYPE_CHECKING:
    from collections.abc import Callable

DATASET_PATH_RED = Path("datasets/wine_quality/wine+quality/winequality-red.csv")
DATASET_PATH_WHITE = Path("datasets/wine_quality/wine+quality/winequality-white.csv")
MNIST_RAW_DIR = Path("datasets/MNIST/raw")
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


def _one_hot_df(features: np.ndarray, labels: np.ndarray, feature_names: list[str], class_names: list[str]) -> pd.DataFrame:
    """App-format DataFrame: float feature columns, boolean one-hot `target_<name>`
    columns (exactly one True per row), and a `row_id` index column. `labels` must be
    integer-coded 0..k-1 aligned with the order of `class_names`.
    """
    df = pd.DataFrame(np.asarray(features, dtype=np.float64), columns=list(feature_names))
    labels = np.asarray(labels)
    for i, name in enumerate(class_names):
        df[f"target_{name}"] = labels == i
    df["row_id"] = df.index
    return df


def _concentric_rings(
    n_per_ring: int = 600, radii: tuple[float, ...] = (1.0, 2.5, 4.0), noise: float = 0.12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords, labels = [], []
    for i, r in enumerate(radii):
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per_ring)
        radius = r + rng.normal(0.0, noise, n_per_ring)
        coords.append(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]))
        labels.append(np.full(n_per_ring, i))
    return np.vstack(coords), np.concatenate(labels)


@st.cache_data
def load_mnist_dataframe() -> pd.DataFrame:
    names = [str(d) for d in range(10)]
    feature_names = [f"px_{i}" for i in range(784)]
    # Prefer the local IDX files (offline, reliable); fall back to an OpenML download.
    if (MNIST_RAW_DIR / "train-images-idx3-ubyte").exists():
        images = np.concatenate(
            [load_mnist_images(MNIST_RAW_DIR / "train-images-idx3-ubyte"), load_mnist_images(MNIST_RAW_DIR / "t10k-images-idx3-ubyte")],
        )
        labels = np.concatenate(
            [load_mnist_labels(MNIST_RAW_DIR / "train-labels-idx1-ubyte"), load_mnist_labels(MNIST_RAW_DIR / "t10k-labels-idx1-ubyte")],
        )
        return _one_hot_df(images.reshape(images.shape[0], 784) / 255.0, labels, feature_names, names)  # .head(10000)  # TODO: REMOVE LIMIT

    data = fetch_openml("mnist_784", version=1, as_frame=False, n_retries=5, delay=2.0)
    return _one_hot_df(data.data / 255.0, data.target.astype(int), feature_names, names)  # .head(10000)  # TODO: REMOVE LIMIT


@st.cache_data
def load_fashion_mnist_dataframe() -> pd.DataFrame:
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, n_retries=5, delay=2.0)
    names = ["tshirt_top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
    return _one_hot_df(data.data / 255.0, data.target.astype(int), [f"px_{i}" for i in range(784)], names)


@st.cache_data
def load_digits_dataframe() -> pd.DataFrame:
    data = load_digits()
    names = [str(d) for d in range(10)]
    return _one_hot_df(data.data / 16.0, data.target, [f"px_{i}" for i in range(data.data.shape[1])], names)


@st.cache_data
def load_iris_dataframe() -> pd.DataFrame:
    data = load_iris()
    return _one_hot_df(data.data, data.target, list(data.feature_names), list(data.target_names))


@st.cache_data
def load_breast_cancer_dataframe() -> pd.DataFrame:
    data = load_breast_cancer()
    return _one_hot_df(data.data, data.target, list(data.feature_names), list(data.target_names))


@st.cache_data
def load_olivetti_faces_dataframe() -> pd.DataFrame:
    data = fetch_olivetti_faces()
    return _one_hot_df(data.data, data.target, [f"px_{i}" for i in range(data.data.shape[1])], [str(i) for i in range(40)])


@st.cache_data
def load_concentric_dataframe() -> pd.DataFrame:
    coords, labels = _concentric_rings()
    return _one_hot_df(coords, labels, ["x", "y"], ["ring_0", "ring_1", "ring_2"])


@st.cache_data
def load_swiss_roll_dataframe() -> pd.DataFrame:
    coords, position = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df["manifold_position"] = position  # continuous non-feature target (position along the roll)
    df["row_id"] = df.index
    return df


# Display label (with size/complexity marker) → loader. Drives the dataset selectbox.
DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "Wine quality (Low)": load_dataset,
    "Iris (Low)": load_iris_dataframe,
    "Digits (Low)": load_digits_dataframe,
    "Breast cancer (Low)": load_breast_cancer_dataframe,
    "Concentric rings (Low)": load_concentric_dataframe,
    "Swiss roll (Low)": load_swiss_roll_dataframe,
    "Olivetti faces (Medium)": load_olivetti_faces_dataframe,
    "Fashion-MNIST (High)": load_fashion_mnist_dataframe,
    "MNIST (High)": load_mnist_dataframe,
}


def export_selection(selected_df: pd.DataFrame, config: Config) -> Path:
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
    selected_features = [f for f in cast("list[str]", st.session_state.get("interactive_features", [])) if f in feature_columns]
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

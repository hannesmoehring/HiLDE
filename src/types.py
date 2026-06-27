from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"


class Config(TypedDict):
    dataset_choice: str
    characteristics_non_feature_only: bool
    selected_indices: list[int]
    selected_df: pd.DataFrame
    latest_selection_config: None
    plot_df: pd.DataFrame
    embedding_full: np.ndarray
    explained_variance_ratio: np.ndarray
    computed_method: None
    interactive_ranges_mode: bool
    interactive_features: list
    clusters_in_original_space: bool
    cluster_method: str
    cluster_n_clusters: int
    cluster_labels: np.ndarray
    hierarchical_mode: bool
    hierarchical_layers: int

    hclust_normalize: bool
    hclust_umap_n_components: int
    hclust_min_samples: int
    hclust_min_cluster_size: int

    analysis_tree: None
    tree_path: list
    cluster_path_for_embed: tuple

    cluster_embedding_full: np.ndarray
    cluster_explained_variance: np.ndarray
    cluster_pca_x_component: int
    cluster_pca_y_component: int
    method: DRMethod
    normalize: bool
    pca_components: int
    pca_x_component: int
    pca_y_component: int
    tsne_perplexity: float
    tsne_learning_rate: float
    tsne_random_state: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_random_state: int

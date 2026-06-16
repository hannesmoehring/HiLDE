from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import fit_cluster_decision_tree
from src.analysis.clustering import compute_clusters, hierarchical_clustering
from src.analysis.dim_reducer import reduce_dimensionality

_KDE_MIN_PTS = 3


class HierarchyObject(TypedDict):
    rel_characteristics: list | None
    rel_position: tuple[float, float] | None
    kde: gaussian_kde | None
    cluster_points: np.ndarray
    outlier_scores: np.ndarray | None
    next_object_layer: list[AnalysisObject] | None


class ExplorationObject(TypedDict):
    is_leaf: Literal[True]
    depth: int
    kde: gaussian_kde | None
    rel_characteristics: list | None
    rel_position: tuple[float, float] | None
    exploration_points: np.ndarray


type AnalysisObject = HierarchyObject | ExplorationObject


def compute_analysis_tree(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> AnalysisObject:
    if config["normalize"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols])
    else:
        X = df[feature_cols].to_numpy()

    return _build_next(
        X=X,
        config=config,
        depth=0,
        feature_cols=feature_cols,
        rel_position=(0.0, 0.0),
        rel_characteristics=[],
    )


def _build_next(
    X: np.ndarray,
    config: dict,
    depth: int,
    feature_cols: list[str],
    rel_position: tuple[float, float],
    rel_characteristics: list,
) -> AnalysisObject:
    if depth >= config["hierarchical_layers"] or len(X) < config["min_cluster_size"] * 2:
        return ExplorationObject(
            is_leaf=True,
            depth=depth,
            kde=None,
            rel_characteristics=rel_characteristics,
            rel_position=rel_position,
            exploration_points=X,
        )
    else:  # TODO(Hannes): maybe also check if clusters are too small or not well separated?
        # TODO(Hannes): check this, should i normalze again and again for each layer?

        labels, outlier_scores = compute_clusters(X, method="HDBSCAN", min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"])
        mask = labels != -1
        X_scaled_df = pd.DataFrame(X, columns=feature_cols)
        centroids = X_scaled_df[mask].groupby(labels[mask]).mean()

        centroids_2d = reduce_dimensionality("MDS", X=centroids.values, n_components=2)
        rel_positions: dict[int, tuple[float, float]] = {
            cluster_id: (centroids_2d[i, 0], centroids_2d[i, 1]) for i, cluster_id in enumerate(centroids.index)
        }

        rel_characteristics_dict: dict[int, list] = {}
        hierarchy_objects = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            temp_mask = labels == cluster_id
            cluster_X = X[temp_mask]
            hierarchy_objects.append(
                _build_next(
                    X=cluster_X,
                    config=config,
                    depth=depth + 1,
                    feature_cols=feature_cols,
                    rel_position=rel_positions[cluster_id],
                    rel_characteristics=rel_characteristics_dict[cluster_id],
                ),
            )
        return HierarchyObject(
            rel_characteristics=rel_characteristics,
            rel_position=rel_position,
            kde=None,
            cluster_points=X,
            outlier_scores=outlier_scores,
            next_object_layer=hierarchy_objects,
        )

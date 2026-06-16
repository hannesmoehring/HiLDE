from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import compute_cluster_characteristics, compute_cluster_kde, fit_cluster_decision_tree
from src.analysis.clustering import compute_clusters, hierarchical_clustering
from src.analysis.dim_reducer import reduce_dimensionality

_KDE_MIN_PTS = 3
_MIN_CLUSTERS_FOR_HIERARCHY = 2


class HierarchyObject(TypedDict):
    rel_characteristics: pd.DataFrame
    rel_position: tuple[float, float] | None
    kde: gaussian_kde | None
    cluster_points: np.ndarray
    outlier_scores: np.ndarray | None
    next_object_layer: list[AnalysisObject] | None


class ExplorationObject(TypedDict):
    is_leaf: Literal[True]
    depth: int
    kde: gaussian_kde | None
    rel_characteristics: pd.DataFrame
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
        rel_characteristics=pd.DataFrame(),  # global characteristics could be computed here if needed
    )


def _build_next(
    X: np.ndarray,
    config: dict,
    depth: int,
    feature_cols: list[str],
    rel_position: tuple[float, float],
    rel_characteristics: pd.DataFrame,
) -> AnalysisObject:
    if depth >= config["hierarchical_layers"] or len(X) < config["min_cluster_size"] * 2:
        return ExplorationObject(
            is_leaf=True,
            depth=depth,
            kde=compute_cluster_kde(X, config["kde_dr_method"], config),
            rel_characteristics=rel_characteristics,
            rel_position=rel_position,
            exploration_points=X,
        )
    else:  # TODO(Hannes): maybe also check if clusters are too small or not well separated?
        # TODO(Hannes): check this, should i normalze again and again for each layer?

        labels, outlier_scores = compute_clusters(X, method="HDBSCAN", min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"])
        valid_cluster_ids = [c for c in np.unique(labels) if c != -1]
        if len(valid_cluster_ids) < _MIN_CLUSTERS_FOR_HIERARCHY:
            return ExplorationObject(
                is_leaf=True,
                depth=depth,
                kde=compute_cluster_kde(X, config["kde_dr_method"], config),
                rel_characteristics=rel_characteristics,
                rel_position=rel_position,
                exploration_points=X,
            )

        mask = labels != -1
        X_scaled_df = pd.DataFrame(X, columns=feature_cols)
        X_scaled_df["cluster"] = labels
        centroids = X_scaled_df.loc[mask, feature_cols].groupby(labels[mask]).mean()

        centroids_2d = reduce_dimensionality("MDS", X=centroids.values, n_components=2)
        rel_positions: dict[int, tuple[float, float]] = {
            cluster_id: (centroids_2d[i, 0], centroids_2d[i, 1]) for i, cluster_id in enumerate(centroids.index)
        }

        rel_characteristics_dict: dict[int, pd.DataFrame] = {
            cluster_id: compute_cluster_characteristics(
                cluster_id=cluster_id,
                df=X_scaled_df,
                X_scaled_df=X_scaled_df,
                feature_cols=feature_cols,
            )
            for cluster_id in valid_cluster_ids
        }
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
            kde=compute_cluster_kde(cluster_X, config["kde_dr_method"], config) if len(cluster_X) >= _KDE_MIN_PTS else None,
            cluster_points=X,
            outlier_scores=outlier_scores,
            next_object_layer=hierarchy_objects,
        )


def print_tree(node: AnalysisObject, prefix: str = "", *, is_last: bool = True) -> None:
    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    if "is_leaf" in node:
        leaf = cast("ExplorationObject", node)
        pts = leaf["exploration_points"].shape[0]
        pos = leaf["rel_position"]
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos else "N/A"
        kde_str = "kde=yes" if leaf["kde"] is not None else "kde=no"
        rc = leaf["rel_characteristics"]
        char_str = "char=yes" if (isinstance(rc, pd.DataFrame) and not rc.empty) or (isinstance(rc, list) and rc) else "char=no"
        print(f"{prefix}{connector}[LEAF] depth={leaf['depth']}  pts={pts}  pos={pos_str}  {kde_str}  {char_str}")
    else:
        hier: HierarchyObject = node  # type: ignore[assignment]
        children = hier["next_object_layer"] or []
        pts = hier["cluster_points"].shape[0]
        pos = hier["rel_position"]
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos else "N/A"
        scores = hier["outlier_scores"]
        outlier_str = f"  outlier_mean={scores.mean():.3f}" if scores is not None else ""
        kde_str = "kde=yes" if hier["kde"] is not None else "kde=no"
        rc = hier["rel_characteristics"]
        char_str = "char=yes" if (isinstance(rc, pd.DataFrame) and not rc.empty) or (isinstance(rc, list) and rc) else "char=no"
        print(f"{prefix}{connector}[NODE] pts={pts}  children={len(children)}  pos={pos_str}{outlier_str}  {kde_str}  {char_str}")
        for i, child in enumerate(children):
            print_tree(child, prefix=child_prefix, is_last=(i == len(children) - 1))

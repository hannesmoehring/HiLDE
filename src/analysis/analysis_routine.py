from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import compute_cluster_characteristics, compute_cluster_kde, fit_cluster_decision_tree
from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import reduce_dimensionality

_KDE_MIN_PTS = 3
_MIN_CLUSTERS_FOR_HIERARCHY = 2


@dataclass
class _NodeCtx:
    X_orig: np.ndarray
    feature_cols: list[str]
    depth: int
    rel_position: tuple[float, float]
    rel_characteristics: pd.DataFrame
    row_indices: np.ndarray


class HierarchyObject(TypedDict):
    rel_characteristics: pd.DataFrame
    rel_position: tuple[float, float] | None
    kde: np.ndarray | None
    cluster_points: np.ndarray
    row_indices: np.ndarray
    outlier_scores: np.ndarray | None
    next_object_layer: list[AnalysisObject] | None


class ExplorationObject(TypedDict):  # add embedded points
    is_leaf: Literal[True]
    depth: int
    kde: np.ndarray | None
    rel_characteristics: pd.DataFrame
    rel_position: tuple[float, float] | None
    exploration_points: np.ndarray
    row_indices: np.ndarray


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

    umap_n_comp = config["hdbscan_umap_n_components"]
    if umap_n_comp and umap_n_comp < X.shape[1]:
        n_comp = min(umap_n_comp, X.shape[1], len(X) - 1)
        n_nbrs = min(30, len(X) - 1)
        X_reduced = reduce_dimensionality("UMAP", X=X, n_components=n_comp, n_neighbors=n_nbrs, min_dist=0.1)
    else:
        X_reduced = X

    return _build_next(
        X=X_reduced,
        config=config,
        ctx=_NodeCtx(
            X_orig=X,
            feature_cols=feature_cols,
            depth=0,
            rel_position=(0.0, 0.0),
            rel_characteristics=pd.DataFrame(),
            row_indices=np.arange(len(df)),
        ),
    )


def _build_next(X: np.ndarray, config: dict, ctx: _NodeCtx) -> AnalysisObject:
    if ctx.depth >= config["hierarchical_layers"] or len(X) < config["min_cluster_size"] * 2:
        return ExplorationObject(
            is_leaf=True,
            depth=ctx.depth,
            kde=compute_cluster_kde(X, config["kde_dr_method"], config) if len(X) >= _KDE_MIN_PTS else None,
            rel_characteristics=ctx.rel_characteristics,
            rel_position=ctx.rel_position,
            exploration_points=X,
            row_indices=ctx.row_indices,
        )
    else:
        labels, outlier_scores = compute_clusters(
            X,
            method="HDBSCAN",
            min_cluster_size=config["min_cluster_size"],
            min_samples=config["min_samples"],
        )
        valid_cluster_ids = [c for c in np.unique(labels) if c != -1]
        if len(valid_cluster_ids) < _MIN_CLUSTERS_FOR_HIERARCHY:
            return ExplorationObject(
                is_leaf=True,
                depth=ctx.depth,
                kde=compute_cluster_kde(X, config["kde_dr_method"], config) if len(X) >= _KDE_MIN_PTS else None,
                rel_characteristics=ctx.rel_characteristics,
                rel_position=ctx.rel_position,
                exploration_points=X,
                row_indices=ctx.row_indices,
            )

        mask = labels != -1
        X_orig_df = pd.DataFrame(ctx.X_orig, columns=ctx.feature_cols)
        X_orig_df["cluster"] = labels
        reduced_cols = [f"dim_{i}" for i in range(X.shape[1])]
        X_reduced_df = pd.DataFrame(X, columns=reduced_cols)
        X_reduced_df["cluster"] = labels
        centroids = X_reduced_df.loc[mask, reduced_cols].groupby(labels[mask]).mean()

        centroids_2d = reduce_dimensionality("MDS", X=centroids.values, n_components=2)
        rel_positions: dict[int, tuple[float, float]] = {
            cluster_id: (centroids_2d[i, 0], centroids_2d[i, 1]) for i, cluster_id in enumerate(centroids.index)
        }

        rel_characteristics_dict: dict[int, pd.DataFrame] = {
            cluster_id: compute_cluster_characteristics(
                cluster_id=cluster_id,
                df=X_orig_df,
                X_scaled_df=X_orig_df,
                feature_cols=ctx.feature_cols,
            )
            for cluster_id in valid_cluster_ids
        }
        hierarchy_objects = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            temp_mask = labels == cluster_id
            hierarchy_objects.append(
                _build_next(
                    X=X[temp_mask],
                    config=config,
                    ctx=_NodeCtx(
                        X_orig=ctx.X_orig[temp_mask],
                        feature_cols=ctx.feature_cols,
                        depth=ctx.depth + 1,
                        rel_position=rel_positions[cluster_id],
                        rel_characteristics=rel_characteristics_dict[cluster_id],
                        row_indices=ctx.row_indices[temp_mask],
                    ),
                ),
            )
        return HierarchyObject(
            rel_characteristics=ctx.rel_characteristics,
            rel_position=ctx.rel_position,
            kde=compute_cluster_kde(X, config["kde_dr_method"], config) if len(X) >= _KDE_MIN_PTS else None,
            cluster_points=X,
            row_indices=ctx.row_indices,
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

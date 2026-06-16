from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import fit_cluster_decision_tree
from src.analysis.clustering import hierarchical_clustering
from src.analysis.dim_reducer import reduce_dimensionality

_KDE_MIN_PTS = 3


class HierarchyObject(TypedDict):
    characteristics_list: list
    position: tuple[float, float]
    kde: gaussian_kde
    cluster_points: list
    next_layer_object: AnalysisLayer


class HierarchicalLayer(TypedDict):
    is_leaf: Literal[False]
    depth: int
    hierarchy_object_list: list[HierarchyObject]


class ExplorationLayer(TypedDict):
    is_leaf: Literal[True]
    depth: int
    exploration_points: list


type AnalysisLayer = HierarchicalLayer | ExplorationLayer


def compute_analysis_tree(
    df: pd.DataFrame,
    # X_scaled: np.ndarray,
    feature_cols: list[str],
    config: dict,
) -> AnalysisLayer:
    return _build_layer(df=df, feature_cols=feature_cols, depth=0, config=config)  # X_scaled=X_scaled,


def _build_layer(
    df: pd.DataFrame,
    # X_scaled: np.ndarray,
    feature_cols: list[str],
    depth: int,
    config: dict,
) -> AnalysisLayer:
    if depth >= config["hierarchical_layers"] or len(df) < config["min_cluster_size"] * 2:
        return ExplorationLayer(
            is_leaf=True,
            depth=depth,
            exploration_points=list(df.index),
        )

    X_hc = StandardScaler().fit_transform(df[feature_cols])
    n_comp = min(config["hclust_umap_n_components"], X_hc.shape[1], len(df) - 1)
    n_nbrs = min(30, len(df) - 1)

    layout_df, labels, _n_outliers, _outlier_scores = hierarchical_clustering(
        df,
        X_hc,
        feature_cols,
        n_components=n_comp,
        n_neighbors=n_nbrs,
        min_samples=config["min_samples"],
        min_cluster_size=config["min_cluster_size"],
        min_dist=config["umap_min_dist"],
    )

    if layout_df.empty:
        return ExplorationLayer(
            is_leaf=True,
            depth=depth,
            exploration_points=list(df.index),
        )

    X_scaled_df = pd.DataFrame(X_hc, columns=feature_cols, index=df.index)
    hierarchy_objects: list[HierarchyObject] = []

    for _, row in layout_df.iterrows():
        cluster_id = int(row["cluster"])
        mask = labels == cluster_id
        cluster_df = df[mask].reset_index(drop=True)
        # cluster_X = X_scaled[mask]
        cluster_points = list(df.index[mask])

        position = (float(row["x"]), float(row["y"]))

        pts = X_scaled_df.loc[mask].to_numpy()
        if len(pts) >= _KDE_MIN_PTS:
            pts_2d = reduce_dimensionality(
                "mds",
                X=pts,
                n_init=config["n_init"],
                normalized_stress="auto",
                dissimilarity="euclidean",
                n_components=2,
            )
            kde = gaussian_kde(pts_2d.T, bw_method="scott")

        rules_str = fit_cluster_decision_tree(
            df,
            feature_cols,
            pd.Series(mask, index=df.index),
        )
        characteristics_list = rules_str.splitlines()

        # next_layer = _build_layer(df=cluster_df, X_scaled=cluster_X, feature_cols=feature_cols, depth=depth + 1, config=config)
        next_layer = _build_layer(df=cluster_df, feature_cols=feature_cols, depth=depth + 1, config=config)

        hierarchy_objects.append(
            HierarchyObject(
                characteristics_list=characteristics_list,
                position=position,
                kde=kde,
                cluster_points=cluster_points,
                next_layer_object=next_layer,
            ),
        )

    return HierarchicalLayer(
        is_leaf=False,
        depth=depth,
        hierarchy_object_list=hierarchy_objects,
    )


def _print_tree(node: AnalysisLayer, indent: int = 0) -> None:
    pad = "  " * indent
    if node["is_leaf"]:
        leaf = cast("ExplorationLayer", node)
        print(f"{pad}ExplorationLayer(depth={leaf['depth']}, points={len(leaf['exploration_points'])})")
    else:
        hier = cast("HierarchicalLayer", node)
        print(f"{pad}HierarchicalLayer(depth={hier['depth']}, clusters={len(hier['hierarchy_object_list'])})")
        for obj in hier["hierarchy_object_list"]:
            print(
                f"{pad}  HierarchyObject pos={obj['position']} pts={len(obj['cluster_points'])}"
                f" kde={obj['kde'] is not None} chars={len(obj['characteristics_list'])}",
            )
            _print_tree(obj["next_layer_object"], indent + 2)

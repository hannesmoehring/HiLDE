from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import fit_cluster_decision_tree
from src.analysis.clustering import hierarchical_clustering
from src.analysis.dim_reducer import reduce_dimensionality

if TYPE_CHECKING:
    import numpy as np

_KDE_MIN_PTS = 3


class HierarchyObject(TypedDict):
    characteristics_list: list
    position: tuple[float, float]
    kde: gaussian_kde | None
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


class _LayerConfig(TypedDict):
    max_depth: int
    min_cluster_size: int
    min_samples: int
    n_components: int
    kde_dr_method: str


def compute_analysis_tree(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> AnalysisLayer:
    return _build_layer(df=df, X_scaled=X_scaled, feature_cols=feature_cols, depth=0)


def _build_layer(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    depth: int,
) -> AnalysisLayer:
    if depth >= st.session_state["hierarchical_layers"] or len(df) < st.session_state["min_cluster_size"] * 2:
        return ExplorationLayer(
            is_leaf=True,
            depth=depth,
            exploration_points=list(df.index),
        )

    X_hc = StandardScaler().fit_transform(X_scaled)
    n_comp = min(st.session_state["hclust_umap_n_components"], X_hc.shape[1], len(df) - 1)
    n_nbrs = min(30, len(df) - 1)

    layout_df, labels, _n_outliers, _outlier_scores = hierarchical_clustering(
        df,
        X_hc,
        feature_cols,
        n_components=n_comp,
        n_neighbors=n_nbrs,
        min_samples=st.session_state["min_samples"],
        min_cluster_size=st.session_state["min_cluster_size"],
        min_dist=st.session_state["umap_min_dist"],
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
        cluster_X = X_scaled[mask]
        cluster_points = list(df.index[mask])

        position = (float(row["x"]), float(row["y"]))

        pts = X_scaled_df.loc[mask].to_numpy()
        if len(pts) >= _KDE_MIN_PTS:
            pts_2d = reduce_dimensionality(st.session_state["explore_method"], X=pts, n_components=2)
            kde = gaussian_kde(pts_2d.T, bw_method="scott")
        else:
            kde = None

        rules_str = fit_cluster_decision_tree(
            df,
            feature_cols,
            pd.Series(mask, index=df.index),
        )
        characteristics_list = rules_str.splitlines()

        next_layer = _build_layer(
            df=cluster_df,
            X_scaled=cluster_X,
            feature_cols=feature_cols,
            depth=depth + 1,
        )

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


def example_run() -> None:
    dataset_path = Path("datasets/wine_quality/wine+quality/winequality-red.csv")
    df = pd.read_csv(dataset_path, sep=";").head(300).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "quality"]
    X_scaled = StandardScaler().fit_transform(df[feature_cols].to_numpy())

    fake_state = {
        "hierarchical_layers": 2,
        "hclust_saved_config": {
            "min_cluster_size": 20,
            "min_samples": 5,
            "umap_n_components": 5,
            "kde_dr_method": "UMAP",
            "umap_n_neighbors": 15,
            "explore_method": "UMAP",
        },
    }
    mock_state = MagicMock()
    mock_state.__getitem__.side_effect = fake_state.__getitem__
    mock_state.get.side_effect = fake_state.get

    with patch.object(st, "session_state", mock_state):
        tree = compute_analysis_tree(df, X_scaled, feature_cols)

    print("\n--- Analysis Tree ---")
    _print_tree(tree)


if __name__ == "__main__":
    example_run()

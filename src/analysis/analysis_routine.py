from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.analysis.characteristics import (
    compute_cluster_characteristics,
)
from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import fit_dimensionality_reducer, reduce_dimensionality
from src.types import Config
from src.util import console as clog

_MIN_CLUSTERS_FOR_HIERARCHY = 2
_MIN_EMBED_DIMS = 2


@dataclass
class _NodeCtx:
    X_orig: np.ndarray
    feature_cols: list[str]
    depth: int
    rel_position: tuple[float, float]
    rel_characteristics: pd.DataFrame
    row_indices: np.ndarray
    X_feat_raw: np.ndarray  # unscaled values of the feature columns (X_orig is scaled)
    X_nonfeat: np.ndarray  # raw values of numeric columns not selected as features
    nonfeat_cols: list[str]


class NodeScores(TypedDict):
    n_points: int
    k: int | None
    trustworthiness: float | None
    continuity: float | None
    mrre_false: float | None
    mrre_missing: float | None
    stress: float | None
    cadi: float | None


class HierarchyObject(TypedDict):
    rel_characteristics: pd.DataFrame
    rel_position: tuple[float, float] | None
    cluster_points: np.ndarray
    embedding_original: np.ndarray | None  # None = this node could not be projected
    embedding_original_variance: np.ndarray | None
    row_indices: np.ndarray
    outlier_scores: np.ndarray | None
    next_object_layer: list[AnalysisObject] | None
    scores: NotRequired[NodeScores]
    scaler: NotRequired[StandardScaler | None]


class ExplorationObject(TypedDict):  # add embedded points
    is_leaf: Literal[True]
    depth: int
    rel_characteristics: pd.DataFrame
    rel_position: tuple[float, float] | None
    exploration_points: np.ndarray
    embedding_original: np.ndarray | None  # None = this node could not be projected
    embedding_original_variance: np.ndarray | None
    row_indices: np.ndarray
    scores: NotRequired[NodeScores]
    scaler: NotRequired[StandardScaler | None]


type AnalysisObject = HierarchyObject | ExplorationObject


def _embed_original(
    X_orig: np.ndarray, config: Config
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """2D projection of a node's original (normalized) features, plus PCA explained
    variance when applicable. Returns `None` when the node cannot be projected — too
    small, or the reducer raised. Never a fabricated zero embedding: an (n, 2) array of
    origins passes every downstream shape check, so a failed projection would be scored
    and reported as a real DR-quality result. (It was, in an earlier revision.)
    """
    n = X_orig.shape[0]
    if n < _MIN_EMBED_DIMS or X_orig.shape[1] < _MIN_EMBED_DIMS:
        return None, None
    try:
        result = fit_dimensionality_reducer(
            method=config["method"], X=X_orig, n_components=2, config=config
        )
    except Exception as exc:
        clog.warn(
            f"Projection failed for a node of {n} points ({type(exc).__name__}: {exc}) — left unembedded"
        )
        return None, None
    return result.embedding, result.explained_variance_ratio


def compute_analysis_tree(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: Config,
) -> AnalysisObject:
    clog.phase("Computing analysis tree")
    if config["normalize"]:
        scaler: StandardScaler | None = StandardScaler()
        X = scaler.fit_transform(df[feature_cols].to_numpy())
    else:
        scaler = None
        X = df[feature_cols].to_numpy()

    # Numeric columns the user did not pick as features — surfaced (in a distinct
    # color) alongside feature characteristics so their cluster behaviour is visible.
    nonfeat_cols = [
        str(c)
        for c in df.columns
        if c not in feature_cols
        and c != "row_id"
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    X_nonfeat = df[nonfeat_cols].to_numpy() if nonfeat_cols else np.empty((len(df), 0))

    umap_n_comp = config["hclust_umap_n_components"]
    if umap_n_comp and umap_n_comp < X.shape[1]:
        config["hclust_umap_n_components"] = min(umap_n_comp, X.shape[1], len(X) - 1)
        config["umap_n_neighbors"] = min(config["umap_n_neighbors"], len(X) - 1)
        clog.substep(
            f"Pre-clustering UMAP: {X.shape[1]}D → {config['hclust_umap_n_components']}D  ({len(X)} points)"
        )
        X_reduced = reduce_dimensionality(
            "UMAP", X=X, n_components=config["hclust_umap_n_components"], config=config
        )
    else:
        X_reduced = X

    root = _build_next(
        X=X_reduced,
        config=config,
        ctx=_NodeCtx(
            X_orig=X,
            feature_cols=feature_cols,
            depth=0,
            rel_position=(0.0, 0.0),
            rel_characteristics=pd.DataFrame(),
            row_indices=np.arange(len(df)),
            X_feat_raw=df[feature_cols].to_numpy(),
            X_nonfeat=X_nonfeat,
            nonfeat_cols=nonfeat_cols,
        ),
    )
    root["scaler"] = (
        scaler  # fit once; reused by scoring and the global predicate scope
    )
    return root


def _build_next(X: np.ndarray, config: Config, ctx: _NodeCtx) -> AnalysisObject:
    if (
        ctx.depth >= config["hierarchical_layers"]
        or len(X) < config["hclust_min_cluster_size"] * 2
    ):
        emb_orig, emb_var = _embed_original(ctx.X_orig, config)
        return ExplorationObject(
            is_leaf=True,
            depth=ctx.depth,
            rel_characteristics=ctx.rel_characteristics,
            rel_position=ctx.rel_position,
            exploration_points=X,
            embedding_original=emb_orig,
            embedding_original_variance=emb_var,
            row_indices=ctx.row_indices,
        )
    else:
        labels, outlier_scores = compute_clusters(X, method="HDBSCAN", config=config)
        valid_cluster_ids = [c for c in np.unique(labels) if c != -1]
        if len(valid_cluster_ids) < _MIN_CLUSTERS_FOR_HIERARCHY:
            emb_orig, emb_var = _embed_original(ctx.X_orig, config)
            return ExplorationObject(
                is_leaf=True,
                depth=ctx.depth,
                rel_characteristics=ctx.rel_characteristics,
                rel_position=ctx.rel_position,
                exploration_points=X,
                embedding_original=emb_orig,
                embedding_original_variance=emb_var,
                row_indices=ctx.row_indices,
            )

        mask = labels != -1
        X_orig_df = pd.DataFrame(ctx.X_orig, columns=ctx.feature_cols)
        X_orig_df["cluster"] = labels
        # df carrying the *raw* feature + non-feature columns, so `raw_mean` reports
        # original units. z-scores come from X_orig_df (the scaled frame).
        char_df = pd.DataFrame(ctx.X_feat_raw, columns=ctx.feature_cols)
        char_df["cluster"] = labels
        for j, col in enumerate(ctx.nonfeat_cols):
            char_df[col] = ctx.X_nonfeat[:, j]
        reduced_cols = [f"dim_{i}" for i in range(X.shape[1])]
        X_reduced_df = pd.DataFrame(X, columns=reduced_cols)
        X_reduced_df["cluster"] = labels
        centroids = X_reduced_df.loc[mask, reduced_cols].groupby(labels[mask]).mean()

        centroids_2d = reduce_dimensionality(
            "MDS", X=centroids.values, n_components=2, config=config
        )
        rel_positions: dict[int, tuple[float, float]] = {
            cluster_id: (centroids_2d[i, 0], centroids_2d[i, 1])
            for i, cluster_id in enumerate(centroids.index)
        }

        rel_characteristics_dict: dict[int, pd.DataFrame] = {
            cluster_id: compute_cluster_characteristics(
                cluster_id=cluster_id,
                df=char_df,
                X_scaled_df=X_orig_df,
                feature_cols=ctx.feature_cols,
                extra_cols=ctx.nonfeat_cols,
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
                        X_feat_raw=ctx.X_feat_raw[temp_mask],
                        X_nonfeat=ctx.X_nonfeat[temp_mask],
                        nonfeat_cols=ctx.nonfeat_cols,
                    ),
                ),
            )
        emb_orig, emb_var = _embed_original(ctx.X_orig, config)
        return HierarchyObject(
            rel_characteristics=ctx.rel_characteristics,
            rel_position=ctx.rel_position,
            cluster_points=X,
            row_indices=ctx.row_indices,
            embedding_original=emb_orig,
            embedding_original_variance=emb_var,
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
        rc = leaf["rel_characteristics"]
        char_str = (
            "char=yes"
            if (isinstance(rc, pd.DataFrame) and not rc.empty)
            or (isinstance(rc, list) and rc)
            else "char=no"
        )
        print(
            f"{prefix}{connector}[LEAF] depth={leaf['depth']}  pts={pts}  pos={pos_str}  {char_str}"
        )
    else:
        hier: HierarchyObject = node  # type: ignore[assignment]
        children = hier["next_object_layer"] or []
        pts = hier["cluster_points"].shape[0]
        pos = hier["rel_position"]
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos else "N/A"
        scores = hier["outlier_scores"]
        outlier_str = (
            f"  outlier_mean={scores.mean():.3f}" if scores is not None else ""
        )
        rc = hier["rel_characteristics"]
        char_str = (
            "char=yes"
            if (isinstance(rc, pd.DataFrame) and not rc.empty)
            or (isinstance(rc, list) and rc)
            else "char=no"
        )
        print(
            f"{prefix}{connector}[NODE] pts={pts}  children={len(children)}  pos={pos_str}{outlier_str}  {char_str}"
        )
        for i, child in enumerate(children):
            print_tree(child, prefix=child_prefix, is_last=(i == len(children) - 1))

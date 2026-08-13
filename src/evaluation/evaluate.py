from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from zadu.zadu import ZADU

from src.analysis.analysis_routine import AnalysisObject, HierarchyObject, NodeScores, compute_analysis_tree
from src.evaluation.neighbor_metrics import neighbor_scores
from src.types import Config
from src.util import console as clog

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

EVAL_K = 20
MIN_PTS_FOR_NEIGHBORS = 10
MIN_PTS_FOR_EMBED = 2
CADI_RANDOM_SEED = 42


def start_evaluation(df: pd.DataFrame, feature_cols: list[str], config: Config) -> AnalysisObject:
    clog.build_banner(config.get("dataset_choice", "?"), feature_cols, config)
    tree = compute_analysis_tree(df, feature_cols, config)
    clog.phase("Evaluating embedding quality (ZADU)")
    _attach_scores(tree, df, feature_cols, tree.get("scaler"))
    n_nodes, n_leaves = _count_tree(tree)
    clog.success("Build complete", nodes=n_nodes, leaves=n_leaves)
    return tree


def _count_tree(node: AnalysisObject) -> tuple[int, int]:
    """(nodes, leaves) for the build-summary line."""
    if "is_leaf" in node:
        return 0, 1
    nodes, leaves = 1, 0
    for child in node["next_object_layer"] or []:
        cn, cl = _count_tree(child)
        nodes, leaves = nodes + cn, leaves + cl
    return nodes, leaves


def _attach_scores(node: AnalysisObject, df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler | None) -> None:
    # Reuse the embedding the tree already computed (the one the UI displays) and the
    # scaler fit once at the root, so scores measure exactly what is shown.
    X = df.iloc[node["row_indices"]][feature_cols].to_numpy()
    if scaler is not None:
        X = scaler.transform(X)

    emb = node["embedding_original"]
    if emb.shape[0] < MIN_PTS_FOR_EMBED or emb.shape[1] < MIN_PTS_FOR_EMBED:
        emb = None

    if "is_leaf" in node:
        node["scores"] = _score_node(X, emb, None)
        return

    node["scores"] = _score_node(X, emb, _child_labels(node))
    for child in node["next_object_layer"] or []:
        _attach_scores(child, df, feature_cols, scaler)


def _score_node(X: np.ndarray, emb: np.ndarray | None, labels: np.ndarray | None) -> NodeScores:
    scores = NodeScores(
        n_points=X.shape[0],
        k=None,
        trustworthiness=None,
        continuity=None,
        mrre_false=None,
        mrre_missing=None,
        stress=None,
        cadi=None,
    )
    if emb is None:
        return scores

    n = X.shape[0]
    k = None
    if n >= MIN_PTS_FOR_NEIGHBORS:
        k = min(EVAL_K, (n - 1) // 2)
        if k >= 1:
            scores["k"] = k
        else:
            k = None

    try:
        # stress + tnc + mrre in one pass, sharing the distance and ranking work.
        # `neighbor_metrics` reproduces ZADU's formulas without its N×N allocations,
        # which a root node of tens of thousands of points cannot afford.
        scores.update(neighbor_scores(X, emb, k))  # type: ignore[typeddict-item]
    except Exception:
        pass

    if labels is not None:
        scores["cadi"] = _cadi(X, emb, labels)

    return scores


def _child_labels(node: HierarchyObject) -> np.ndarray:
    """Label each of the node's points by which child cluster it fell into (-1 = noise/no child)."""
    pos = {ri: i for i, ri in enumerate(node["row_indices"])}
    labels = np.full(len(node["row_indices"]), -1, dtype=int)
    for child_idx, child in enumerate(node["next_object_layer"] or []):
        for ri in child["row_indices"]:
            labels[pos[ri]] = child_idx
    return labels


def _cadi(X: np.ndarray, emb: np.ndarray, labels: np.ndarray) -> float | None:
    valid = labels >= 0
    lab = labels[valid]
    uniq, counts = np.unique(lab, return_counts=True)
    if lab.shape[0] < 3 or len(uniq) < 2 or counts.max() < 2:
        return None
    try:
        result = ZADU(
            [{"id": "cadi", "params": {"random_seed": CADI_RANDOM_SEED}}],
            orig=X[valid],
        ).measure(emb[valid], label=lab)[0]
        return float(result["Class Angular Distortion Index"])
    except Exception:
        return None

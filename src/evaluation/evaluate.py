from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from zadu.zadu import ZADU

from src.analysis.analysis_routine import AnalysisObject, HierarchyObject, NodeScores, compute_analysis_tree
from src.analysis.dim_reducer import reduce_dimensionality
from src.types import Config

EVAL_DR_METHOD = "PCA"
EVAL_K = 20
MIN_PTS_FOR_NEIGHBORS = 10
CADI_RANDOM_SEED = 42


def start_evaluation(df: pd.DataFrame, feature_cols: list[str], config: Config) -> AnalysisObject:
    tree = compute_analysis_tree(df, feature_cols, config)
    _attach_scores(tree, df, feature_cols, config)
    return tree


def _attach_scores(node: AnalysisObject, df: pd.DataFrame, feature_cols: list[str], config: Config) -> None:
    X = df.iloc[node["row_indices"]][feature_cols].to_numpy()
    if config["normalize"]:
        X = StandardScaler().fit_transform(X)

    emb = _embed_2d(X, config)

    if "is_leaf" in node:
        node["scores"] = _score_node(X, emb, None)
        return

    node["scores"] = _score_node(X, emb, _child_labels(node))
    for child in node["next_object_layer"] or []:
        _attach_scores(child, df, feature_cols, config)


def _embed_2d(X: np.ndarray, config: Config) -> np.ndarray | None:
    n = X.shape[0]
    if n < 2 or X.shape[1] < 2:
        return None
    method = config.get("eval_dr_method", EVAL_DR_METHOD)
    return reduce_dimensionality(method, X=X, n_components=2, config=config)


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

    try:
        stress = ZADU([{"id": "stress"}], orig=X).measure(emb)[0]
        scores["stress"] = float(stress["stress"])
    except Exception:
        pass

    if n >= MIN_PTS_FOR_NEIGHBORS:
        k = min(EVAL_K, (n - 1) // 2)
        if k >= 1:
            scores["k"] = k
            try:
                tnc, mrre = ZADU(
                    [{"id": "tnc", "params": {"k": k}}, {"id": "mrre", "params": {"k": k}}],
                    orig=X,
                ).measure(emb)
                scores["trustworthiness"] = float(tnc["trustworthiness"])
                scores["continuity"] = float(tnc["continuity"])
                scores["mrre_false"] = float(mrre["mrre_false"])
                scores["mrre_missing"] = float(mrre["mrre_missing"])
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

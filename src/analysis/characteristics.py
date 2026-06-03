from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def fit_cluster_decision_tree(
    df: pd.DataFrame,
    feature_cols: list[str],
    in_cluster: pd.Series,
    tree_depth: int = 3,
) -> str:
    tree = DecisionTreeClassifier(max_depth=tree_depth, class_weight="balanced", random_state=0)
    tree.fit(df[feature_cols].to_numpy(), in_cluster.to_numpy().astype(int))
    return export_text(tree, feature_names=list(feature_cols))

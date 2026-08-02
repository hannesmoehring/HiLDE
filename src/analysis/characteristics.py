import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def compute_cluster_characteristics(
    cluster_id: int,
    df: pd.DataFrame,
    X_scaled_df: pd.DataFrame,
    feature_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by column name with z_mean, z_std, raw_mean."""
    in_cluster = df["cluster"] == cluster_id
    order = sorted(feature_cols)
    pts = X_scaled_df.loc[in_cluster, feature_cols]

    rows = pd.DataFrame(
        {
            "z_mean": pts.mean()[order],
            "z_std": pts.std()[order],
            "raw_mean": df.loc[in_cluster, feature_cols].mean()[order],
            "is_feature": True,
        },
    )

    if extra_cols:
        extra_order = sorted(extra_cols)
        g_mean = df[extra_order].mean()
        g_std = df[extra_order].std().replace(0, 1)
        c_mean = df.loc[in_cluster, extra_order].mean()
        extra_rows = pd.DataFrame(
            {
                "z_mean": (c_mean - g_mean) / g_std,
                "z_std": df.loc[in_cluster, extra_order].std(),
                "raw_mean": c_mean,
                "is_feature": False,
            },
        )
        rows = pd.concat([rows, extra_rows])

    return rows


def fit_cluster_decision_tree(
    df: pd.DataFrame,
    feature_cols: list[str],
    in_cluster: pd.Series,
    tree_depth: int = 3,
) -> str:
    tree = DecisionTreeClassifier(max_depth=tree_depth, class_weight="balanced", random_state=0)
    tree.fit(df[feature_cols].to_numpy(), in_cluster.to_numpy().astype(int))
    return export_text(tree, feature_names=list(feature_cols))

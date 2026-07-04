import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier, export_text

from src.analysis.dim_reducer import reduce_dimensionality
from src.types import Config


def compute_cluster_kde(
    pts: np.ndarray,
    config: Config,
) -> np.ndarray:
    pts_2d = reduce_dimensionality(method=config["method"], X=pts, n_components=2, config=config)  # type: ignore[arg-type]

    kde = gaussian_kde(pts_2d.T, bw_method="scott")
    pad = 0.5 * pts_2d.std(axis=0).max()
    lx_min, lx_max = pts_2d[:, 0].min() - pad, pts_2d[:, 0].max() + pad
    ly_min, ly_max = pts_2d[:, 1].min() - pad, pts_2d[:, 1].max() + pad

    # Normalised grid [-0.5, 0.5] so callers can scale/position using rel_position + cluster size
    lx_norm = np.linspace(-0.5, 0.5, 60)
    ly_norm = np.linspace(-0.5, 0.5, 60)
    lx_actual = lx_norm * (lx_max - lx_min) + (lx_min + lx_max) / 2
    ly_actual = ly_norm * (ly_max - ly_min) + (ly_min + ly_max) / 2
    LX, LY = np.meshgrid(lx_actual, ly_actual)
    return kde(np.vstack([LX.ravel(), LY.ravel()])).reshape(LX.shape)


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

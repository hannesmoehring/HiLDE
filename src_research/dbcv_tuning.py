import numpy as np
import optuna
import pandas as pd

# kDBCV's type hints reference np.float_, which NumPy 2.0 removed. Its actual
# DBCV computation is NumPy-2/SciPy-1.18 compatible, so restore the alias to let
# the import (evaluated at def time) succeed. See override-dependencies in pyproject.toml.
if not hasattr(np, "float_"):
    np.float_ = np.float64
from kDBCV.DBCV import DBCV_score
from sklearn.preprocessing import StandardScaler

from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import reduce_dimensionality
from src.config_defaults import init_state
from src.datasets import load_dataset
from src.types import Config
from src_research.pipeline_tuning import feature_columns

SEED = 42
N_TRIALS = 200


def clustering_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The standardised feature matrix, features only.

    `df.to_numpy()` took all 14 columns, including `target_is_red` (the binary ground
    truth), `target_quality` and `row_id` — and since the frame is red-then-white
    concatenated, `row_id < 1599` *is* `is_red`. Optimising DBCV in a space where the
    answer is a feature makes any "the clusters recover the wine type" reading circular,
    and `dtype=object` from the mixed bool/int/float mix ran StandardScaler on an object
    array. `feature_columns` is the rule pipeline_tuning already uses: drop row_id and
    target_*.
    """
    feature_cols = feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return StandardScaler().fit_transform(X), feature_cols


def make_objective(X: np.ndarray, config: Config):
    n_dims = X.shape[
        1
    ]  # the real feature count (11), not 14: it bounds the component range

    def objective(trial):
        cfg: Config = config.copy()
        cfg["umap_n_neighbors"] = trial.suggest_int("umap_n_neighbors", 5, 50)
        cfg["umap_min_dist"] = trial.suggest_float("umap_min_dist", 0.0, 0.5)
        cfg["hclust_min_cluster_size"] = trial.suggest_int(
            "hclust_min_cluster_size", 2, 50
        )
        cfg["hclust_min_samples"] = trial.suggest_int("hclust_min_samples", 1, 25)
        cfg["hclust_umap_n_components"] = trial.suggest_int(
            "hclust_umap_n_components", 2, n_dims
        )

        X_reduced = reduce_dimensionality(
            method="UMAP", X=X, n_components=cfg["hclust_umap_n_components"], config=cfg
        )
        labels, _ = compute_clusters(X_reduced, method="HDBSCAN", config=cfg)

        # guard degenerate solutions
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = np.mean(labels == -1)
        if n_clusters < 2 or noise_frac > 0.5:
            return -1.0  # DBCV is in [-1, 1]; worst possible

        score, _ = DBCV_score(
            X, labels
        )  # returns (score, None); scored in ORIGINAL space (optimized)
        # diagnostic: DBCV in the embedding space — if much higher, UMAP is sharpening faint structure
        score_embedded, _ = DBCV_score(X_reduced, labels)
        trial.set_user_attr("dbcv_embedded", float(score_embedded))
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("noise_frac", float(noise_frac))
        return score

    return objective


def main() -> None:
    config: Config = init_state()
    X, feature_cols = clustering_matrix(load_dataset())
    print(f"tuning on {X.shape[0]}x{X.shape[1]}: {feature_cols}")
    # Seeded sampler: an unseeded study cannot be reproduced, so its best_params are not a
    # result anyone else can check.
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(make_objective(X, config), n_trials=N_TRIALS)
    print(study.best_params)
    print("DBCV (original space):", study.best_value)
    print("DBCV (embedding space):", study.best_trial.user_attrs["dbcv_embedded"])
    print(
        "n_clusters:",
        study.best_trial.user_attrs["n_clusters"],
        "| noise_frac:",
        study.best_trial.user_attrs["noise_frac"],
    )


# Everything above is definition only; the dataset load and the 200-trial study run from
# `main`, so importing this module does not start a multi-minute optimisation.
if __name__ == "__main__":
    main()

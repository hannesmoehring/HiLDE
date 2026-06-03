import hdbscan  # sklearn contrib hdbscan version
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture

from src.analysis.dim_reducer import ReductionResult, reduce_dimensionality

# TODO(Hannes): thinking about computing cluster in original space and projected space and then compare
# the computed clusters to compute how well information is preserved? if that makes sense?
# use Adjusted Rand Index or similar to compare clusterings?


def compute_clusters(X_scaled: np.ndarray, method: str, **kwargs) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    match method:
        case "KMeans":
            model = KMeans(n_clusters=kwargs["n_clusters"], random_state=42)
            return model.fit_predict(X_scaled)
        case "GMM":
            model = GaussianMixture(n_components=kwargs["n_components"], random_state=42)
            return model.fit_predict(X_scaled)
        case "DBSCAN":
            model = DBSCAN(eps=kwargs["eps"], min_samples=kwargs["min_samples"])
            return model.fit_predict(X_scaled)
        case "HDBSCAN":
            model = clusterer = hdbscan.HDBSCAN(min_cluster_size=kwargs["min_cluster_size"], min_samples=kwargs["min_samples"])
            labels = model.fit_predict(X_scaled)
            glosh_scores = model.outlier_scores_
            return labels, glosh_scores
            # model = HDBSCAN(min_cluster_size=kwargs["min_cluster_size"], min_samples=kwargs["min_samples"])
            # return model.fit_predict(X_scaled)

        case _:
            raise ValueError(f"Unknown clustering method: {method}")


def hierarchical_clustering(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    min_samples: int,
    min_cluster_size: int,
) -> tuple[pd.DataFrame, np.ndarray, int, np.ndarray]:

    X_umap = reduce_dimensionality("UMAP", X=X_scaled, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    labels, outlier_scores = compute_clusters(X_umap, method="HDBSCAN", min_cluster_size=min_cluster_size, min_samples=min_samples)

    mask = labels != -1
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    centroids = X_scaled_df[mask].groupby(labels[mask]).mean()

    centroids_2d = reduce_dimensionality(
        "MDS",
        X=centroids.values,
        n_components=2,
        n_init=8,
        normalized_stress="auto",
        dissimilarity="euclidean",
    )

    all_sizes = pd.Series(labels).value_counts().sort_index()
    n_outliers = int(all_sizes.get(-1, 0))
    sizes = all_sizes.drop(-1, errors="ignore")

    layout_df = pd.DataFrame(
        {
            "x": centroids_2d[:, 0],
            "y": centroids_2d[:, 1],
            "cluster": centroids.index,
            "size": sizes.to_numpy(),
        },
    )
    return layout_df, labels, n_outliers, outlier_scores

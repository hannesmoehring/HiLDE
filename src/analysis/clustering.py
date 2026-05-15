import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture

# TODO(Hannes): thinking about computing cluster in original space and projected space and then compare
# the computed clusters to compute how well information is preserved? if that makes sense?
# use Adjusted Rand Index or similar to compare clusterings?


def compute_clusters(X_scaled: np.ndarray, method: str, n_clusters: int = 5):
    match method:
        case "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            return model.fit_predict(X_scaled)
        case "GMM":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            return model.fit_predict(X_scaled)
        case "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            return model.fit_predict(X_scaled)

        case _:
            raise ValueError(f"Unknown clustering method: {method}")


def hierarchical_clustering(
    df: pd.DataFrame, X_scaled: np.ndarray, feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0)
    X_umap = reducer.fit_transform(X_scaled)
    model = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = model.fit_predict(X_umap)

    mask = labels != -1
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    centroids = X_scaled_df[mask].groupby(labels[mask]).mean()

    # 2D layout that preserves pairwise centroid distances
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_init=8, normalized_stress="auto")
    centroids_2d = mds.fit_transform(centroids.values)

    sizes = pd.Series(labels[mask]).value_counts().sort_index()

    layout_df = pd.DataFrame(
        {
            "x": centroids_2d[:, 0],
            "y": centroids_2d[:, 1],
            "cluster": centroids.index,
            "size": sizes.to_numpy(),
        },
    )
    return layout_df, labels

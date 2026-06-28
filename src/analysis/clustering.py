import hdbscan  # sklearn contrib hdbscan version
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from src.types import Config


def compute_clusters(X_scaled: np.ndarray, method: str, config: Config) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    match method:
        case "KMeans":
            model = KMeans(n_clusters=config["cluster_n_clusters"], random_state=42)
            return model.fit_predict(X_scaled)
        case "GMM":
            model = GaussianMixture(n_components=config["hclust_umap_n_components"], random_state=42)
            return model.fit_predict(X_scaled)
        case "DBSCAN":
            model = DBSCAN(eps=config["dbscan_eps"], min_samples=config["hclust_min_samples"])
            return model.fit_predict(X_scaled)
        case "HDBSCAN":
            model = hdbscan.HDBSCAN(min_cluster_size=config["hclust_min_cluster_size"], min_samples=config["hclust_min_samples"])
            labels = model.fit_predict(X_scaled)
            return labels, model.outlier_scores_
        case _:
            raise ValueError(f"Unknown clustering method: {method}")

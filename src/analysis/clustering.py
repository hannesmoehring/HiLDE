import hdbscan  # sklearn contrib hdbscan version
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


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
            model = hdbscan.HDBSCAN(min_cluster_size=kwargs["min_cluster_size"], min_samples=kwargs["min_samples"])
            labels = model.fit_predict(X_scaled)
            return labels, model.outlier_scores_
        case _:
            raise ValueError(f"Unknown clustering method: {method}")

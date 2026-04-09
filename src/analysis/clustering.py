import numpy as np
from sklearn.cluster import KMeans
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
        case _:
            raise ValueError(f"Unknown clustering method: {method}")

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def reduce_dimensionality(method: str, X: np.ndarray, n_components: int = 2, normalize: bool = True) -> np.ndarray:
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    match method:
        case "PCA":
            return _pca(X, n_components)
        case "t-SNE":
            return _tsne(X, n_components)
        case "UMAP":
            return _umap(X, n_components)
        case _:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")


def _pca(X: np.ndarray, n_components: int) -> np.ndarray:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def _tsne(X: np.ndarray, n_components: int) -> np.ndarray:
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(X)


def _umap(X: np.ndarray, n_components: int) -> np.ndarray:
    umap_reducer = umap.UMAP(n_components=n_components)
    return umap_reducer.fit_transform(X)

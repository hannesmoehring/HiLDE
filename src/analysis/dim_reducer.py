from dataclasses import dataclass

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass
class ReductionResult:
    embedding: np.ndarray
    reducer: object
    explained_variance_ratio: np.ndarray | None = None


def reduce_dimensionality(method: str, X: np.ndarray, n_components: int = 2, *, normalize: bool = True) -> np.ndarray:
    result = fit_dimensionality_reducer(
        method=method,
        X=X,
        n_components=n_components,
        normalize=normalize,
    )
    return result.embedding


def fit_dimensionality_reducer(
    method: str,
    X: np.ndarray,
    n_components: int = 2,
    *,
    normalize: bool = True,
    **kwargs: object,
) -> ReductionResult:
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    match method:
        case "PCA":
            return _pca(X, n_components, **kwargs)
        case "t-SNE":
            return _tsne(X, n_components, **kwargs)
        case "UMAP":
            return _umap(X, n_components, **kwargs)
        case _:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")


def _pca(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    pca = PCA(n_components=n_components, **kwargs)
    embedding = pca.fit_transform(X)
    return ReductionResult(
        embedding=embedding,
        reducer=pca,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def _tsne(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    tsne = TSNE(n_components=n_components, **kwargs)
    embedding = tsne.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=tsne)


def _umap(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    umap_reducer = umap.UMAP(n_components=n_components, **kwargs)
    embedding = umap_reducer.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=umap_reducer)

from dataclasses import dataclass

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

KDE_NONLINEAR_MIN_PTS = 15  # minimum points before using UMAP/t-SNE locally


@dataclass
class ReductionResult:
    embedding: np.ndarray
    reducer: object
    explained_variance_ratio: np.ndarray | None = None


def reduce_dimensionality(
    method: str,
    X: np.ndarray,
    n_components: int = 2,
    *,
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    result = fit_dimensionality_reducer(
        method=method,
        X=X,
        n_components=n_components,
        normalize=normalize,
        **kwargs,
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
            return _pca(X, n_components=n_components)
        case "t-SNE":
            return _tsne(X, n_components=n_components, perplexity=kwargs["perplexity"], learning_rate=kwargs["learning_rate"])
        case "UMAP":
            return _umap(X, n_components=n_components, n_neighbors=kwargs["n_neighbors"], min_dist=kwargs["min_dist"])
        case "MDS":
            return _mds(
                X,
                n_components=n_components,
                n_init=kwargs["n_init"],
                normalized_stress=kwargs["normalized_stress"],
                dissimilarity=kwargs["dissimilarity"],
            )
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


def _mds(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    mds = MDS(n_components=n_components, **kwargs)
    embedding = mds.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=mds)

from dataclasses import dataclass

import numpy as np
import streamlit as st
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

KDE_NONLINEAR_MIN_PTS = 15  # minimum points before using UMAP/t-SNE locally


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


def local_2d(pts: np.ndarray, method: str) -> np.ndarray:
    """Project a small cluster to 2D for KDE. Falls back to PCA for tiny clusters."""
    match method:
        case "UMAP":
            if len(pts) >= KDE_NONLINEAR_MIN_PTS:
                print(f"UMAP settings: n_neighbors={st.session_state['umap_n_neighbors']}, min_dist={st.session_state['umap_min_dist']}")
                return umap.UMAP(
                    n_components=2,
                    random_state=st.session_state["umap_random_state"],
                    n_neighbors=st.session_state["umap_n_neighbors"],
                    min_dist=st.session_state["umap_min_dist"],
                ).fit_transform(pts)
        case "t-SNE":
            if len(pts) >= KDE_NONLINEAR_MIN_PTS:
                perplexity = (
                    (len(pts) - 1) // 3 if len(pts) <= st.session_state["tsne_perplexity"] else st.session_state["tsne_perplexity"]
                )  # TODO: double check fallback value
                return TSNE(
                    n_components=2,
                    random_state=st.session_state["tsne_random_state"],
                    perplexity=perplexity,
                    learning_rate=st.session_state["tsne_learning_rate"],
                ).fit_transform(pts)
        case "PCA":
            return PCA(n_components=2).fit_transform(pts)

    raise ValueError(f"Unsupported KDE DR method: {method}")

from dataclasses import dataclass

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from src.types import Config
from src.util import console as clog

KDE_NONLINEAR_MIN_PTS = 15  # minimum points before using UMAP/t-SNE locally


@dataclass
class ReductionResult:
    embedding: np.ndarray
    reducer: object
    explained_variance_ratio: np.ndarray | None = None


def reduce_dimensionality(
    method: str,
    X: np.ndarray,
    config: Config,
    n_components: int = 2,
) -> np.ndarray:
    result = fit_dimensionality_reducer(
        method=method,
        X=X,
        n_components=n_components,
        config=config,
    )
    return result.embedding


def fit_dimensionality_reducer(
    method: str,
    X: np.ndarray,
    config: Config,
    n_components: int = 2,
) -> ReductionResult:
    clog.substep(
        f"Dim reduction: {method.upper()}  {X.shape[0]}x{X.shape[1]} -> {n_components}D"
    )
    match method.lower():
        case "pca":
            return _pca(X, n_components=n_components)
        case "t-sne":
            return _tsne(
                X,
                n_components=n_components,
                perplexity=config["tsne_perplexity"],
                learning_rate=config["tsne_learning_rate"],
                random_state=config["tsne_random_state"],
            )
        case "umap":
            return _umap(
                X,
                n_components=n_components,
                n_neighbors=config["umap_n_neighbors"],
                min_dist=config["umap_min_dist"],
                random_state=config["umap_random_state"],
            )
        case "mds":
            return _mds(
                X,
                n_components=n_components,
                metric_mds=config["mds_metric"],
                n_init=config["mds_n_init"],
                max_iter=config["mds_max_iter"],
                random_state=config["mds_random_state"],
            )
        case _:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")


def _pca(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    # svd_solver="auto" picks the *randomized* solver on wide or small matrices (e.g.
    # Olivetti's 400x4096 root), which draws from the unseeded process-global RNG.
    # covariance_eigh consults no RNG at all — and is what "auto" already chooses for
    # the tall/narrow shapes, so those embeddings are unchanged.
    pca = PCA(n_components=n_components, svd_solver="covariance_eigh", **kwargs)
    embedding = pca.fit_transform(X)
    return ReductionResult(
        embedding=embedding,
        reducer=pca,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def _tsne(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    # sklearn requires perplexity < n_samples; clamp so t-SNE works on small clusters
    # (e.g. hierarchical sub-regions), not just the full dataset.
    if "perplexity" in kwargs:
        kwargs["perplexity"] = min(
            float(kwargs["perplexity"]), max(1.0, X.shape[0] - 1.0)
        )
    tsne = TSNE(n_components=n_components, **kwargs)
    embedding = tsne.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=tsne)


def _umap(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    # `random_state` alone does not make UMAP reproducible: with the default
    # init="spectral" a disconnected fuzzy graph falls into `multi_component_layout`,
    # which places the components outside the seeded path. A PCA init is deterministic
    # whatever the graph structure.
    umap_reducer = umap.UMAP(n_components=n_components, init="pca", **kwargs)
    embedding = umap_reducer.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=umap_reducer)


def _mds(X: np.ndarray, n_components: int, **kwargs: object) -> ReductionResult:
    mds = MDS(n_components=n_components, init="random", n_jobs=-1, **kwargs)
    embedding = mds.fit_transform(X)
    return ReductionResult(embedding=embedding, reducer=mds)

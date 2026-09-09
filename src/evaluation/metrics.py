"""Stress / T&C / MRRE computed in row blocks, matching ZADU's values exactly.

ZADU materialises three n x n arrays per space — the distance matrix, its
argsort, and the rank matrix (`measures/utils/knn.py::knn_with_ranking`). At
n=70000 that is ~235 GB across both spaces, so scoring a large node fails
outright rather than merely running long.

The measures do not need any of it. `trustworthiness_continuity` and
`mean_relative_rank_error` index the rank matrix only at the *other* space's
k-NN, i.e. k values per row, and every rank they take of a space's own k-NN is
1..k by construction. Stress reduces both matrices to two scalars. So one row
block at a time suffices: memory is O(block * n), and distances come from a
BLAS GEMM rather than scipy's single-threaded cdist, so the work spreads across
all cores.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Per space, per in-flight block. Peaks near 4 GB at n=70000. Bigger blocks buy
# only a few percent; the loop is memory-bandwidth bound, which is also why the
# worker count stops paying off past ~12 (measured 10.3 busy cores on 14).
# Workers are additionally capped at cpu_count-2, so a smaller host scales down.
_BLOCK_BYTES = 64 * 1024 * 1024
_MAX_WORKERS = 12


def _block_size(n: int) -> int:
    return max(1, min(n, _BLOCK_BYTES // (8 * n)))


def _map_blocks(n: int, step: int, fn: Callable[[int, int], object]) -> list:
    """Run `fn(lo, hi)` over row blocks, in parallel where there is more than one.

    Blocks touch disjoint output rows, and numpy releases the GIL for the
    distance and comparison work, so threads scale here. Results come back in
    block order, so float accumulation stays deterministic.
    """
    bounds = [(lo, min(lo + step, n)) for lo in range(0, n, step)]
    workers = min(_MAX_WORKERS, len(bounds), max(1, (os.cpu_count() or 1) - 2))
    if workers == 1:
        return [fn(lo, hi) for lo, hi in bounds]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda b: fn(*b), bounds))


def _sq_norms(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def _dist_block(Xb: np.ndarray, X: np.ndarray, sq_b: np.ndarray, sq_all: np.ndarray) -> np.ndarray:
    """Euclidean distances from rows `Xb` to every row of `X`, as one GEMM.

    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, accumulated in place to hold a single
    block-sized array. The expansion can leave tiny negatives on near-duplicate
    points, so clip before the square root.
    """
    d2 = Xb @ X.T
    d2 *= -2.0
    d2 += sq_b[:, None]
    d2 += sq_all[None, :]
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def _knn_from_block(D: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest rows, ascending by distance (self already +inf)."""
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def _knn_and_stress(X: np.ndarray, emb: np.ndarray, k: int | None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """First pass: the stress accumulators, and k-NN indices for both spaces."""
    n = X.shape[0]
    step = _block_size(n)
    sq_x, sq_e = _sq_norms(X), _sq_norms(emb)
    knn_o = knn_e = None
    if k is not None:
        knn_o = np.empty((n, k), dtype=np.int64)
        knn_e = np.empty((n, k), dtype=np.int64)

    def block(lo: int, hi: int) -> tuple[float, float]:
        d_o = _dist_block(X[lo:hi], X, sq_x[lo:hi], sq_x)
        d_e = _dist_block(emb[lo:hi], emb, sq_e[lo:hi], sq_e)

        diff = d_o - d_e
        partial = (float(np.einsum("ij,ij->", diff, diff)), float(np.einsum("ij,ij->", d_o, d_o)))

        if k is not None:
            rows, cols = np.arange(hi - lo), np.arange(lo, hi)
            d_o[rows, cols] = np.inf  # a point is never its own neighbour
            d_e[rows, cols] = np.inf
            knn_o[lo:hi] = _knn_from_block(d_o, k)
            knn_e[lo:hi] = _knn_from_block(d_e, k)
        return partial

    partials = _map_blocks(n, step, block)
    num = sum(p[0] for p in partials)
    den = sum(p[1] for p in partials)
    return knn_o, knn_e, float(np.sqrt(num / den))


def _cross_ranks(X: np.ndarray, emb: np.ndarray, knn_o: np.ndarray, knn_e: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Second pass: original-space ranks of each point's embedding neighbours, and vice versa.

    rank(i, j) = |{m : d(i, m) < d(i, j)}|, which is what ZADU's
    argsort(argsort(D)) produces: the zero self-distance puts every other point
    at rank >= 1. Exact distance ties (duplicate points) are the one case where
    the two can order differently.
    """
    n = X.shape[0]
    step = _block_size(n)
    sq_x, sq_e = _sq_norms(X), _sq_norms(emb)
    rank_o = np.empty((n, k), dtype=np.int64)  # orig rank of the embedding's neighbours
    rank_e = np.empty((n, k), dtype=np.int64)  # embedding rank of the original's neighbours

    def block(lo: int, hi: int) -> None:
        d_o = _dist_block(X[lo:hi], X, sq_x[lo:hi], sq_x)
        d_e = _dist_block(emb[lo:hi], emb, sq_e[lo:hi], sq_e)
        cut_o = np.take_along_axis(d_o, knn_e[lo:hi], axis=1)
        cut_e = np.take_along_axis(d_e, knn_o[lo:hi], axis=1)
        for c in range(k):
            rank_o[lo:hi, c] = (d_o < cut_o[:, c, None]).sum(axis=1)
            rank_e[lo:hi, c] = (d_e < cut_e[:, c, None]).sum(axis=1)

    _map_blocks(n, step, block)
    return rank_o, rank_e


def _rows_isin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, which entries of `a` also appear in `b` (both n x k)."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2)


def node_scores(X: np.ndarray, emb: np.ndarray, k: int | None = None) -> dict[str, float]:
    """Stress, and (when `k` is given) trustworthiness/continuity/MRRE.

    Values match `ZADU([{"id": "stress"}, {"id": "tnc"}, {"id": "mrre"}], orig=X).measure(emb)`.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    emb = np.ascontiguousarray(emb, dtype=np.float64)
    n = X.shape[0]

    knn_o, knn_e, stress = _knn_and_stress(X, emb, k)
    if k is None:
        return {"stress": stress}

    rank_o, rank_e = _cross_ranks(X, emb, knn_o, knn_e, k)

    # Trustworthiness/continuity: rank penalty over the neighbours one space has
    # that the other lost, normalised as in zadu's tnc_computation.
    norm = 2 / (k * (2 * n - 3 * k - 1))
    trust_pen = ((rank_o - k) * ~_rows_isin(knn_e, knn_o)).sum(axis=1)
    cont_pen = ((rank_e - k) * ~_rows_isin(knn_o, knn_e)).sum(axis=1)

    # MRRE: the rank each space assigns its own k-NN is 1..k by construction, so
    # only the opposite space's ranks (rank_o / rank_e) have to be looked up.
    own = np.arange(1, k + 1, dtype=np.float64)
    c = float(np.sum(np.abs(n - 2 * own + 1) / own))
    false_pen = (np.abs(rank_o - own) / own).sum(axis=1)
    missing_pen = (np.abs(rank_e - own) / own).sum(axis=1)

    return {
        "stress": stress,
        "trustworthiness": float(np.mean(1 - trust_pen * norm)),
        "continuity": float(np.mean(1 - cont_pen * norm)),
        "mrre_false": float(np.mean(1 - false_pen / c)),
        "mrre_missing": float(np.mean(1 - missing_pen / c)),
    }

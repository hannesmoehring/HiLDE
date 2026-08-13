"""Chunked stress / trustworthiness-continuity / MRRE.

Adapted from ZADU (https://github.com/hj-n/zadu, `measures/stress.py`,
`trustworthiness_continuity.py`, `mean_relative_rank_error.py`). The formulas and
their normalisation constants are ZADU's, unchanged — this returns the same numbers.

What differs is the bookkeeping. ZADU computes the pairwise distance matrix and then
`argsort(argsort(...))` for the rankings, in both spaces: six N×N arrays live at once,
which is ~40 GB at N=30k and gets the process OOM-killed. Every quantity those formulas
actually read is row-local (each point needs only the ranks of its own ~2k neighbours),
so the rows are walked in chunks and only the per-row values are kept: O(chunk×N).

ZADU is still the right tool for everything else — this covers only the three measures
whose cost is quadratic in memory.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# Scratch budget per chunk: 2 float64 distance blocks, plus 4 int64 order/ranking
# blocks when the neighbour measures are wanted.
CHUNK_BYTES = 256 * 1024**2


def _chunk_rows(n: int, *, with_ranking: bool) -> int:
    return max(1, min(n, CHUNK_BYTES // ((48 if with_ranking else 16) * n)))


def _order_and_ranking(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZADU's `argsort(d)` and `argsort(argsort(d))`, for a block of rows.

    The outer argsort only inverts a permutation, so scattering each position back
    through the order gives the identical ranking without a second sort.
    """
    order = np.argsort(dist, axis=1)
    ranking = np.empty_like(order)
    np.put_along_axis(ranking, order, np.arange(dist.shape[1]), axis=1)
    return order, ranking


def neighbor_scores(orig: np.ndarray, emb: np.ndarray, k: int | None) -> dict[str, float | None]:
    """Stress, plus — when `k` is given — trustworthiness, continuity and both MRRE terms.

    `k` is None for nodes too small for a meaningful neighbourhood, where only the
    stress accumulation runs and the ranking work is skipped entirely.
    """
    n = orig.shape[0]
    step = _chunk_rows(n, with_ranking=k is not None)

    diff_squared_sum = 0.0
    orig_squared_sum = 0.0
    trust = np.empty(n)
    cont = np.empty(n)
    mrre_false = np.empty(n)
    mrre_missing = np.empty(n)

    for start in range(0, n, step):
        stop = min(start + step, n)
        orig_dist = cdist(orig[start:stop], orig)
        emb_dist = cdist(emb[start:stop], emb)

        diff_squared_sum += float(np.square(orig_dist - emb_dist).sum())
        orig_squared_sum += float(np.square(orig_dist).sum())
        if k is None:
            continue

        orig_order, orig_rank = _order_and_ranking(orig_dist)
        emb_order, emb_rank = _order_and_ranking(emb_dist)
        del orig_dist, emb_dist

        # Column 0 is the point itself, as in ZADU's `sorted_indices[:, 1 : k + 1]`.
        orig_knn = orig_order[:, 1 : k + 1]
        emb_knn = emb_order[:, 1 : k + 1]

        for row in range(stop - start):
            i = start + row
            near_orig, near_emb = orig_knn[row], emb_knn[row]

            trust[i] = (orig_rank[row, np.setdiff1d(near_emb, near_orig)] - k).sum()
            cont[i] = (emb_rank[row, np.setdiff1d(near_orig, near_emb)] - k).sum()

            target = emb_rank[row, near_emb]
            mrre_false[i] = (np.abs(orig_rank[row, near_emb] - target) / target).sum()
            target = orig_rank[row, near_orig]
            mrre_missing[i] = (np.abs(emb_rank[row, near_orig] - target) / target).sum()

    # A node whose points are identical in feature space has no distances to
    # preserve: the ratio is 0/0. Reporting `None` keeps the four neighbourhood
    # measures, which are perfectly well defined there (ZADU returns nan/1.0).
    stress = float(np.sqrt(diff_squared_sum / orig_squared_sum)) if orig_squared_sum > 0 else None
    if k is None:
        return {"stress": stress, "trustworthiness": None, "continuity": None, "mrre_false": None, "mrre_missing": None}

    tnc_norm = 2 / (k * (2 * n - 3 * k - 1))
    mrre_norm = sum(abs(n - 2 * i + 1) / i for i in range(1, k + 1))
    return {
        "stress": stress,
        "trustworthiness": float(np.mean(1 - trust * tnc_norm)),
        "continuity": float(np.mean(1 - cont * tnc_norm)),
        "mrre_false": float(np.mean(1 - mrre_false / mrre_norm)),
        "mrre_missing": float(np.mean(1 - mrre_missing / mrre_norm)),
    }

"""The 24h commit threaded `random_state` into umap.UMAP -- but that does NOT make the
embedding reproducible when the fuzzy kNN graph is disconnected, because UMAP's spectral
initialisation (umap/spectral.py::multi_component_layout) draws per-component offsets
outside the seeded path.

This matters because the DEFAULT dataset ("Wine quality (Low)", 6497 rows) produces a
disconnected graph, and the pre-clustering UMAP at analysis_routine.py:115 feeds HDBSCAN --
so the entire hierarchy changes between two identical builds.

Run (needs the wine CSVs, so run from the repo root, not /tmp):
  cd <repo> && PYTHONPATH=. .venv/bin/python review_repro/10_umap_still_nondeterministic.py
"""

import warnings

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler

from backend.datasets import default_feature_cols
from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import reduce_dimensionality
from src.config_defaults import default_config
from src.datasets import DATASETS

warnings.filterwarnings("ignore")

cfg = default_config()
df = DATASETS["Wine quality (Low)"]()
feats = default_feature_cols(df)
X = StandardScaler().fit_transform(df[feats].to_numpy())
print(f"X: {X.shape}   umap_random_state: {cfg['umap_random_state']}   "
      f"n_neighbors: {cfg['umap_n_neighbors']}")
print()

print("=== 1. three identical UMAP calls through the repo's own code path ===")
embs = [reduce_dimensionality("UMAP", X=X, n_components=2, config=cfg) for _ in range(3)]
scale = float(np.abs(embs[0]).max())
for i in (1, 2):
    same = bool(np.array_equal(embs[0], embs[i]))
    print(f"  run0 vs run{i}: identical={same}   max|diff|={np.abs(embs[0] - embs[i]).max():.4f}"
          f"   (coordinate scale {scale:.2f})")

print()
print("=== 2. what that does to the clustering the hierarchy is built from ===")
for i, e in enumerate(embs):
    labels, _ = compute_clusters(e, method="HDBSCAN", config=cfg)
    uniq = [c for c in np.unique(labels) if c != -1]
    print(f"  run{i}: n_clusters={len(uniq):<4} n_noise={int((labels == -1).sum())}")

print()
print("=== 3. why: the fuzzy graph is disconnected at this size ===")
try:
    from umap.umap_ import fuzzy_simplicial_set
    from sklearn.neighbors import NearestNeighbors

    k = cfg["umap_n_neighbors"]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, idx = nn.kneighbors(X)
    graph, _, _ = fuzzy_simplicial_set(X, k, np.random.RandomState(42), "euclidean",
                                       knn_indices=idx, knn_dists=d)
    n_comp, _ = connected_components(graph, directed=False)
    print(f"  connected components in the fuzzy graph: {n_comp}")
    print("  (1 => spectral init is deterministic; >1 => multi_component_layout is not)")
except Exception as e:  # noqa: BLE001
    print(f"  (could not compute directly: {type(e).__name__}: {e})")

print()
print("=== 4. the fix: a deterministic init ===")
import umap

for init in ("spectral", "pca", "random"):
    a = umap.UMAP(n_components=2, n_neighbors=cfg["umap_n_neighbors"],
                  min_dist=cfg["umap_min_dist"], random_state=42, init=init).fit_transform(X)
    b = umap.UMAP(n_components=2, n_neighbors=cfg["umap_n_neighbors"],
                  min_dist=cfg["umap_min_dist"], random_state=42, init=init).fit_transform(X)
    print(f"  init={init:<9} identical={bool(np.array_equal(a, b))}   "
          f"max|diff|={np.abs(a - b).max():.4f}")
print()
print("  dim_reducer._umap passes no `init`, so UMAP's default 'spectral' is used.")

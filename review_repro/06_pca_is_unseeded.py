"""Claim under test: src/analysis/dim_reducer.py:81 constructs PCA with no random_state,
and sklearn's svd_solver="auto" selects the RANDOMIZED solver for wide/small matrices,
so the DEFAULT method ("PCA", src/config_defaults.py:47) is nondeterministic.

Adversarial framing: try to REFUTE this. If PCA were deterministic on every shape the
repo actually produces, the finding is void.
"""

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import PCA as _PCA

from src.analysis.dim_reducer import fit_dimensionality_reducer
from src.config_defaults import default_config

warnings.filterwarnings("ignore")


def chosen_solver(n, p, k=2):
    """Replicate sklearn's svd_solver='auto' decision for this shape."""
    est = _PCA(n_components=k, svd_solver="auto")
    X = np.zeros((n, p))
    try:
        return est._fit(X) and "?"
    except Exception:
        pass
    # sklearn exposes the decision after a fit; do a real tiny fit instead
    est = PCA(n_components=k, svd_solver="auto").fit(np.random.default_rng(0).normal(size=(n, p)))
    return est.svd_solver_


print("=== 1. which solver does svd_solver='auto' pick for the repo's real shapes? ===")
shapes = [
    (6497, 11, "wine root"),
    (400, 4096, "olivetti root"),
    (70000, 784, "MNIST root"),
    (3000, 784, "MNIST subcluster"),
    (1200, 784, "MNIST subcluster"),
    (30000, 19, "QM9 root"),
    (150, 4, "iris root"),
]
for n, p, tag in shapes:
    if n * p > 20_000_000:
        print(f"  n={n:<6} p={p:<5} {tag:<20} -> (skipped, too large to fit here)")
        continue
    print(f"  n={n:<6} p={p:<5} {tag:<20} -> svd_solver_ = {chosen_solver(n, p)}")

print()
print("=== 2. does the REPO's own code path reproduce across runs? ===")
print("    (fit_dimensionality_reducer('pca', ...), global RNG perturbed in between)")

for n, p, tag in [(600, 12, "narrow -> covariance_eigh"), (400, 3000, "wide -> randomized")]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, p))
    cfg = default_config()
    embs = []
    for trial in range(3):
        np.random.seed(trial * 7919)          # perturb the process-global RNG
        np.random.random(1000)                 # advance it
        embs.append(fit_dimensionality_reducer(method="pca", X=X, config=cfg, n_components=2).embedding)
    same = all(np.array_equal(embs[0], e) for e in embs[1:])
    maxdiff = max(float(np.abs(embs[0] - e).max()) for e in embs[1:])
    print(f"  n={n:<5} p={p:<5} {tag:<28} identical across 3 runs: {same}   max|diff| = {maxdiff:.3e}")

print()
print("=== 3. worst case: near-degenerate spectrum (two nearly equal eigenvalues) ===")
rng = np.random.default_rng(5)
base = rng.normal(size=(1200, 6))
# build a matrix whose 2nd and 3rd components are nearly tied -> unstable subspace
W = rng.normal(size=(6, 784))
X = base @ W + 0.01 * rng.normal(size=(1200, 784))
cfg = default_config()
embs = []
for trial in range(4):
    np.random.seed(trial * 104729)
    np.random.random(5000)
    embs.append(fit_dimensionality_reducer(method="pca", X=X, config=cfg, n_components=2).embedding)
for i, e in enumerate(embs):
    print(f"  run{i}: first point = {e[0, 0]:+.9f}, {e[0, 1]:+.9f}   max|diff vs run0| = {np.abs(embs[0] - e).max():.3e}")

print()
print("=== 4. is a seed available but unused? ===")
cfg = default_config()
print("  config keys containing 'random_state':",
      [k for k in cfg if "random_state" in k])
print("  -> t-SNE/UMAP/MDS each have one and it IS forwarded (dim_reducer.py:57,65,74).")
print("     There is no pca_random_state, and _pca() (dim_reducer.py:80-87) takes no seed.")

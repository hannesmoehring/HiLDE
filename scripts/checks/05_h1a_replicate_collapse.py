"""H1a replicate collapse -- and the check that it is fixed.

src_research/hierarchical_vs_flat.py documents the design:

    "the ``SEEDS`` loop functions as *replicates* that capture embedding variance -
     the unit of the paired H1a test is a (region x seed) pair."

The 24h window (1d85891, src/analysis/dim_reducer.py) started threading
config["umap_random_state"] / ["tsne_random_state"] -- both hardcoded to 42 in
src/config_defaults.py -- into the constructors. run_cell() built cfg from init_state()
and set ONLY cfg["method"] and cfg["hierarchical_layers"]; `seed` went into the label
dict and nowhere else. So every replicate saw random_state=42 and came out byte-identical.

B1 fixes that: run_cell now writes the replicate seed into all three random_state keys.

This script runs both arms side by side -- the old config-building and the new one, on the
same data through the same code path -- and then shows what each does to the Wilcoxon the
H1a summary reports.

Run:
  cd /tmp && PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/scripts/checks/05_h1a_replicate_collapse.py
"""

import warnings

import numpy as np

from src.analysis.analysis_routine import _embed_original
from src.config_defaults import init_state

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 8))

SEEDS = list(range(5))  # mirrors hierarchical_vs_flat.SEEDS


def embed(method: str, seed: int, thread_seed: bool):
    """The config run_cell builds, with and without B1's line."""
    cfg = init_state(init_streamlit=False)
    cfg["method"] = method
    cfg["hierarchical_layers"] = 1
    if thread_seed:  # <- B1: hierarchical_vs_flat.run_cell now does exactly this
        cfg["umap_random_state"] = cfg["tsne_random_state"] = cfg["mds_random_state"] = seed
    emb, _ = _embed_original(X, cfg)
    return emb


print("Replicates of one cell, same data, same code path.\n")
print(f"{'method':6s}  {'arm':<22s}  {'replicate 0 vs 1..4 identical':<32s}  max|diff|")
print("-" * 88)
for method in ("UMAP", "t-SNE", "PCA"):
    for label, thread_seed in (("BEFORE B1 (seed unused)", False), ("AFTER  B1 (seed used) ", True)):
        embs = [embed(method, seed, thread_seed) for seed in SEEDS]
        identical = [bool(np.array_equal(embs[0], e)) for e in embs[1:]]
        maxdiff = max(float(np.abs(embs[0] - e).max()) for e in embs[1:])
        print(f"{method:6s}  {label:<22s}  {str(identical):<32s}  {maxdiff:.3e}")

print()
print("UMAP varies after B1: the collapse is gone. MDS does too (not shown here; same check).")
print()
print("PCA is identical in both arms by construction -- _pca takes no random_state and is")
print("deterministic on these shapes, so its replicates never carried embedding variance.")
print()
print("t-SNE is identical in both arms for a DIFFERENT and less obvious reason: the seed IS")
print("threaded now, but _tsne passes no `init`, and sklearn's default is init='pca', under")
print("which the embedding never consults random_state. So B1 is a no-op for t-SNE and its")
print("SEEDS levels remain repeats, not replicates:")
from sklearn.manifold import TSNE  # noqa: E402 - local to this demonstration

_a = TSNE(n_components=2, random_state=0).fit_transform(X)
_b = TSNE(n_components=2, random_state=7).fit_transform(X)
_c = TSNE(n_components=2, random_state=0, init="random").fit_transform(X)
_d = TSNE(n_components=2, random_state=7, init="random").fit_transform(X)
print(f"    sklearn TSNE, default init='pca'  -> max|diff| across random_state = {np.abs(_a - _b).max():.3e}")
print(f"    sklearn TSNE, init='random'       -> max|diff| across random_state = {np.abs(_c - _d).max():.3e}")
print("  The fix for that lives in src/analysis/dim_reducer._tsne, which is frozen.")
print()

# ---------------------------------------------------------------------------
# What each arm hands scipy.stats.wilcoxon, via _paired_deltas' (seed, region) pivot.
# ---------------------------------------------------------------------------
from scipy.stats import wilcoxon  # noqa: E402 - after the demonstration above, by design

print("Consequence for the H1a test (_paired_deltas pivots on ['seed','region']):")
print()
print(f"  {'R':>3s}  {'collapsed: n=5R, R distinct':>30s}  {'fixed: n=5R, all distinct':>28s}  {'per-region: n=R':>20s}")
print("  " + "-" * 86)

for R in (8, 12, 20):
    truth = rng.normal(loc=0.03, scale=0.06, size=R)  # R regions, one small real effect

    # BEFORE: 5 identical replicates -> the pivot yields each region's delta 5 times.
    collapsed = np.tile(truth, 5)

    # AFTER: 5 genuinely different embeddings -> 5 distinct measurements per region.
    fixed = np.concatenate([truth + rng.normal(scale=0.02, size=R) for _ in range(5)])

    # The conservative unit: one value per region (replicates averaged), n = R.
    per_region = fixed.reshape(5, R).mean(axis=0)

    p_collapsed = wilcoxon(collapsed).pvalue
    p_fixed = wilcoxon(fixed).pvalue
    p_region = wilcoxon(per_region).pvalue
    print(f"  {R:3d}  n={collapsed.size:3d} p={p_collapsed:<20.7f}  n={fixed.size:3d} p={p_fixed:<18.7f}  n={per_region.size:3d} p={p_region:.7f}")

print()
print("Column 1 is the defect: R numbers repeated 5x, so n is inflated 5x and p shrinks")
print("accordingly -- the duplication itself supplies the significance.")
print("Column 2 is what B1 restores: 5R genuinely distinct measurements, which is the")
print("pre-registered (region x seed) unit.")
print("Column 3 is the honest-n reading kept alongside it: the 5 replicates of a region are")
print("repeated measures of that region, so n=R is the conservative unit. B1 removes the")
print("duplication; it does not by itself make (region x seed) pairs independent.")

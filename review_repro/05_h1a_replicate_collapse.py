"""H1a replicate collapse.

src_research/hierarchical_vs_flat.py:33-36 documents the design:

    "Reproducibility caveat: ``reduce_dimensionality`` does not thread a seed into
     UMAP / t-SNE, so those embeddings are stochastic. The subsample is seeded (fixed
     across runs), so the ``SEEDS`` loop functions as *replicates* that capture
     embedding variance - the unit of the paired H1a test is a (region x seed) pair."

The 24h window (1d85891, src/analysis/dim_reducer.py) started threading
config["umap_random_state"] / ["tsne_random_state"] -- both hardcoded to 42 in
src/config_defaults.py -- into the constructors.

run_cell() (hierarchical_vs_flat.py:259-277) builds cfg from init_state() and sets ONLY
cfg["method"] and cfg["hierarchical_layers"]; `seed` is written into the label dict at
:277 and used nowhere else. So every replicate now sees random_state=42.

This reproduces exactly that: same cfg, different "seed", same code path.
"""

import warnings

import numpy as np

from src.analysis.analysis_routine import _embed_original
from src.config_defaults import init_state

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 8))

SEEDS = list(range(5))  # mirrors hierarchical_vs_flat.SEEDS

for method in ("UMAP", "t-SNE", "PCA"):
    embs = []
    for seed in SEEDS:
        # verbatim the three lines run_cell uses to build its config
        cfg = init_state(init_streamlit=False)
        cfg["method"] = method
        cfg["hierarchical_layers"] = 1
        # `seed` goes only into the label dict in run_cell -- it never touches cfg
        emb, _ = _embed_original(X, cfg)
        embs.append(emb)

    identical = [bool(np.array_equal(embs[0], e)) for e in embs[1:]]
    maxdiff = max(float(np.abs(embs[0] - e).max()) for e in embs[1:])
    print(f"{method:6s}  replicate 0 vs 1..4 identical: {identical}   max|diff| = {maxdiff:.3e}")

print()
print("Consequence for the H1a test (_paired_deltas, hierarchical_vs_flat.py:385-392):")
print("  wide = sub.pivot_table(index=['seed','region'], ...)  ->  5 x R rows")
print("  but only R DISTINCT values, each repeated 5x.")
print("  scipy.stats.wilcoxon then receives n = 5R 'independent' paired deltas.")
print()

# Show what that does to a Wilcoxon: same data, honestly n=R vs duplicated to n=5R.
from scipy.stats import wilcoxon

for R, loc in ((8, 0.04), (12, 0.03), (20, 0.02)):
    base = rng.normal(loc=loc, scale=0.06, size=R)  # R regions, small real effect
    dup = np.tile(base, 5)  # exactly what the pivot hands the test
    p_honest = wilcoxon(base).pvalue
    p_inflated = wilcoxon(dup).pvalue
    print(
        f"  R={R:3d} regions | honest n={base.size:3d} p={p_honest:.5f}"
        f"   ->  inflated n={dup.size:3d} p={p_inflated:.7f}"
        f"   ({'CROSSES 0.05' if p_honest >= 0.05 > p_inflated else 'same side of 0.05'})"
    )
print("  ^ identical data, 5x duplicated. n is inflated 5x and p shrinks accordingly.")

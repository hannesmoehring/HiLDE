"""MRRE direction is inverted in the H1a summary.

src_research/hierarchical_vs_flat.py:105-106

    # Higher-is-better metrics: T&C improve when up, MRRE/stress are errors (improve when down).
    HIGHER_IS_BETTER = {"trustworthiness": True, "continuity": True,
                        "mrre_false": False, "mrre_missing": False, "stress": False}

But both the repo and ZADU return MRRE already inverted into a similarity:

    src/evaluation/neighbor_metrics.py:99-100   1 - mrre_false / mrre_norm
    zadu/measures/mean_relative_rank_error.py:78  local_distortion_list = 1 - local_distortion_list / c

so mrre_false / mrre_missing are HIGHER-is-better in [0, 1], exactly like trustworthiness.
`stress` is genuinely lower-is-better (it is a raw normalised error, not 1 - x), so that
entry is correct; only the two MRRE entries are wrong.

Consequence: win_rate and rank_biserial for both MRRE metrics come out exactly backwards
in h1a_summary.csv, while median_delta (which does not consult HIGHER_IS_BETTER) is right.
A row can therefore read median_delta=+0.09 next to win_rate=0.02.

Run:
  cd /tmp && PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/scripts/checks/09_mrre_direction_inverted.py
"""

import warnings

import numpy as np

from src.evaluation.neighbor_metrics import neighbor_scores
from src_research.hierarchical_vs_flat import HIGHER_IS_BETTER

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
n, d, k = 300, 10, 20

# A structured original space.
X = rng.normal(size=(n, d))

# A GOOD embedding: the leading 2 principal directions (preserves neighbourhoods).
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
good = Xc @ Vt[:2].T

# A BAD embedding: independent noise (destroys neighbourhoods).
bad = rng.normal(size=(n, 2))

g = neighbor_scores(X, good, k)
b = neighbor_scores(X, bad, k)

print(
    f"{'metric':<18} {'GOOD emb':>12} {'BAD emb':>12}   {'good>bad?':>10}   "
    f"{'declared':>16}   verdict"
)
print("-" * 92)
for m in ("trustworthiness", "continuity", "mrre_false", "mrre_missing", "stress"):
    gv, bv = float(g[m]), float(b[m])
    higher_is_better_empirically = gv > bv
    declared = HIGHER_IS_BETTER[m]
    ok = higher_is_better_empirically == declared
    print(
        f"{m:<18} {gv:>12.6f} {bv:>12.6f}   {higher_is_better_empirically!s:>10}   "
        f"{'higher=better' if declared else 'lower=better':>16}   {'ok' if ok else '*** INVERTED ***'}"
    )

inverted = [
    m
    for m in ("trustworthiness", "continuity", "mrre_false", "mrre_missing", "stress")
    if (float(g[m]) > float(b[m])) != HIGHER_IS_BETTER[m]
]

print()
print("A good embedding scores HIGHER than a bad one on both MRRE terms, so they are")
print(
    "higher-is-better. HIGHER_IS_BETTER used to declare them lower-is-better, which flipped"
)
print(
    "win_rate and rank_biserial in h1a_summary.csv while median_delta (computed without the"
)
print(
    "map) stayed right -- which is why the SHIPPED summaries contain rows whose two effect"
)
print("sizes disagree in sign, e.g.:")
print(
    "  Breast cancer (Low) PCA hier_leaf mrre_false   median_delta=+0.094339  win_rate=0.023  rbc=-0.955"
)
print(
    "  Breast cancer (Low) PCA hier_leaf mrre_missing median_delta=+0.072784  win_rate=0.000  rbc=-1.000"
)
print()
if inverted:
    print(f"*** STILL INVERTED: {inverted} -- B2 is not (or no longer) applied. ***")
else:
    print(
        "B2 applied: every declared direction now matches the measured one, so a re-derived"
    )
    print("summary agrees with itself. The shipped CSVs above are corrected by")
    print(
        "`python -m src_research.rederive` (see outputs/experiments/*/rederived_*/DELTAS.md)."
    )

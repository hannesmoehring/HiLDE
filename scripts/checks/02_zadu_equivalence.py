"""neighbor_metrics.py claims: 'The formulas and their normalisation constants are
ZADU's, unchanged -- this returns the same numbers.' Test that claim directly."""

import warnings

import numpy as np
from zadu.zadu import ZADU

from src.evaluation.neighbor_metrics import neighbor_scores

warnings.filterwarnings("ignore")

print(
    f"{'n':>5} {'d':>3} {'k':>3} | {'measure':<16} {'repo':>12} {'zadu':>12} {'absdiff':>11}  ok"
)
print("-" * 78)

worst = 0.0
bad = []
for seed, n, d, k in [
    (0, 50, 8, 5),
    (1, 50, 8, 20),
    (2, 120, 10, 20),
    (3, 200, 4, 1),
    (4, 200, 4, 2),
    (5, 300, 16, 20),
    (6, 61, 3, 20),  # k = (n-1)//2 boundary
    (7, 10, 5, 4),  # smallest node that gets neighbour metrics
]:
    rng = np.random.default_rng(seed)
    orig = rng.normal(size=(n, d))
    emb = rng.normal(size=(n, 2))

    repo = neighbor_scores(orig, emb, k)
    z = ZADU(
        [
            {"id": "stress"},
            {"id": "tnc", "params": {"k": k}},
            {"id": "mrre", "params": {"k": k}},
        ],
        orig=orig,
    ).measure(emb)
    ref = {
        "stress": z[0]["stress"],
        "trustworthiness": z[1]["trustworthiness"],
        "continuity": z[1]["continuity"],
        "mrre_false": z[2]["mrre_false"],
        "mrre_missing": z[2]["mrre_missing"],
    }
    for m in ref:
        a, b = float(repo[m]), float(ref[m])
        diff = abs(a - b)
        worst = max(worst, diff)
        ok = diff < 1e-9
        if not ok:
            bad.append((n, d, k, m, a, b, diff))
        print(
            f"{n:>5} {d:>3} {k:>3} | {m:<16} {a:>12.8f} {b:>12.8f} {diff:>11.2e}  {'OK' if ok else 'MISMATCH'}"
        )
    print()

print(f"worst absolute difference across all cases: {worst:.3e}")
print(f"MISMATCHES: {len(bad)}")
for row in bad:
    print("   ", row)

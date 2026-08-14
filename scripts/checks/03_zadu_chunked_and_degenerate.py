"""The chunking in neighbor_metrics is the whole point of the rewrite, but it only
engages above n ~= 2365 with the shipped CHUNK_BYTES (256MB). Force multi-chunk by
shrinking CHUNK_BYTES and re-check equivalence with ZADU. Also probe degenerate data."""

import warnings

import numpy as np
from zadu.zadu import ZADU

import src.evaluation.neighbor_metrics as nm

warnings.filterwarnings("ignore")

MEAS = ["stress", "trustworthiness", "continuity", "mrre_false", "mrre_missing"]


def zadu_ref(orig, emb, k):
    z = ZADU(
        [
            {"id": "stress"},
            {"id": "tnc", "params": {"k": k}},
            {"id": "mrre", "params": {"k": k}},
        ],
        orig=orig,
    ).measure(emb)
    return {
        "stress": z[0]["stress"],
        "trustworthiness": z[1]["trustworthiness"],
        "continuity": z[1]["continuity"],
        "mrre_false": z[2]["mrre_false"],
        "mrre_missing": z[2]["mrre_missing"],
    }


print("=== A. multi-chunk equivalence (CHUNK_BYTES shrunk to force >1 chunk) ===")
orig_bytes = nm.CHUNK_BYTES
n, d, k = 150, 6, 10
rng = np.random.default_rng(7)
orig = rng.normal(size=(n, d))
emb = rng.normal(size=(n, 2))
ref = zadu_ref(orig, emb, k)

for cb in (orig_bytes, 48 * n * 37, 48 * n * 7, 48 * n * 1):
    nm.CHUNK_BYTES = cb
    step = nm._chunk_rows(n, with_ranking=True)
    got = nm.neighbor_scores(orig, emb, k)
    diffs = {m: abs(float(got[m]) - float(ref[m])) for m in MEAS}
    worst = max(diffs.values())
    nchunks = -(-n // step)
    print(
        f"  step={step:>4} ({nchunks:>3} chunks)  worst_diff={worst:.3e}  {'OK' if worst < 1e-9 else 'MISMATCH ' + str(diffs)}"
    )
nm.CHUNK_BYTES = orig_bytes

print()
print(
    "=== B. duplicate points (argsort ties -> is column 0 still the point itself?) ==="
)
base = rng.normal(size=(40, 4))
orig_d = np.vstack([base, base])  # every point has an exact duplicate
emb_d = np.vstack([rng.normal(size=(40, 2))] * 2)
k = 5
got = nm.neighbor_scores(orig_d, emb_d, k)
ref = zadu_ref(orig_d, emb_d, k)
for m in MEAS:
    diff = abs(float(got[m]) - float(ref[m]))
    print(
        f"  {m:<16} repo={float(got[m]):>12.8f} zadu={float(ref[m]):>12.8f} diff={diff:.2e} "
        f"{'OK' if diff < 1e-9 else 'MISMATCH'}"
    )

print()
print("=== C. all-identical points (stress denominator == 0) ===")
flat = np.ones((20, 4))
flat_emb = np.ones((20, 2))
try:
    got = nm.neighbor_scores(flat, flat_emb, 5)
    print("  returned:", got)
except Exception as e:  # noqa: BLE001
    print(f"  RAISED {type(e).__name__}: {e}")
    print("  -> in evaluate.py this is swallowed by `except Exception: pass`,")
    print("     leaving stress/trustworthiness/continuity/mrre all None for the node")

print()
print("=== D. what does ZADU do on the same all-identical input? ===")
try:
    print("  zadu:", zadu_ref(flat, flat_emb, 5))
except Exception as e:  # noqa: BLE001
    print(f"  zadu RAISED {type(e).__name__}: {e}")

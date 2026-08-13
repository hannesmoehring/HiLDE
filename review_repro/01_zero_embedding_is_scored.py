"""Does _embed_original's `except Exception -> zeros` path actually trigger, and
what do the reported quality scores become when it does?"""

import warnings

import numpy as np

from src.analysis.analysis_routine import _embed_original
from src.config_defaults import default_config
from src.evaluation.evaluate import _score_node

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)


def probe(method, n, d):
    cfg = default_config()
    cfg["method"] = method
    X = rng.normal(size=(n, d))
    emb, var = _embed_original(X, cfg)
    degenerate = bool(np.all(emb == 0.0))
    print(f"  {method:6s} n={n:<4d} d={d:<3d} -> emb{emb.shape} all_zero={degenerate}")
    return X, emb, degenerate


print("=== 1. which (method, n, d) fall into the zeros fallback? ===")
cases = []
for method in ("pca", "umap", "t-sne", "mds"):
    for n, d in ((3, 5), (5, 5), (11, 5), (12, 3), (30, 5)):
        try:
            cases.append((method, n, d, *probe(method, n, d)))
        except Exception as e:  # noqa: BLE001
            print(f"  {method:6s} n={n:<4d} d={d:<3d} -> RAISED OUT: {type(e).__name__}: {e}")

print()
print("=== 2. scores reported for a node whose embedding silently became zeros ===")
for method, n, d, X, emb, degenerate in cases:
    if not degenerate or n < 10:
        continue
    s = _score_node(X, emb, None)
    print(f"  {method} n={n} d={d}:")
    print(f"    k={s['k']} stress={s['stress']} trust={s['trustworthiness']} cont={s['continuity']}")
    print("    ^ these are reported to the UI as embedding-quality scores")

print()
print("=== 3. control: a real (non-degenerate) embedding for comparison ===")
X = rng.normal(size=(30, 5))
cfg = default_config()
cfg["method"] = "pca"
emb, _ = _embed_original(X, cfg)
s = _score_node(X, emb, None)
print(f"  pca n=30 d=5: k={s['k']} stress={s['stress']} trust={s['trustworthiness']} cont={s['continuity']}")

print()
print("=== 4. forced failure: monkeypatch the reducer to raise, as a broken node would ===")
import src.analysis.analysis_routine as ar

orig_fn = ar.fit_dimensionality_reducer
ar.fit_dimensionality_reducer = lambda **kw: (_ for _ in ()).throw(RuntimeError("reducer blew up"))
X = rng.normal(size=(30, 5))
emb, var = _embed_original(X, default_config())
ar.fit_dimensionality_reducer = orig_fn
print(f"  reducer raised -> emb{emb.shape} all_zero={bool(np.all(emb == 0))} variance={var}")
s = _score_node(X, emb, None)
print(f"  reported scores: k={s['k']} stress={s['stress']} trust={s['trustworthiness']} "
      f"cont={s['continuity']} mrre_false={s['mrre_false']}")
print("  -> a totally failed projection is reported with finite, plausible-looking scores")

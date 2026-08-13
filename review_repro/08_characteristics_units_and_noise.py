"""Three independent defects in the characteristics / hierarchy path.

Run:
  cd /tmp && PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/review_repro/08_characteristics_units_and_noise.py

(1) NON-FEATURE z_std IS IN RAW UNITS.
    src/analysis/characteristics.py:33 standardises the extras' z_mean as
    (c_mean - g_mean)/g_std, but line 34 sets z_std to the cluster's RAW std with no
    division by g_std. Feature rows get z_std from the already-scaled frame. Both are
    drawn on one "z-score" axis by CharacteristicsBar, whose y-domain spans z_mean +/- z_std.

(2) FEATURE AND EXTRA z-SCORES USE DIFFERENT ddof.
    Features go through StandardScaler (population std, ddof=0); extras through
    pandas .std() (sample std, ddof=1). Two identical columns get bar lengths differing
    by exactly sqrt(n/(n-1)).

(3) HDBSCAN NOISE POINTS LEAVE THE HIERARCHY.
    analysis_routine.py:198-201 skips label -1, so noise rows sit in the parent's
    row_indices and in no child's. They are counted in the parent's n_points, drawn in
    the parent scatter, and unreachable by drilling down. Nothing reports the count.
"""

import warnings

import numpy as np
import pandas as pd

from backend.datasets import default_feature_cols
from src.analysis.analysis_routine import compute_analysis_tree
from src.analysis.characteristics import compute_cluster_characteristics
from src.config_defaults import default_config
from src.datasets import DATASETS

warnings.filterwarnings("ignore")

print("=" * 78)
print("(1)+(2)  z_std units and ddof, on a hand-built frame with a KNOWN answer")
print("=" * 78)

rng = np.random.default_rng(0)
n = 60
# `big` is a non-feature column in raw units ~N(500, 40); `feat` is a feature.
raw = pd.DataFrame({
    "feat": rng.normal(0, 1, n),
    "big": rng.normal(500, 40, n),
    "twin": rng.normal(0, 1, n),   # used for the ddof check
})
raw["twin_extra"] = raw["twin"]     # an EXACT copy, carried as a non-feature column
labels = np.zeros(n, dtype=int)
labels[:20] = 1                     # cluster 1 = first 20 rows

feature_cols = ["feat", "twin"]
extra_cols = ["big", "twin_extra"]

scaled = pd.DataFrame(
    (raw[feature_cols] - raw[feature_cols].mean()) / raw[feature_cols].std(ddof=0),
    columns=feature_cols,
)
scaled["cluster"] = labels
df = raw.copy()
df["cluster"] = labels

rows = compute_cluster_characteristics(
    cluster_id=1, df=df, X_scaled_df=scaled, feature_cols=feature_cols, extra_cols=extra_cols
)

in_c = labels == 1
print(f"{'column':<12} {'is_feature':<11} {'z_std EMITTED':>15} {'z_std CORRECT':>15}  {'ratio':>8}")
for name in rows.index:
    emitted = float(rows.loc[name, "z_std"])
    if bool(rows.loc[name, "is_feature"]):
        correct = emitted  # features already come from the scaled frame
    else:
        correct = float(df.loc[in_c, name].std() / df[name].std())
    print(f"{name:<12} {str(bool(rows.loc[name, 'is_feature'])):<11} "
          f"{emitted:>15.6f} {correct:>15.6f}  {emitted / correct:>8.2f}x")

print()
print("  -> 'big' is reported with a z_std in mg/L-scale units on a z-score axis.")
print("     CharacteristicsBar derives its y-domain from z_mean +/- z_std, so this one")
print("     column sets the scale for the whole chart.")

print()
print("--- ddof: 'twin' (feature) and 'twin_extra' (extra) are the SAME column ---")
zf = float(rows.loc["twin", "z_mean"])
ze = float(rows.loc["twin_extra", "z_mean"])
print(f"  z_mean as a FEATURE : {zf:+.9f}   (StandardScaler, ddof=0)")
print(f"  z_mean as an EXTRA  : {ze:+.9f}   (pandas .std(), ddof=1)")
print(f"  ratio               : {zf / ze:.9f}")
print(f"  sqrt(n/(n-1))       : {np.sqrt(n / (n - 1)):.9f}   <- exact match confirms the cause")

print()
print("=" * 78)
print("(3)  noise points that leave the hierarchy, on a real dataset")
print("=" * 78)

df_iris = DATASETS["Iris (Low)"]()
feats = default_feature_cols(df_iris)
cfg = default_config()
cfg["hierarchical_layers"] = 2
tree = compute_analysis_tree(df_iris, feats, cfg)


def walk(node, path="root"):
    kids = node.get("next_object_layer") or []
    if kids:
        parent_n = len(node["row_indices"])
        child_n = sum(len(k["row_indices"]) for k in kids)
        lost = parent_n - child_n
        flag = "  <-- LOST" if lost else ""
        print(f"  {path:<12} n_points={parent_n:<5} sum(children)={child_n:<5} unreachable={lost}{flag}")
        for i, k in enumerate(kids):
            walk(k, f"{path}/{i}")


print(f"{'node':<14} parent vs children")
walk(tree)
print()
print("  Any 'unreachable' > 0 is points visible in the parent scatter and counted in")
print("  the parent's n_points, that no drill-down can reach. No field reports the count.")

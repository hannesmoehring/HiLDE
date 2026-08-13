"""backend/characteristics.py (new in the 24h window) says:

    "The tree stores one `rel_characteristics` frame per cluster, contrasting that
     cluster with its parent's space. Exploration needs the same contrast for an
     arbitrary lasso selection inside a node, so this reuses the unchanged calc layer"

Test whether the two paths really produce the same contrast for the same points.

Tree path   : src/analysis/analysis_routine._build_next -> compute_cluster_characteristics
              with X_scaled_df = ctx.X_orig, which is the ROOT scaler applied once
              and then only row-masked. Feature z-scores are therefore global.
Endpoint    : backend/characteristics.compute_selection_characteristics fits a FRESH
              StandardScaler on the node's rows. Feature z-scores are node-local.

Both are rendered by the same CharacteristicsBar component.
"""

import warnings

import numpy as np
import pandas as pd

from backend.characteristics import compute_selection_characteristics
from src.analysis.analysis_routine import compute_analysis_tree
from src.config_defaults import default_config

warnings.filterwarnings("ignore")

rng = np.random.default_rng(3)

# Three well-separated blobs, each internally structured, so the tree splits twice.
blobs = []
for centre in ([0, 0, 0, 0], [14, 14, 0, 0], [0, 0, 14, 14]):
    a = rng.normal(size=(220, 4)) + np.array(centre)
    b = rng.normal(size=(220, 4)) + np.array(centre) + np.array([4, 0, 4, 0])
    blobs += [a, b]
X = np.vstack(blobs)
feature_cols = [f"f{i}" for i in range(4)]
df = pd.DataFrame(X, columns=feature_cols)
df["row_id"] = df.index

cfg = default_config()
cfg["hierarchical_layers"] = 3
cfg["hclust_min_cluster_size"] = 25
cfg["method"] = "PCA"
tree = compute_analysis_tree(df, feature_cols, cfg)


# We need a parent that is NOT the root: at the root the endpoint's node-local scaler
# and the tree's root scaler are fit on the same rows, so the two paths trivially agree.
def find_non_root_parent(node, depth=0):
    for kid in node.get("next_object_layer") or []:
        grandkids = kid.get("next_object_layer") or []
        if depth + 1 >= 1 and grandkids:
            return kid, grandkids[0]  # parent at depth>=1, child at depth>=2
        found = find_non_root_parent(kid, depth + 1)
        if found:
            return found
    return None


pair = find_non_root_parent(tree)
assert pair is not None, "tree did not reach depth 2; adjust the synthetic data"
parent, child = pair
assert len(parent["row_indices"]) < len(tree["row_indices"]), "parent must not be the root"

print(f"parent node rows: {len(parent['row_indices'])}")
print(f"child  node rows: {len(child['row_indices'])}")
print()

tree_chars = child["rel_characteristics"]
print("=== TREE path: child['rel_characteristics'] (z_mean per feature) ===")
print(tree_chars[["z_mean", "raw_mean"]].to_string())

# Same points, same parent space, via the new endpoint.
pos = {ri: i for i, ri in enumerate(parent["row_indices"])}
sel_local = [pos[ri] for ri in child["row_indices"]]
recs = compute_selection_characteristics(
    df, feature_cols, list(parent["row_indices"]), sel_local
)
ep = pd.DataFrame(recs).set_index("feature") if recs else pd.DataFrame()
print()
print("=== ENDPOINT path: compute_selection_characteristics, same points ===")
print(ep.to_string())

print()
print("=== SIDE BY SIDE (feature z_mean) ===")
print(f"{'feature':<10} {'tree z_mean':>14} {'endpoint z_mean':>18} {'abs diff':>12}")
worst = 0.0
for f in feature_cols:
    t = float(tree_chars.loc[f, "z_mean"])
    e = float(ep.loc[f, "z_mean"]) if f in ep.index else float("nan")
    worst = max(worst, abs(t - e))
    print(f"{f:<10} {t:>14.6f} {e:>18.6f} {abs(t - e):>12.6f}")
print()
print(f"worst |tree - endpoint| on feature z_mean = {worst:.6f}")
print("Both are labelled 'characteristics vs. the parent' and drawn by the same bar chart.")

# Show the direct cause: the tree's scaled frame is the ROOT scaler, not the node's.
sub = df.iloc[parent["row_indices"]][feature_cols].to_numpy()
print()
print("cause: parent-node feature std under the ROOT scaler (tree's basis) vs 1.0 by")
print("construction under a node-local scaler (endpoint's basis):")
root_scaler = tree["scaler"]
print("  parent rows, root-scaled std :", np.round(root_scaler.transform(sub).std(axis=0, ddof=0), 4))
print("  parent rows, node-scaled std :", np.round(np.ones(len(feature_cols)), 4))

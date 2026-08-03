"""Precompute the eight MNIST / Fashion-MNIST runs into the hosting-mode run cache.

One-shot, and it takes hours: every run clusters 70000 points in the full 784-D
feature space, which is an O(n^2) single-threaded HDBSCAN of roughly an hour on
its own. Progress is printed per stage; a re-run skips whatever is already
cached, so an interrupted job can simply be started again.

    PYTHONPATH=. .venv/bin/python scripts/precompute_cache.py

Entries land in `.cache/hilde_runs` (or $HILDE_CACHE_DIR) — the same directory
docker-compose mounts, and the same keys the UI looks up, so the cached runs are
served without recomputing.
"""

from __future__ import annotations

import os
import time
import traceback

# Must precede the backend import path that reads it: without hosting mode
# `_build` computes the tree but never writes it to disk.
os.environ.setdefault("HILDE_HOSTING", "1")

from backend import datasets as ds  # noqa: E402
from backend import run_cache  # noqa: E402
from backend.app import AnalysisRequest, _build, _cache_key, _tree_cache  # noqa: E402

# DATASETS = ["MNIST (High)", "Fashion-MNIST (High)"]
# LAYERS = [1, 2]
# METHODS = ["UMAP", "PCA"]

DATASETS = [
    "Wine quality (Low)",
    "Iris (Low)",
    "Digits (Low)",
    "Breast cancer (Low)",
    "Concentric rings (Low)",
    "Swiss roll (Low)",
    "Olivetti faces (Medium)",
]
LAYERS = [1, 2, 3, 4]
METHODS = ["UMAP", "PCA", "t-SNE"]  # , "MDS"

# Mirrors frontend/src/config.ts::DEFAULT_CONFIG. The run-cache key is a
# json.dumps of exactly this dict, so the values have to match what the browser
# posts down to their JSON types — ints where the UI sends ints, 0.1 as a float.
# Any drift here yields a key the UI will never look up.
# BASE_CONFIG: dict[str, object] = {
#     "hclust_normalize": True,
#     "hierarchical_layers": 1,
#     "hclust_umap_n_components": 2,
#     "hclust_min_samples": 5,
#     "hclust_min_cluster_size": 25,
#     "normalize": True,
#     "method": "UMAP",
#     "pca_components": 4,
#     "tsne_perplexity": 30,
#     "tsne_learning_rate": 200,
#     "tsne_random_state": 42,
#     "umap_n_neighbors": 15,
#     "umap_min_dist": 0.1,
#     "umap_random_state": 42,
#     "mds_metric": True,
#     "mds_n_init": 2,
#     "mds_max_iter": 100,
#     "mds_random_state": 42,
# }
BASE_CONFIG: dict[str, object] = {
    "hclust_normalize": True,
    "hierarchical_layers": 1,
    "hclust_umap_n_components": 2,
    "hclust_min_samples": 5,
    "hclust_min_cluster_size": 25,
    "normalize": True,
    "method": "UMAP",
    "pca_components": 4,
    "tsne_perplexity": 30,
    "tsne_learning_rate": 200,
    "tsne_random_state": 42,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_random_state": 42,
    "mds_metric": True,
    "mds_n_init": 2,
    "mds_max_iter": 100,
    "mds_random_state": 42,
}


def _config(n_features: int, layers: int, method: str) -> dict[str, object]:
    config = dict(BASE_CONFIG)
    config["hierarchical_layers"] = layers
    config["method"] = method
    # App.tsx raises this to the feature count as soon as a dataset loads, so
    # clustering runs on the full space (analysis_routine skips the
    # pre-reduction once n_components is not below n_features).
    config["hclust_umap_n_components"] = max(2, n_features)
    return config


def _hms(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def main() -> int:
    started = time.perf_counter()
    print(f"cache dir: {run_cache.cache_dir()}")
    runs = [(d, layer, m) for d in DATASETS for layer in LAYERS for m in METHODS]
    print(f"{len(runs)} runs queued\n")

    failed: list[tuple[str, str]] = []
    for i, (dataset, layers, method) in enumerate(runs, start=1):
        label = f"{dataset}  layers={layers}  {method}"
        df = ds.load(dataset)  # cached by @cache after the first load per dataset
        feature_cols = ds.default_feature_cols(df)
        config = _config(len(feature_cols), layers, method)
        req = AnalysisRequest(dataset=dataset, feature_cols=feature_cols, config=config)
        key = _cache_key(req.dataset, req.feature_cols, req.config)

        if run_cache.load(key) is not None:
            print(f"[{i}/{len(runs)}] {label} — already cached, skipping")
            continue

        print(f"[{i}/{len(runs)}] {label} — building ({len(df)} rows x {len(feature_cols)} features)", flush=True)
        t0 = time.perf_counter()
        try:
            _build(req, df, key)
        except Exception:  # one bad config must not cost the remaining runs
            failed.append((label, traceback.format_exc()))
            print(f"[{i}/{len(runs)}] {label} — FAILED after {_hms(time.perf_counter() - t0)}", flush=True)
        else:
            print(f"[{i}/{len(runs)}] {label} — done in {_hms(time.perf_counter() - t0)}", flush=True)
        # Payloads are large and the script only needs them on disk.
        _tree_cache.clear()

    print(f"\ntotal {_hms(time.perf_counter() - started)}, {len(runs) - len(failed)}/{len(runs)} cached")
    for label, tb in failed:
        print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

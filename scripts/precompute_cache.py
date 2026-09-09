"""Precompute analysis runs into the hosting-mode run cache.

RUN_CONFIG is BASE_CONFIG with a list of values per knob; the queue is its
cartesian product, run against every dataset in DATASETS. Knobs that only one DR
method reads are expanded only under that method, so sweeping `mds_metric` costs
two extra runs instead of doubling the whole grid.

The queue runs WORKERS runs at a time in separate processes. A single build is
effectively single-threaded — UMAP pins n_jobs to 1 as soon as a random_state is
set — so the parallelism has to come from running whole runs side by side rather
than from inside one. Each worker's per-stage output goes to /dev/null: ten
interleaved progress logs are unreadable, and a failure comes back as a captured
traceback anyway.

The job is long and restartable: a re-run skips whatever is already cached, so an
interrupted one can simply be started again.

    PYTHONPATH=. .venv/bin/python scripts/precompute_cache.py

Entries land in `.cache/hilde_runs` (or $HILDE_CACHE_DIR) — the same directory
docker-compose mounts, and the same keys the UI looks up, so the cached runs are
served without recomputing.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# Must precede the backend import path that reads it: without hosting mode
# `_build` computes the tree but never writes it to disk.
os.environ.setdefault("HILDE_HOSTING", "1")

# The sweep parallelises across runs, so the numeric libraries inside a run must
# not also fan out — WORKERS builds each claiming every core would oversubscribe
# the machine badly. LOKY_MAX_CPU_COUNT is the one that collapses dim_reducer's
# `MDS(n_jobs=-1)`; the rest cap whichever BLAS numpy was built against. These
# must be set before the backend import, because numpy reads them at import time.
# (hdbscan asks joblib for 4 workers explicitly, which no env var caps — hence the
# headroom left in WORKERS below.)
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_var, "1")

from backend import datasets as ds  # noqa: E402
from backend import run_cache  # noqa: E402
from backend.app import AnalysisRequest, _build, _cache_key, _tree_cache  # noqa: E402

# DATASETS = ["MNIST (High)", "Fashion-MNIST (High)"]

DATASETS = [
    "Wine quality (Low)",
    "Iris (Low)",
    "Digits (Low)",
    "Breast cancer (Low)",
    "Concentric rings (Low)",
    "Swiss roll (Low)",
    "Olivetti faces (Medium)",
]

# Runs built concurrently. Each is one core's worth of work (see the thread caps
# above), so this is the sweep's real parallelism. The spare cores absorb
# hdbscan's own 4-way fan-out and the ~100 MB of tree payload a worker holds
# while serializing; raise it and the machine thrashes instead of going faster.
WORKERS = max(1, (os.cpu_count() or 4) - 4)

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

# The values to sweep, one list per BASE_CONFIG knob. Every combination becomes a
# run, so each list multiplies the queue — except the DR knobs, which only expand
# under their own method (see METHOD_KNOBS).
RUN_CONFIG: dict[str, list[object]] = {
    "hclust_normalize": [True],  # ignored: follows `normalize` (see _FOLLOWS)
    "hierarchical_layers": [0, 1, 2, 3, 4],
    "hclust_umap_n_components": [None],  # None = the dataset's feature count
    "hclust_min_samples": [5, 3, 10],
    "hclust_min_cluster_size": [25, 50, 10],
    "normalize": [True, False],
    "method": ["UMAP", "PCA", "t-SNE", "MDS"],
    "pca_components": [4],  # the config panel never shows it; the UI always posts 4
    "tsne_perplexity": [30],
    "tsne_learning_rate": [200],
    "tsne_random_state": [42],
    "umap_n_neighbors": [15],
    "umap_min_dist": [0.1],
    "umap_random_state": [42],
    "mds_metric": [True, False],
    "mds_n_init": [2],
    "mds_max_iter": [100],
    "mds_random_state": [42],
}

assert RUN_CONFIG.keys() == BASE_CONFIG.keys(), "RUN_CONFIG must mirror BASE_CONFIG"

# Knobs a single DR method reads, which ConfigPanel.tsx renders only while that
# method is selected. Expanding them under any other method would spend runs on
# configs the UI can reach only by selecting that method, changing the knob and
# switching back. PCA exposes no knobs of its own.
METHOD_KNOBS: dict[str, tuple[str, ...]] = {
    "PCA": (),
    "t-SNE": ("tsne_perplexity", "tsne_learning_rate", "tsne_random_state"),
    "UMAP": ("umap_n_neighbors", "umap_min_dist", "umap_random_state"),
    "MDS": ("mds_metric", "mds_n_init", "mds_max_iter", "mds_random_state"),
}
_METHOD_ONLY = {knob for knobs in METHOD_KNOBS.values() for knob in knobs}

# One checkbox drives both — ConfigPanel.tsx sends `{hclust_normalize: v, normalize: v}`
# — so no browser can post them differing. Sweeping them apart would make half the
# normalize runs unreachable. Follower -> the knob it copies.
_FOLLOWS = {"hclust_normalize": "normalize"}


def _swept(method: str) -> list[str]:
    """The RUN_CONFIG keys expanded for `method`: shared knobs plus its own."""
    shared = [
        k
        for k in RUN_CONFIG
        if k != "method" and k not in _METHOD_ONLY and k not in _FOLLOWS
    ]
    return shared + list(METHOD_KNOBS[method])


def _configs() -> list[dict[str, object]]:
    """RUN_CONFIG's cartesian product, one branch per DR method.

    Knobs outside a branch stay at their BASE_CONFIG value: the browser posts all
    eighteen whatever the method, so a config carrying a swept value for a knob the
    UI keeps at its default is a key nothing will ever look up.
    """
    out: list[dict[str, object]] = []
    for method in RUN_CONFIG["method"]:
        keys = _swept(str(method))
        for values in product(*(RUN_CONFIG[k] for k in keys)):
            config = dict(BASE_CONFIG)
            config["method"] = method
            config.update(zip(keys, values, strict=True))
            for follower, leader in _FOLLOWS.items():
                config[follower] = config[leader]
            out.append(config)
    return out


def _for_dataset(config: dict[str, object], n_features: int) -> dict[str, object]:
    """Resolve the one knob whose value depends on the dataset.

    App.tsx raises the pre-reduction width to the feature count as soon as a dataset
    loads, so clustering runs on the full space (analysis_routine skips the
    pre-reduction once n_components is not below n_features). `None` in RUN_CONFIG
    means that value; a number is used as written.
    """
    if config["hclust_umap_n_components"] is not None:
        return config
    return {**config, "hclust_umap_n_components": max(2, n_features)}


def _label(dataset: str, config: dict[str, object]) -> str:
    """Dataset, method, and the knobs this sweep actually varies."""
    knobs = [k for k in _swept(str(config["method"])) if len(RUN_CONFIG[k]) > 1]
    return "  ".join(
        [dataset, str(config["method"]), *(f"{k}={config[k]}" for k in knobs)]
    )


def _hms(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


# What a worker needs to build one run: dataset, feature_cols, config, cache key,
# plus the label the parent prints when it comes back.
_Task = tuple[str, list[str], dict[str, object], str, str]


def _queue(configs: list[dict[str, object]]) -> list[_Task]:
    """One task per run, with everything dataset-dependent already resolved.

    Resolved in the parent so a worker receives plain picklable values and never
    has to reason about the sweep. The DataFrames stay out of the pickle — sending
    one per task would cost more than the build; each worker loads its own.
    """
    tasks: list[_Task] = []
    for dataset in DATASETS:
        df = ds.load(dataset)  # cached by @cache after the first load per dataset
        feature_cols = ds.default_feature_cols(df)
        print(f"  {dataset}: {len(df)} rows x {len(feature_cols)} features", flush=True)
        for swept in configs:
            config = _for_dataset(swept, len(feature_cols))
            key = _cache_key(dataset, feature_cols, config)
            tasks.append((dataset, feature_cols, config, key, _label(dataset, config)))
    return tasks


def _quiet() -> None:
    """Point a worker's output at /dev/null, once, when the process starts.

    Redirected at the file-descriptor level so numba's and the BLAS libraries'
    native chatter goes with it, not just what `src.util.console` prints.
    """
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def _run_one(
    dataset: str, feature_cols: list[str], config: dict[str, object], key: str
) -> tuple[str, float, str]:
    """Build one run into the cache, in a worker process.

    Returns (status, seconds, traceback) rather than the payload: `_build` has
    already written it to disk, and shipping an 8-10 MB tree back through the pool
    would cost more than the build that produced it.
    """
    if run_cache.load(key) is not None:
        return "skipped", 0.0, ""

    df = ds.load(dataset)
    req = AnalysisRequest(dataset=dataset, feature_cols=feature_cols, config=config)
    t0 = time.perf_counter()
    try:
        _build(req, df, key)
    except Exception:  # one bad config must not cost the remaining runs
        return "failed", time.perf_counter() - t0, traceback.format_exc()
    finally:
        # Payloads are large and the script only needs them on disk.
        _tree_cache.clear()
    return "done", time.perf_counter() - t0, ""


def main() -> int:
    started = time.perf_counter()
    print(f"cache dir: {run_cache.cache_dir()}")
    configs = _configs()
    runs = _queue(configs)
    print(
        f"{len(runs)} runs queued "
        f"({len(DATASETS)} datasets x {len(configs)} configs), {WORKERS} at a time\n"
    )

    failed: list[tuple[str, str]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_quiet) as pool:
        pending = {
            pool.submit(_run_one, dataset, feature_cols, config, key): label
            for dataset, feature_cols, config, key, label in runs
        }
        for future in as_completed(pending):
            label = pending[future]
            completed += 1
            try:
                status, seconds, tb = future.result()
            except Exception:  # the worker itself died — crash, OOM, cancelled pool
                status, seconds, tb = "failed", 0.0, traceback.format_exc()

            if status == "skipped":
                print(f"[{completed}/{len(runs)}] {label} — already cached, skipping")
                continue
            if status == "failed":
                failed.append((label, tb))
            verb = "done in" if status == "done" else "FAILED after"
            print(
                f"[{completed}/{len(runs)}] {label} — {verb} {_hms(seconds)}",
                flush=True,
            )

    print(
        f"\ntotal {_hms(time.perf_counter() - started)}, {len(runs) - len(failed)}/{len(runs)} cached"
    )
    for label, tb in failed:
        print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

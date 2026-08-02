"""Pipeline-level configuration tuning for dataset presets (EQ1b).

Design and pre-registration: ``src_research/EXPERIMENT_pipeline_tuning.md``.
Read it before changing anything here; the guards, objectives and acceptance
criteria in this file are pre-registered and must not be tuned to taste.

Unlike ``src_research/hyperparameter_tuning.py`` (which optimises a *flat*
``DR -> cluster`` pipeline on a 500-row subsample), this harness optimises the
**deployed recursive pipeline** at full ``n``: every trial calls the same
``compute_analysis_tree`` the FastAPI backend serves, and scores the tree that
comes out. One build yields both objectives.

Run with::

    python -m src_research.pipeline_tuning            # all datasets
    python -m src_research.pipeline_tuning --datasets "Iris (Low)" --trials 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# kdbcv 1.0.0 type-hints reference the removed np.float_ / np.int_ aliases; the project
# overrides its stale numpy cap (see pyproject.toml), so the aliases are shimmed here.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]

import optuna
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Pre-registered constants (EXPERIMENT_pipeline_tuning.md sections 4-6).       #
# --------------------------------------------------------------------------- #

SEED = 42
N_TRIALS = 40
N_STARTUP_TRIALS = 10
BUILD_TIMEOUT_S = 180.0
EVAL_K = 10  # fixed k for O2, so T&C is comparable across tree shapes
MIN_NODE_PTS_FOR_SCORE = 2 * EVAL_K + 1  # 21
MIN_CLUSTER_SIZE_FLOOR = MIN_NODE_PTS_FOR_SCORE  # every leaf must be scoreable
MAX_NOISE_FRAC = 0.5
MIN_SCORED_COVERAGE = 0.8
MDS_MAX_N = 2000  # MDS offered as a view method only below this n
N_BASELINE_BUILDS = 10  # 1-5 select, 6-10 test
N_VALIDATION_BUILDS = 5
WORST = (-1.0, 0.0)

OUTPUT_ROOT = Path("outputs/experiments")

DATASETS_TO_RUN = [
    "Iris (Low)",
    "Breast cancer (Low)",
    "Concentric rings (Low)",
    "Digits (Low)",
    "Wine quality (Low)",
]

# Label-bearing / bookkeeping columns that must never be clustering features.
# (backend/datasets.py::default_feature_cols returns *every* column but row_id,
# which feeds these into the clustering; see design section 2.)
_NON_FEATURE_COLS = {"row_id", "is_red", "quality", "manifold_position"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and not str(c).startswith("target_")]


def ground_truth_labels(df: pd.DataFrame) -> np.ndarray | None:
    target_cols = [c for c in df.columns if str(c).startswith("target_")]
    if target_cols:
        return df[target_cols].to_numpy().argmax(axis=1)
    if "is_red" in df.columns:
        return df["is_red"].to_numpy().astype(int)
    return None


# --------------------------------------------------------------------------- #
# Metrics computed inside the build subprocess.                                #
# --------------------------------------------------------------------------- #


def _leaf_partition(tree: Any, n_rows: int) -> np.ndarray:
    """Leaf id per row; -1 for rows dropped as HDBSCAN noise at any level."""
    labels = np.full(n_rows, -1, dtype=int)
    counter = [0]

    def walk(node: Any) -> None:
        if "is_leaf" in node:
            labels[node["row_indices"]] = counter[0]
            counter[0] += 1
            return
        for child in node["next_object_layer"] or []:
            walk(child)

    walk(tree)
    return labels


def _depth1_partition(tree: Any, n_rows: int) -> np.ndarray:
    labels = np.full(n_rows, -1, dtype=int)
    if "is_leaf" in tree:
        labels[tree["row_indices"]] = 0
        return labels
    for i, child in enumerate(tree["next_object_layer"] or []):
        labels[child["row_indices"]] = i
    return labels


def _iter_nodes(node: Any):
    yield node
    if "is_leaf" not in node:
        for child in node["next_object_layer"] or []:
            yield from _iter_nodes(child)


def _tnc_at_fixed_k(X: np.ndarray, emb: np.ndarray) -> float | None:
    from zadu.zadu import ZADU

    if X.shape[0] < MIN_NODE_PTS_FOR_SCORE:
        return None
    scores = ZADU([{"id": "tnc", "params": {"k": EVAL_K}}], X).measure(emb)[0]
    t, c = scores.get("trustworthiness"), scores.get("continuity")
    if t is None or c is None or not np.isfinite(t) or not np.isfinite(c):
        return None
    return float((t + c) / 2.0)


def _dbcv(X: np.ndarray, labels: np.ndarray) -> tuple[float | None, str]:
    """(score, status). kDBCV returns a legitimate-looking -1.0 for degenerate
    input, so the degenerate cases are detected from the labels beforehand."""
    import contextlib
    import io

    n_clusters = len({int(v) for v in labels} - {-1})
    if n_clusters < 2:
        return None, f"degenerate:{n_clusters}_clusters"
    from kDBCV.DBCV import DBCV_score

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            value = float(DBCV_score(X, labels)[0])
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return None, f"dbcv_error:{type(exc).__name__}"
    if not np.isfinite(value):
        return None, "dbcv_nonfinite"
    return value, "ok"


def _build_and_score(dataset: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """One build + scoring. Runs in a child process; returns a flat dict."""
    from src.analysis.analysis_routine import compute_analysis_tree
    from src.config_defaults import default_config
    from src.datasets import DATASETS
    from src.util import console as clog

    for name in ("phase", "substep", "success", "build_banner", "warn", "info"):
        if hasattr(clog, name):
            setattr(clog, name, lambda *a, **k: None)

    df = DATASETS[dataset]()
    fcols = feature_columns(df)
    y = ground_truth_labels(df)

    config = default_config()  # fresh per build: compute_analysis_tree mutates it
    config["dataset_choice"] = dataset
    config["hclust_umap_n_components"] = max(2, len(fcols))  # App.tsx behaviour
    config.update(overrides)

    t0 = time.time()
    tree = compute_analysis_tree(df, fcols, config)  # type: ignore[arg-type]
    build_s = time.time() - t0

    scaler = tree.get("scaler")
    X_all = df[fcols].to_numpy(dtype=np.float64)
    if scaler is not None:
        X_all = scaler.transform(X_all)

    leaf_labels = _leaf_partition(tree, len(df))
    noise_frac = float(np.mean(leaf_labels == -1))
    leaf_sizes = [int(np.sum(leaf_labels == c)) for c in sorted(set(leaf_labels.tolist()) - {-1})]

    dbcv, dbcv_status = _dbcv(X_all, leaf_labels)
    keep = leaf_labels != -1
    dbcv_nf, _ = _dbcv(X_all[keep], leaf_labels[keep]) if keep.any() else (None, "")

    # O2 at fixed k, over nodes whose embedding actually exists.
    tnc_vals: list[float] = []
    zero_embed = 0
    scored_rows = 0
    for node in _iter_nodes(tree):
        emb = node["embedding_original"]
        idx = node["row_indices"]
        if emb.shape[0] < MIN_NODE_PTS_FOR_SCORE or emb.shape[1] < 2:
            continue
        # _embed_original swallows reducer failures and returns zeros; ZADU would
        # happily score that constant embedding at ~0.55 (design section 9, m2).
        if node.get("embedding_original_variance") is None and not np.any(emb):
            zero_embed += 1
            continue
        val = _tnc_at_fixed_k(X_all[idx], emb)
        if val is not None:
            tnc_vals.append(val)
            if "is_leaf" in node:
                scored_rows += len(idx)

    leaf_rows = int(np.sum(keep))
    out: dict[str, Any] = {
        "build_seconds": build_s,
        "dbcv_leaf": dbcv,
        "dbcv_status": dbcv_status,
        "dbcv_leaf_noisefree": dbcv_nf,
        "tnc_mean": float(np.mean(tnc_vals)) if tnc_vals else None,
        "n_scored_nodes": len(tnc_vals),
        "scored_coverage": (scored_rows / leaf_rows) if leaf_rows else 0.0,
        "zero_embed_nodes": zero_embed,
        "n_leaves": len(leaf_sizes),
        "median_leaf_size": float(np.median(leaf_sizes)) if leaf_sizes else 0.0,
        "min_leaf_size": min(leaf_sizes) if leaf_sizes else 0,
        "noise_frac": noise_frac,
        "preclustering_skipped": bool(config["hclust_umap_n_components"] >= len(fcols)),
        "n_rows": len(df),
        "n_features": len(fcols),
    }
    if y is not None and len(leaf_sizes) >= 1:
        out["ari"] = float(adjusted_rand_score(y[keep], leaf_labels[keep])) if keep.any() else None
        out["ami"] = float(adjusted_mutual_info_score(y[keep], leaf_labels[keep])) if keep.any() else None
        d1 = _depth1_partition(tree, len(df))
        k1 = d1 != -1
        out["ari_depth1"] = float(adjusted_rand_score(y[k1], d1[k1])) if k1.any() else None
    else:
        out["ari"] = out["ami"] = out["ari_depth1"] = None
    return out


def _worker(dataset: str, overrides: dict[str, Any], q: mp.Queue) -> None:
    try:
        q.put(("ok", _build_and_score(dataset, overrides)))
    except Exception:  # noqa: BLE001 - exceptions are a pre-registered outcome (A4)
        q.put(("error", traceback.format_exc(limit=3)))


def build(dataset: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Run one build under a hard wall-clock abort (design section 4)."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(dataset, overrides, q))
    t0 = time.time()
    p.start()
    p.join(BUILD_TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"exception": "timeout", "build_seconds": time.time() - t0, "degenerate": True}
    try:
        status, payload = q.get_nowait()
    except Exception:  # noqa: BLE001 - child died without putting a result
        return {"exception": "child_died", "build_seconds": time.time() - t0, "degenerate": True}
    if status == "error":
        return {"exception": payload.strip().splitlines()[-1][:200], "build_seconds": time.time() - t0, "degenerate": True}
    payload["exception"] = None
    payload["degenerate"] = _is_degenerate(payload)
    return payload


def _is_degenerate(m: dict[str, Any]) -> bool:
    """Pre-registered degeneracy guards (design section 4)."""
    return bool(
        m.get("n_leaves", 0) < 2
        or m.get("dbcv_leaf") is None
        or m.get("tnc_mean") is None
        or m.get("noise_frac", 1.0) >= MAX_NOISE_FRAC
        or m.get("scored_coverage", 0.0) < MIN_SCORED_COVERAGE,
    )


def objectives(m: dict[str, Any]) -> tuple[float, float]:
    return WORST if m.get("degenerate") else (float(m["dbcv_leaf"]), float(m["tnc_mean"]))


# --------------------------------------------------------------------------- #
# Search space.                                                                #
# --------------------------------------------------------------------------- #


def suggest_config(trial: optuna.Trial, n: int, d: int) -> dict[str, Any]:
    f_mcs_lo = MIN_CLUSTER_SIZE_FLOOR / n
    f_mcs = trial.suggest_float("f_min_cluster_size", min(f_mcs_lo, 0.25), 0.25)
    mcs = max(MIN_CLUSTER_SIZE_FLOOR, round(f_mcs * n))
    r_ms = trial.suggest_float("r_min_samples", 0.05, 1.0)

    cfg: dict[str, Any] = {
        "hierarchical_layers": trial.suggest_int("hierarchical_layers", 1, 3),
        "hclust_umap_n_components": trial.suggest_int("hclust_umap_n_components", 2, max(2, d)),
        "hclust_min_cluster_size": mcs,
        "hclust_min_samples": max(1, round(r_ms * mcs)),
        "umap_n_neighbors": trial.suggest_int("umap_n_neighbors", 5, 50),
        "umap_min_dist": trial.suggest_float("umap_min_dist", 0.0, 0.5),
    }
    methods = ["PCA", "UMAP", "t-SNE"] + (["MDS"] if n <= MDS_MAX_N else [])
    method = trial.suggest_categorical("method", methods)
    cfg["method"] = method
    if method == "t-SNE":
        cfg["tsne_perplexity"] = trial.suggest_float("tsne_perplexity", 5.0, 50.0)
        cfg["tsne_learning_rate"] = trial.suggest_float("tsne_learning_rate", 10.0, 1000.0)
    elif method == "MDS":
        cfg["mds_n_init"] = trial.suggest_int("mds_n_init", 1, 4)
        cfg["mds_max_iter"] = trial.suggest_int("mds_max_iter", 50, 300)
    return cfg


PRESET_KEYS = (
    "hierarchical_layers",
    "hclust_umap_n_components",
    "hclust_min_cluster_size",
    "hclust_min_samples",
    "umap_n_neighbors",
    "umap_min_dist",
    "method",
    "tsne_perplexity",
    "tsne_learning_rate",
    "mds_n_init",
    "mds_max_iter",
)


# --------------------------------------------------------------------------- #
# Per-dataset run.                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class DatasetRun:
    dataset: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    baseline: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    candidate: dict[str, Any] | None = None
    verdict: dict[str, Any] = field(default_factory=dict)


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def reference_dbcv(dataset: str) -> dict[str, Any]:
    """DBCV of the ground-truth partition, same space as O1 (design section 3)."""
    from sklearn.preprocessing import StandardScaler

    from src.datasets import DATASETS

    df = DATASETS[dataset]()
    fcols = feature_columns(df)
    y = ground_truth_labels(df)
    if y is None:
        return {"dataset": dataset, "dbcv_ground_truth": None}
    X = StandardScaler().fit_transform(df[fcols].to_numpy(dtype=np.float64))
    value, status = _dbcv(X, y)
    return {"dataset": dataset, "dbcv_ground_truth": value, "status": status, "n_classes": len(set(y.tolist()))}


def run_dataset(dataset: str, n_trials: int, out_dir: Path) -> DatasetRun:
    from src.datasets import DATASETS

    run = DatasetRun(dataset=dataset)
    df = DATASETS[dataset]()
    fcols = feature_columns(df)
    n, d = len(df), len(fcols)
    _log(f"=== {dataset}  n={n} d={d} ===")

    # ---- baseline: 10 builds, 1-5 select / 6-10 test -----------------------
    for i in range(N_BASELINE_BUILDS):
        m = build(dataset, {})
        m |= {"dataset": dataset, "arm": "baseline", "build_index": i, "split": "select" if i < 5 else "test"}
        run.baseline.append(m)
        _log(f"  baseline {i + 1}/{N_BASELINE_BUILDS}: dbcv={m.get('dbcv_leaf')} tnc={m.get('tnc_mean')} leaves={m.get('n_leaves')} {m.get('build_seconds', 0):.0f}s")

    tnc_select = [b["tnc_mean"] for b in run.baseline if b["split"] == "select" and b.get("tnc_mean") is not None]
    tnc_floor = (float(np.mean(tnc_select)) - 0.01) if tnc_select else -np.inf
    _log(f"  selection floor tnc >= {tnc_floor:.4f}")

    # ---- tuning ------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        cfg = suggest_config(trial, n, d)
        m = build(dataset, cfg)
        m |= {"dataset": dataset, "arm": "trial", "trial": trial.number, **{f"p_{k}": v for k, v in cfg.items()}}
        run.trials.append(m)
        pd.DataFrame(run.trials).to_csv(out_dir / f"trials_{_slug(dataset)}.csv", index=False)
        o = objectives(m)
        _log(f"  trial {trial.number:2d}: {cfg['method']:5s} L{cfg['hierarchical_layers']} mcs={cfg['hclust_min_cluster_size']:4d} -> dbcv={o[0]:+.4f} tnc={o[1]:.4f} leaves={m.get('n_leaves')} {m.get('build_seconds', 0):.0f}s{' DEGEN' if m.get('degenerate') else ''}{' ' + str(m.get('exception')) if m.get('exception') else ''}")
        return o

    study.optimize(objective, n_trials=n_trials, catch=())

    # ---- selection rule ----------------------------------------------------
    front = [t for t in study.best_trials]
    front_rows = [run.trials[t.number] for t in front if t.number < len(run.trials)]
    eligible = [
        r for r in front_rows
        if not r.get("degenerate") and r.get("tnc_mean") is not None and r["tnc_mean"] >= tnc_floor
    ]
    if eligible:
        eligible.sort(key=lambda r: (-r["dbcv_leaf"], r["build_seconds"], r["p_hierarchical_layers"]))
        run.candidate = eligible[0]
        _log(f"  candidate: trial {run.candidate['trial']} dbcv={run.candidate['dbcv_leaf']:+.4f} tnc={run.candidate['tnc_mean']:.4f}")
    else:
        _log("  no eligible Pareto point -> defaults retained")
        run.verdict = {"dataset": dataset, "adopted": False, "reason": "no Pareto point met the view floor"}
        return run

    # ---- validation --------------------------------------------------------
    cfg = {k[2:]: v for k, v in run.candidate.items() if k.startswith("p_")}
    for i in range(N_VALIDATION_BUILDS):
        m = build(dataset, cfg)
        m |= {"dataset": dataset, "arm": "preset", "build_index": i}
        run.validation.append(m)
        _log(f"  preset {i + 1}/{N_VALIDATION_BUILDS}: dbcv={m.get('dbcv_leaf')} tnc={m.get('tnc_mean')} leaves={m.get('n_leaves')} {m.get('build_seconds', 0):.0f}s")

    run.verdict = judge(dataset, run, cfg, fcols)
    return run


def _vals(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def judge(dataset: str, run: DatasetRun, cfg: dict[str, Any], fcols: list[str]) -> dict[str, Any]:
    """Apply A1-A6 exactly as pre-registered (design section 6)."""
    test = [b for b in run.baseline if b["split"] == "test"]
    pre = run.validation

    b_dbcv, p_dbcv = _vals(test, "dbcv_leaf"), _vals(pre, "dbcv_leaf")
    b_tnc, p_tnc = _vals(test, "tnc_mean"), _vals(pre, "tnc_mean")
    b_sec, p_sec = _vals(test, "build_seconds"), _vals(pre, "build_seconds")
    b_noise, p_noise = _vals(test, "noise_frac"), _vals(pre, "noise_frac")
    b_ari, p_ari = _vals(test, "ari"), _vals(pre, "ari")

    n_ok = sum(1 for m in pre if not m.get("exception"))
    a4 = n_ok == N_VALIDATION_BUILDS
    a1 = bool(p_dbcv and b_dbcv and min(p_dbcv) > max(b_dbcv))
    a2 = bool(p_tnc and b_tnc and float(np.mean(p_tnc)) >= float(np.mean(b_tnc)) - 0.01)
    a3 = bool(p_sec and b_sec and float(np.median(p_sec)) <= min(BUILD_TIMEOUT_S, 3.0 * float(np.median(b_sec))))
    a5 = bool(
        pre
        and all((m.get("median_leaf_size") or 0) >= MIN_NODE_PTS_FOR_SCORE for m in pre)
        and p_noise
        and b_noise
        and float(np.mean(p_noise)) <= float(np.mean(b_noise)) + 0.05,
    )
    a6 = bool(not b_ari or not p_ari or float(np.mean(p_ari)) >= float(np.mean(b_ari)) - 0.05)

    adopted = all([a1, a2, a3, a4, a5, a6])
    return {
        "dataset": dataset,
        "adopted": adopted,
        "A1_separation": a1,
        "A2_view": a2,
        "A3_interactive": a3,
        "A4_reliable": a4,
        "A5_granularity": a5,
        "A6_ari": a6,
        "baseline_dbcv_mean": float(np.mean(b_dbcv)) if b_dbcv else None,
        "baseline_dbcv_max": float(np.max(b_dbcv)) if b_dbcv else None,
        "preset_dbcv_mean": float(np.mean(p_dbcv)) if p_dbcv else None,
        "preset_dbcv_min": float(np.min(p_dbcv)) if p_dbcv else None,
        "baseline_tnc_mean": float(np.mean(b_tnc)) if b_tnc else None,
        "preset_tnc_mean": float(np.mean(p_tnc)) if p_tnc else None,
        "baseline_ari_mean": float(np.mean(b_ari)) if b_ari else None,
        "preset_ari_mean": float(np.mean(p_ari)) if p_ari else None,
        "baseline_noise_mean": float(np.mean(b_noise)) if b_noise else None,
        "preset_noise_mean": float(np.mean(p_noise)) if p_noise else None,
        "baseline_seconds_median": float(np.median(b_sec)) if b_sec else None,
        "preset_seconds_median": float(np.median(p_sec)) if p_sec else None,
        "baseline_leaves": [b.get("n_leaves") for b in test],
        "preset_leaves": [m.get("n_leaves") for m in pre],
        "config": cfg,
        "feature_cols": fcols,
    }


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS_TO_RUN)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = args.out or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"run {stamp} -> {out_dir}  ({len(args.datasets)} datasets, {args.trials} trials each)")

    from src.config_defaults import default_config

    base = {k: v for k, v in default_config().items() if isinstance(v, (int, float, str, bool))}
    (out_dir / "baseline.json").write_text(
        json.dumps({"note": "hclust_umap_n_components is overwritten with n_features by App.tsx", "config_defaults": base}, indent=2),
    )

    pd.DataFrame([reference_dbcv(ds) for ds in args.datasets]).to_csv(out_dir / "reference_dbcv.csv", index=False)

    runs: list[DatasetRun] = []
    for ds in args.datasets:
        run = run_dataset(ds, args.trials, out_dir)
        runs.append(run)
        rows = run.baseline + run.validation
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"validation_{_slug(ds)}.csv", index=False)
        (out_dir / f"verdict_{_slug(ds)}.json").write_text(json.dumps(run.verdict, indent=2, default=str))
        _log(f"  VERDICT {ds}: {'ADOPTED' if run.verdict.get('adopted') else 'defaults retained'}")

    presets = {
        r.dataset: {k: v for k, v in r.verdict["config"].items() if k in PRESET_KEYS}
        | {"_feature_cols": r.verdict["feature_cols"], "_run": stamp}
        for r in runs
        if r.verdict.get("adopted")
    }
    (out_dir / "presets.json").write_text(json.dumps(presets, indent=2))
    pd.DataFrame([r.verdict for r in runs]).to_csv(out_dir / "verdicts.csv", index=False)
    _log(f"done. adopted: {list(presets) or 'none'}")


if __name__ == "__main__":
    main()

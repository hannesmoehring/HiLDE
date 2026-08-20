"""Hyperparameter-tuning effectiveness experiment (thesis).

This script measures *how effective* automated hyperparameter tuning is for the
dimensionality-reduction -> clustering pipeline used throughout this project. It
sweeps a configurable grid of datasets, DR methods, clustering methods and
optimisation objectives, and for every grid cell quantifies tuning effectiveness
along four axes:

    1. Default-vs-tuned gain   - best tuned score minus the score obtained with
                                 the project's default hyperparameters.
    2. TPE-vs-random search    - an Optuna TPE study vs. a random-search study of
                                 the same budget. Shows whether *intelligent*
                                 search beats brute luck (i.e. tuning, not chance).
    3. Convergence speed       - the best-so-far curve over trials.
    4. External validation     - agreement of the resulting clusters with the
                                 ground-truth labels (ARI / NMI), to check whether
                                 optimising an internal metric yields *real* gains.

Two objective tracks are run (the objective metric defines which factors vary):

    Track A - "clustering quality".  Objective = DBCV (original space). The whole
        pipeline is tuned: DR hyperparameters *including the embedding dimension*
        plus the clustering hyperparameters. Factors: dataset x DR x clustering.

    Track B - "DR faithfulness".  Objective = ZADU trustworthiness & continuity
        (mean) of a 2-D embedding. Only the DR hyperparameters are tuned
        (n_components fixed at 2 - the visualisation embedding); the clustering
        axis collapses. For external validation the tuned 2-D embedding is
        clustered with default HDBSCAN. Factors: dataset x DR.

Outputs (written to ``outputs/experiments/<timestamp>/``):
    * results_summary.csv - one row per (track, dataset, DR, clustering, sampler).
    * trials.csv          - every individual trial (for convergence analysis).
    * plots/*.png         - gain heatmaps, TPE-vs-random bars, convergence curves,
                            internal-vs-external scatter.

Reproducibility: ``reduce_dimensionality`` threads ``config["*_random_state"]``
into UMAP / t-SNE / MDS, and ``init_state`` pins all three, so a trial's embedding
is a deterministic function of its suggested hyperparameters. Together with the
seeded Optuna samplers and the seeded subsample, a rerun reproduces the grid.
Note the flip side: this harness therefore measures no embedding variance at all,
so a "gain" here is a gain at one seed, not an expected gain.

Run with::

    uv run python -m src_research.hyperparameter_tuning
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

# kDBCV's type hints reference np.float_ / np.int_, which NumPy 2.0 removed, and they are
# evaluated at def time — so without this the import below raises AttributeError and this
# module cannot be imported at all. `pipeline_tuning` and `dbcv_tuning` both carry the same
# shim; this one was missing, which is why the Track A/B experiment could not be regenerated.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]

from joblib import Parallel, delayed, parallel_config
from kDBCV.DBCV import DBCV_score
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from zadu.zadu import ZADU

from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import reduce_dimensionality
from src.config_defaults import init_state
from src.datasets import DATASETS
from src.types import Config

# --------------------------------------------------------------------------- #
# CONFIG - edit this block to change the experiment. Defaults = "Medium" grid. #
# --------------------------------------------------------------------------- #

SEED = 42
N_TRIALS = 100  # trials per study (one TPE + one random study per cell)
SUBSAMPLE_CAP = 500  # cap rows per dataset (seeded) to keep t-SNE/UMAP tractable
PARALLEL_JOBS = (
    -1
)  # grid cells to run concurrently; 1 = serial, -1 = all cores. See note below.
# Cells (dataset x DR x clustering) are independent and run in separate processes when
# PARALLEL_JOBS != 1. Each worker's inner BLAS/UMAP threads are capped to 1 to avoid
# oversubscription, so a good value is ~ (physical cores) with a little headroom.

DATASETS_TO_RUN = [
    "Wine quality (Low)",  # low-dim (11) real tabular, binary ground truth
    "Digits (Low)",  # high-dim (64), 10 classes, manifold structure
    "Breast cancer (Low)",  # mid-dim (30) real tabular, binary ground truth
    "Concentric rings (Low)",  # non-convex density clusters; density-based vs convex contrast
]
DR_METHODS = ["UMAP", "t-SNE", "PCA", "MDS"]  # also supported: "MDS" (slow)
CLUSTER_METHODS = [
    "HDBSCAN",
    "DBSCAN",
    "KMeans",
]  # Track A only; Track B collapses clustering axis #, "GMM"
TRACKS = ["A", "B"]  # A = DBCV pipeline, B = ZADU DR faithfulness

OUTPUT_ROOT = Path("outputs/experiments")

# --------------------------------------------------------------------------- #
# Quiet third-party noise so the rich output stays readable.                   #
# --------------------------------------------------------------------------- #


def _silence_noise() -> None:
    """Mute third-party chatter. Called at import and re-asserted in each worker
    process, since loky children re-import the module and Optuna reconfigures its
    own logger lazily on first use.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")


_silence_noise()
console = Console()


# --------------------------------------------------------------------------- #
# Metric / track specifications.                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Metric:
    """An optimisation objective: its name, direction and worst-case value.

    ``worst`` is returned for degenerate or failed trials so Optuna treats them
    as the least desirable outcome regardless of search direction.
    """

    name: str
    direction: str  # "maximize" or "minimize"
    worst: float


DBCV_METRIC = Metric("dbcv", "maximize", -1.0)  # DBCV is in [-1, 1]
TNC_METRIC = Metric(
    "tnc", "maximize", 0.0
)  # mean(trustworthiness, continuity) in [0, 1]


def finite_or_worst(value: float | None, metric: Metric) -> float:
    """Coerce None / NaN / inf to the metric's worst value.

    Optuna rejects a NaN objective ("The value nan is not acceptable") and marks
    the trial failed; DBCV returns NaN for some degenerate clusterings. Mapping
    non-finite scores to ``worst`` keeps the trial valid and just ranks it last.
    """
    if value is None or not np.isfinite(value):
        return metric.worst
    return float(value)


# --------------------------------------------------------------------------- #
# Dataset preparation.                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    name: str
    X: np.ndarray  # standardised feature matrix (n, d)
    y: np.ndarray | None  # integer ground-truth labels, or None if unavailable
    n_dims: int


def prepare_dataset(display_name: str) -> Dataset:
    """Load a dataset from the project registry and split it into features / labels.

    Every label column in the registry is named ``target_*``, so features are always
    "everything but row_id and target_*". Only the label convention varies:
      * wine quality (``target_is_red``) -> labels = is_red (binary ground truth);
      * swiss roll (``target_manifold_position``) -> continuous, no discrete labels;
      * one-hot ``target_*`` block -> labels = argmax.
    Features are standardised; large datasets are subsampled to ``SUBSAMPLE_CAP``.
    """
    df = DATASETS[display_name]()
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [
        c for c in df.columns if c != "row_id" and not c.startswith("target_")
    ]

    if "target_is_red" in df.columns:  # wine quality
        y = df["target_is_red"].to_numpy().astype(int)
    elif (
        "target_manifold_position" in df.columns
    ):  # swiss roll - continuous, no classes
        y = None
    elif target_cols:
        y = df[target_cols].to_numpy().argmax(axis=1)
    else:
        y = None

    X = df[feature_cols].to_numpy(dtype=np.float64)

    if X.shape[0] > SUBSAMPLE_CAP:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], SUBSAMPLE_CAP, replace=False)
        X = X[idx]
        y = y[idx] if y is not None else None

    X = StandardScaler().fit_transform(X)
    return Dataset(name=display_name, X=X, y=y, n_dims=X.shape[1])


# --------------------------------------------------------------------------- #
# Search spaces.                                                               #
# --------------------------------------------------------------------------- #


def suggest_dr_params(
    trial: optuna.Trial, method: str, n_dims: int, fixed_n_components: int | None
) -> tuple[dict, int]:
    """Suggest DR hyperparameters. Returns (config overrides, n_components).

    ``fixed_n_components`` pins the embedding dimension (Track B uses 2); when
    None (Track A) the dimension itself is part of the search space.
    """
    overrides: dict = {}
    if method == "UMAP":
        overrides["umap_n_neighbors"] = trial.suggest_int("umap_n_neighbors", 5, 50)
        overrides["umap_min_dist"] = trial.suggest_float("umap_min_dist", 0.0, 0.5)
        n_components = fixed_n_components or trial.suggest_int(
            "n_components", 2, min(n_dims, 15)
        )
    elif method == "t-SNE":
        overrides["tsne_perplexity"] = trial.suggest_float("tsne_perplexity", 5.0, 50.0)
        overrides["tsne_learning_rate"] = trial.suggest_float(
            "tsne_learning_rate", 10.0, 1000.0
        )
        n_components = 2  # sklearn 'barnes_hut' only supports 2 components
    elif method == "PCA":
        n_components = fixed_n_components or trial.suggest_int(
            "n_components", 2, n_dims
        )
    elif method == "MDS":
        n_components = fixed_n_components or trial.suggest_int(
            "n_components", 2, min(n_dims, 10)
        )
    else:
        raise ValueError(f"Unknown DR method: {method}")
    return overrides, n_components


def suggest_cluster_params(trial: optuna.Trial, method: str) -> dict:
    """Suggest clustering hyperparameters as direct ``Config`` key overrides.

    Note the GMM quirk: ``compute_clusters`` reads ``hclust_umap_n_components`` as
    the number of mixture components, so that is the key we tune for GMM.
    """
    if method == "HDBSCAN":
        return {
            "hclust_min_cluster_size": trial.suggest_int(
                "hclust_min_cluster_size", 2, 50
            ),
            "hclust_min_samples": trial.suggest_int("hclust_min_samples", 1, 25),
        }
    if method == "DBSCAN":
        return {
            "dbscan_eps": trial.suggest_float("dbscan_eps", 0.1, 5.0),
            "hclust_min_samples": trial.suggest_int("hclust_min_samples", 2, 25),
        }
    if method == "KMeans":
        return {"cluster_n_clusters": trial.suggest_int("cluster_n_clusters", 2, 15)}
    if method == "GMM":
        return {
            "hclust_umap_n_components": trial.suggest_int("gmm_n_components", 2, 15)
        }
    raise ValueError(f"Unknown clustering method: {method}")


def dr_is_tunable(method: str, fixed_n_components: int | None) -> bool:
    """True if the DR method has any hyperparameter to search in this track.

    PCA with a pinned dimension (Track B) is deterministic and has nothing to
    tune, so its cells are reported baseline-only.
    """
    return not (method == "PCA" and fixed_n_components is not None)


# --------------------------------------------------------------------------- #
# Pipeline + metric computation.                                               #
# --------------------------------------------------------------------------- #


def run_pipeline(
    X: np.ndarray, dr_method: str, n_components: int, cluster_method: str, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce then cluster. Normalises the clustering return-type asymmetry
    (HDBSCAN returns ``(labels, outlier_scores)``; others return ``labels``)."""
    embedding = reduce_dimensionality(
        method=dr_method, X=X, config=cfg, n_components=n_components
    )
    out = compute_clusters(embedding, method=cluster_method, config=cfg)
    labels = out[0] if isinstance(out, tuple) else out
    return embedding, np.asarray(labels)


def cluster_stats(labels: np.ndarray) -> tuple[int, float]:
    """Return (n_clusters excluding noise, noise fraction)."""
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise_frac = float(np.mean(labels == -1))
    return n_clusters, noise_frac


def _dbcv(data: np.ndarray, labels: np.ndarray) -> float | None:
    """DBCV score, or None on failure. kDBCV prints "Not enough clusters..." to
    stdout for degenerate inputs, so we swallow its stdout to keep the log clean."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return float(DBCV_score(data, labels)[0])
    except Exception:
        return None


def clustering_metrics(
    X: np.ndarray, embedding: np.ndarray, labels: np.ndarray, y: np.ndarray | None
) -> dict:
    """Internal (DBCV, silhouette) and external (ARI, NMI) clustering metrics.

    Returns a dict of floats / None. ``dbcv`` is scored in the original space
    (robust but pessimistic in high dimensions); ``dbcv_embedded`` in the
    embedding for comparison.
    """
    out: dict = {
        "dbcv": None,
        "dbcv_embedded": None,
        "silhouette": None,
        "ari": None,
        "nmi": None,
    }
    n_clusters, _ = cluster_stats(labels)
    if n_clusters < 2:
        return out

    out["dbcv"] = _dbcv(X, labels)
    out["dbcv_embedded"] = _dbcv(embedding, labels)

    # silhouette ignores noise points; needs >=2 surviving clusters
    mask = labels >= 0
    if mask.sum() >= 3 and len(set(labels[mask].tolist())) >= 2:
        try:
            out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        except Exception:
            pass

    if y is not None:
        out["ari"] = float(adjusted_rand_score(y, labels))
        out["nmi"] = float(normalized_mutual_info_score(y, labels))
    return out


def dr_metrics(X: np.ndarray, embedding: np.ndarray) -> dict:
    """ZADU trustworthiness, continuity and stress for a 2-D embedding."""
    out: dict = {"trustworthiness": None, "continuity": None, "stress": None}
    n = X.shape[0]
    specs: list[dict] = [{"id": "stress"}]
    k = min(20, (n - 1) // 2) if n >= 10 else None
    if k and k >= 1:
        specs.append({"id": "tnc", "params": {"k": k}})
    try:
        results = ZADU(specs, orig=X).measure(embedding)
        out["stress"] = float(results[0]["stress"])
        if k:
            out["trustworthiness"] = float(results[1]["trustworthiness"])
            out["continuity"] = float(results[1]["continuity"])
    except Exception:
        pass
    return out


def tnc_mean(dr_scores: dict) -> float | None:
    """Mean of trustworthiness & continuity (the Track-B objective), or None."""
    t, c = dr_scores.get("trustworthiness"), dr_scores.get("continuity")
    if t is None or c is None:
        return None
    return (t + c) / 2.0


# --------------------------------------------------------------------------- #
# Objective construction (one closure per grid cell).                          #
# --------------------------------------------------------------------------- #


def make_objective(
    track: str,
    ds: Dataset,
    dr_method: str,
    cluster_method: str | None,
    base_cfg: Config,
):
    """Build an Optuna objective for a single grid cell.

    Side effect: every trial records its secondary metrics as ``user_attrs`` so
    the best trial's full metric vector can be recovered afterwards.
    """
    metric = DBCV_METRIC if track == "A" else TNC_METRIC
    fixed_nc = None if track == "A" else 2

    def objective(trial: optuna.Trial) -> float:
        cfg: Config = base_cfg.copy()
        try:
            dr_overrides, n_components = suggest_dr_params(
                trial, dr_method, ds.n_dims, fixed_nc
            )
            cfg.update(dr_overrides)

            if track == "A":
                cfg.update(suggest_cluster_params(trial, cluster_method))
                embedding, labels = run_pipeline(
                    ds.X, dr_method, n_components, cluster_method, cfg
                )
                n_clusters, noise_frac = cluster_stats(labels)
                if n_clusters < 2 or noise_frac > 0.5:
                    return metric.worst
                cm = clustering_metrics(ds.X, embedding, labels, ds.y)
                _record(trial, n_clusters=n_clusters, noise_frac=noise_frac, **cm)
                return finite_or_worst(cm["dbcv"], metric)

            # Track B: faithfulness of a 2-D embedding; cluster with default HDBSCAN for ARI
            embedding = reduce_dimensionality(
                method=dr_method, X=ds.X, config=cfg, n_components=2
            )
            dm = dr_metrics(ds.X, embedding)
            score = tnc_mean(dm)
            ext = {"ari": None, "nmi": None}
            if ds.y is not None:
                labels = np.asarray(
                    compute_clusters(embedding, method="HDBSCAN", config=cfg)[0]
                )
                if cluster_stats(labels)[0] >= 2:
                    ext["ari"] = float(adjusted_rand_score(ds.y, labels))
                    ext["nmi"] = float(normalized_mutual_info_score(ds.y, labels))
            _record(trial, **dm, **ext)
            return finite_or_worst(score, metric)
        except Exception:
            return metric.worst

    return objective, metric


def _record(trial: optuna.Trial, **attrs: object) -> None:
    for key, value in attrs.items():
        trial.set_user_attr(key, value)


# --------------------------------------------------------------------------- #
# Running one grid cell (baseline + TPE study + random study).                 #
# --------------------------------------------------------------------------- #

SAMPLERS = {
    "TPE": lambda: optuna.samplers.TPESampler(seed=SEED),
    "Random": lambda: optuna.samplers.RandomSampler(seed=SEED),
}


def evaluate_baseline(
    track: str,
    ds: Dataset,
    dr_method: str,
    cluster_method: str | None,
    base_cfg: Config,
) -> dict:
    """Score the pipeline with the project's default hyperparameters (n_components=2)."""
    cfg: Config = base_cfg.copy()
    try:
        if track == "A":
            embedding, labels = run_pipeline(ds.X, dr_method, 2, cluster_method, cfg)
            if cluster_stats(labels)[0] < 2:
                return {"baseline": DBCV_METRIC.worst}
            cm = clustering_metrics(ds.X, embedding, labels, ds.y)
            return {"baseline": finite_or_worst(cm["dbcv"], DBCV_METRIC), **cm}
        embedding = reduce_dimensionality(
            method=dr_method, X=ds.X, config=cfg, n_components=2
        )
        dm = dr_metrics(ds.X, embedding)
        return {"baseline": finite_or_worst(tnc_mean(dm), TNC_METRIC), **dm}
    except Exception:
        return {"baseline": (DBCV_METRIC if track == "A" else TNC_METRIC).worst}


def best_so_far(values: list[float], direction: str) -> list[float]:
    """Running best of a value sequence (for convergence curves)."""
    out, cur = [], (-np.inf if direction == "maximize" else np.inf)
    better = max if direction == "maximize" else min
    for v in values:
        cur = better(cur, v)
        out.append(cur)
    return out


def run_cell(
    track: str, ds: Dataset, dr_method: str, cluster_method: str | None
) -> tuple[list[dict], list[dict]]:
    """Run baseline + both samplers for one grid cell.

    Returns (summary_rows, trial_rows). One summary row per sampler.
    """
    _silence_noise()  # re-assert in each (possibly worker) process
    base_cfg = init_state(init_streamlit=False)
    objective, metric = make_objective(track, ds, dr_method, cluster_method, base_cfg)
    base = evaluate_baseline(track, ds, dr_method, cluster_method, base_cfg)

    cell = {
        "track": track,
        "dataset": ds.name,
        "dr_method": dr_method,
        "cluster_method": cluster_method or "-",
        "objective": metric.name,
    }

    # Deterministic cell with nothing to tune (PCA, Track B): baseline only.
    if not dr_is_tunable(dr_method, None if track == "A" else 2):
        row = {
            **cell,
            "sampler": "none",
            "baseline": base["baseline"],
            "best": base["baseline"],
            "gain": 0.0,
            "gain_pct": 0.0,
            "best_params": "{}",
            "tunable": False,
            **_baseline_external(base),
            **_secondary(base),
        }
        return [row], []

    summary_rows: list[dict] = []
    trial_rows: list[dict] = []
    for sampler_name, make_sampler in SAMPLERS.items():
        study = optuna.create_study(direction=metric.direction, sampler=make_sampler())
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        values = [
            t.value if t.value is not None else metric.worst for t in study.trials
        ]
        bsf = best_so_far(values, metric.direction)
        for t, b in zip(study.trials, bsf):
            trial_rows.append(
                {
                    **cell,
                    "sampler": sampler_name,
                    "trial": t.number,
                    "value": t.value,
                    "best_so_far": b,
                }
            )

        gain = study.best_value - base["baseline"]
        denom = abs(base["baseline"]) if base["baseline"] != 0 else 1.0
        summary_rows.append(
            {
                **cell,
                "sampler": sampler_name,
                "baseline": base["baseline"],
                "best": study.best_value,
                "gain": gain,
                "gain_pct": 100.0 * gain / denom,
                "best_params": json.dumps(study.best_params),
                "tunable": True,
                **_baseline_external(base),
                **_secondary(study.best_trial.user_attrs),
            }
        )
    return summary_rows, trial_rows


SECONDARY_KEYS = [
    "dbcv_embedded",
    "silhouette",
    "ari",
    "nmi",
    "trustworthiness",
    "continuity",
    "stress",
    "n_clusters",
    "noise_frac",
]


def _secondary(attrs: dict) -> dict:
    return {k: attrs.get(k) for k in SECONDARY_KEYS}


def _baseline_external(base: dict) -> dict:
    """The baseline's own external metrics, on the same row as the tuned ones.

    Without these the only way to reach a baseline ARI was to join against the
    ``sampler == "none"`` rows — which exist only for untunable cells, i.e. never for
    Track A — so the join produced an all-NaN column.
    """
    return {"ari_base": base.get("ari"), "nmi_base": base.get("nmi")}


# --------------------------------------------------------------------------- #
# Grid driver.                                                                 #
# --------------------------------------------------------------------------- #


def build_cells() -> list[tuple[str, str, str, str | None]]:
    """Enumerate (track, dataset, dr_method, cluster_method) grid cells.

    Track A varies the clustering axis; Track B collapses it (DR-only objective).
    """
    cells: list[tuple[str, str, str, str | None]] = []
    for track in TRACKS:
        for dataset in DATASETS_TO_RUN:
            for dr in DR_METHODS:
                if track == "A":
                    cells.extend((track, dataset, dr, cm) for cm in CLUSTER_METHODS)
                else:
                    cells.append((track, dataset, dr, None))
    return cells


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full grid, returning (summary_df, trials_df)."""
    cells = build_cells()
    datasets = {name: prepare_dataset(name) for name in DATASETS_TO_RUN}
    for name, ds in datasets.items():
        note = (
            "no ground truth" if ds.y is None else f"{len(set(ds.y.tolist()))} classes"
        )
        console.print(
            f"  loaded [bold]{name}[/]: {ds.X.shape[0]}x{ds.X.shape[1]}, {note}"
        )

    summaries: list[dict] = []
    trials: list[dict] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    desc = f"Running grid (jobs={PARALLEL_JOBS})"
    with progress:
        task = progress.add_task(desc, total=len(cells))
        if PARALLEL_JOBS == 1:
            for track, dataset, dr, cm in cells:
                label = f"[{track}] {dataset} / {dr}" + (f" / {cm}" if cm else "")
                progress.update(task, description=label[:48])
                s, t = run_cell(track, datasets[dataset], dr, cm)
                summaries.extend(s)
                trials.extend(t)
                progress.advance(task)
        else:
            # Cells are independent -> run in worker processes. inner_max_num_threads=1
            # stops each worker's UMAP/BLAS pools from oversubscribing the cores.
            # return_as="generator" yields results as cells finish, so the bar stays live.
            jobs = (
                delayed(run_cell)(track, datasets[dataset], dr, cm)
                for track, dataset, dr, cm in cells
            )
            with parallel_config(backend="loky", inner_max_num_threads=1):
                for s, t in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
                    summaries.extend(s)
                    trials.extend(t)
                    progress.advance(task)

    return pd.DataFrame(summaries), pd.DataFrame(trials)


# --------------------------------------------------------------------------- #
# Output: CSVs, plots, console tables.                                         #
# --------------------------------------------------------------------------- #


def save_outputs(summary: pd.DataFrame, trials: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    trials.to_csv(out_dir / "trials.csv", index=False)


def make_plots(summary: pd.DataFrame, trials: pd.DataFrame, out_dir: Path) -> None:
    """Thesis figures: gain heatmaps, TPE-vs-random, convergence, internal-vs-external."""
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    tpe = summary[(summary["sampler"] == "TPE") & summary["tunable"]]

    # 1. Track-A gain heatmap (DR x clustering) faceted by dataset.
    a = tpe[tpe["track"] == "A"]
    if not a.empty:
        datasets = a["dataset"].unique()
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(4 * len(datasets), 3.5), squeeze=False
        )
        for ax, dname in zip(axes[0], datasets):
            piv = a[a["dataset"] == dname].pivot_table(
                index="dr_method", columns="cluster_method", values="gain"
            )
            sns.heatmap(
                piv, annot=True, fmt=".2f", center=0, cmap="RdYlGn", ax=ax, cbar=False
            )
            ax.set_title(dname, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle("Track A - DBCV gain (tuned - baseline), TPE")
        fig.tight_layout()
        fig.savefig(plots / "trackA_gain_heatmap.png", dpi=150)
        plt.close(fig)

    # 2. TPE vs random: mean best per track.
    tv = summary[summary["tunable"]]
    if not tv.empty:
        agg = tv.groupby(["track", "sampler"])["best"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=agg, x="track", y="best", hue="sampler", ax=ax)
        ax.set_title("Mean best objective: TPE vs random search")
        fig.tight_layout()
        fig.savefig(plots / "tpe_vs_random.png", dpi=150)
        plt.close(fig)

    # 3. Convergence: mean best-so-far across cells, per track, TPE vs random.
    if not trials.empty:
        conv = (
            trials.groupby(["track", "sampler", "trial"])["best_so_far"]
            .mean()
            .reset_index()
        )
        tracks = conv["track"].unique()
        fig, axes = plt.subplots(
            1, len(tracks), figsize=(5 * len(tracks), 4), squeeze=False
        )
        for ax, tr in zip(axes[0], tracks):
            sub = conv[conv["track"] == tr]
            for sampler_name in sub["sampler"].unique():
                ss = sub[sub["sampler"] == sampler_name]
                ax.plot(ss["trial"], ss["best_so_far"], label=sampler_name)
            ax.set_title(f"Track {tr} convergence")
            ax.set_xlabel("trial")
            ax.set_ylabel("mean best-so-far")
            ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "convergence.png", dpi=150)
        plt.close(fig)

    # 4. Internal vs external: does DBCV gain track an ARI gain? (Track A, needs ground truth)
    # `ari_base` is recorded on the row by run_cell, not joined in from the
    # `sampler == "none"` rows: those exist only for untunable cells and therefore never for
    # Track A, so that join returns an all-NaN column, and a `.fillna(0)` on it would turn
    # the y axis into raw tuned ARI under an "ARI change vs baseline" label — a cell whose
    # tuning *lowered* ARI would plot as a gain. No fill: a pair without both halves is not
    # plotted, and the count that dropped out is printed.
    a_ext = a.dropna(subset=["ari", "ari_base"]) if not a.empty else a
    if not a_ext.empty:
        dropped = len(a) - len(a_ext)
        if dropped:
            console.print(
                f"[yellow]internal-vs-external: {dropped} of {len(a)} Track-A TPE cells lack a tuned or a baseline ARI and are not plotted.[/]"
            )
        paired = a_ext.assign(ari_gain=a_ext["ari"] - a_ext["ari_base"])
        r = float(paired["gain"].corr(paired["ari_gain"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=paired, x="gain", y="ari_gain", hue="dataset", style="dr_method", ax=ax
        )
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel(
            "DBCV gain: best tuned DBCV - default-config DBCV (original space)"
        )
        ax.set_ylabel("ARI gain: best-trial ARI - default-config ARI (vs ground truth)")
        ax.set_title(
            f"Does optimising DBCV improve agreement with ground truth?\nTrack A, TPE, n={len(paired)} cells, Pearson r={r:.3f}"
        )
        fig.tight_layout()
        fig.savefig(plots / "internal_vs_external.png", dpi=150)
        plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    """Render per-track rich tables of baseline / best / gain (TPE rows)."""
    for track in summary["track"].unique():
        sub = summary[
            (summary["track"] == track) & (summary["sampler"].isin(["TPE", "none"]))
        ]
        if sub.empty:
            continue
        metric = sub["objective"].iloc[0]
        table = Table(title=f"Track {track} - objective: {metric} (best = TPE)")
        for col in [
            "dataset",
            "dr_method",
            "cluster_method",
            "baseline",
            "best",
            "gain",
            "ari",
        ]:
            table.add_column(
                col,
                justify="right"
                if col in {"baseline", "best", "gain", "ari"}
                else "left",
            )
        for _, r in sub.iterrows():
            gain = r["gain"]
            gain_str = (
                f"[green]{gain:+.3f}[/]"
                if gain > 0
                else (f"[red]{gain:+.3f}[/]" if gain < 0 else f"{gain:+.3f}")
            )
            ari = "-" if pd.isna(r.get("ari")) else f"{r['ari']:.3f}"
            table.add_row(
                r["dataset"],
                r["dr_method"],
                r["cluster_method"],
                f"{r['baseline']:.3f}",
                f"{r['best']:.3f}",
                gain_str,
                ari,
            )
        console.print(table)

    tv = summary[summary["tunable"]]
    if not tv.empty:
        agg = tv.groupby(["track", "sampler"])["best"].mean().unstack()
        console.print("\n[bold]Mean best objective (TPE vs Random):[/]")
        console.print(agg.to_string())


def main() -> None:
    np.random.seed(SEED)
    console.rule("[bold]Hyperparameter-tuning effectiveness experiment")
    console.print(
        f"datasets={len(DATASETS_TO_RUN)}  DR={DR_METHODS}  clustering={CLUSTER_METHODS}  tracks={TRACKS}  trials/study={N_TRIALS}\n"
    )

    summary, trials = run_experiment()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    save_outputs(summary, trials, out_dir)
    make_plots(summary, trials, out_dir)

    console.print()
    print_summary(summary)
    console.print(f"\n[bold green]Done.[/] Results + plots in [underline]{out_dir}[/]")


if __name__ == "__main__":
    main()

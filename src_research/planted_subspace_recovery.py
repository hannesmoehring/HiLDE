"""RQ1-S experiment: planted-subspace recovery (thesis).

The clean synthetic counterpart to the benchmark H1b result in
``hierarchical_vs_flat.py``: that prior work showed the recursive hierarchy is *not* a
better clustering of benchmark class labels. This experiment plants the one structure
class labels cannot express - **nested, multi-scale subspace clusters that are globally
hidden but conditionally visible** - and asks whether recursion recovers the *fine*
sub-clusters better than a single flat clustering, and where the crossover is.

Hypotheses (design SS1):
    H2a - conditional recovery.  The hierarchy recovers the fine (level-2) sub-clusters better
        than flat full-space clustering, because recursion conditions on the coarse group
        before resolving fine structure. Headline metric: within-group ARI.
    H2b - it is the nesting that matters.  The advantage appears for nested multi-scale
        structure and *disappears* for non-nested (single-level) structure. A negative control
        on ourselves: if the hierarchy "wins everywhere" the win is a generator artefact.
    H2c - crossover.  A scale-separation knob ``rho`` controls a transition where flat goes
        from adequate to failing while the hierarchy stays adequate. We want the curve, not a
        single rigged point.

Four conditions, all sharing identical HDBSCAN hyperparameters (the isolated variable is
recursion / subspace scope, not "HDBSCAN vs anything else"):
    * flat_full          - one HDBSCAN run on the full standardised space (mirrors the tree's
                           optional UMAP pre-reduction via ``clustering_space``); the real competitor.
    * hierarchical       - leaf membership of ``start_evaluation``'s recursive tree; the method under test.
    * flat_oracle_B      - one HDBSCAN run on block B only (the planted fine subspace, global);
                           must fail when rotation diversity is high (subspace smearing).
    * oracle_conditional - HDBSCAN on block B *within each true coarse group*; the upper bound a
                           perfect level-1 split would give, bounding the hierarchy's headroom.

Metrics (design SS5): the prior lesson that ARI punishes over-segmentation is designed into the
metric set. We report homogeneity / completeness / V (a fragmenting hierarchy scores high h,
lower c) alongside ARI/NMI (read together, never alone), and lead with within-group ARI.

Noise rule: metrics vs ``y_fine`` are computed on each condition's own non-noise rows
(``label != -1``); ``noise_frac`` is reported separately so the exclusion is transparent.

Reproducibility: ``reduce_dimensionality`` threads ``config["*_random_state"]`` into UMAP /
t-SNE / MDS, so ``run_cell`` writes the cell's seed into those three keys as well as into the
generator. The ``SEEDS`` loop therefore captures both generator and embedding variance, and
every cell is reproducible - synthetic data is cheap, so we use many seeds for real CIs.

Outputs (written to ``outputs/experiments/<timestamp>/``):
    * subspace_recovery.csv     - one row per (rho, nesting, seed, condition).
    * subspace_summary.csv      - per (rho, nesting): median delta (hier-flat), win rate,
                                  Wilcoxon p, rank-biserial, oracle-relative recovery.
    * subspace_faithfulness.csv - per (rho, seed, region): local vs global fine-label kNN agreement.
    * plots/*.png               - crossover curve, homogeneity-vs-completeness, rotation diagnostic.

Run with::

    uv run python -m src_research.planted_subspace_recovery

``design SSN`` below marks a rule fixed by the pre-registered design, which is
recorded in the thesis and no longer kept in this repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_config
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.linalg import expm
from scipy.stats import wilcoxon
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.analysis.analysis_routine import _embed_original
from src.analysis.clustering import compute_clusters
from src.config_defaults import init_state
from src.evaluation.evaluate import start_evaluation
from src.types import Config

# Lift the small tree helpers from the sibling RQ1 harness rather than reimplement them
# (importing the module is side-effect-safe: it only runs the experiment under __main__).
from src_research.hierarchical_vs_flat import (
    _silence_noise,
    clustering_space,
    collect_leaves,
    leaf_partition,
    standardised_X,
)

# --------------------------------------------------------------------------- #
# CONFIG - edit this block to change the experiment.                          #
# --------------------------------------------------------------------------- #

SEED = (
    42  # seeds the global NumPy state (per-cell generators are seeded by the seed loop)
)
SEEDS = list(
    range(20)
)  # >=20 replicates for real CIs (design SS12.5); synthetic data is cheap
RHO_GRID = [
    1,
    2,
    4,
    8,
    16,
    32,
]  # primary scale-separation sweep, the crossover knob (SS12.4)
NESTINGS = [
    "nested",
    "non_nested",
]  # H2b control; non_nested is the negative control on ourselves
ADEQUACY = 0.7  # pre-registered within-group-ARI adequacy threshold; rho* = where flat drops below it
DR_METHOD = "UMAP"  # UI default = the headline DR method
HIER_LAYERS = (
    2  # tree depth: layer 1 = coarse split, layer 2 = fine split (nested needs 2)
)
RUN_FAITHFULNESS = True  # planted-label H1a add-on (design SS5/SS10)

# Generator params. Single values = headline; widen a list-free knob to run a SS8 robustness sweep.
G, K = 3, 4  # coarse groups, fine sub-clusters per group -> G*K planted fine labels
D_A, D_B, D_NOISE = 3, 3, 10  # block A (coarse), block B (fine), isotropic noise dims
N_PER = 60  # points per planted fine cluster
SIGMA_A, SIGMA_B, SIGMA_NOISE = 0.4, 0.4, 1.0  # within-cluster spreads
BASE_SEP = 3.0  # centroid magnitude in A and B at rho=1 (fixed, reported; sets |c|/sigma_B detectability)
ROTATION_DIVERSITY = 1.0  # spread of per-group block-B rotations R[g] (0 = shared subspace, larger = smeared)
ROTATE_BLOCKS = (
    True  # apply a global rotation Q so the planted blocks are not axis-aligned
)
MIN_CLUSTER_SIZE = (
    15  # HDBSCAN min_cluster_size, identical across ALL conditions (asserted)
)

PARALLEL_JOBS = -1  # grid cells run concurrently; 1 = serial, -1 = all cores
OUTPUT_ROOT = Path("outputs/experiments")

CONDITIONS = ["hierarchical", "flat_full", "flat_oracle_B", "oracle_conditional"]
RECOVERY_METRICS = [
    "h",
    "c",
    "v",
    "ari",
    "nmi",
    "within_g_ari",
    "coarse_ari",
    "noise_frac",
    "n_clusters",
    "frag_ratio",
]

_silence_noise()
console = Console()


# --------------------------------------------------------------------------- #
# Planted-subspace generator (design SS3).                                     #
# --------------------------------------------------------------------------- #


def _unit_dirs(rng: np.random.Generator, m: int, d: int) -> np.ndarray:
    """``m`` well-separated unit direction vectors in R^d (rows). Orthonormal when m <= d
    (maximal separation); random unit vectors otherwise."""
    if m <= d:
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return q[:, :m].T
    v = rng.standard_normal((m, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _rotation(rng: np.random.Generator, d: int, diversity: float) -> np.ndarray:
    """A random rotation matrix whose angle scales with ``diversity`` (0 -> identity).
    ``expm`` of a scaled random skew-symmetric matrix is orthonormal with det +1."""
    if diversity <= 0.0:
        return np.eye(d)
    a = rng.standard_normal((d, d))
    return expm(diversity * (a - a.T))


def make_nested_subspace(
    rho: float,
    nesting: str,
    seed: int,
    g_groups: int = G,
    k_sub: int = K,
    d_a: int = D_A,
    d_b: int = D_B,
    d_noise: int = D_NOISE,
    n_per: int = N_PER,
    sigma_a: float = SIGMA_A,
    sigma_b: float = SIGMA_B,
    sigma_noise: float = SIGMA_NOISE,
    base_sep: float = BASE_SEP,
    rotation_diversity: float = ROTATION_DIVERSITY,
    rotate_blocks: bool = ROTATE_BLOCKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate nested (or non-nested control) subspace structure.

    Returns ``(X, y_fine, y_coarse, B)``:
      * ``X``       - the feature matrix all whole-space conditions cluster (optionally rotated by Q).
      * ``y_fine``  - planted sub-cluster id in 0..G*K-1 (the primary target).
      * ``y_coarse``- planted group id = ``y_fine // K`` (used for within-group scoring).
      * ``B``       - the *unrotated* block-B submatrix (the planted fine subspace). A global Q
                      preserves Euclidean distances so it leaves flat/hierarchical clustering
                      invariant; only the column-selecting oracles depend on which columns are
                      block B, so they consume ``B`` directly (the fair reading of "block B only").

    nested (primary): block A separates the G coarse groups (scaled by rho); within a group the
        K sub-clusters are separated in block B along a *per-group rotated* direction, so the
        block-B marginal pooled over groups is unimodal - globally hidden, locally clean.
    non_nested (control): rho is forced to 1 and every one of the G*K clusters gets its own
        block-A centroid, so a single flat pass resolves them all -> the hierarchy should not help.
    """
    rng = np.random.default_rng(seed)
    n_clusters = g_groups * k_sub

    mu_a_group = base_sep * _unit_dirs(
        rng, g_groups, d_a
    )  # G coarse centroids (nested)
    mu_a_full = base_sep * _unit_dirs(
        rng, n_clusters, d_a
    )  # G*K centroids (non_nested)
    c_fine = base_sep * _unit_dirs(
        rng, k_sub, d_b
    )  # K base fine centroids, shared before rotation
    rot = [
        np.eye(d_b) if g == 0 else _rotation(rng, d_b, rotation_diversity)
        for g in range(g_groups)
    ]

    xa_parts, xb_parts, y_fine, y_coarse = [], [], [], []
    for g in range(g_groups):
        for k in range(k_sub):
            cid = g * k_sub + k
            if nesting == "nested":
                centre_a = rho * mu_a_group[g]
                centre_b = rot[g] @ c_fine[k]
            else:  # non_nested: rho=1 enforced by the driver; every cluster its own A-centroid
                centre_a = mu_a_full[cid]
                centre_b = rot[g] @ c_fine[k]
            xa_parts.append(centre_a + sigma_a * rng.standard_normal((n_per, d_a)))
            xb_parts.append(centre_b + sigma_b * rng.standard_normal((n_per, d_b)))
            y_fine += [cid] * n_per
            y_coarse += [g] * n_per

    x_a = np.vstack(xa_parts)
    x_b = np.vstack(xb_parts)
    x_noise = sigma_noise * rng.standard_normal((len(y_fine), d_noise))
    x_design = np.hstack([x_a, x_b, x_noise])

    block_b = x_b.copy()  # unrotated block B for the oracles
    if rotate_blocks:
        q = _rotation(rng, x_design.shape[1], 1.0)
        x = x_design @ q.T
    else:
        x = x_design
    return x, np.asarray(y_fine), np.asarray(y_coarse), block_b


def _make_df(x: np.ndarray, y_fine: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Wrap ``X`` as the one-hot app DataFrame ``start_evaluation`` expects (f-columns +
    ``target_*`` from ``y_fine`` + ``row_id``), mirroring ``src.datasets._one_hot_df``."""
    feature_cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_cols).astype(np.float64)
    for cid in sorted(set(y_fine.tolist())):
        df[f"target_c{cid}"] = y_fine == cid
    df["row_id"] = np.arange(len(df))
    return df, feature_cols


# --------------------------------------------------------------------------- #
# Conditions (design SS4).                                                     #
# --------------------------------------------------------------------------- #


def within_group_hdbscan(
    block_b: np.ndarray, y_coarse: np.ndarray, cfg: Config
) -> np.ndarray:
    """Oracle-conditional: HDBSCAN on block B *within each true coarse group*, relabelled to
    globally-unique cluster ids. The upper bound a perfect level-1 split would hand the method."""
    part = np.full(len(block_b), -1, dtype=int)
    nxt = 0
    for g in np.unique(y_coarse):
        idx = np.where(y_coarse == g)[0]
        labels = np.asarray(
            compute_clusters(block_b[idx], method="HDBSCAN", config=cfg)[0]
        )
        for lab in np.unique(labels):
            if lab == -1:
                continue
            part[idx[labels == lab]] = nxt
            nxt += 1
    return part


# --------------------------------------------------------------------------- #
# Metrics (design SS5).                                                        #
# --------------------------------------------------------------------------- #


def _to_modal(part: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map each predicted cluster to the modal value of ``y`` among its members (noise -> -1)."""
    out = np.full_like(part, -1)
    for cluster in np.unique(part):
        if cluster == -1:
            continue
        mask = part == cluster
        vals, counts = np.unique(y[mask], return_counts=True)
        out[mask] = vals[counts.argmax()]
    return out


def mean_within_group_ari(
    part: np.ndarray, y_fine: np.ndarray, y_coarse: np.ndarray
) -> float:
    """Headline metric: for each true coarse group, ARI of predicted vs ``y_fine`` restricted to
    that group's (non-noise) rows, averaged over groups. Isolates the level-2 split both methods
    do NOT get for free from the level-1 split they both do."""
    vals = []
    for g in np.unique(y_coarse):
        mask = (y_coarse == g) & (part != -1)
        if mask.sum() < 2 or len(np.unique(y_fine[mask])) < 2:
            continue
        vals.append(adjusted_rand_score(y_fine[mask], part[mask]))
    return float(np.mean(vals)) if vals else float("nan")


def score_condition(
    part: np.ndarray, y_fine: np.ndarray, y_coarse: np.ndarray, n_planted: int
) -> dict:
    """Full metric vector for one condition's partition (design SS5). Fine-label metrics use this
    condition's own non-noise rows; ``noise_frac`` is reported separately."""
    keep = part != -1
    h = c = v = ari = nmi = float("nan")
    if keep.sum() >= 2 and len(np.unique(y_fine[keep])) >= 1:
        h, c, v = homogeneity_completeness_v_measure(y_fine[keep], part[keep])
        ari = float(adjusted_rand_score(y_fine[keep], part[keep]))
        nmi = float(normalized_mutual_info_score(y_fine[keep], part[keep]))
    n_clusters = len(set(part[keep].tolist()))
    coarse = float("nan")
    if keep.sum() >= 2:
        coarse = float(
            adjusted_rand_score(y_coarse[keep], _to_modal(part, y_coarse)[keep])
        )
    return {
        "h": float(h),
        "c": float(c),
        "v": float(v),
        "ari": ari,
        "nmi": nmi,
        "within_g_ari": mean_within_group_ari(part, y_fine, y_coarse),
        "coarse_ari": coarse,
        "noise_frac": float(np.mean(part == -1)),
        "n_clusters": n_clusters,
        "frag_ratio": n_clusters / n_planted,
    }


def label_knn_agreement(emb: np.ndarray, labels: np.ndarray) -> float | None:
    """Fraction of each point's k nearest neighbours (in the 2D embedding) sharing its label,
    averaged over points. Neighbourhood size mirrors ``_score_node`` (k = min(20, (n-1)//2),
    guarded at n >= 10). The planted-label H1a faithfulness statistic."""
    n = len(labels)
    if n < 10:
        return None
    k = min(20, (n - 1) // 2)
    if k < 1:
        return None
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]  # drop self
    return float(np.mean(labels[idx] == labels[:, None]))


# --------------------------------------------------------------------------- #
# Running one grid cell (rho x nesting x seed).                                #
# --------------------------------------------------------------------------- #


def run_cell(rho: float, nesting: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Generate the planted data once, run all four conditions + metrics (+ the faithfulness
    add-on when enabled). Returns (recovery_rows, faithfulness_rows)."""
    _silence_noise()  # re-assert before init_state touches streamlit in a worker process
    cfg: Config = init_state(init_streamlit=False)
    cfg["method"] = DR_METHOD
    cfg["hierarchical_layers"] = HIER_LAYERS
    cfg["hclust_min_cluster_size"] = MIN_CLUSTER_SIZE
    assert (
        cfg["hclust_min_cluster_size"] == MIN_CLUSTER_SIZE
    )  # identical HDBSCAN size across conditions
    # The cell's seed must reach the reducers too, not just the generator: `init_state`
    # hardcodes 42 for all three, so the embedding half of every replicate would be constant.
    cfg["umap_random_state"] = cfg["tsne_random_state"] = cfg["mds_random_state"] = seed

    eff_rho = (
        rho if nesting == "nested" else 1.0
    )  # non_nested control forces rho=1 (design SS3)
    x, y_fine, y_coarse, block_b = make_nested_subspace(eff_rho, nesting, seed)
    n_planted = G * K

    df, feature_cols = _make_df(x, y_fine)
    tree = start_evaluation(df, feature_cols, cfg)
    x_all = standardised_X(df, feature_cols, tree)
    n = len(x_all)

    block_b_std = StandardScaler().fit_transform(block_b)
    parts = {
        "hierarchical": leaf_partition(collect_leaves(tree), n),
        "flat_full": np.asarray(
            compute_clusters(
                clustering_space(x_all, cfg), method="HDBSCAN", config=cfg
            )[0]
        ),
        "flat_oracle_B": np.asarray(
            compute_clusters(block_b_std, method="HDBSCAN", config=cfg)[0]
        ),
        "oracle_conditional": within_group_hdbscan(block_b_std, y_coarse, cfg),
    }

    base = {
        "rho": rho,
        "nesting": nesting,
        "seed": seed,
        "rotation_diversity": ROTATION_DIVERSITY,
        "d_noise": D_NOISE,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "dr_method": DR_METHOD,
    }
    rec = [
        {
            **base,
            "condition": name,
            **score_condition(part, y_fine, y_coarse, n_planted),
        }
        for name, part in parts.items()
    ]

    faith: list[dict] = []
    if RUN_FAITHFULNESS:
        # `_embed_original` returns None for a region it could not project, rather than a
        # fabricated all-zeros embedding — whose kNN agreement is meaningless but finite.
        # An unprojected arm is recorded as None and flagged, never scored and never fatal.
        e_global, _ = _embed_original(x_all, cfg)
        for g in np.unique(y_coarse):
            idx = np.where(y_coarse == g)[0]
            e_local, _ = _embed_original(x_all[idx], cfg)
            faith.append(
                {
                    "rho": rho,
                    "nesting": nesting,
                    "seed": seed,
                    "region": int(g),
                    "local": label_knn_agreement(e_local, y_fine[idx])
                    if e_local is not None
                    else None,
                    "global": label_knn_agreement(e_global[idx], y_fine[idx])
                    if e_global is not None
                    else None,
                    "unprojected": bool(e_local is None or e_global is None),
                }
            )
    return rec, faith


# --------------------------------------------------------------------------- #
# Grid driver.                                                                 #
# --------------------------------------------------------------------------- #


def build_cells() -> list[tuple[float, str, int]]:
    """Enumerate (rho, nesting, seed) grid cells - all independent.

    ``run_cell`` forces ``eff_rho = 1`` for the non_nested control (design SS3), so
    enumerating that arm across the whole RHO_GRID recomputes one condition len(RHO_GRID)
    times. The duplicates are not free: H2b then averages len(RHO_GRID) copies of a single
    control condition against that many *distinct* nested conditions. The control is run at
    the first rho only.
    """
    cells: list[tuple[float, str, int]] = []
    for nesting in NESTINGS:
        rhos = RHO_GRID if nesting == "nested" else RHO_GRID[:1]
        cells.extend((rho, nesting, seed) for rho in rhos for seed in SEEDS)
    return cells


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full grid, returning (recovery_df, faithfulness_df)."""
    cells = build_cells()
    rec_rows: list[dict] = []
    faith_rows: list[dict] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task(
            f"Running grid (jobs={PARALLEL_JOBS})", total=len(cells)
        )
        if PARALLEL_JOBS == 1:
            for rho, nesting, seed in cells:
                progress.update(
                    task, description=f"rho={rho} / {nesting} / s{seed}"[:48]
                )
                r, f = run_cell(rho, nesting, seed)
                rec_rows.extend(r)
                faith_rows.extend(f)
                progress.advance(task)
        else:
            # Cells are independent -> worker processes. inner_max_num_threads=1 stops each
            # worker's UMAP/BLAS pools oversubscribing the cores. Generator keeps the bar live.
            jobs = (
                delayed(run_cell)(rho, nesting, seed) for rho, nesting, seed in cells
            )
            with parallel_config(backend="loky", inner_max_num_threads=1):
                for r, f in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
                    rec_rows.extend(r)
                    faith_rows.extend(f)
                    progress.advance(task)

    return pd.DataFrame(rec_rows), pd.DataFrame(faith_rows)


# --------------------------------------------------------------------------- #
# Statistics (design SS9).                                                     #
# --------------------------------------------------------------------------- #


def drop_duplicate_control_cells(rec: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse the non_nested control to the one rho it actually realises.

    ``run_cell`` forces ``eff_rho = 1`` for that arm, so its RHO_GRID levels are repeats of
    a single condition. ``build_cells`` no longer generates them, but a recovery CSV from
    before that change still carries them, and any aggregate over ``nesting`` would weight
    the control by len(RHO_GRID). Returns (frame, rows dropped) so the caller can say so.
    """
    ctrl = rec["nesting"] == "non_nested"
    if not ctrl.any():
        return rec, 0
    keep_rho = rec.loc[ctrl, "rho"].min()
    dup = ctrl & (rec["rho"] != keep_rho)
    return rec.loc[~dup].copy(), int(dup.sum())


def summarise(rec: pd.DataFrame) -> pd.DataFrame:
    """Per (rho, nesting): paired hier-vs-flat within-group ARI delta across seeds (median,
    win rate, Wilcoxon p, rank-biserial) plus oracle-relative recovery. The unit is a seed.

    Every aggregate in a row comes from the *same* dropna'd paired sample. They did not:
    hier_mean / flat_mean / oracle came from the undropped pivot while median_delta came
    from the pairs, so a seed whose hierarchical arm failed to score still moved the means -
    and those drops correlate with the condition being hard, i.e. exactly with the rows the
    means are used to compare. ``n_seeds`` vs ``n_pairs`` exposes the attrition.
    """
    rec, n_dropped = drop_duplicate_control_cells(rec)
    if n_dropped:
        console.print(
            f"[yellow]Dropped {n_dropped} duplicate non_nested rows[/] (eff_rho is forced to 1; only one rho level is a distinct control)."
        )
    rows: list[dict] = []
    for (rho, nesting), sub in rec.groupby(["rho", "nesting"]):
        piv = sub.pivot_table(index="seed", columns="condition", values="within_g_ari")
        if "hierarchical" not in piv or "flat_full" not in piv:
            continue
        pair = piv[["hierarchical", "flat_full"]].dropna()
        if pair.empty:
            continue
        d = (pair["hierarchical"] - pair["flat_full"]).to_numpy()
        nz = d[d != 0]
        p = (
            float(wilcoxon(nz).pvalue)
            if nz.size >= 1 and not np.allclose(d, 0)
            else float("nan")
        )
        rbc = (
            float((np.sum(d > 0) - np.sum(d < 0)) / nz.size)
            if nz.size
            else float("nan")
        )
        oracle = (
            piv["oracle_conditional"].reindex(pair.index)
            if "oracle_conditional" in piv
            else pd.Series(dtype=float)
        )
        orc = float(oracle.mean()) if oracle.notna().any() else float("nan")
        hier_mean = float(pair["hierarchical"].mean())
        rows.append(
            {
                "rho": rho,
                "nesting": nesting,
                "metric": "within_g_ari",
                "n_seeds": int(piv.shape[0]),
                "n_pairs": int(d.size),
                "n_oracle": int(oracle.notna().sum()),
                "hier_mean": hier_mean,
                "flat_mean": float(pair["flat_full"].mean()),
                "median_delta": float(np.median(d)),
                "win_rate": float(np.mean(d > 0)),
                "wilcoxon_p": p,
                "rank_biserial": rbc,
                "oracle_relative": hier_mean / orc
                if orc and np.isfinite(orc) and orc != 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def crossover_rho(rec: pd.DataFrame) -> float | None:
    """Smallest rho where the nested flat_full mean within-group ARI drops below ADEQUACY."""
    nested = rec[(rec["nesting"] == "nested") & (rec["condition"] == "flat_full")]
    means = nested.groupby("rho")["within_g_ari"].mean().sort_index()
    below = means[means < ADEQUACY]
    return float(below.index[0]) if not below.empty else None


# --------------------------------------------------------------------------- #
# Output: CSVs, plots, console tables.                                         #
# --------------------------------------------------------------------------- #


def save_outputs(
    rec: pd.DataFrame, faith: pd.DataFrame, summary: pd.DataFrame, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.to_csv(out_dir / "subspace_recovery.csv", index=False)
    summary.to_csv(out_dir / "subspace_summary.csv", index=False)
    faith.to_csv(out_dir / "subspace_faithfulness.csv", index=False)


def make_plots(rec: pd.DataFrame, out_dir: Path) -> None:
    """Thesis figures: crossover curve, homogeneity-vs-completeness, rotation diagnostic."""
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # fig:rq1s-crossover - within-group ARI vs rho, four conditions (nested), mean +- CI over seeds.
    nested = rec[rec["nesting"] == "nested"]
    if not nested.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=nested,
            x="rho",
            y="within_g_ari",
            hue="condition",
            marker="o",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.axhline(
            ADEQUACY, color="grey", lw=0.8, ls="--", label=f"adequacy={ADEQUACY}"
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rho (coarse-vs-fine scale separation)")
        ax.set_ylabel("within-group ARI")
        ax.set_title(
            "RQ1-S crossover: fine-structure recovery vs scale separation (nested)"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "rq1s_crossover.png", dpi=150)
        plt.close(fig)

    # fig:rq1s-homcomp - homogeneity vs completeness scatter (the over-segmentation signature).
    if not nested.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=nested,
            x="h",
            y="c",
            hue="condition",
            style="condition",
            s=40,
            alpha=0.7,
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
        ax.set_xlabel("homogeneity (leaves pure)")
        ax.set_ylabel("completeness (cluster not split)")
        ax.set_title(
            "Over-segmentation signature: high h, lower c => found but split (nested)"
        )
        fig.tight_layout()
        fig.savefig(plots / "rq1s_homcomp.png", dpi=150)
        plt.close(fig)

    # fig:rq1s-rotation - subspace-smearing diagnostic. Only meaningful with a rotation sweep;
    # with a single ROTATION_DIVERSITY we cannot draw the curve, so skip cleanly (design SS10).
    if rec["rotation_diversity"].nunique() > 1:
        diag = rec[
            (rec["nesting"] == "nested")
            & (rec["condition"].isin(["hierarchical", "flat_oracle_B"]))
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=diag,
            x="rotation_diversity",
            y="within_g_ari",
            hue="condition",
            marker="o",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_xlabel("rotation diversity (subspace smearing)")
        ax.set_ylabel("within-group ARI")
        ax.set_title("Subspace smearing: flat_oracle_B degrades, hierarchical holds")
        fig.tight_layout()
        fig.savefig(plots / "rq1s_rotation.png", dpi=150)
        plt.close(fig)
    else:
        console.print(
            "[dim]Skipped rotation diagnostic: single ROTATION_DIVERSITY (no sweep).[/]"
        )


def print_summary(rec: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Render the crossover table (per rho, nested) and the H2b nested-vs-non_nested gap."""
    nested = summary[summary["nesting"] == "nested"].sort_values("rho")
    if not nested.empty:
        table = Table(
            title="RQ1-S crossover (nested) - within-group ARI, delta = hierarchical - flat_full"
        )
        for col in [
            "rho",
            "hier_mean",
            "flat_mean",
            "median_delta",
            "win_rate",
            "wilcoxon_p",
            "oracle_relative",
        ]:
            table.add_column(col, justify="right")
        for _, r in nested.iterrows():
            md = r["median_delta"]
            md_str = (
                f"[green]{md:+.3f}[/]"
                if md > 0
                else (f"[red]{md:+.3f}[/]" if md < 0 else f"{md:+.3f}")
            )
            p = r["wilcoxon_p"]
            table.add_row(
                f"{r['rho']:g}",
                f"{r['hier_mean']:.3f}",
                f"{r['flat_mean']:.3f}",
                md_str,
                f"{r['win_rate']:.2f}",
                "-" if pd.isna(p) else f"{p:.3g}",
                "-" if pd.isna(r["oracle_relative"]) else f"{r['oracle_relative']:.2f}",
            )
        console.print(table)

    rho_star = crossover_rho(rec)
    console.print(
        f"\nEstimated crossover rho* (flat within-group ARI < {ADEQUACY}): [bold]{rho_star if rho_star else 'none in grid'}[/]"
    )

    # H2b: the hier - flat gap should be clearly positive for nested and ~zero for non_nested.
    if not summary.empty:
        gap = (
            summary.groupby("nesting")["median_delta"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        table = Table(
            title="H2b - hier vs flat within-group ARI gap (should be +ve nested, ~0 non_nested)"
        )
        # n_rho makes the asymmetry explicit: the nested arm spans the rho sweep, the control
        # is a single condition (eff_rho is forced to 1), not len(RHO_GRID) copies of one.
        for col in ["nesting", "mean_gap", "median_gap", "n_rho"]:
            table.add_column(col, justify="right" if col != "nesting" else "left")
        for _, r in gap.iterrows():
            table.add_row(
                r["nesting"],
                f"{r['mean']:+.3f}",
                f"{r['median']:+.3f}",
                str(int(r["count"])),
            )
        console.print(table)


def main() -> None:
    np.random.seed(SEED)
    console.rule("[bold]RQ1-S: planted-subspace recovery")
    console.print(
        f"rho={RHO_GRID}  nestings={NESTINGS}  seeds={len(SEEDS)}  DR={DR_METHOD}  "
        f"G,K={G},{K}  d_A/d_B/d_noise={D_A}/{D_B}/{D_NOISE}  faithfulness={RUN_FAITHFULNESS}\n"
    )

    rec, faith = run_experiment()
    n_unprojected = (
        int(faith["unprojected"].sum())
        if not faith.empty and "unprojected" in faith
        else 0
    )
    if n_unprojected:
        console.print(
            f"[bold yellow]Unprojected faithfulness regions: {n_unprojected}[/] of {len(faith)} — reported as null, see subspace_faithfulness.csv"
        )
    summary = summarise(rec)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    save_outputs(rec, faith, summary, out_dir)
    make_plots(rec, out_dir)

    console.print()
    print_summary(rec, summary)
    console.print(f"\n[bold green]Done.[/] Results + plots in [underline]{out_dir}[/]")


if __name__ == "__main__":
    main()

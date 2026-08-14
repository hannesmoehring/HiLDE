"""RQ2 experiment: predicate stability under relaxation (thesis).

Implements the design in ``src_research/EXPERIMENT_predicate_stability.md``. It tests H2:
as the relaxation threshold ``t`` is lowered from 1.0, the predicate describing a selection
becomes more stable under small perturbations of that selection, while its specificity (F1
against the unperturbed selection) degrades only gradually - so a favourable operating
point exists. Decomposed:

    H2a - stability gain.       Mean pairwise Jaccard of the admitted sets across perturbed
        re-selections rises as t drops through {0.95, 0.9, 0.8}; F1 variance falls.
    H2b - graceful cost.        Median F1 vs the base selection stays above the pre-specified
        floor (0.9 x strict) for at least one t < 1.0 (the joint operating-point criterion
        lives in ``predicate_stability_analysis``).
    H2c - severity split ablation.  The severity-proportional tail split
        (``_tail_removal_shares``) retains higher F1 than a naive 50/50 split at matched t
        on *skewed* within-cluster data, and is indistinguishable on symmetric data
        (negative control on ourselves).

Selections come from a simulated user: hierarchy leaves on wine (the tree is rebuilt per
seed, with the seed threaded into UMAP's ``random_state``, so seeds are distinct-but-
reproducible tree-rebuild replicates) and planted clusters on synthetic data
(no tree needed - keeps recovery scoring independent of tree quality). Perturbation is
boundary jitter: drop a fraction delta of the selection, add the same number from the
K_NN-nearest non-selected neighbours of its members in the standardised full space.

Axis-parallel generator (the one new generator, design SS3): C clusters in a shared
D = C*r + d_noise feature space; each cluster owns r relevant dims (disjoint across
clusters - "sampled without replacement" - so on any cluster's relevant dim every other
point is N(0,1) background and the margin knob has clean semantics). Deliberately
axis-aligned: range predicates can express these boxes, the mirror image of the RQ1
generator being fair to *that* method (stated threat, design SS8). Resolved knobs:
    * skew: within-cluster distribution on relevant dims - gaussian (symmetric control),
      standardised lognormal with the heavy tail pointing TOWARD the background (the
      friendly case for severity-splitting, per the design's own threat note), and an
      exploratory adversarial ``bimodal`` shape (minor mode toward the background) where
      the median-based severity heuristic is misleading (design SS8/SS9.3).
    * margin: cluster-centre offset from the N(0,1) background along relevant dims -
      wide=4.0 (clean separation) vs tight=1.5 (real contest at the box edge).

Stability metrics are aggregated to ONE number per (selection, method, split, t, delta)
before any test - the unit of analysis is the selection, never the replicate pair
(the SS6.6 pseudo-replication lesson, built in).

Outputs (written to ``outputs/experiments/<timestamp>/``):
    * stability_records.csv - one row per (arm, seed, selection, method, split, t, delta).
    * recovery.csv          - synthetic arms: relevant-dim P/R/F1 + membership F1 per t
                              (fills thesis table ``tab:synthetic-recovery``).
    * seed_stability.csv    - pass (b): matched-leaf predicate agreement across tree
                              rebuilds (wine); the leaf-matching confound is reported, not
                              hidden (match_jaccard / weak_match columns).
    * stability_summary.csv, h2c_summary.csv, verdicts.csv, plots/*.png - via
      ``predicate_stability_analysis`` (re-runnable standalone on this output dir).

Run with::

    uv run python -m src_research.predicate_stability
"""

from __future__ import annotations

import zlib
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.analysis.analysis_routine import compute_analysis_tree
from src.analysis.predicate_generator import _f1, generate_predicate
from src.config_defaults import init_state
from src.types import Config
from src_research import predicate_stability_analysis as psa

# Lift the shared helpers from the sibling RQ1 harness rather than reimplement them
# (importing the module is side-effect-safe: it only runs the experiment under __main__).
from src_research.hierarchical_vs_flat import (
    _silence_noise,
    collect_leaves,
    prepare_dataset,
    standardised_X,
)

# --------------------------------------------------------------------------- #
# CONFIG - edit this block to change the experiment.                          #
# --------------------------------------------------------------------------- #

SEED = 42  # root entropy for the per-selection perturbation RNGs
SEEDS_WINE = list(
    range(5)
)  # tree-rebuild replicates, threaded into the DR seed (tree builds dominate cost)
SEEDS_SYNTH = list(
    range(20)
)  # generator replicates (cheap; matches the RQ1-S standard)

T_GRID = [
    1.0,
    0.95,
    0.9,
    0.8,
]  # pre-specified in the thesis - do not add levels after seeing results
METHODS = [
    "threshold",
    "db",
]  # threshold = primary; db = DimBridge-style baseline (secondary)
SPLITS = ["severity", "symmetric"]  # H2c ablation arms of the tail split
DELTAS = [0.1, 0.05, 0.2]  # perturbation strength; 0.1 = pre-registered headline
M_REPLICATES = 20  # perturbed re-selections per (selection, delta)
K_NN = 10  # neighbours per member forming the boundary-jitter candidate pool
MIN_SEL = 20  # quantiles on fewer points are noise (design SS4); sizes are reported

WINE_DATASET = "Wine quality (Low)"
DR_METHOD = "UMAP"  # UI default; drives the per-seed tree builds on wine
HIER_LAYERS = 2  # matches the RQ1 headline depth

# Axis-parallel generator, fixed headline (design SS3): D = C*r + d_noise = 27 features.
C_CLUSTERS = 4
R_REL = 3  # relevant dims per cluster
D_NOISE = 15  # dims that are noise for every cluster
N_PER = 60  # points per planted cluster (= selection size)
SIGMA_REL = 0.5  # within-cluster spread on relevant dims
SKEW_SIGMA = 0.75  # lognormal shape (moderate skew ~2.9)
BIMODAL_WEIGHT, BIMODAL_SHIFT, BIMODAL_SIGMA = (
    0.15,
    3.0,
    0.35,
)  # adversarial shape params

SYNTH_ARMS: dict[str, dict] = {
    "synth-sym-wide": {"skew": "gaussian", "margin": 4.0},
    "synth-sym-tight": {"skew": "gaussian", "margin": 1.5},
    "synth-skew-wide": {"skew": "lognormal", "margin": 4.0},
    "synth-skew-tight": {"skew": "lognormal", "margin": 1.5},
    "synth-bimodal-tight": {"skew": "bimodal", "margin": 1.5},  # exploratory (SS9.3)
}

RUN_SEED_PERTURBATION = (
    True  # pass (b), wine only - first thing cut under time pressure (SS9)
)
PARALLEL_JOBS = -1  # grid cells run concurrently; 1 = serial, -1 = all cores
OUTPUT_ROOT = Path("outputs/experiments")

_silence_noise()
console = Console()


# --------------------------------------------------------------------------- #
# Axis-parallel planted generator (design SS3).                                #
# --------------------------------------------------------------------------- #


def _within_cluster(rng: np.random.Generator, n: int, skew: str) -> np.ndarray:
    """Zero-mean, unit-variance within-cluster samples for a relevant dim. Skewed shapes
    put their heavy tail on the NEGATIVE side, i.e. toward the N(0,1) background sitting
    below the +margin cluster centre - the contested region severity-splitting should trim."""
    if skew == "gaussian":
        return rng.standard_normal(n)
    if skew == "lognormal":
        v = np.exp(SKEW_SIGMA * rng.standard_normal(n))
        mean = np.exp(SKEW_SIGMA**2 / 2)
        sd = np.sqrt((np.exp(SKEW_SIGMA**2) - 1) * np.exp(SKEW_SIGMA**2))
        return -(v - mean) / sd
    if skew == "bimodal":  # dense major mode + small far mode toward the background:
        # the median sits in the major mode, so the distance-to-extreme severity heuristic
        # trims straight into the minor mode's real structure (the adversarial case).
        minor = rng.random(n) < BIMODAL_WEIGHT
        v = BIMODAL_SIGMA * rng.standard_normal(n) - BIMODAL_SHIFT * minor
        mean = -BIMODAL_WEIGHT * BIMODAL_SHIFT
        var = (
            BIMODAL_SIGMA**2 + BIMODAL_WEIGHT * (1 - BIMODAL_WEIGHT) * BIMODAL_SHIFT**2
        )
        return (v - mean) / np.sqrt(var)
    raise ValueError(f"Unknown skew shape: {skew}")


def make_axis_parallel(
    skew: str, margin: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-level axis-aligned planted clusters (no nesting - RQ2 needs recoverable boxes).

    Returns ``(X, y, rel)``: the raw feature matrix, planted membership (0..C-1), and the
    (C, r) relevant-dimension sets. Every cluster's r dims are disjoint from every other's
    (one global permutation), so along cluster c's relevant dims all non-members are exactly
    N(0,1) background at distance ``margin`` from the cluster centre."""
    rng = np.random.default_rng(seed)
    d_total = C_CLUSTERS * R_REL + D_NOISE
    rel = rng.permutation(d_total)[: C_CLUSTERS * R_REL].reshape(C_CLUSTERS, R_REL)
    X = rng.standard_normal((C_CLUSTERS * N_PER, d_total))
    y = np.repeat(np.arange(C_CLUSTERS), N_PER)
    for c in range(C_CLUSTERS):
        block = slice(c * N_PER, (c + 1) * N_PER)
        for j in rel[c]:
            X[block, j] = margin + SIGMA_REL * _within_cluster(rng, N_PER, skew)
    return X, y, rel


# --------------------------------------------------------------------------- #
# Simulated-user perturbation + predicate helpers (design SS2/SS4).            #
# --------------------------------------------------------------------------- #


def knn_pool(X_all: np.ndarray, sel_idx: np.ndarray) -> np.ndarray:
    """Boundary-jitter candidates: union of the K_NN nearest NON-selected neighbours of the
    selection's members in the standardised full space. Computed once per selection and
    shared by every delta so replicates differ only in how much they swap."""
    inside = np.zeros(len(X_all), dtype=bool)
    inside[sel_idx] = True
    outside = np.where(~inside)[0]
    k = min(K_NN, len(outside))
    nn = NearestNeighbors(n_neighbors=k).fit(X_all[outside])
    neigh = nn.kneighbors(X_all[sel_idx], return_distance=False)
    return np.unique(outside[neigh.ravel()])


def perturb(
    sel_idx: np.ndarray, pool: np.ndarray, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """One perturbed re-selection: drop round(delta*n) members uniformly at random, add the
    same number from the kNN boundary pool (mimics a user lassoing slightly differently -
    moves the boundary, not the core)."""
    n = len(sel_idx)
    n_swap = max(1, round(delta * n))
    keep = rng.choice(n, size=n - n_swap, replace=False)
    added = rng.choice(pool, size=min(n_swap, len(pool)), replace=False)
    return np.concatenate([sel_idx[keep], added])


def build_predicate(
    method: str,
    split: str,
    t: float,
    idx: np.ndarray,
    X_all: np.ndarray,
    feature_cols: list[str],
) -> list[dict]:
    """One calc-layer predicate for the rows ``idx``: background = the shared standardised
    matrix, selection = those rows of it (the app's global predicate scope)."""
    sel_df = pd.DataFrame(X_all[idx], columns=feature_cols)
    return generate_predicate(  # type: ignore[return-value]
        method,
        sel_df,
        X_all,
        threshold=t,
        selected_indices=idx.tolist() if method == "db" else None,
        tail_split=split,
    )


def admitted_mask(
    rows: list[dict], X_all: np.ndarray, feature_index: dict[str, int]
) -> np.ndarray:
    """Membership of every dataset row under the predicate: AND over interval clauses.
    Threshold rows all apply (the conjunction over all features); db rows apply only when
    the greedy step selected them (``in_predicate``) - a clause-less db predicate is the
    empty conjunction and admits everything."""
    mask = np.ones(len(X_all), dtype=bool)
    for row in rows:
        if row.get("in_predicate") is False:
            continue
        j = feature_index[str(row["feature"])]
        mask &= (X_all[:, j] >= row["sel_min"]) & (X_all[:, j] <= row["sel_max"])
    return mask


# --------------------------------------------------------------------------- #
# Per-record metrics (design SS5).                                             #
# --------------------------------------------------------------------------- #


def mean_pairwise_mask_jaccard(masks: np.ndarray) -> float:
    """Mean pairwise Jaccard of the m admitted sets (boolean (m, n) matrix). Two empty
    sets count as identical (Jaccard 1)."""
    m = masks.astype(np.int32)
    inter = m @ m.T
    sizes = m.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    iu = np.triu_indices(len(masks), k=1)
    inter_u = inter[iu].astype(float)
    union_u = union[iu].astype(float)
    vals = np.where(union_u > 0, inter_u / np.maximum(union_u, 1.0), 1.0)
    return float(vals.mean())


def mean_pairwise_clause_jaccard(rep_rows: list[list[dict]]) -> float:
    """db only: mean pairwise Jaccard of the greedily selected clause sets - the metric the
    greedy step puts genuinely at risk (one moved point can reroute the greedy path)."""
    sets = [
        frozenset(i for i, r in enumerate(rows) if r["in_predicate"])
        for rows in rep_rows
    ]
    vals = [
        len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in combinations(sets, 2)
    ]
    return float(np.mean(vals))


def bound_sd(rep_rows: list[list[dict]]) -> float:
    """Mean per-feature sd of (sel_min, sel_max) across replicates - the diagnostic for
    *why* the admitted set moves. Rows are in feature order for both methods."""
    mins = np.array([[r["sel_min"] for r in rows] for rows in rep_rows])
    maxs = np.array([[r["sel_max"] for r in rows] for rows in rep_rows])
    return float(
        (mins.std(axis=0, ddof=1).mean() + maxs.std(axis=0, ddof=1).mean()) / 2
    )


def dim_recovery(
    rows: list[dict], rel_dims: np.ndarray, method: str, feature_index: dict[str, int]
) -> dict:
    """Relevant-dimension recovery (synthetic only, design SS2): db reads its clause set
    directly; threshold has no clause selection, so rank features by selectivity
    sel_range/global_range and score precision@r (pre-specified cutoff r = planted count)."""
    truth = {int(j) for j in rel_dims}
    if method == "db":
        found = {feature_index[str(r["feature"])] for r in rows if r["in_predicate"]}
    else:
        ranked = sorted(
            (
                (r["sel_max"] - r["sel_min"])
                / max(r["global_max"] - r["global_min"], 1e-12),
                feature_index[str(r["feature"])],
            )
            for r in rows
        )
        found = {j for _, j in ranked[: len(truth)]}
    tp = len(found & truth)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "dim_precision": precision,
        "dim_recall": recall,
        "dim_f1": f1,
        "n_dims_found": len(found),
    }


# --------------------------------------------------------------------------- #
# One selection through the full condition sweep (design SS6 pseudocode).      #
# --------------------------------------------------------------------------- #


def selection_records(
    common: dict,
    sel_idx: np.ndarray,
    X_all: np.ndarray,
    feature_cols: list[str],
    rel_dims: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """All records for one base selection. The same m replicates per delta are shared
    across every (method, split, t) so comparisons across conditions are paired."""
    y_base = np.zeros(len(X_all), dtype=bool)
    y_base[sel_idx] = True
    feature_index = {str(c): j for j, c in enumerate(feature_cols)}
    pool = knn_pool(X_all, sel_idx)
    reps = {
        delta: [perturb(sel_idx, pool, delta, rng) for _ in range(M_REPLICATES)]
        for delta in DELTAS
    }

    records: list[dict] = []
    recovery: list[dict] = []
    for method in METHODS:
        for split in SPLITS:
            for t in T_GRID:
                base_rows = build_predicate(
                    method, split, t, sel_idx, X_all, feature_cols
                )
                base_mask = admitted_mask(base_rows, X_all, feature_index)
                f1, precision, recall = _f1(base_mask, y_base)
                quality = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "coverage": float(base_mask.mean()),
                    "length": int(sum(bool(r.get("in_predicate")) for r in base_rows))
                    if method == "db"
                    else len(feature_cols),
                }
                if rel_dims is not None:
                    recovery.append(
                        {
                            **common,
                            "method": method,
                            "split": split,
                            "t": t,
                            **dim_recovery(base_rows, rel_dims, method, feature_index),
                            "membership_f1": f1,  # selection == planted cluster, so they coincide
                            "coverage": quality["coverage"],
                        }
                    )
                for delta in DELTAS:
                    rep_rows = [
                        build_predicate(method, split, t, r, X_all, feature_cols)
                        for r in reps[delta]
                    ]
                    rep_masks = np.array(
                        [admitted_mask(rw, X_all, feature_index) for rw in rep_rows]
                    )
                    rep_f1 = [_f1(mask, y_base)[0] for mask in rep_masks]
                    records.append(
                        {
                            **common,
                            "method": method,
                            "split": split,
                            "t": t,
                            "delta": delta,
                            "m_reps": len(rep_rows),
                            "jaccard_admitted": mean_pairwise_mask_jaccard(rep_masks),
                            "f1_sd": float(np.std(rep_f1, ddof=1)),
                            "bound_sd": bound_sd(rep_rows),
                            "jaccard_clauses": mean_pairwise_clause_jaccard(rep_rows)
                            if method == "db"
                            else float("nan"),
                            **quality,
                        }
                    )
    return records, recovery


# --------------------------------------------------------------------------- #
# Running one grid cell (arm x seed).                                          #
# --------------------------------------------------------------------------- #


def run_cell(
    arm: str, seed: int
) -> tuple[list[dict], list[dict], list[np.ndarray] | None]:
    """Build selections for one (arm, seed) and sweep them. Wine: tree leaves (the tree is
    the varying element - the seed is threaded into UMAP's random_state). Synthetic: planted
    clusters, no tree. Returns (records, recovery_rows, wine_leaf_indices-or-None) - the leaf
    index sets feed the seed-perturbation pass (b) in the driver."""
    _silence_noise()  # re-assert in worker processes
    if arm == "wine":
        df, feature_cols, _y = prepare_dataset(WINE_DATASET)
        cfg: Config = init_state(init_streamlit=False)
        cfg["method"] = DR_METHOD
        cfg["hierarchical_layers"] = HIER_LAYERS
        # Without this the seed reaches nothing: `init_state` hardcodes 42 for all three
        # reducers, so all SEEDS_WINE "rebuilds" would be one tree and pass (b) would
        # compare a leaf against itself.
        cfg["umap_random_state"] = cfg["tsne_random_state"] = cfg[
            "mds_random_state"
        ] = seed
        tree = compute_analysis_tree(df, feature_cols, cfg)
        X_all = standardised_X(df, feature_cols, tree)
        leaf_idx = [np.asarray(leaf["row_indices"]) for leaf in collect_leaves(tree)]
        selections = [(f"leaf{i}", idx) for i, idx in enumerate(leaf_idx)]
        rel_by_sel: dict[str, np.ndarray] = {}
        meta = {"skew": "real", "margin": float("nan")}
    else:
        spec = SYNTH_ARMS[arm]
        X, y, rel = make_axis_parallel(spec["skew"], spec["margin"], seed)
        X_all = StandardScaler().fit_transform(X)
        feature_cols = [f"f{j}" for j in range(X_all.shape[1])]
        leaf_idx = None
        selections = [(f"cluster{c}", np.where(y == c)[0]) for c in range(C_CLUSTERS)]
        rel_by_sel = {f"cluster{c}": rel[c] for c in range(C_CLUSTERS)}
        meta = {"skew": spec["skew"], "margin": spec["margin"]}

    records: list[dict] = []
    recovery: list[dict] = []
    for pos, (sel_id, sel_idx) in enumerate(selections):
        # Min-size filter (quantiles on tiny selections are noise) + enough outside points
        # for the boundary pool. Sizes are recorded; skipped leaves show up as the gap
        # between the tree's leaf count and the selections in the records.
        if len(sel_idx) < MIN_SEL or len(X_all) - len(sel_idx) < K_NN:
            continue
        rng = np.random.default_rng([SEED, zlib.crc32(arm.encode()), seed, pos])
        common = {
            "arm": arm,
            **meta,
            "seed": seed,
            "sel_id": sel_id,
            "n_sel": len(sel_idx),
            "n_all": len(X_all),
        }
        rec, rcv = selection_records(
            common, sel_idx, X_all, feature_cols, rel_by_sel.get(sel_id), rng
        )
        records.extend(rec)
        recovery.extend(rcv)
    return records, recovery, leaf_idx


# --------------------------------------------------------------------------- #
# Seed-perturbation pass (b) - wine, matched leaves across tree rebuilds.      #
# --------------------------------------------------------------------------- #


def greedy_match(
    leaves_a: list[tuple[int, np.ndarray]], leaves_b: list[tuple[int, np.ndarray]]
) -> list[tuple]:
    """One-to-one leaf matching across two rebuilds by member-set Jaccard, greedy on the
    best remaining pair. Returns [((pos_a, idx_a), (pos_b, idx_b), jaccard), ...]."""
    sets_a = [set(idx.tolist()) for _, idx in leaves_a]
    sets_b = [set(idx.tolist()) for _, idx in leaves_b]
    jac = np.array(
        [[len(a & b) / len(a | b) if (a | b) else 0.0 for b in sets_b] for a in sets_a]
    )
    matches: list[tuple] = []
    while jac.size and jac.max() > 0:
        ia, ib = np.unravel_index(int(jac.argmax()), jac.shape)
        matches.append((leaves_a[ia], leaves_b[ib], float(jac[ia, ib])))
        jac[ia, :] = -1.0
        jac[:, ib] = -1.0
    return matches


def seed_perturbation_records(leaf_sets: dict[int, list[np.ndarray]]) -> pd.DataFrame:
    """Pass (b): compare predicates of leaves matched across tree rebuilds (design SS4b).
    Known confound, stated in advance: this conflates tree instability with predicate
    instability - match_jaccard is therefore recorded per pair and weak matches (< 0.5)
    are flagged as tree-level instability, counted but excluded from the predicate claim."""
    df, feature_cols, _y = prepare_dataset(WINE_DATASET)
    # Fresh scaler on the same rows == the root scaler every tree fits (normalize=True).
    X_all = StandardScaler().fit_transform(df[feature_cols].to_numpy(dtype=np.float64))
    feature_index = {str(c): j for j, c in enumerate(feature_cols)}
    eligible = {
        s: [(i, idx) for i, idx in enumerate(leaves) if len(idx) >= MIN_SEL]
        for s, leaves in leaf_sets.items()
    }

    masks: dict[tuple, np.ndarray] = {}
    for s, leaves in eligible.items():
        for i, idx in leaves:
            for method in METHODS:
                for t in T_GRID:
                    rows = build_predicate(
                        method, "severity", t, idx, X_all, feature_cols
                    )
                    masks[(s, i, method, t)] = admitted_mask(rows, X_all, feature_index)

    out: list[dict] = []
    for a, b in combinations(sorted(eligible), 2):
        for (ia, idx_a), (ib, idx_b), match in greedy_match(eligible[a], eligible[b]):
            for method in METHODS:
                for t in T_GRID:
                    ma, mb = masks[(a, ia, method, t)], masks[(b, ib, method, t)]
                    union = int(np.count_nonzero(ma | mb))
                    out.append(
                        {
                            "seed_a": a,
                            "seed_b": b,
                            "leaf_a": ia,
                            "leaf_b": ib,
                            "match_jaccard": match,
                            "n_a": len(idx_a),
                            "n_b": len(idx_b),
                            "method": method,
                            "t": t,
                            "jaccard_admitted": int(np.count_nonzero(ma & mb)) / union
                            if union
                            else 1.0,
                            "weak_match": match < 0.5,
                        }
                    )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Grid driver.                                                                 #
# --------------------------------------------------------------------------- #


def build_cells() -> list[tuple[str, int]]:
    """Enumerate (arm, seed) cells - all independent. Wine first: its tree builds dominate
    wall-clock, so they should start earliest."""
    return [("wine", s) for s in SEEDS_WINE] + [
        (arm, s) for arm in SYNTH_ARMS for s in SEEDS_SYNTH
    ]


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Run the full grid, returning (records, recovery, seedpass-or-None)."""
    cells = build_cells()
    records: list[dict] = []
    recovery: list[dict] = []
    wine_leaves: dict[int, list[np.ndarray]] = {}
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
            results = []
            for arm, seed in cells:
                progress.update(task, description=f"{arm} / s{seed}"[:48])
                results.append(run_cell(arm, seed))
                progress.advance(task)
        else:
            # Cells are independent -> worker processes. inner_max_num_threads=1 stops each
            # worker's UMAP/BLAS pools oversubscribing the cores. Generator (order-preserving)
            # keeps the bar live and lets us zip results back to their cells.
            jobs = (delayed(run_cell)(arm, seed) for arm, seed in cells)
            results = []
            with parallel_config(backend="loky", inner_max_num_threads=1):
                for res in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
                    results.append(res)
                    progress.advance(task)
    for (arm, seed), (rec, rcv, leaves) in zip(cells, results):
        records.extend(rec)
        recovery.extend(rcv)
        if arm == "wine" and leaves is not None:
            wine_leaves[seed] = leaves

    seedpass = None
    if RUN_SEED_PERTURBATION and len(wine_leaves) >= 2:
        console.print(
            "Seed-perturbation pass (b): matching leaves across wine tree rebuilds ..."
        )
        seedpass = seed_perturbation_records(wine_leaves)
    return pd.DataFrame(records), pd.DataFrame(recovery), seedpass


def main() -> None:
    np.random.seed(SEED)
    console.rule("[bold]RQ2: predicate stability under relaxation")
    console.print(
        f"arms=wine+{list(SYNTH_ARMS)}  seeds={len(SEEDS_WINE)}/{len(SEEDS_SYNTH)}  t={T_GRID}  "
        f"methods={METHODS}  splits={SPLITS}  deltas={DELTAS}  m={M_REPLICATES}  k={K_NN}\n"
    )

    records, recovery, seedpass = run_experiment()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_dir / "stability_records.csv", index=False)
    recovery.to_csv(out_dir / "recovery.csv", index=False)
    if seedpass is not None:
        seedpass.to_csv(out_dir / "seed_stability.csv", index=False)

    psa.check_records(
        records
    )  # t=1.0 split-invariance: catches a mis-threaded tail_split (records are already on disk)

    results = psa.analyse(records, recovery, seedpass)
    psa.save_outputs(results, out_dir)
    psa.make_plots(records, out_dir)
    psa.print_report(results, console)
    console.print(f"\n[bold green]Done.[/] Results + plots in [underline]{out_dir}[/]")


if __name__ == "__main__":
    main()

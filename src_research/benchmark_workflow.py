"""Benchmark workflow walk (thesis §6.5).

Implements the design in ``src_research/EXPERIMENT_benchmark_workflow.md`` — the
descriptive census of what the *shipped defaults* produce on the three real benchmark
datasets, plus the four pre-registered consistency checks (C1–C4) derived from the
completed experiments. Unlike the other harnesses this tests no hypothesis; every
reporting and aggregation rule is fixed in the design doc *before* the run, and the
operationalisation of each pass/fail verdict is fixed in this file before any result
was seen. If a check fails, the failure is the finding — degenerate builds are
reported, never re-rolled (re-rolling a build until the tree looks better is silent
p-hacking, design §8).

Readouts (design §1):
    D1 hierarchy shape (depth, leaves, sizes, noise fraction — first-class),
    D2 per-node shipped ZADU scores, D3 predicate readability over all eligible
    leaves, D4 purity/enrichment + faithfulness–purity association.
Checks:
    C1 dense-recall compounding out of sample (bc d=30, digits d=64 vs the t^d bound),
    C2 H1a replication arm (paired leaf-vs-root, k-matched, the run's only p-value),
    C3 noise fractions vs the H1b UMAP ranges,
    C4 sparse (db) brevity replication (wine 10→6 ± 2 at t=0.95; direction elsewhere).

Reuse, don't reimplement (design §2): ``prepare_dataset`` / ``standardised_X`` /
``collect_leaves`` from the RQ1 harness; ``build_predicate`` / ``admitted_mask`` and
the output/timestamp/Parallel conventions from the RQ2 harness; the driver core is
``start_evaluation`` (tree + shipped node scores).

Outputs (``outputs/experiments/<timestamp>/``, design §10): tree_shape.csv,
node_scores.csv, leaf_predicates.csv, leaf_purity.csv (D4 records), consistency.csv
(verdict lines; the C2 paired records live in c2_paired.csv for a tidy schema — a
deliberate, purely additive split of §10's "consistency.csv + C2 paired records"),
figures benchmark_shape_faithfulness.png / benchmark_compounding.png, run_meta.json,
summary.md.

Run with::

    uv run python -m src_research.benchmark_workflow
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_config
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from scipy.stats import spearmanr, wilcoxon

from src.analysis.analysis_routine import AnalysisObject
from src.analysis.predicate_generator import _f1
from src.config_defaults import default_config
from src.evaluation.evaluate import _score_node, start_evaluation
from src.types import Config
from src_research.hierarchical_vs_flat import _silence_noise, collect_leaves, prepare_dataset, standardised_X
from src_research.predicate_stability import admitted_mask, build_predicate

# --------------------------------------------------------------------------- #
# CONFIG — fixed by the design doc (§3, §4). No new levels, no new datasets.   #
# --------------------------------------------------------------------------- #

DATASETS_TO_RUN = ["Wine quality (Low)", "Breast cancer (Low)", "Digits (Low)"]  # design §3
N_BUILDS = 5  # replicates; the build id is threaded into the DR seed (SEEDS_WINE convention)
HIER_LAYERS_CAP = 4  # the single named deviation from shipped defaults (design §4)
T_GRID = [1.0, 0.95, 0.9, 0.8]  # pre-specified in the thesis; no new levels
PREDICATE_METHODS = ["threshold", "db"]  # dense (C1) + sparse (C4 / table example)
TAIL_SPLIT = "severity"  # shipped default; not swept (the ablation is RQ2's, done)
MIN_SELECTION = 20  # leaves below get no predicates; counted (RQ2 convention)
C2_MIN_N = 41  # k = min(20, (n-1)//2) = 20 ⇔ n ≥ 41: the H1a parity condition

PARALLEL_JOBS = -1
OUTPUT_ROOT = Path("outputs/experiments")

# Feature counts d per dataset (design §3) — used for the t^d bound and C4 direction.
DATASET_D = {"Wine quality (Low)": 11, "Breast cancer (Low)": 30, "Digits (Low)": 64}

# --------------------------------------------------------------------------- #
# Pre-registered predictions (design §1) and their mechanical                  #
# operationalisations. Everything in this block was fixed BEFORE the run;     #
# the raw records let any reader re-judge a borderline case.                  #
# --------------------------------------------------------------------------- #

# C1: at t = 0.95, dense (threshold-method) median recall of leaf members must lie at
# or above the independence bound t^d for the two out-of-sample dimensionalities
# (breast cancer d=30, digits d=64), and digits must stay far below 1 — concretised in
# the design as digits < 0.5.
C1_T_STAR = 0.95
C1_OOS_DATASETS = ["Breast cancer (Low)", "Digits (Low)"]
C1_DIGITS_CEILING = 0.5

# C2: pooled over eligible leaves (n ≥ C2_MIN_N) of all datasets and builds:
# "small positive median Δtrust" with |median Δtrust| ≤ 0.05, win rate in [0.5, 0.8]
# (prediction from run 20260628_184827, UMAP, hier_leaf). Δ = leaf-local − root-sliced;
# win = Δ > 0. "Positive" is enforced (median > 0) — the stricter of the two readings
# of the prediction, adopted before any result was seen. Degenerate builds' single
# leaf is its own root (Δ ≡ 0, a self-comparison): recorded in c2_paired.csv with
# eligible=False, excluded from the check by this rule. One Wilcoxon signed-rank on
# the pooled Δtrust — the experiment's only p-value, reported descriptively (the
# prediction constrains median and win rate, not p).
C2_MEDIAN_BAND = 0.05
C2_WIN_RANGE = (0.5, 0.8)

# C3: predicted per-build noise fractions = the observed H1b UMAP ranges
# (h1b_recovery.csv, 5 seeds). "Modest excursions are expected; factor-level
# disagreement means a harness bug or config drift" (design §1). Operationalisation,
# fixed pre-run: PASS if every build lies inside the predicted range;
# PASS-WITH-EXCURSION if every build lies inside the widened band
# [lower/2, upper·1.5] but not all inside the range; FAIL (factor-level) otherwise.
C3_NOISE_RANGES = {
    "Wine quality (Low)": (0.00, 0.19),
    "Breast cancer (Low)": (0.15, 0.32),
    "Digits (Low)": (0.12, 0.17),
}
C3_BAND_LO_FACTOR = 0.5
C3_BAND_HI_FACTOR = 1.5

# C4: RQ2 measured wine median db length 10 (strict, t=1.0) → 6 (relaxed, t=0.95).
# "Within ±2 clauses" ⇒ strict median ∈ [8, 12] and relaxed median ∈ [4, 8].
# Breast cancer / digits: direction only — relaxed median < strict median ≪ d,
# with "≪ d" operationalised pre-run as strict median < d/2; magnitude unconstrained.
C4_WINE_STRICT = (8, 12)
C4_WINE_RELAXED = (4, 8)
C4_RELAXED_T = 0.95

_silence_noise()
console = Console()


# --------------------------------------------------------------------------- #
# Tree walking.                                                                #
# --------------------------------------------------------------------------- #


def walk_nodes(node: AnalysisObject, depth: int = 0):
    """Yield (node, depth) over the whole tree, root first. HierarchyObject carries no
    depth field, so depth is tracked here; for leaves it must agree with the stored one."""
    yield node, depth
    if "is_leaf" not in node:
        for child in node.get("next_object_layer") or []:
            yield from walk_nodes(child, depth + 1)


# --------------------------------------------------------------------------- #
# One build = one grid cell (dataset × build); the build id seeds the reducer.  #
# --------------------------------------------------------------------------- #


def run_build(dataset: str, data: tuple[pd.DataFrame, list[str], np.ndarray | None], build: int) -> dict[str, list[dict]]:
    """Build the tree with shipped defaults (cap HIER_LAYERS_CAP) and extract every
    record the design needs. Returns lists of plain dicts (trees stay in the worker)."""
    _silence_noise()
    cfg: Config = default_config()
    cfg["dataset_choice"] = dataset  # display-only (console banner); no algorithmic effect
    cfg["hierarchical_layers"] = HIER_LAYERS_CAP  # the single named deviation (design §4)
    # `default_config()` pins all three random_state keys to 42, so without this every one
    # of the N_BUILDS "replicates" is byte-identical and C2's Wilcoxon receives each leaf
    # N_BUILDS times. (The shipped 20260728_185329 run predates DR seeding, so its builds
    # are genuine replicates; a rerun without this line would not be.)
    cfg["umap_random_state"] = cfg["tsne_random_state"] = cfg["mds_random_state"] = build

    df, feature_cols, y = data
    n_total = len(df)
    cell = {"dataset": dataset, "build": build}

    tree = start_evaluation(df, feature_cols, cfg)
    X_all = standardised_X(df, feature_cols, tree)
    feature_index = {str(c): j for j, c in enumerate(feature_cols)}
    leaves = collect_leaves(tree)
    degenerate = "is_leaf" in tree  # root found < 2 clusters → single-leaf tree (§8)

    # ---- D2: every node's shipped scores --------------------------------- #
    node_rows: list[dict] = []
    leaf_depths: list[int] = []
    for node_id, (node, depth) in enumerate(walk_nodes(tree)):
        is_leaf = "is_leaf" in node
        if is_leaf:
            assert node["depth"] == depth, f"walk depth {depth} != stored {node['depth']}"
            leaf_depths.append(depth)
        s = node.get("scores") or {}
        node_rows.append(
            {
                **cell,
                "node_id": node_id,
                "depth": depth,
                "is_leaf": is_leaf,
                "n_points": s.get("n_points"),
                "k": s.get("k"),
                "trustworthiness": s.get("trustworthiness"),
                "continuity": s.get("continuity"),
                "mrre_false": s.get("mrre_false"),
                "mrre_missing": s.get("mrre_missing"),
                "stress": s.get("stress"),
                "cadi": s.get("cadi"),
            }
        )

    # ---- D1: hierarchy shape --------------------------------------------- #
    leaf_sizes = [int(leaf["row_indices"].shape[0]) for leaf in leaves]
    covered = int(np.sum(leaf_sizes))
    shape_row = {
        **cell,
        "n_rows": n_total,
        "degenerate": degenerate,
        "realized_depth": int(max(leaf_depths)) if leaf_depths else 0,
        "cap_censored": bool(leaf_depths and max(leaf_depths) >= HIER_LAYERS_CAP),
        "n_leaves": len(leaves),
        "median_leaf_size": float(np.median(leaf_sizes)) if leaf_sizes else np.nan,
        "min_leaf_size": int(min(leaf_sizes)) if leaf_sizes else 0,
        "max_leaf_size": int(max(leaf_sizes)) if leaf_sizes else 0,
        "noise_fraction": 1.0 - covered / n_total,
        "n_leaves_scored": int(sum(1 for leaf in leaves if (leaf.get("scores") or {}).get("trustworthiness") is not None)),
        "n_leaves_predicated": int(sum(1 for s in leaf_sizes if s >= MIN_SELECTION)),
        # Projections that failed outright, so the census says how much of the tree it saw.
        "root_unprojected": root_emb is None,
        "n_leaves_unprojected": int(sum(1 for leaf in leaves if leaf["embedding_original"] is None)),
    }

    # ---- per-leaf records: D3 predicates, C2 paired arm, D4 purity -------- #
    pred_rows: list[dict] = []
    c2_rows: list[dict] = []
    purity_rows: list[dict] = []
    root_emb = tree["embedding_original"]
    # `_embed_original` returns None for a projection that failed (it used to fabricate an
    # all-zeros embedding, which scores ~0.55 and would enter C2 as a real comparator).
    # `root_emb[idx]` on None takes the whole build down, so the state is named, carried
    # into c2_paired.csv per row, and counted in tree_shape.csv — never silently absorbed.
    root_unprojected = root_emb is None

    for leaf_id, leaf in enumerate(leaves):
        idx = leaf["row_indices"]
        n_leaf = int(idx.shape[0])
        leaf_scores = leaf.get("scores") or {}
        leaf_cell = {**cell, "leaf_id": leaf_id, "leaf_n": n_leaf, "leaf_depth": int(leaf["depth"])}

        # D3 / C1 / C4 — predicates for every eligible leaf, both methods, full t grid.
        if n_leaf >= MIN_SELECTION:
            y_leaf = np.zeros(n_total, dtype=bool)
            y_leaf[idx] = True
            for method in PREDICATE_METHODS:
                for t in T_GRID:
                    rows = build_predicate(method, TAIL_SPLIT, t, idx, X_all, feature_cols)
                    mask = admitted_mask(rows, X_all, feature_index)
                    f1, precision, recall = _f1(mask, y_leaf)
                    rec = {
                        **leaf_cell,
                        "method": method,
                        "t": t,
                        "length": int(sum(bool(r.get("in_predicate")) for r in rows)) if method == "db" else len(feature_cols),
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "coverage": float(mask.mean()),
                    }
                    # Verbatim clause dump (db, strict + relaxed) for EVERY eligible
                    # leaf — the example leaf is only known after the fixed rule (§5)
                    # sees all builds, so no leaf's clauses may be discarded here.
                    if method == "db" and t in (1.0, C4_RELAXED_T):
                        rec["clauses_json"] = json.dumps(
                            [
                                {
                                    "feature": str(r["feature"]),
                                    "sel_min": float(r["sel_min"]),
                                    "sel_max": float(r["sel_max"]),
                                    "in_predicate": bool(r.get("in_predicate")),
                                    "step": r.get("predicate_step"),
                                }
                                for r in rows
                            ]
                        )
                    pred_rows.append(rec)

        # C2 — paired leaf-vs-root, identical scoring path (_score_node both sides).
        # All leaves recorded, including a degenerate build's root-leaf (Δ ≡ 0 by
        # construction — a self-comparison, marked ineligible per the rule fixed in
        # the CONFIG block); the check restricts to n ≥ C2_MIN_N (k = 20 parity).
        X_leaf = X_all[idx]
        leaf_emb = leaf["embedding_original"]
        s_leaf = _score_node(X_leaf, leaf_emb, None)
        s_root = _score_node(X_leaf, None if root_unprojected else root_emb[idx], None)
        # k parity is the fairness invariant, and it is only defined when both arms scored;
        # an unprojected arm reports k=None and is excluded below rather than asserted on.
        if s_leaf["k"] is not None and s_root["k"] is not None:
            assert s_leaf["k"] == s_root["k"], f"k mismatch: leaf={s_leaf['k']} root={s_root['k']}"
        c2_rows.append(
            {
                **leaf_cell,
                "k": s_leaf["k"],
                "degenerate": degenerate,
                "root_unprojected": root_unprojected,
                "leaf_unprojected": leaf_emb is None,
                "eligible": (n_leaf >= C2_MIN_N) and not degenerate and not root_unprojected and leaf_emb is not None,
                "trust_leaf": s_leaf["trustworthiness"],
                "trust_root": s_root["trustworthiness"],
                "cont_leaf": s_leaf["continuity"],
                "cont_root": s_root["continuity"],
                "delta_trust": None
                if s_leaf["trustworthiness"] is None or s_root["trustworthiness"] is None
                else s_leaf["trustworthiness"] - s_root["trustworthiness"],
                "delta_cont": None
                if s_leaf["continuity"] is None or s_root["continuity"] is None
                else s_leaf["continuity"] - s_root["continuity"],
            }
        )

        # D4 — purity / enrichment (labels exist on all three datasets).
        if y is not None and n_leaf > 0:
            members = y[idx]
            classes, counts = np.unique(members, return_counts=True)
            maj = int(classes[np.argmax(counts)])
            purity = float(counts.max() / n_leaf)
            global_share = float(np.mean(y == maj))
            purity_rows.append(
                {
                    **leaf_cell,
                    "majority_class": maj,
                    "purity": purity,
                    "global_share": global_share,
                    "enrichment": purity / global_share if global_share > 0 else np.nan,
                    "trustworthiness": leaf_scores.get("trustworthiness"),
                    "continuity": leaf_scores.get("continuity"),
                }
            )

    return {
        "shape": [shape_row],
        "nodes": node_rows,
        "predicates": pred_rows,
        "c2": c2_rows,
        "purity": purity_rows,
    }


# --------------------------------------------------------------------------- #
# Aggregation — every rule below restates design §5; nothing is chosen here.   #
# --------------------------------------------------------------------------- #


def _med_range(values: pd.Series | list, fmt: str = "{:.2f}") -> str:
    v = pd.Series(values).dropna()
    if v.empty:
        return "—"
    return f"{fmt.format(v.median())} ({fmt.format(v.min())}–{fmt.format(v.max())})"


def build_leaf_means(pred_or_scores: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Per (dataset, build): unweighted mean over scored leaves (every leaf one view,
    design §5) plus the size-weighted variant (CSV only, never the table)."""
    rows = []
    for (dataset, build), g in pred_or_scores.groupby(["dataset", "build"]):
        rec = {"dataset": dataset, "build": build, "n_leaves_scored": int(g[value_cols[0]].notna().sum())}
        for c in value_cols:
            gv = g.dropna(subset=[c])
            rec[f"{c}_mean"] = float(gv[c].mean()) if len(gv) else np.nan
            rec[f"{c}_wmean"] = float(np.average(gv[c], weights=gv["n_points"])) if len(gv) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def pick_example_leaf(shape: pd.DataFrame, predicates: pd.DataFrame, dataset: str) -> dict | None:
    """The fixed example rule (§5): the build with the median leaf count (tie → earlier
    build id); within it the leaf of median size (even count → the larger middle; equal
    sizes → the later leaf id, a fixed arbitrary tiebreak). Returns the strict and
    relaxed db records for that leaf."""
    ds_shape = shape[shape["dataset"] == dataset].sort_values("build")
    if ds_shape.empty:
        return None
    counts = ds_shape["n_leaves"].to_numpy()
    median_count = int(np.median(counts))
    cand = ds_shape[ds_shape["n_leaves"] == median_count]
    if cand.empty:  # median not attained (possible only with an even build count)
        cand = ds_shape.iloc[(ds_shape["n_leaves"] - median_count).abs().argsort()]
    build = int(cand.iloc[0]["build"])

    leafs = (
        predicates[(predicates["dataset"] == dataset) & (predicates["build"] == build)][["leaf_id", "leaf_n"]]
        .drop_duplicates()
        .sort_values(["leaf_n", "leaf_id"])
        .reset_index(drop=True)
    )
    if leafs.empty:
        return None
    chosen = leafs.iloc[len(leafs) // 2]
    leaf_id = int(chosen["leaf_id"])

    sel = predicates[
        (predicates["dataset"] == dataset)
        & (predicates["build"] == build)
        & (predicates["leaf_id"] == leaf_id)
        & (predicates["method"] == "db")
    ]
    if sel[sel["t"] == 1.0].empty or sel[sel["t"] == C4_RELAXED_T].empty:
        return None
    strict = sel[sel["t"] == 1.0].iloc[0].to_dict()
    relaxed = sel[sel["t"] == C4_RELAXED_T].iloc[0].to_dict()
    return {"dataset": dataset, "build": build, "leaf_id": leaf_id, "leaf_n": int(chosen["leaf_n"]), "strict": strict, "relaxed": relaxed}


# --------------------------------------------------------------------------- #
# Consistency checks C1–C4 — prediction / observed / verdict, one line each.   #
# --------------------------------------------------------------------------- #


def check_c1(predicates: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    # "Median over leaves and builds" (design §5) = pooled median over (leaf × build)
    # records — the natural reading; the two-stage per-build variant (median over the
    # 5 build medians) is emitted alongside so a reader can re-judge borderline cases.
    dense = predicates[predicates["method"] == "threshold"]
    detail = dense.groupby(["dataset", "t"])["recall"].agg(median="median", count="size").reset_index()
    two_stage = dense.groupby(["dataset", "build", "t"])["recall"].median().groupby(["dataset", "t"]).median().rename("median_of_build_medians").reset_index()
    detail = detail.merge(two_stage, on=["dataset", "t"])
    obs: dict[tuple[str, float], float] = {(r["dataset"], r["t"]): r["median"] for _, r in detail.iterrows()}
    parts, ok = [], True
    for ds in C1_OOS_DATASETS:
        d = DATASET_D[ds]
        bound = C1_T_STAR**d
        med = obs.get((ds, C1_T_STAR), np.nan)
        ds_ok = bool(med >= bound)
        if ds == "Digits (Low)":
            ds_ok = ds_ok and bool(med < C1_DIGITS_CEILING)
        ok &= ds_ok
        parts.append(f"{ds}: median {med:.4f} vs bound {bound:.4f}" + (f" (< {C1_DIGITS_CEILING})" if ds == "Digits (Low)" else ""))
    wine_med = obs.get(("Wine quality (Low)", C1_T_STAR), np.nan)
    line = {
        "check": "C1",
        "prediction": f"dense recall at t={C1_T_STAR} ≥ t^d and ≪ 1 for d=30, 64; digits < {C1_DIGITS_CEILING}",
        "observed": "; ".join(parts) + f"; wine (in-sample): {wine_med:.4f} vs bound {C1_T_STAR**11:.4f}",
        "verdict": "PASS" if ok else "FAIL",
    }
    return [line], detail


def check_c2(c2: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    assert not c2.empty, "C2 frame empty — harness bug"
    elig = c2[c2["eligible"] & c2["delta_trust"].notna()]
    d = elig["delta_trust"].to_numpy(dtype=float)
    med = float(np.median(d)) if d.size else np.nan
    win = float(np.mean(d > 0)) if d.size else np.nan
    med_c = float(elig["delta_cont"].median()) if d.size else np.nan
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if nz.size >= 1 and not np.allclose(d, 0) else np.nan
    ok = bool(d.size and 0 < med <= C2_MEDIAN_BAND and C2_WIN_RANGE[0] <= win <= C2_WIN_RANGE[1])
    per_ds = (
        elig.groupby("dataset")["delta_trust"].agg(median="median", win_rate=lambda s: float(np.mean(s > 0)), n="size").reset_index()
    )
    # Pairs that were eligible on size but carry no Δ (an arm was never projected, or the
    # scorer failed): a shrinking denominator is stated, not absorbed.
    n_no_delta = int((c2["eligible"] & c2["delta_trust"].isna()).sum())
    line = {
        "check": "C2",
        "prediction": f"small positive median Δtrust (0 < median ≤ {C2_MEDIAN_BAND}), win rate in [{C2_WIN_RANGE[0]}, {C2_WIN_RANGE[1]}] (pooled, n≥{C2_MIN_N}, non-degenerate)",
        "observed": f"median Δtrust {med:+.4f}, win rate {win:.2f}, n={d.size}, median Δcont {med_c:+.4f}, Wilcoxon p={p:.3g}, unscored pairs excluded={n_no_delta}",
        "verdict": "PASS" if ok else "FAIL",
    }
    return [line], per_ds


def check_c3(shape: pd.DataFrame) -> list[dict]:
    lines = []
    for ds, (lo, hi) in C3_NOISE_RANGES.items():
        fr = shape[shape["dataset"] == ds]["noise_fraction"].to_numpy(dtype=float)
        assert fr.size, f"C3: no builds for {ds} — harness bug"
        band = (lo * C3_BAND_LO_FACTOR, hi * C3_BAND_HI_FACTOR)
        all_in_range = bool(np.all((fr >= lo) & (fr <= hi)))
        all_in_band = bool(np.all((fr >= band[0]) & (fr <= band[1])))
        verdict = "PASS" if all_in_range else ("PASS-WITH-EXCURSION" if all_in_band else "FAIL")
        lines.append(
            {
                "check": f"C3 [{ds}]",
                "prediction": f"per-build noise fraction in [{lo:.2f}, {hi:.2f}] (H1b UMAP range); band [{band[0]:.2f}, {band[1]:.2f}]",
                "observed": f"builds: {np.array2string(np.sort(fr), precision=3)}, median {np.median(fr):.3f}",
                "verdict": verdict,
            }
        )
    return lines


def check_c4(predicates: pd.DataFrame) -> list[dict]:
    # Pooled (leaf × build) medians, same reading as C1; the per-build two-stage
    # variant is derivable from leaf_predicates.csv.
    db = predicates[predicates["method"] == "db"]
    lines = []
    for ds in DATASETS_TO_RUN:
        d = DATASET_D[ds]
        strict = db[(db["dataset"] == ds) & (db["t"] == 1.0)]["length"]
        relaxed = db[(db["dataset"] == ds) & (db["t"] == C4_RELAXED_T)]["length"]
        s_med, r_med = float(strict.median()), float(relaxed.median())
        if ds == "Wine quality (Low)":
            ok = C4_WINE_STRICT[0] <= s_med <= C4_WINE_STRICT[1] and C4_WINE_RELAXED[0] <= r_med <= C4_WINE_RELAXED[1]
            pred = f"strict median in {C4_WINE_STRICT}, relaxed (t={C4_RELAXED_T}) in {C4_WINE_RELAXED} (RQ2: 10 → 6 ± 2)"
        else:
            ok = (r_med < s_med) and (s_med < d / 2)
            pred = f"direction: relaxed median < strict median ≪ d={d} (strict < d/2)"
        lines.append(
            {
                "check": f"C4 [{ds}]",
                "prediction": pred,
                "observed": f"strict median {s_med:.1f}, relaxed median {r_med:.1f} (n_leaf-records {len(strict)}/{len(relaxed)})",
                "verdict": "PASS" if ok else "FAIL",
            }
        )
    return lines


# --------------------------------------------------------------------------- #
# Figures (design §10).                                                        #
# --------------------------------------------------------------------------- #


def fig_shape_faithfulness(nodes: pd.DataFrame, out_dir: Path) -> None:
    """Per dataset: pooled leaf T&C distributions with the per-build root scores as
    reference marks (the root is the view the user starts from)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(DATASETS_TO_RUN), figsize=(11.5, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, DATASETS_TO_RUN, strict=True):
        sub = nodes[nodes["dataset"] == ds]
        leaf = sub[sub["is_leaf"]]
        root = sub[sub["depth"] == 0]
        dd = leaf.melt(value_vars=["trustworthiness", "continuity"], var_name="measure", value_name="score").dropna()
        sns.boxplot(data=dd, x="measure", y="score", hue="measure", ax=ax, width=0.5, fliersize=2, legend=False)
        for i, m in enumerate(["trustworthiness", "continuity"]):
            vals = root[m].dropna()
            ax.scatter(np.full(len(vals), i), vals, marker="D", s=28, zorder=5, facecolor="none", edgecolor="crimson", label="root (per build)" if i == 0 else None)
        ax.set_title(ds.replace(" (Low)", ""), fontsize=10)
        ax.set_xlabel("")
        ax.set_xticks([0, 1], ["trust.", "cont."])
        ax.set_ylabel("score" if ds == DATASETS_TO_RUN[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:1], ["root embedding (one mark per build)"], loc="lower center", ncol=1, frameon=False)
    fig.suptitle("Leaf-local faithfulness (pooled over 5 builds) vs the root view", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "benchmark_shape_faithfulness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_compounding(c1_detail: pd.DataFrame, out_dir: Path) -> None:
    """C1 on the rq2_compounding axes: median dense recall vs t (log y), against the
    t^d independence bounds. Extends the RQ2 figure by d = 30 and d = 64."""
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ts = np.array(sorted(T_GRID, reverse=True))
    colors = {"Wine quality (Low)": "C1", "Breast cancer (Low)": "C2", "Digits (Low)": "C0"}
    markers = {"Wine quality (Low)": "s", "Breast cancer (Low)": "^", "Digits (Low)": "o"}
    floor = None
    med = {(r["dataset"], r["t"]): r["median"] for _, r in c1_detail.iterrows()}
    positive = [m for m in med.values() if m > 0]
    floor = min(positive) / 3 if positive else 1e-4
    for ds in DATASETS_TO_RUN:
        d = DATASET_D[ds]
        obs = np.array([med.get((ds, t), np.nan) for t in ts])
        shown = np.where(obs > 0, obs, floor)
        label = ds.replace(" quality (Low)", "").replace(" (Low)", "").lower()
        ax.plot(ts, shown, marker=markers[ds], color=colors[ds], label=f"{label}, observed ($d={d}$)")
        ax.plot(ts, ts**d, ls=":", color=colors[ds], label=f"$t^{{{d}}}$")
        for t, o in zip(ts, obs, strict=True):
            if o == 0:
                ax.annotate(f"recall $= 0$ at $t={t}$", xy=(t, floor), xytext=(t + 0.035, floor * 2.6), fontsize=9, arrowprops={"arrowstyle": "->", "lw": 0.9})
    ax.set_yscale("log")
    ax.set_xlim(1.005, 0.79)  # t decreasing rightwards, as in rq2_compounding
    ax.set_xticks(ts, [f"{t:.2f}" for t in ts])
    ax.set_xlabel("relaxation threshold $t$")
    ax.set_ylabel("median dense recall of leaf members")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_compounding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver.                                                                      #
# --------------------------------------------------------------------------- #


def run_experiment() -> dict[str, pd.DataFrame]:
    cells = [(ds, b) for ds in DATASETS_TO_RUN for b in range(N_BUILDS)]
    datasets = {name: prepare_dataset(name) for name in DATASETS_TO_RUN}
    for name, (df, fc, y) in datasets.items():
        console.print(f"  loaded [bold]{name}[/]: {len(df)} rows, d={len(fc)}, {len(np.unique(y))} classes")
        assert len(fc) == DATASET_D[name], f"{name}: d={len(fc)} != design {DATASET_D[name]}"

    buckets: dict[str, list[dict]] = {"shape": [], "nodes": [], "predicates": [], "c2": [], "purity": []}
    progress = Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    )
    with progress:
        task = progress.add_task(f"Benchmark walk (jobs={PARALLEL_JOBS})", total=len(cells))
        jobs = (delayed(run_build)(ds, datasets[ds], b) for ds, b in cells)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            for result in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
                for key in buckets:
                    buckets[key].extend(result[key])
                progress.advance(task)

    return {
        "shape": pd.DataFrame(buckets["shape"]).sort_values(["dataset", "build"]).reset_index(drop=True),
        "nodes": pd.DataFrame(buckets["nodes"]),
        "predicates": pd.DataFrame(buckets["predicates"]),
        "c2": pd.DataFrame(buckets["c2"]),
        "purity": pd.DataFrame(buckets["purity"]),
    }


def summarise(frames: dict[str, pd.DataFrame], out_dir: Path) -> str:
    shape, nodes, predicates, c2, purity = (frames[k] for k in ["shape", "nodes", "predicates", "c2", "purity"])

    # ---- Table 6.2 cells (§5) -------------------------------------------- #
    leaf_scores = nodes[nodes["is_leaf"]]
    per_build = build_leaf_means(leaf_scores, ["trustworthiness", "continuity"])
    per_build.to_csv(out_dir / "leaf_means_per_build.csv", index=False)

    table_rows: list[dict] = []
    for ds in DATASETS_TO_RUN:
        s = shape[shape["dataset"] == ds]
        pb = per_build[per_build["dataset"] == ds]
        depth_cells = [f"{int(r.realized_depth)}{'†' if r.cap_censored else ''}" for r in s.itertuples()]
        depth_med = f"{int(np.median(s['realized_depth']))}{'†' if s['cap_censored'].any() else ''}"
        table_rows.append(
            {
                "dataset": ds,
                "depth_median": depth_med,
                "depth_builds": " ".join(depth_cells),
                "n_leaves": _med_range(s["n_leaves"], "{:.0f}"),
                "noise_fraction": _med_range(s["noise_fraction"], "{:.2f}"),
                "median_leaf_size": _med_range(s["median_leaf_size"], "{:.0f}"),
                "mean_trust": _med_range(pb["trustworthiness_mean"]),
                "mean_cont": _med_range(pb["continuity_mean"]),
                "degenerate_builds": int(s["degenerate"].sum()),
            }
        )
    table = pd.DataFrame(table_rows)
    table.to_csv(out_dir / "table_6_2_cells.csv", index=False)

    # ---- example predicates (§5 fixed rule) ------------------------------- #
    examples = {ds: pick_example_leaf(shape, predicates, ds) for ds in DATASETS_TO_RUN}
    with (out_dir / "example_predicates.json").open("w") as fh:
        json.dump(examples, fh, indent=2, default=str)

    # ---- checks ----------------------------------------------------------- #
    c1_lines, c1_detail = check_c1(predicates)
    c1_detail.to_csv(out_dir / "c1_recall_detail.csv", index=False)
    c2_lines, c2_per_ds = check_c2(c2)
    c2_per_ds.to_csv(out_dir / "c2_per_dataset.csv", index=False)
    verdicts = c1_lines + c2_lines + check_c3(shape) + check_c4(predicates)
    pd.DataFrame(verdicts).to_csv(out_dir / "consistency.csv", index=False)

    # ---- D4 association --------------------------------------------------- #
    # D4 association: ρ with n only — NO p-value (C2's Wilcoxon is the experiment's
    # only p-value, design §5/§7; adversarial review 2026-07-28 finding 1).
    d4_lines = []
    for ds in DATASETS_TO_RUN:
        sub = purity[(purity["dataset"] == ds)].dropna(subset=["trustworthiness", "continuity", "purity"])
        if len(sub) >= 3:
            tc = (sub["trustworthiness"] + sub["continuity"]) / 2
            rho, _ = spearmanr(tc, sub["purity"])
            d4_lines.append({"dataset": ds, "spearman_rho": float(rho), "n_leaves": len(sub), "median_purity": float(sub["purity"].median()), "median_enrichment": float(sub["enrichment"].median())})
    d4 = pd.DataFrame(d4_lines)
    d4.to_csv(out_dir / "d4_purity_association.csv", index=False)

    # ---- figures ----------------------------------------------------------- #
    fig_shape_faithfulness(nodes, out_dir)
    fig_compounding(c1_detail, out_dir)

    # ---- console + summary.md --------------------------------------------- #
    rt = Table(title="Table 6.2 cells — median (min–max) over 5 builds")
    for col in table.columns:
        rt.add_column(col)
    for _, r in table.iterrows():
        rt.add_row(*[str(v) for v in r])
    console.print(rt)
    vt = Table(title="Consistency checks")
    for col in ["check", "prediction", "observed", "verdict"]:
        vt.add_column(col, overflow="fold")
    for v in verdicts:
        colour = {"PASS": "green", "PASS-WITH-EXCURSION": "yellow"}.get(v["verdict"], "red")
        vt.add_row(v["check"], v["prediction"], v["observed"], f"[{colour}]{v['verdict']}[/]")
    console.print(vt)

    lines = ["# Benchmark workflow walk — run summary", "", "## Table 6.2 cells", "", table.to_markdown(index=False), "", "## Consistency checks", ""]
    for v in verdicts:
        lines.append(f"- **{v['check']}** — {v['verdict']}. Prediction: {v['prediction']}. Observed: {v['observed']}.")
    lines += ["", "## D4 purity association (Spearman ρ, leaf mean T&C vs purity)", "", d4.to_markdown(index=False) if not d4.empty else "(insufficient records)", ""]
    for ds, ex in examples.items():
        if ex is None:
            continue
        lines.append(f"### Example leaf predicate — {ds} (build {ex['build']}, leaf {ex['leaf_id']}, n={ex['leaf_n']})")
        for arm in ["strict", "relaxed"]:
            rec = ex[arm]
            clauses = [c for c in json.loads(rec["clauses_json"]) if c["in_predicate"]]
            lines.append(f"- {arm} (t={rec['t']}): length {rec['length']}, F1 {rec['f1']:.3f}, coverage {rec['coverage']:.3f}")
            for c in sorted(clauses, key=lambda c: (c["step"] is None, c["step"])):
                lines.append(f"    - `{c['feature']}` ∈ [{c['sel_min']:.2f}, {c['sel_max']:.2f}]")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> None:
    console.rule("[bold]§6.5 benchmark workflow walk")
    console.print(f"datasets={DATASETS_TO_RUN}  builds={N_BUILDS}  cap={HIER_LAYERS_CAP}  t={T_GRID}  split={TAIL_SPLIT}\n")
    frames = run_experiment()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    frames["shape"].to_csv(out_dir / "tree_shape.csv", index=False)
    frames["nodes"].to_csv(out_dir / "node_scores.csv", index=False)
    frames["predicates"].to_csv(out_dir / "leaf_predicates.csv", index=False)
    frames["c2"].to_csv(out_dir / "c2_paired.csv", index=False)
    frames["purity"].to_csv(out_dir / "leaf_purity.csv", index=False)

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "timestamp_utc": timestamp,
                "design_doc": "src_research/EXPERIMENT_benchmark_workflow.md",
                "datasets": DATASETS_TO_RUN,
                "n_builds": N_BUILDS,
                "hierarchical_layers": HIER_LAYERS_CAP,
                "t_grid": T_GRID,
                "tail_split": TAIL_SPLIT,
                "min_selection": MIN_SELECTION,
                "c2_min_n": C2_MIN_N,
                "versions": {
                    "python": platform.python_version(),
                    **{p: pkg_version(p) for p in ["numpy", "pandas", "scikit-learn", "umap-learn", "hdbscan", "zadu", "scipy"]},
                },
                "platform": platform.platform(),
            },
            indent=2,
        )
    )

    summarise(frames, out_dir)
    console.print(f"\n[bold green]Done.[/] Results + figures in [underline]{out_dir}[/]")


if __name__ == "__main__":
    main()

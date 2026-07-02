"""RQ1 experiment: hierarchical vs. flat embeddings (thesis).

Implements the design in ``src_research/EXPERIMENT_hierarchical_vs_flat.md``. It tests
whether the project's recursive, density-based hierarchical decomposition surfaces and
isolates subspace structure better than a single global ("flat") projection, along two
separable claims:

    H1a - faithfulness of local views (no ground truth needed).  For each hierarchy
        region, is the leaf's *local* re-embedding more faithful to that region's
        internal structure than the *global* embedding restricted to the same rows?
        A paired comparison: same region, same points, same neighbourhood ``k`` - so the
        only thing that varies is local-vs-global projection (region size cannot confound
        it). Scored with the project's own ``_score_node`` (ZADU trustworthiness,
        continuity, MRRE, stress).

    H1b - discovery of nested structure (needs ground-truth labels).  Does the recursive
        HDBSCAN partition (the tree's leaves) recover a known partition better than a
        single non-recursive HDBSCAN run on the same pre-processed space? Same algorithm
        with/without recursion, so recursion is the isolated variable. Scored with
        ARI / NMI plus a "merge diagnostic".

Fairness is by construction (design SS2, SS7): both conditions reuse the exact same
embedding code path (``_embed_original``) and the exact same scorer (``_score_node``),
differing only in input scope (one region vs. the whole dataset). Pairing fixes ``n`` and
therefore ``k`` per region (asserted).

Two robustness factors from the design are exposed in the CONFIG block:
    * DR method as a factor (SS8.2): repeat over {UMAP, t-SNE, PCA}.
    * Method-neutral regions for H1a (SS8.1): also score regions defined by ground-truth
      classes (independent of the hierarchy), answering the "self-selected regions" caveat.
    * Depth sweep (SS8.3): ``HIER_LAYERS`` is a list.

Reproducibility caveat: ``reduce_dimensionality`` does not thread a seed into UMAP / t-SNE,
so those embeddings are stochastic. The subsample is seeded (fixed across runs), so the
``SEEDS`` loop functions as *replicates* that capture embedding variance - the unit of the
paired H1a test is a (region x seed) pair. PCA replicates are ~identical (deterministic).

Outputs (written to ``outputs/experiments/<timestamp>/``):
    * h1a_regions.csv   - one row per (dataset, method, seed, region, condition, region_def).
    * h1a_summary.csv   - per (dataset, method, region_def, metric): median delta, win rate,
                          Wilcoxon p, rank-biserial effect size.
    * h1b_recovery.csv  - per (dataset, method, seed): ari/nmi for both, merge diagnostic,
                          noise fractions.
    * plots/*.png       - paired-delta distributions (H1a), ARI hier-vs-flat bars (H1b).

Run with::

    uv run python -m src_research.hierarchical_vs_flat
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_config
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from streamlit import logger as streamlit_logger

from src.analysis.analysis_routine import AnalysisObject, ExplorationObject, _embed_original
from src.analysis.clustering import compute_clusters
from src.analysis.dim_reducer import reduce_dimensionality
from src.evaluation.evaluate import _score_node, start_evaluation
from src.types import Config
from src.ui.data import DATASETS
from src.ui.state import init_state

# --------------------------------------------------------------------------- #
# CONFIG - edit this block to change the experiment.                          #
# --------------------------------------------------------------------------- #

SEED = 42  # seeds the subsample (fixed across replicates)
SUBSAMPLE_CAP = 1000  # cap rows per dataset (seeded) to keep t-SNE/UMAP + O(n^2) measures tractable
SEEDS = list(range(5))  # replicate ids - capture stochastic-embedding variance (unit of the paired test)
DR_METHODS = ["UMAP", "t-SNE", "PCA", "MDS"]  # DR method as a factor (SS8.2)
HIER_LAYERS = [2]  # depth sweep (SS8.3); a single value = headline. Default tree depth.
RUN_NEUTRAL_REGIONS = True  # SS8.1 method-neutral H1a variant (regions = ground-truth classes)

DATASETS_TO_RUN = [
    "Concentric rings (Low)",  # non-convex nested density - the case the hierarchy should win (H1a + H1b)
    "Wine quality (Low)",  # interpretable low-dim tabular (H1a + H1b)
    "Breast cancer (Low)",  # moderate dim (H1a + H1b)
    "Digits (Low)",  # manifold structure, higher dim (H1a + H1b)
    "Swiss roll (Low)",  # continuous manifold, no classes (H1a only)
]

PARALLEL_JOBS = -1  # grid cells run concurrently; 1 = serial, -1 = all cores
OUTPUT_ROOT = Path("outputs/experiments")

# Metrics scored per region (keys of NodeScores we carry into the CSVs).
H1A_METRIC_KEYS = ["trustworthiness", "continuity", "mrre_false", "mrre_missing", "stress"]
# Higher-is-better metrics: T&C improve when up, MRRE/stress are errors (improve when down).
HIGHER_IS_BETTER = {"trustworthiness": True, "continuity": True, "mrre_false": False, "mrre_missing": False, "stress": False}
PRIMARY_METRICS = ["trustworthiness", "continuity"]  # pre-registered primary (SS9)

# --------------------------------------------------------------------------- #
# Quiet third-party noise so the rich output stays readable.                   #
# --------------------------------------------------------------------------- #


def _silence_noise() -> None:
    """Mute third-party chatter. Re-asserted in each worker process, since loky children
    re-import the module and Streamlit/Optuna reconfigure their loggers lazily on first use.
    Identical to the tuning harness; see its docstring for the Streamlit-logger rationale.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    streamlit_logger.set_log_level("error")
    warnings.filterwarnings("ignore")


_silence_noise()
console = Console()


# --------------------------------------------------------------------------- #
# Dataset preparation. Returns the RAW df (start_evaluation standardises it).  #
# --------------------------------------------------------------------------- #


def prepare_dataset(display_name: str) -> tuple[pd.DataFrame, list[str], np.ndarray | None]:
    """Load a registry dataset; return (df, feature_cols, y).

    Label conventions (mirrors the tuning harness ``prepare_dataset``):
      * one-hot ``target_*`` columns -> y = argmax, features = the rest;
      * wine quality (``is_red``)    -> y = is_red (binary), drop quality/is_red from features;
      * swiss roll (``manifold_position``) -> continuous, y = None.
    Seeded subsample to ``SUBSAMPLE_CAP`` on the df; the index is reset so a node's
    ``row_indices`` (positional 0..n-1 into this df) line up with the standardised matrix.
    """
    df = DATASETS[display_name]()
    target_cols = [c for c in df.columns if c.startswith("target_")]

    if target_cols:
        feature_cols = [c for c in df.columns if c != "row_id" and not c.startswith("target_")]
        y = df[target_cols].to_numpy().argmax(axis=1)
    elif "is_red" in df.columns:  # wine quality
        feature_cols = [c for c in df.columns if c not in {"row_id", "is_red", "quality"}]
        y = df["is_red"].to_numpy().astype(int)
    elif "manifold_position" in df.columns:  # swiss roll - continuous, no classes
        feature_cols = [c for c in df.columns if c not in {"row_id", "manifold_position"}]
        y = None
    else:
        feature_cols = [c for c in df.columns if c != "row_id"]
        y = None

    if len(df) > SUBSAMPLE_CAP:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(df), SUBSAMPLE_CAP, replace=False)
        df = df.iloc[idx]
        y = y[idx] if y is not None else None

    df = df.reset_index(drop=True)
    return df, feature_cols, y


# --------------------------------------------------------------------------- #
# Tree helpers.                                                                #
# --------------------------------------------------------------------------- #


def standardised_X(df: pd.DataFrame, feature_cols: list[str], tree: AnalysisObject) -> np.ndarray:
    """The exact feature space the tree's leaves and ``_score_node`` operate in: the root
    scaler applied to the raw features (raw features if ``normalize`` is off)."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    scaler = tree.get("scaler")
    return scaler.transform(X) if scaler is not None else X


def collect_leaves(tree: AnalysisObject) -> list[ExplorationObject]:
    """Walk ``next_object_layer`` to a flat list of leaf (ExplorationObject) nodes."""
    if "is_leaf" in tree:
        return [tree]  # type: ignore[list-item]
    leaves: list[ExplorationObject] = []
    for child in tree.get("next_object_layer") or []:
        leaves.extend(collect_leaves(child))
    return leaves


def clustering_space(X_all: np.ndarray, config: Config) -> np.ndarray:
    """Mirror the tree's optional UMAP pre-reduction (analysis_routine.compute_analysis_tree)
    so the flat HDBSCAN sees the same representation the recursive one starts from."""
    umap_n_comp = config["hclust_umap_n_components"]
    if umap_n_comp and umap_n_comp < X_all.shape[1]:
        n_comp = min(umap_n_comp, X_all.shape[1], len(X_all) - 1)
        config["hclust_umap_n_components"] = n_comp
        config["umap_n_neighbors"] = min(config["umap_n_neighbors"], len(X_all) - 1)
        return reduce_dimensionality("UMAP", X=X_all, n_components=n_comp, config=config)
    return X_all


def leaf_partition(leaves: list[ExplorationObject], n: int) -> np.ndarray:
    """Point -> leaf id. Rows absent from every leaf (dropped as noise mid-recursion) = -1."""
    part = np.full(n, -1, dtype=int)
    for leaf_id, leaf in enumerate(leaves):
        part[leaf["row_indices"]] = leaf_id
    return part


def merge_diagnostic(part_f: np.ndarray, part_h: np.ndarray, y: np.ndarray) -> int:
    """Structure the flat view merged but the hierarchy separated (design SS4).

    For each flat cluster, count the distinct ground-truth classes it contains that end up
    in *different* hierarchical leaves; sum over flat clusters. Noise (-1) is ignored on
    both axes.
    """
    total = 0
    for fc in np.unique(part_f):
        if fc == -1:
            continue
        in_fc = part_f == fc
        # classes present in this flat cluster that the hierarchy splits across >1 leaf
        for cls in np.unique(y[in_fc]):
            leaves_for_cls = np.unique(part_h[in_fc & (y == cls) & (part_h != -1)])
            if len(leaves_for_cls) > 1:
                total += 1
    return total


# --------------------------------------------------------------------------- #
# Scoring.                                                                     #
# --------------------------------------------------------------------------- #


def _score(X_r: np.ndarray, emb: np.ndarray) -> dict:
    """Score one region with the project's own scorer. ``emb`` is set to None when too small
    to embed (matches ``_attach_scores``), so degenerate regions return all-None cleanly."""
    e = None if emb.shape[0] < 2 or emb.shape[1] < 2 else emb
    return dict(_score_node(X_r, e, None))


def _region_rows(cell: dict, region_id: str, region_def: str, n: int, X_r: np.ndarray, emb_h: np.ndarray, emb_f: np.ndarray) -> list[dict]:
    """Paired (hierarchical, flat) rows for one region. Asserts the neighbourhood k matches:
    equal n => equal k, the invariant that makes the comparison fair."""
    s_h = _score(X_r, emb_h)
    s_f = _score(X_r, emb_f)
    assert s_h["k"] == s_f["k"], f"k mismatch ({region_id}): hier={s_h['k']} flat={s_f['k']}"
    rows = []
    for cond, s in (("hier", s_h), ("flat", s_f)):
        rows.append({**cell, "region": region_id, "region_def": region_def, "cond": cond, "n": n, "k": s["k"], **{m: s[m] for m in H1A_METRIC_KEYS}})
    return rows


# --------------------------------------------------------------------------- #
# Running one grid cell (dataset x method x seed x layers).                    #
# --------------------------------------------------------------------------- #


def run_cell(
    dataset: str,
    data: tuple[pd.DataFrame, list[str], np.ndarray | None],
    dr_method: str,
    seed: int,
    layers: int,
) -> tuple[list[dict], list[dict]]:
    """Build the hierarchy + the flat global embedding once, then run H1a (paired regions)
    and H1b (structure recovery). Returns (h1a_rows, h1b_rows).

    ``data`` is the preloaded ``(df, feature_cols, y)`` for this dataset (prepared once in
    the driver, not per cell) so workers neither reload nor re-subsample it."""
    _silence_noise()  # re-assert before init_state touches streamlit in a worker process
    cfg: Config = init_state(init_streamlit=False)
    cfg["method"] = dr_method
    cfg["hierarchical_layers"] = layers

    df, feature_cols, y = data
    cell = {"dataset": dataset, "method": dr_method, "seed": seed, "layers": layers}

    # ---- hierarchical condition: tree + per-node scores (cfg is standardised inside) ----
    tree = start_evaluation(df, feature_cols, cfg)
    X_all = standardised_X(df, feature_cols, tree)
    leaves = collect_leaves(tree)

    # ---- flat condition: one global 2D embedding via the identical code path ----
    E_global, _ = _embed_original(X_all, cfg)

    h1a: list[dict] = []

    # H1a primary: hierarchy-leaf regions (paired, same points, same k).
    for i, leaf in enumerate(leaves):
        idx = leaf["row_indices"]
        h1a.extend(_region_rows(cell, f"leaf{i}", "hier_leaf", len(idx), X_all[idx], leaf["embedding_original"], E_global[idx]))

    # H1a robustness: method-neutral regions = ground-truth classes (SS8.1).
    if RUN_NEUTRAL_REGIONS and y is not None:
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            emb_h, _ = _embed_original(X_all[idx], cfg)  # local re-embed of the class
            h1a.extend(_region_rows(cell, f"class{cls}", "gt_class", len(idx), X_all[idx], emb_h, E_global[idx]))

    # ---- H1b: structure recovery (needs labels) ----
    h1b: list[dict] = []
    if y is not None:
        part_h = leaf_partition(leaves, len(df))
        part_f = np.asarray(compute_clusters(clustering_space(X_all, cfg), method="HDBSCAN", config=cfg)[0])
        # Exclude noise consistently: score only rows non-noise in BOTH partitions.
        keep = (part_h != -1) & (part_f != -1)
        ari_h = nmi_h = ari_f = nmi_f = None
        if keep.sum() >= 2 and len(np.unique(y[keep])) >= 2:
            ari_h = float(adjusted_rand_score(y[keep], part_h[keep]))
            nmi_h = float(normalized_mutual_info_score(y[keep], part_h[keep]))
            ari_f = float(adjusted_rand_score(y[keep], part_f[keep]))
            nmi_f = float(normalized_mutual_info_score(y[keep], part_f[keep]))
        h1b.append(
            {
                **cell,
                "ari_h": ari_h,
                "nmi_h": nmi_h,
                "ari_f": ari_f,
                "nmi_f": nmi_f,
                "merges": merge_diagnostic(part_f, part_h, y),
                "noise_h": float(np.mean(part_h == -1)),
                "noise_f": float(np.mean(part_f == -1)),
            }
        )

    return h1a, h1b


# --------------------------------------------------------------------------- #
# Grid driver.                                                                 #
# --------------------------------------------------------------------------- #


def build_cells() -> list[tuple[str, str, int, int]]:
    """Enumerate (dataset, dr_method, seed, layers) grid cells - all independent."""
    return [(ds, dr, seed, layers) for ds in DATASETS_TO_RUN for dr in DR_METHODS for seed in SEEDS for layers in HIER_LAYERS]


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full grid, returning (h1a_df, h1b_df)."""
    cells = build_cells()
    # Prepare each dataset once (load + seeded subsample), not per cell: avoids redundant
    # reloads and concentrates the streamlit cache warnings in the driver process.
    datasets = {name: prepare_dataset(name) for name in DATASETS_TO_RUN}
    for name, (df, _, y) in datasets.items():
        note = "no ground truth" if y is None else f"{len(set(y.tolist()))} classes"
        console.print(f"  loaded [bold]{name}[/]: {len(df)} rows, {note}")
    h1a_rows: list[dict] = []
    h1b_rows: list[dict] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task(f"Running grid (jobs={PARALLEL_JOBS})", total=len(cells))
        if PARALLEL_JOBS == 1:
            for ds, dr, seed, layers in cells:
                progress.update(task, description=f"{ds} / {dr} / s{seed} / L{layers}"[:48])
                a, b = run_cell(ds, datasets[ds], dr, seed, layers)
                h1a_rows.extend(a)
                h1b_rows.extend(b)
                progress.advance(task)
        else:
            # Cells are independent -> worker processes. inner_max_num_threads=1 stops each
            # worker's UMAP/BLAS pools oversubscribing the cores. Generator keeps the bar live.
            jobs = (delayed(run_cell)(ds, datasets[ds], dr, seed, layers) for ds, dr, seed, layers in cells)
            with parallel_config(backend="loky", inner_max_num_threads=1):
                for a, b in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
                    h1a_rows.extend(a)
                    h1b_rows.extend(b)
                    progress.advance(task)

    return pd.DataFrame(h1a_rows), pd.DataFrame(h1b_rows)


# --------------------------------------------------------------------------- #
# Statistics (design SS9).                                                     #
# --------------------------------------------------------------------------- #


def _paired_deltas(h1a: pd.DataFrame, dataset: str, method: str, region_def: str, metric: str) -> np.ndarray:
    """Per-(region x seed) delta = hier - flat for one metric, dropping pairs with any NaN."""
    sub = h1a[(h1a["dataset"] == dataset) & (h1a["method"] == method) & (h1a["region_def"] == region_def)]
    wide = sub.pivot_table(index=["seed", "region"], columns="cond", values=metric)
    if "hier" not in wide or "flat" not in wide:
        return np.array([])
    pair = wide[["hier", "flat"]].dropna()
    return (pair["hier"] - pair["flat"]).to_numpy()


def h1a_summary(h1a: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method, region_def, metric): median delta, win rate, Wilcoxon p,
    rank-biserial effect size. 'Win' is direction-aware (lower-is-better for MRRE/stress).
    Regions are NOT pooled across datasets (scales differ)."""
    rows: list[dict] = []
    for (dataset, method, region_def), _ in h1a.groupby(["dataset", "method", "region_def"]):
        for metric in H1A_METRIC_KEYS:
            d = _paired_deltas(h1a, dataset, method, region_def, metric)
            if d.size == 0:
                continue
            higher = HIGHER_IS_BETTER[metric]
            wins = d > 0 if higher else d < 0  # "hierarchical better than flat"
            # Wilcoxon needs at least one non-zero delta and n>=1; guard degenerate cases.
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if nz.size >= 1 and not np.allclose(d, 0) else float("nan")
            # matched-pairs rank-biserial = (#favourable - #unfavourable) / #non-zero pairs
            rbc = float((np.sum(wins) - np.sum(~wins & (d != 0))) / nz.size) if nz.size else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "region_def": region_def,
                    "metric": metric,
                    "n_pairs": int(d.size),
                    "median_delta": float(np.median(d)),
                    "win_rate": float(np.mean(wins)),
                    "wilcoxon_p": p,
                    "rank_biserial": rbc,
                    "primary": metric in PRIMARY_METRICS,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Output: CSVs, plots, console tables.                                         #
# --------------------------------------------------------------------------- #


def save_outputs(h1a: pd.DataFrame, h1b: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h1a.to_csv(out_dir / "h1a_regions.csv", index=False)
    summary.to_csv(out_dir / "h1a_summary.csv", index=False)
    h1b.to_csv(out_dir / "h1b_recovery.csv", index=False)


def make_plots(h1a: pd.DataFrame, h1b: pd.DataFrame, out_dir: Path) -> None:
    """Thesis figures: paired-delta distributions (H1a) and ARI hier-vs-flat bars (H1b)."""
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # fig:rq1-h1a - paired-delta (hier - flat) distributions per dataset, primary metrics,
    # leaf regions. A box centred above 0 => the local view is more faithful.
    leaf = h1a[h1a["region_def"] == "hier_leaf"]
    delta_rows: list[dict] = []
    for (dataset, method), _ in leaf.groupby(["dataset", "method"]):
        for metric in PRIMARY_METRICS:
            for d in _paired_deltas(leaf, dataset, method, "hier_leaf", metric):
                delta_rows.append({"dataset": dataset, "method": method, "metric": metric, "delta": d})
    dd = pd.DataFrame(delta_rows)
    if not dd.empty:
        g = sns.catplot(data=dd, x="method", y="delta", col="dataset", row="metric", kind="box", height=3, aspect=1.1, sharey=False)
        for ax in g.axes.flat:
            ax.axhline(0, color="grey", lw=0.8, ls="--")
        g.figure.suptitle("H1a: paired delta (hierarchical local - flat global), leaf regions", y=1.02)
        g.savefig(plots / "h1a_paired_deltas.png", dpi=150, bbox_inches="tight")
        plt.close(g.figure)

    # fig:rq1-h1b - mean ARI hierarchical vs flat per dataset (averaged over seeds), per method.
    if not h1b.empty and {"ari_h", "ari_f"} <= set(h1b.columns):
        long = h1b.melt(id_vars=["dataset", "method"], value_vars=["ari_h", "ari_f"], var_name="condition", value_name="ari")
        long["condition"] = long["condition"].map({"ari_h": "hierarchical", "ari_f": "flat"})
        g = sns.catplot(data=long, x="dataset", y="ari", hue="condition", col="method", kind="bar", height=4, aspect=1.0, errorbar="sd")
        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=30)
        g.figure.suptitle("H1b: ground-truth recovery (ARI), hierarchical vs flat", y=1.02)
        g.savefig(plots / "h1b_ari.png", dpi=150, bbox_inches="tight")
        plt.close(g.figure)


def print_summary(summary: pd.DataFrame, h1b: pd.DataFrame) -> None:
    """Render the primary H1a summary (T&C) and the H1b recovery means."""
    prim = summary[summary["primary"]] if not summary.empty else summary
    if not prim.empty:
        table = Table(title="H1a (primary: trustworthiness & continuity) - delta = hierarchical - flat")
        for col in ["dataset", "method", "region_def", "metric", "median_delta", "win_rate", "wilcoxon_p"]:
            table.add_column(col, justify="right" if col in {"median_delta", "win_rate", "wilcoxon_p"} else "left")
        for _, r in prim.sort_values(["dataset", "method", "region_def", "metric"]).iterrows():
            md = r["median_delta"]
            md_str = f"[green]{md:+.3f}[/]" if md > 0 else (f"[red]{md:+.3f}[/]" if md < 0 else f"{md:+.3f}")
            p = r["wilcoxon_p"]
            table.add_row(r["dataset"], r["method"], r["region_def"], r["metric"], md_str, f"{r['win_rate']:.2f}", "-" if pd.isna(p) else f"{p:.3g}")
        console.print(table)

    if not h1b.empty:
        agg = h1b.groupby(["dataset", "method"])[["ari_h", "ari_f"]].mean().reset_index()
        table = Table(title="H1b - mean ARI vs ground truth (hierarchical vs flat)")
        for col in ["dataset", "method", "ari_h", "ari_f"]:
            table.add_column(col, justify="right" if col.startswith("ari") else "left")
        for _, r in agg.iterrows():
            ah, af = r["ari_h"], r["ari_f"]
            ah_str = f"[green]{ah:.3f}[/]" if pd.notna(ah) and pd.notna(af) and ah > af else (f"{ah:.3f}" if pd.notna(ah) else "-")
            table.add_row(r["dataset"], r["method"], ah_str, "-" if pd.isna(af) else f"{af:.3f}")
        console.print(table)


def main() -> None:
    np.random.seed(SEED)
    console.rule("[bold]RQ1: hierarchical vs flat embeddings")
    console.print(
        f"datasets={len(DATASETS_TO_RUN)}  DR={DR_METHODS}  seeds={len(SEEDS)}  layers={HIER_LAYERS}  neutral_regions={RUN_NEUTRAL_REGIONS}\n"
    )

    h1a, h1b = run_experiment()
    summary = h1a_summary(h1a)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    save_outputs(h1a, h1b, summary, out_dir)
    make_plots(h1a, h1b, out_dir)

    console.print()
    print_summary(summary, h1b)
    console.print(f"\n[bold green]Done.[/] Results + plots in [underline]{out_dir}[/]")


if __name__ == "__main__":
    main()

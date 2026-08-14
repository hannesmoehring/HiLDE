"""Re-derivation 3: the internal-vs-external figure and its correlation (H12e).

The shipped figure joined `ari_base` in from the ``sampler == "none"`` rows, which exist
only for untunable cells and therefore never for Track A, then applied ``.fillna(0)``. Its
y axis is labelled "ARI change vs baseline" and actually carries **raw tuned ARI**.

**What can be re-derived, and what cannot.** The corrected quantity needs the ARI the
pipeline scored under the *default* configuration. That number was computed at run time
(``evaluate_baseline`` returns it) and then discarded: ``results_summary.csv`` keeps only
the baseline's DBCV in its ``baseline`` column, and ``trials.csv`` carries no external
metric at all — its columns are track, dataset, dr_method, cluster_method, objective,
sampler, trial, value, best_so_far. The fix records ``ari_base`` on every summary row, so
future runs carry it, but for these two shipped runs the baseline ARI does not exist on
disk in any form.

So this module re-derives the *plotted* relationship exactly, states the corrected label
for it, and reports the corrected r for the quantity that is recoverable — and refuses to
manufacture the one that is not. It does not rerun the experiment.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src_research.rederive import TUNING_RUNS, fmt, out_dir, run_dir, write_deltas

BASE_JOIN_KEYS = ["track", "dataset", "dr_method", "cluster_method"]


def _shipped_merge(summary: pd.DataFrame) -> pd.DataFrame:
    """Exactly what make_plots used to do, so the claim about it is checkable."""
    a = summary[
        (summary["sampler"] == "TPE") & summary["tunable"] & (summary["track"] == "A")
    ]
    a_ext = a.dropna(subset=["ari"])
    merged = a_ext.merge(
        summary[summary["sampler"] == "none"][[*BASE_JOIN_KEYS, "ari"]],
        on=BASE_JOIN_KEYS,
        how="left",
        suffixes=("", "_base"),
    )
    merged["ari_gain_shipped"] = merged["ari"] - merged["ari_base"].fillna(0)
    return merged


def rederive_run(run: str) -> dict:
    summary = pd.read_csv(run_dir(run) / "results_summary.csv")
    trials = pd.read_csv(run_dir(run) / "trials.csv")
    merged = _shipped_merge(summary)

    n_base_rows = int((summary["sampler"] == "none").sum())
    n_base_trackA = int(
        ((summary["sampler"] == "none") & (summary["track"] == "A")).sum()
    )
    n_base_ari = int(summary.loc[summary["sampler"] == "none", "ari"].notna().sum())
    all_nan = bool(merged["ari_base"].isna().all())
    identical = bool((merged["ari_gain_shipped"] - merged["ari"]).abs().max() == 0)

    # The relationship the shipped figure actually drew, with the label it should have had.
    r_plotted = float(merged["gain"].corr(merged["ari_gain_shipped"]))
    r_raw_ari = float(merged["gain"].corr(merged["ari"]))

    corrected = merged[
        [
            *BASE_JOIN_KEYS,
            "sampler",
            "baseline",
            "best",
            "gain",
            "ari",
            "ari_base",
            "ari_gain_shipped",
        ]
    ].copy()
    corrected = corrected.rename(
        columns={"ari": "ari_tuned", "ari_gain_shipped": "y_actually_plotted"}
    )
    corrected["ari_gain_corrected"] = (
        pd.NA
    )  # not derivable: baseline ARI was never persisted
    corrected.to_csv(out_dir(run) / "internal_vs_external_records.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sns.scatterplot(
        data=merged, x="gain", y="ari", hue="dataset", style="dr_method", ax=ax
    )
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("DBCV gain: best tuned DBCV - default-config DBCV (original space)")
    ax.set_ylabel("ARI of the best trial's clustering (NOT a change vs baseline)")
    ax.set_title(
        "What the shipped figure plots, correctly labelled\n"
        f"Track A, TPE, n={len(merged)} cells, Pearson r={r_raw_ari:.3f}  ·  baseline ARI not on disk, so no gain axis is derivable"
    )
    fig.tight_layout()
    fig.savefig(out_dir(run) / "internal_vs_external_corrected_labels.png", dpi=150)
    plt.close(fig)

    sections = [
        "## 3. internal_vs_external — the join, the label, and the number that is missing (H12e)",
        "",
        f"Source: `results_summary.csv` ({len(summary)} rows) and `trials.csv` ({len(trials)} rows).",
        "",
        "### 3a. The join returned nothing, and fillna(0) hid that",
        "",
        f'- `sampler == "none"` rows in this run: **{n_base_rows}**, of which **{n_base_trackA}** are Track A '
        f"and **{n_base_ari}** carry a non-null `ari`.",
        f"- Track-A TPE rows entering the figure: **{len(merged)}**.",
        f"- `ari_base` after the join is entirely null: **{all_nan}**.",
        f"- Therefore the plotted y equals raw tuned ARI exactly: **{identical}** "
        "(`ari - ari_base.fillna(0)` == `ari`).",
        "",
        "A cell whose tuning *lowered* ARI plotted above zero, because zero was never the baseline.",
        "",
        "### 3b. Corrected r",
        "",
        "| quantity | Pearson r vs DBCV gain | derivable from shipped CSVs? |",
        "|---|---|---|",
        f"| ARI of the best trial (what the figure actually plots) | **{fmt(r_raw_ari)}** | yes |",
        f"| the shipped y series, verbatim | {fmt(r_plotted)} | yes (identical to the row above) |",
        "| ARI change vs the default configuration (what the axis claims) | **not derivable** | no — see 3c |",
        "",
        f"Re-plotted with honest labels: `{out_dir(run).name}/internal_vs_external_corrected_labels.png`.",
        "",
        "### 3c. Why the labelled quantity cannot be re-derived",
        "",
        "The corrected y needs the ARI of the pipeline under the *default* configuration.",
        "`evaluate_baseline` computed it at run time and the summary row kept only the baseline's",
        "DBCV (`baseline`); `trials.csv` carries no external metric at all — its columns are",
        f"`{', '.join(trials.columns)}`.",
        "",
        "**The baseline ARI for these two runs is not on disk in any form, so no re-derivation can",
        "produce the corrected figure.** Recomputing it would mean rerunning the pipeline, which is",
        "out of scope here and would in any case be a new measurement rather than a correction of",
        "this one. The code fix records `ari_base` and `nmi_base` on every summary row, so the next",
        "run of this harness carries what this one discarded.",
        "",
    ]
    write_deltas(run, f"Corrected internal-vs-external readout — {run}", sections)
    print(
        f"  {run}: n={len(merged)} Track-A TPE cells, r(gain, tuned ARI)={r_raw_ari:.4f}; ari_base all-null={all_nan}; corrected gain NOT derivable"
    )
    return {"run": run, "n_cells": len(merged), "r": r_raw_ari, "all_nan": all_nan}


def main() -> list[dict]:
    return [rederive_run(run) for run in TUNING_RUNS]

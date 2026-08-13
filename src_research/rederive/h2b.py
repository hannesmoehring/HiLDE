"""Re-derivation 2: RQ1-S summary re-aggregated under the H12d rules (20260628_195214).

Two aggregation defects, both fixed in ``planted_subspace_recovery`` and both re-applied
here to the shipped ``subspace_recovery.csv``:

* the ``non_nested`` control forces ``eff_rho = 1``, so its six rho levels are six
  recomputations of one condition — and the H2b nesting gap averaged those six duplicates
  against six genuinely distinct nested conditions;
* ``hier_mean`` / ``flat_mean`` / ``oracle_relative`` came from the undropped pivot while
  ``median_delta`` came from the dropna'd pairs, so the two halves of a row described
  different samples.

The per-cell measurements are untouched, so this is a pure re-aggregation.
"""

from __future__ import annotations

import pandas as pd

from src_research.planted_subspace_recovery import drop_duplicate_control_cells, summarise
from src_research.rederive import H2B_RUN, deltas_table, diff_rows, fmt, out_dir, run_dir, write_deltas

KEYS = ["rho", "nesting"]
VALUES = ["n_pairs", "hier_mean", "flat_mean", "median_delta", "win_rate", "wilcoxon_p", "rank_biserial", "oracle_relative"]


def _gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The H2b readout: the hier-minus-flat gap per nesting arm."""
    return summary.groupby("nesting")["median_delta"].agg(["mean", "median", "count"]).reset_index()


def main() -> dict:
    run = H2B_RUN
    rec = pd.read_csv(run_dir(run) / "subspace_recovery.csv")
    old = pd.read_csv(run_dir(run) / "subspace_summary.csv")

    deduped, n_dropped = drop_duplicate_control_cells(rec)
    corrected = summarise(rec)
    corrected.to_csv(out_dir(run) / "subspace_summary.csv", index=False)

    old_gap, new_gap = _gap_table(old), _gap_table(corrected)
    old_gap.to_csv(out_dir(run) / "h2b_gap_shipped.csv", index=False)
    new_gap.to_csv(out_dir(run) / "h2b_gap_corrected.csv", index=False)

    rows = diff_rows(old, corrected, KEYS, VALUES)
    gap_rows = diff_rows(old_gap, new_gap, ["nesting"], ["mean", "median", "count"])

    # How much of the grid was duplicate work, stated in cells rather than in prose.
    ctrl = rec[rec["nesting"] == "non_nested"]
    per_condition = ctrl.groupby(["rho"]).size()

    sections = [
        "## 2. subspace_summary.csv — duplicated control + mixed samples (H12d)",
        "",
        f"Source: `subspace_recovery.csv` ({len(rec)} condition records) -> "
        f"`{out_dir(run).name}/subspace_summary.csv` ({len(corrected)} rows, was {len(old)}).",
        "",
        "### 2a. The duplicated control",
        "",
        f"`non_nested` was enumerated at {ctrl['rho'].nunique()} rho levels "
        f"({sorted(ctrl['rho'].unique().tolist())}), each carrying {int(per_condition.iloc[0])} records, "
        f"but `run_cell` forces `eff_rho = 1` for that arm — so they are "
        f"{ctrl['rho'].nunique()} recomputations of one condition. "
        f"{n_dropped} of {len(rec)} recovery records ({100 * n_dropped / len(rec):.1f}%) are dropped as duplicates, "
        f"leaving {len(deduped)}.",
        "",
        "### 2b. The H2b nesting gap (the number the control exists to produce)",
        "",
        "| nesting | mean gap (shipped) | mean gap (corrected) | median (shipped) | median (corrected) | rho levels (shipped -> corrected) |",
        "|---|---|---|---|---|---|",
    ]
    for nesting in sorted(set(old_gap["nesting"]) | set(new_gap["nesting"])):
        o = old_gap[old_gap["nesting"] == nesting]
        n = new_gap[new_gap["nesting"] == nesting]
        get = lambda f, c: fmt(f[c].iloc[0]) if len(f) else "—"  # noqa: E731 - local formatting shorthand
        sections.append(
            f"| {nesting} | {get(o, 'mean')} | {get(n, 'mean')} | {get(o, 'median')} | {get(n, 'median')} | "
            f"{get(o, 'count')} -> {get(n, 'count')} |"
        )
    sections += ["", *deltas_table(gap_rows, "nesting (H2b gap)")]

    sections += [
        "### 2c. Per-(rho, nesting) summary rows",
        "",
        f"{len(rows)} entries changed (a dropped row is a changed table).",
        "",
        *deltas_table(rows, "rho · nesting"),
    ]

    # The mixed-sample defect only bites where a seed was dropped; say whether it did here.
    attrition = corrected[corrected["n_seeds"] != corrected["n_pairs"]]
    sections += [
        "### 2d. Sample attrition (the mixed-sample half of H12d)",
        "",
        (
            f"{len(attrition)} of {len(corrected)} corrected rows lost a seed to `dropna` "
            f"(n_seeds != n_pairs). Where none did, `hier_mean` / `flat_mean` were already computed "
            "over the same seeds as `median_delta` and are unchanged by this half of the fix — the "
            "guarantee is now structural rather than incidental."
            if len(attrition) == 0
            else f"{len(attrition)} of {len(corrected)} corrected rows lost at least one seed to `dropna`; "
            "for those, the shipped `hier_mean` / `flat_mean` / `oracle_relative` averaged seeds that "
            "`median_delta` excluded."
        ),
        "",
    ]
    if len(attrition):
        sections += [
            "| rho · nesting | n_seeds | n_pairs | n_oracle |",
            "|---|---|---|---|",
            *[f"| {r.rho} · {r.nesting} | {r.n_seeds} | {r.n_pairs} | {r.n_oracle} |" for r in attrition.itertuples()],
            "",
        ]

    write_deltas(run, f"Corrected RQ1-S / H2b summary — {run}", sections)
    print(f"  {run}: {len(old)} -> {len(corrected)} summary rows, {n_dropped} duplicate control records dropped, {len(rows) + len(gap_rows)} numbers changed")
    return {"run": run, "n_changed": len(rows) + len(gap_rows), "n_dropped": n_dropped}

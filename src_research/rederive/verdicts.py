"""Re-derivation 4: RQ2 verdicts with honest n columns (20260711_115849, H12c).

``verdicts.csv`` reported ``n_selections = n_pairs.max()``, i.e. seeds x selections. The
unit of analysis is the selection, and on the wine arm the seeds are rebuilds of one
dataset — the same leaves, five times over — so the column overstated the sample fivefold.

The test statistic is unchanged: the pairs entering the Wilcoxon are the same pairs. What
changes is that the sample is now described truthfully, as n_selections distinct selections
observed over n_seeds seeds. Re-derived from ``stability_records.csv`` through the fixed
``stability_summary`` / ``verdicts``, so this is a pure re-aggregation.
"""

from __future__ import annotations

import pandas as pd

from src_research.predicate_stability_analysis import stability_summary
from src_research.predicate_stability_analysis import verdicts as compute_verdicts
from src_research.rederive import STABILITY_RUN, deltas_table, diff_rows, out_dir, run_dir, write_deltas

KEYS = ["arm", "method", "split", "delta"]


def main() -> dict:
    run = STABILITY_RUN
    records = pd.read_csv(run_dir(run) / "stability_records.csv")
    old = pd.read_csv(run_dir(run) / "verdicts.csv")

    summary = stability_summary(records)
    corrected = compute_verdicts(summary)
    summary.to_csv(out_dir(run) / "stability_summary.csv", index=False)
    corrected.to_csv(out_dir(run) / "verdicts.csv", index=False)

    # The scientific verdict must not move: only the description of the sample does.
    verdict_rows = diff_rows(old, corrected, KEYS, ["t_star", "h2_supported", "primary"])
    n_rows = diff_rows(old, corrected, KEYS, ["n_selections"])

    per_arm = (
        corrected.groupby("arm")[["n_pairs", "n_selections", "n_seeds"]].max().reset_index().sort_values("arm")
    )

    sections = [
        "## 4. verdicts.csv — n_selections was n_pairs (H12c)",
        "",
        f"Source: `stability_records.csv` ({len(records)} records) -> `{out_dir(run).name}/verdicts.csv` "
        f"({len(corrected)} verdict rows) and `{out_dir(run).name}/stability_summary.csv` ({len(summary)} rows).",
        "",
        "### 4a. What the sample actually was",
        "",
        "| arm | n_pairs (was reported as n_selections) | n_selections (distinct) | n_seeds | overstatement |",
        "|---|---|---|---|---|",
    ]
    for r in per_arm.itertuples():
        factor = r.n_pairs / r.n_selections if r.n_selections else float("nan")
        sections.append(f"| {r.arm} | {r.n_pairs} | {r.n_selections} | {r.n_seeds} | {factor:.1f}x |")
    sections += [
        "",
        "On `wine` the five seeds are tree rebuilds of one dataset, so the distinct selections are the",
        "leaves themselves. On the synthetic arms each seed draws a fresh dataset, so a `cluster0` at",
        "seed 3 is genuinely not the `cluster0` at seed 7 — the pairs there are far closer to",
        "independent, which is precisely why the two counts have to be reported separately rather than",
        "collapsed into one number.",
        "",
        "### 4b. Changed numbers",
        "",
        f"`n_selections` re-stated on {len(n_rows)} of {len(corrected)} verdict rows:",
        "",
        *deltas_table(n_rows, "arm · method · split · delta"),
        "### 4c. Verdicts themselves",
        "",
        (
            "`t_star` and `h2_supported` are unchanged on every row — as they must be: the fix renames "
            "and decomposes the reported sample size, it does not touch the pairs the Wilcoxon ran on."
            if not verdict_rows
            else "**`t_star` / `h2_supported` moved on some rows — investigate before citing:**"
        ),
        "",
        *([] if not verdict_rows else deltas_table(verdict_rows, "arm · method · split · delta")),
        "The p-values are still computed over n_pairs. Aggregating to one value per selection before",
        "testing would change a pre-registered analysis, which is out of scope for a re-derivation; the",
        "honest n is reported so a reader can judge the pseudo-replication for themselves.",
        "",
    ]
    write_deltas(run, f"Corrected RQ2 verdicts — {run}", sections)
    print(f"  {run}: {len(corrected)} verdicts re-derived, n_selections restated on {len(n_rows)} rows, verdicts moved on {len(verdict_rows)}")
    return {"run": run, "n_changed": len(n_rows), "n_verdicts_moved": len(verdict_rows)}

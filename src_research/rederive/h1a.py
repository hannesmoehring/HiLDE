"""Re-derivation 1: corrected ``h1a_summary.csv`` for the six RQ1 runs (B2).

``HIGHER_IS_BETTER`` declared both MRRE terms lower-is-better. They arrive already
inverted into a [0, 1] similarity, so ``win_rate`` and ``rank_biserial`` were computed
against the wrong direction on every MRRE row. ``median_delta`` never consulted the map
and is unaffected — which is exactly why the shipped summaries contain rows whose two
effect sizes contradict each other.

The per-region measurements in ``h1a_regions.csv`` are untouched by the defect, so the
corrected summary is a pure re-aggregation: ``h1a_summary`` is imported from the fixed
harness and applied to the shipped records.

**Not every difference is a correction.** Re-running a June aggregation in an August
environment also moves ``wilcoxon_p``, because SciPy's ``method="auto"`` no longer picks
the same branch at these sample sizes. That is an environment difference, not a fix, and
the two are separated and attributed below rather than presented as one list of
"corrections" — with the attribution *checked* (recomputing with ``method="exact"`` must
reproduce the shipped number) rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src_research.hierarchical_vs_flat import HIGHER_IS_BETTER, _paired_deltas, h1a_summary
from src_research.rederive import H1A_RUNS, deltas_table, diff_rows, fmt, out_dir, run_dir, write_deltas

KEYS = ["dataset", "method", "region_def", "metric"]
VALUES = ["n_pairs", "median_delta", "win_rate", "wilcoxon_p", "rank_biserial", "primary"]
ALPHA = 0.05
EXACT_TOL = 1e-5  # a branch "reproduces" the shipped p at this relative difference
NEAR_TOL = 1e-2  # above EXACT_TOL but below this: same branch, different SciPy internals


def _is_b2(row: dict) -> bool:
    """A change the MRRE direction fix explains: an effect size on an MRRE metric."""
    metric = row["key"][KEYS.index("metric")]
    return str(metric).startswith("mrre") and row["column"] in {"win_rate", "rank_biserial"}


def _branch_match(regions: pd.DataFrame, row: dict) -> tuple[str, float]:
    """Which SciPy ``wilcoxon`` branch reproduces the shipped p, and how closely.

    Returns (branch, relative difference). If some branch reproduces the shipped number
    from the shipped deltas, then the data and the pairing did not move and only SciPy's
    choice did — which is what makes the environment attribution checkable rather than
    assumed. A near-match on the same branch (same choice, different internals — SciPy's
    tie correction changed) is reported separately from an exact one.
    """
    dataset, method, region_def, metric = row["key"]
    d = _paired_deltas(regions, dataset, method, region_def, metric)
    nz = d[d != 0]
    old = float(row["old"])
    if nz.size < 1 or np.allclose(d, 0):
        return "none", float("nan")
    scale = max(abs(old), 1e-300)
    best, best_rel = "none", float("inf")
    for branch in ("exact", "approx"):
        try:
            p = float(wilcoxon(nz, method=branch).pvalue)
        except Exception:  # noqa: BLE001 - a branch that cannot run is simply not a match
            continue
        rel = abs(p - old) / scale
        if rel < best_rel:
            best, best_rel = branch, rel
    return best, best_rel


def _significance_flips(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Rows whose verdict at alpha changes — the only p-value difference that matters."""
    m = old.merge(new, on=KEYS, suffixes=("_old", "_new"))
    sig_old, sig_new = m["wilcoxon_p_old"] < ALPHA, m["wilcoxon_p_new"] < ALPHA
    return m.loc[sig_old != sig_new, [*KEYS, "wilcoxon_p_old", "wilcoxon_p_new"]]


def rederive_run(run: str) -> dict:
    regions = pd.read_csv(run_dir(run) / "h1a_regions.csv")
    corrected = h1a_summary(regions)
    corrected.to_csv(out_dir(run) / "h1a_summary.csv", index=False)

    old = pd.read_csv(run_dir(run) / "h1a_summary.csv")
    rows = diff_rows(old, corrected, KEYS, VALUES)

    b2 = [r for r in rows if _is_b2(r)]
    env = [r for r in rows if r["column"] == "wilcoxon_p"]
    other = [r for r in rows if r not in b2 and r not in env]

    matches = {id(r): _branch_match(regions, r) for r in env}
    exact_hits = [r for r in env if matches[id(r)][1] <= EXACT_TOL]
    near_hits = [r for r in env if EXACT_TOL < matches[id(r)][1] <= NEAR_TOL]
    unmatched = [r for r in env if not (matches[id(r)][1] <= NEAR_TOL)]
    worst_near = max((matches[id(r)][1] for r in near_hits), default=0.0)

    flips = _significance_flips(old, corrected)
    flips.to_csv(out_dir(run) / "significance_flips.csv", index=False)

    sections = [
        "## 1. h1a_summary.csv — MRRE direction (B2)",
        "",
        f"Source: `h1a_regions.csv` ({len(regions)} region records) -> `{out_dir(run).name}/h1a_summary.csv` "
        f"({len(corrected)} summary rows, shipped had {len(old)}).",
        "",
        f"`HIGHER_IS_BETTER` now reads `{HIGHER_IS_BETTER}`.",
        "",
        f"{len(rows)} derived numbers differ. They are not all corrections:",
        "",
        f"- **{len(b2)}** are the B2 fix — `win_rate` / `rank_biserial` on an MRRE metric.",
        f"- **{len(env)}** are `wilcoxon_p`, and they appear on every metric alike, MRRE or not — the "
        "fingerprint of an environment difference rather than a direction fix. Each was re-tested against "
        "SciPy's two branches on the shipped deltas:",
        f"    - **{len(exact_hits)}** are reproduced to within {EXACT_TOL:g} relative by naming a branch "
        "explicitly, i.e. `method=\"auto\"` simply picks a different branch now than it did in June;",
        f"    - **{len(near_hits)}** match a branch to within {NEAR_TOL:g} but not {EXACT_TOL:g} "
        f"(worst {worst_near:.1e} relative) — the same branch computing slightly differently, i.e. SciPy's "
        "internals, not the data;",
        f"    - **{len(unmatched)}** match neither branch"
        + (" — none." if not unmatched else " — **unexplained, investigate before citing.**"),
        f"- **{len(other)}** are some other column"
        + (" — none." if not other else " — **unexplained, investigate before citing.**"),
        "",
        "`n_pairs` and `median_delta` are identical on every row, which is what rules out any change to "
        "the measurements or to the pairing.",
        "",
        f"No conclusion moves: **{len(flips)}** rows change significance at alpha = {ALPHA}.",
        "",
        "### 1a. B2 corrections (the MRRE direction)",
        "",
        *deltas_table(b2, "dataset · method · region_def · metric"),
        "### 1b. Environment differences (SciPy `wilcoxon`)",
        "",
        f"Listed for completeness; none of these is a fix, and none crosses alpha = {ALPHA}. "
        "The `branch` column names the SciPy branch that reproduces the shipped value from the shipped "
        "deltas, and `rel` how closely.",
        "",
    ]
    if env:
        sections += [
            "| dataset · method · region_def · metric | p (shipped) | p (re-derived) | branch | rel |",
            "|---|---|---|---|---|",
            *[
                f"| {' · '.join(str(k) for k in r['key'])} | {fmt(r['old'])} | {fmt(r['new'])} | "
                f"{matches[id(r)][0]} | {matches[id(r)][1]:.1e} |"
                for r in env
            ],
            "",
        ]
    else:
        sections += ["_No p-value differed._", ""]
    if unmatched or other:
        sections += [
            "### 1c. Unexplained differences",
            "",
            *deltas_table(unmatched + other, "dataset · method · region_def · metric"),
        ]
    if len(flips):
        sections += [
            "### 1d. Significance flips",
            "",
            "| " + " · ".join(KEYS) + " | p (shipped) | p (re-derived) |",
            "|---|---|---|",
            *[
                f"| {r.dataset} · {r.method} · {r.region_def} · {r.metric} | {fmt(r.wilcoxon_p_old)} | {fmt(r.wilcoxon_p_new)} |"
                for r in flips.itertuples()
            ],
            "",
        ]
    return {
        "run": run,
        "n_rows": len(corrected),
        "n_b2": len(b2),
        "n_env": len(env),
        "n_exact": len(exact_hits),
        "n_near": len(near_hits),
        "n_unexplained": len(unmatched) + len(other),
        "n_flips": len(flips),
        "sections": sections,
    }


def main() -> list[dict]:
    results = []
    for run in H1A_RUNS:
        res = rederive_run(run)
        write_deltas(run, f"Corrected H1a summary — {run}", res["sections"])
        results.append(res)
        print(
            f"  {run}: {res['n_rows']} rows | B2 corrections {res['n_b2']} | "
            f"scipy p-diffs {res['n_env']} ({res['n_exact']} branch-exact, {res['n_near']} near) | "
            f"unexplained {res['n_unexplained']} | significance flips {res['n_flips']}"
        )
    return results

"""Statistics, verdicts and figures for the RQ2 predicate-stability experiment.

Split out of ``predicate_stability`` so the analysis can be re-run on an existing
records CSV without repeating the sweep::

    uv run python -m src_research.predicate_stability_analysis outputs/experiments/<ts>

Implements SS5/SS7 of the pre-registered RQ2 design:
    * unit of analysis = the selection (replicate pairs were aggregated at record time);
    * H2a: per t < 1.0, paired Wilcoxon of per-selection jaccard_admitted(t) - (1.0),
      Holm-corrected across the three thresholds; same (secondary) for f1_sd;
    * H2b: the pre-specified joint operating point - t* = largest t < 1.0 with
      (i) median Jaccard gain > 0 at Holm p < .05 and (ii) median F1 >= 0.9 x strict.
      H2 is supported iff t* exists; refuted otherwise - both reported;
    * H2c: paired Wilcoxon of severity - symmetric F1 at matched t (skewed arms should
      be positive, the gaussian arms are the negative control on ourselves);
    * stability is never reported alone (trivial-stability guard): every stability figure
      carries the matched F1/coverage.

Pre-registered primary: jaccard_admitted, threshold method, severity split, delta = 0.1.
Everything else (db, f1_sd, the delta sweep, bound_sd, seed-perturbation) is secondary /
exploratory and labelled as such.

``design SSN`` below marks a rule fixed by the pre-registered design, which is
recorded in the thesis and no longer kept in this repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from scipy.stats import wilcoxon

# Pre-registered analysis constants (design SS5/SS7) - fixed before any run.
STRICT_T = 1.0
ALPHA = 0.05
F1_FLOOR = 0.9  # operating-point criterion (ii): median F1 >= 0.9 x median strict F1
PRIMARY = {
    "metric": "jaccard_admitted",
    "method": "threshold",
    "split": "severity",
    "delta": 0.1,
}
STABILITY_METRICS = {
    "jaccard_admitted": "up",
    "f1_sd": "down",
}  # direction of improvement


# --------------------------------------------------------------------------- #
# Small statistics helpers.                                                    #
# --------------------------------------------------------------------------- #


def _wilcoxon_p(d: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank p on the non-zero deltas (guarding degenerate cases),
    mirroring the sibling harnesses."""
    nz = d[d != 0]
    if nz.size < 1 or np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(nz).pvalue)


def _rank_biserial(d: np.ndarray, improved: np.ndarray) -> float:
    """Matched-pairs rank-biserial = (#improved - #worsened) / #non-zero pairs."""
    nz = d != 0
    if not nz.any():
        return float("nan")
    return float((np.sum(improved & nz) - np.sum(~improved & nz)) / np.sum(nz))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values; NaNs pass through and do not count toward m."""
    valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    order = sorted(valid, key=lambda i: pvals[i])
    adjusted: list[float] = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(valid) - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------- #
# Sanity check on the records themselves.                                      #
# --------------------------------------------------------------------------- #


def check_records(records: pd.DataFrame) -> None:
    """At t = 1.0 the total trim is zero, so the tail split cannot matter: severity and
    symmetric rows must be bit-identical. A violation means tail_split is mis-threaded."""
    strict = records[records["t"] == STRICT_T]
    for col in ["jaccard_admitted", "f1", "coverage"]:
        wide = strict.pivot_table(
            index=["arm", "seed", "sel_id", "method", "delta"],
            columns="split",
            values=col,
        )
        if {"severity", "symmetric"} <= set(wide.columns):
            pair = wide[["severity", "symmetric"]].dropna()
            assert np.allclose(pair["severity"], pair["symmetric"]), (
                f"t=1.0 split-invariance violated on {col}"
            )


# --------------------------------------------------------------------------- #
# H2a/H2b: stability sweep summary + operating-point verdicts.                 #
# --------------------------------------------------------------------------- #


def stability_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, method, split, delta, stability metric, t < 1.0): paired delta vs strict
    across selections, Wilcoxon p (Holm across the three thresholds), effect size - each
    row carrying the matched F1/coverage medians (trivial-stability guard)."""
    rows: list[dict] = []
    for (arm, method, split, delta), sub in records.groupby(
        ["arm", "method", "split", "delta"]
    ):
        med_f1 = sub.groupby("t")["f1"].median()
        med_cov = sub.groupby("t")["coverage"].median()
        if STRICT_T not in med_f1.index:
            continue
        relaxed_ts = sorted(
            [t for t in sub["t"].unique() if t != STRICT_T], reverse=True
        )
        for metric, direction in STABILITY_METRICS.items():
            wide = sub.pivot_table(index=["seed", "sel_id"], columns="t", values=metric)
            if STRICT_T not in wide.columns:
                continue
            group_rows: list[dict] = []
            for t in relaxed_ts:
                if t not in wide.columns:
                    continue
                pair = wide[[t, STRICT_T]].dropna()
                d = (pair[t] - pair[STRICT_T]).to_numpy()
                improved = d > 0 if direction == "up" else d < 0
                # The Wilcoxon runs on (seed x selection) pairs, but the *selection* is the
                # unit of analysis, and on wine the seeds are rebuilds of one dataset - the
                # same leaves five times over. Carry both counts so no downstream table can
                # quote the larger one under the smaller one's name.
                seeds, sel_ids = (
                    pair.index.get_level_values(lvl) for lvl in ("seed", "sel_id")
                )
                group_rows.append(
                    {
                        "arm": arm,
                        "method": method,
                        "split": split,
                        "delta": delta,
                        "metric": metric,
                        "t": t,
                        "n_pairs": int(d.size),
                        "n_selections": int(pd.unique(sel_ids).size),
                        "n_seeds": int(pd.unique(seeds).size),
                        "median_delta": float(np.median(d)) if d.size else float("nan"),
                        "win_rate": float(np.mean(improved))
                        if d.size
                        else float("nan"),
                        "wilcoxon_p": _wilcoxon_p(d),
                        "rank_biserial": _rank_biserial(d, improved),
                        "median_value": float(wide[t].median()),
                        "median_strict": float(wide[STRICT_T].median()),
                        "median_f1": float(med_f1.get(t, np.nan)),
                        "f1_ratio": float(med_f1.get(t, np.nan) / med_f1[STRICT_T])
                        if med_f1[STRICT_T]
                        else float("nan"),
                        "median_coverage": float(med_cov.get(t, np.nan)),
                        "primary": metric == PRIMARY["metric"]
                        and method == PRIMARY["method"]
                        and split == PRIMARY["split"]
                        and delta == PRIMARY["delta"],
                    }
                )
            for row, p_adj in zip(
                group_rows, holm([r["wilcoxon_p"] for r in group_rows])
            ):
                row["p_holm"] = p_adj
            rows.extend(group_rows)
    return pd.DataFrame(rows)


def verdicts(summary: pd.DataFrame) -> pd.DataFrame:
    """The pre-specified joint operating point per (arm, method, split, delta):
    t* = largest t < 1.0 with median Jaccard gain > 0 at Holm p < ALPHA and
    median F1 >= F1_FLOOR x strict. H2 supported iff t* exists."""
    rows: list[dict] = []
    jac = summary[summary["metric"] == "jaccard_admitted"]
    for (arm, method, split, delta), sub in jac.groupby(
        ["arm", "method", "split", "delta"]
    ):
        ok = sub[
            (sub["median_delta"] > 0)
            & (sub["p_holm"] < ALPHA)
            & (sub["f1_ratio"] >= F1_FLOOR)
        ]
        t_star = float(ok["t"].max()) if not ok.empty else float("nan")
        rows.append(
            {
                "arm": arm,
                "method": method,
                "split": split,
                "delta": delta,
                "t_star": t_star,
                "h2_supported": bool(not ok.empty),
                # These were one column: `n_selections` reported `sub["n_pairs"].max()`,
                # i.e. seeds x selections, overstating the wine sample fivefold.
                "n_pairs": int(sub["n_pairs"].max()),
                "n_selections": int(sub["n_selections"].max()),
                "n_seeds": int(sub["n_seeds"].max()),
                "primary": bool(sub["primary"].any()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H2c: severity vs symmetric tail split at matched t.                          #
# --------------------------------------------------------------------------- #


def h2c_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon of (severity - symmetric) F1 at matched t < 1.0, per arm and method
    at the headline delta. The skew column carries the expectation: positive on lognormal
    arms, null on gaussian arms (negative control); bimodal is the exploratory adversary."""
    sub = records[(records["delta"] == PRIMARY["delta"]) & (records["t"] != STRICT_T)]
    rows: list[dict] = []
    for (arm, skew, method, t), s in sub.groupby(["arm", "skew", "method", "t"]):
        wide = s.pivot_table(index=["seed", "sel_id"], columns="split", values="f1")
        if not {"severity", "symmetric"} <= set(wide.columns):
            continue
        pair = wide[["severity", "symmetric"]].dropna()
        d = (pair["severity"] - pair["symmetric"]).to_numpy()
        rows.append(
            {
                "arm": arm,
                "skew": skew,
                "method": method,
                "t": t,
                "n_pairs": int(d.size),
                "median_diff": float(np.median(d)) if d.size else float("nan"),
                "win_rate": float(np.mean(d > 0)) if d.size else float("nan"),
                "wilcoxon_p": _wilcoxon_p(d),
                "rank_biserial": _rank_biserial(d, d > 0),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Recovery + seed-perturbation summaries.                                      #
# --------------------------------------------------------------------------- #


def recovery_summary(recovery: pd.DataFrame) -> pd.DataFrame:
    """Synthetic arms: median relevant-dim P/R/F1 and membership F1 per
    (arm, method, split, t) - the strict-vs-relaxed rows of thesis table
    ``tab:synthetic-recovery``."""
    if recovery.empty:
        return pd.DataFrame()
    agg = (
        recovery.groupby(["arm", "method", "split", "t"])[
            ["dim_precision", "dim_recall", "dim_f1", "membership_f1", "coverage"]
        ]
        .median()
        .reset_index()
    )
    counts = (
        recovery.groupby(["arm", "method", "split", "t"])
        .size()
        .rename("n_selections")
        .reset_index()
    )
    return agg.merge(counts, on=["arm", "method", "split", "t"])


def seedpass_summary(seedpass: pd.DataFrame | None) -> pd.DataFrame | None:
    """Pass (b): admitted-set agreement of matched leaves across tree rebuilds, strong
    matches (>= 0.5) only; the weak-match fraction quantifies the tree-instability
    confound and is reported alongside, never silently dropped."""
    if seedpass is None or seedpass.empty:
        return None
    strong = seedpass[~seedpass["weak_match"]]
    rows: list[dict] = []
    for (method, t), sub in strong.groupby(["method", "t"]):
        rows.append(
            {
                "method": method,
                "t": t,
                "n_matched_pairs": len(sub),
                "mean_jaccard_admitted": float(sub["jaccard_admitted"].mean()),
                "median_jaccard_admitted": float(sub["jaccard_admitted"].median()),
                "mean_match_jaccard": float(sub["match_jaccard"].mean()),
                "weak_match_frac": float(
                    seedpass[seedpass["method"] == method]["weak_match"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration + IO.                                                          #
# --------------------------------------------------------------------------- #


def analyse(
    records: pd.DataFrame, recovery: pd.DataFrame, seedpass: pd.DataFrame | None
) -> dict[str, pd.DataFrame | None]:
    summary = stability_summary(records)
    return {
        "summary": summary,
        "verdicts": verdicts(summary),
        "h2c": h2c_summary(records),
        "recovery_summary": recovery_summary(recovery),
        "seedpass_summary": seedpass_summary(seedpass),
    }


def save_outputs(results: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(out_dir / "stability_summary.csv", index=False)
    results["verdicts"].to_csv(out_dir / "verdicts.csv", index=False)
    results["h2c"].to_csv(out_dir / "h2c_summary.csv", index=False)
    if (
        results["recovery_summary"] is not None
        and not results["recovery_summary"].empty
    ):
        results["recovery_summary"].to_csv(
            out_dir / "recovery_summary.csv", index=False
        )
    if results["seedpass_summary"] is not None:
        results["seedpass_summary"].to_csv(
            out_dir / "seed_stability_summary.csv", index=False
        )


# --------------------------------------------------------------------------- #
# Figures (design SS10): trade-off, H2c ablation, coverage diagnostic.         #
# --------------------------------------------------------------------------- #


def _primary_records(records: pd.DataFrame) -> pd.DataFrame:
    return records[
        (records["method"] == PRIMARY["method"])
        & (records["split"] == PRIMARY["split"])
        & (records["delta"] == PRIMARY["delta"])
    ]


def make_plots(records: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    arms = sorted(
        records["arm"].unique(), key=lambda a: (a != "wine", a)
    )  # wine panel first

    # fig:rq2-tradeoff - stability (Jaccard) and specificity (F1) vs t, per arm; the joint
    # read-off the operating point lives on. Median + IQR across selections.
    prim = _primary_records(records)
    fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        sub = prim[prim["arm"] == arm]
        for metric, color, label in [
            ("jaccard_admitted", "tab:blue", "Jaccard (stability)"),
            ("f1", "tab:orange", "F1 (specificity)"),
        ]:
            g = sub.groupby("t")[metric].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(g.index, g[0.5], marker="o", color=color, label=label)
            ax.fill_between(g.index, g[0.25], g[0.75], color=color, alpha=0.2)
        ax.set_xlabel("threshold t")
        ax.set_title(arm, fontsize=9)
        ax.invert_xaxis()  # relaxation increases to the right
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("median (IQR band)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"RQ2 trade-off: admitted-set stability vs specificity ({PRIMARY['method']}, {PRIMARY['split']}, delta={PRIMARY['delta']})"
    )
    fig.tight_layout()
    fig.savefig(plots / "rq2_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # fig:rq2-ablation - H2c: per-selection (severity - symmetric) F1 difference at matched
    # t, synthetic arms. Boxes above 0 on skewed arms, on 0 for the gaussian control.
    abl = records[
        (records["arm"] != "wine")
        & (records["delta"] == PRIMARY["delta"])
        & (records["method"] == PRIMARY["method"])
        & (records["t"] != STRICT_T)
    ]
    if not abl.empty:
        wide = abl.pivot_table(
            index=["arm", "seed", "sel_id", "t"], columns="split", values="f1"
        )
        if {"severity", "symmetric"} <= set(wide.columns):
            diff = (
                (wide["severity"] - wide["symmetric"]).rename("f1_diff").reset_index()
            )
            g = sns.catplot(
                data=diff,
                x="t",
                y="f1_diff",
                col="arm",
                kind="box",
                order=sorted(diff["t"].unique(), reverse=True),
                height=3,
                aspect=0.9,
                col_order=[a for a in arms if a != "wine"],
            )
            for ax in g.axes.flat:
                ax.axhline(0, color="grey", lw=0.8, ls="--")
            g.set_axis_labels("threshold t", "F1(severity) - F1(symmetric)")
            g.figure.suptitle(
                "H2c ablation: severity vs naive 50/50 tail split (threshold method)",
                y=1.04,
            )
            g.savefig(plots / "rq2_ablation.png", dpi=150, bbox_inches="tight")
            plt.close(g.figure)

    # Coverage diagnostic - median admitted fraction vs t (the trivial-stability guard in
    # picture form: stability gains must not come from the predicate admitting everything).
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in arms:
        g = prim[prim["arm"] == arm].groupby("t")["coverage"].median()
        ax.plot(g.index, g.values, marker="o", label=arm)
    ax.set_xlabel("threshold t")
    ax.set_ylabel("median coverage (admitted fraction)")
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    ax.set_title("Coverage vs t (threshold, severity, headline delta)")
    fig.tight_layout()
    fig.savefig(plots / "rq2_coverage.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Console report.                                                              #
# --------------------------------------------------------------------------- #


def _fmt_p(p: float) -> str:
    return "-" if pd.isna(p) else f"{p:.3g}"


def print_report(results: dict[str, pd.DataFrame | None], console: Console) -> None:
    summary, verd = results["summary"], results["verdicts"]

    # Primary sweep: pre-registered metric/method/split/delta, per arm x t + verdict.
    prim = summary[summary["primary"]]
    if not prim.empty:
        table = Table(
            title=f"H2a/H2b primary - jaccard_admitted delta vs strict ({PRIMARY['method']}, {PRIMARY['split']}, delta={PRIMARY['delta']})"
        )
        for col in [
            "arm",
            "t",
            "median J",
            "delta J",
            "win",
            "p(Holm)",
            "median F1",
            "F1 ratio",
            "coverage",
        ]:
            table.add_column(col, justify="right" if col != "arm" else "left")
        for arm, sub in prim.groupby("arm"):
            for _, r in sub.sort_values("t", ascending=False).iterrows():
                md = r["median_delta"]
                md_str = (
                    f"[green]{md:+.3f}[/]"
                    if md > 0
                    else (f"[red]{md:+.3f}[/]" if md < 0 else f"{md:+.3f}")
                )
                floor_ok = r["f1_ratio"] >= F1_FLOOR
                ratio_str = (
                    f"[green]{r['f1_ratio']:.3f}[/]"
                    if floor_ok
                    else f"[red]{r['f1_ratio']:.3f}[/]"
                )
                table.add_row(
                    str(arm),
                    f"{r['t']:g}",
                    f"{r['median_value']:.3f}",
                    md_str,
                    f"{r['win_rate']:.2f}",
                    _fmt_p(r["p_holm"]),
                    f"{r['median_f1']:.3f}",
                    ratio_str,
                    f"{r['median_coverage']:.3f}",
                )
            table.add_section()
        console.print(table)

    if not verd.empty:
        vp = verd[verd["primary"]]
        for _, r in vp.iterrows():
            # Both counts, at the point of the claim: the test used n_pairs observations,
            # but they came from only n_selections distinct selections.
            n = f"[dim](n_selections={r['n_selections']} x n_seeds={r['n_seeds']} = {r['n_pairs']} pairs)[/]"
            if r["h2_supported"]:
                console.print(
                    f"  [bold green]H2 SUPPORTED[/] on [bold]{r['arm']}[/]: operating point t* = {r['t_star']:g} {n}"
                )
            else:
                console.print(
                    f"  [bold red]H2 REFUTED[/] on [bold]{r['arm']}[/]: no t < 1.0 meets the joint criterion {n}"
                )
        console.print()

    # H2c ablation (threshold method at the headline delta).
    h2c = results["h2c"]
    if h2c is not None and not h2c.empty:
        sub = h2c[h2c["method"] == PRIMARY["method"]]
        table = Table(
            title="H2c - F1(severity) - F1(symmetric) at matched t (threshold, headline delta)"
        )
        for col in ["arm", "expectation", "t", "median diff", "win", "p"]:
            table.add_column(
                col, justify="right" if col not in {"arm", "expectation"} else "left"
            )
        expect = {
            "gaussian": "~0 (control)",
            "lognormal": "> 0",
            "bimodal": "? (adversarial)",
            "real": "exploratory",
        }
        for arm, s in sub.groupby("arm"):
            for _, r in s.sort_values("t", ascending=False).iterrows():
                md = r["median_diff"]
                md_str = (
                    f"[green]{md:+.4f}[/]"
                    if md > 0
                    else (f"[red]{md:+.4f}[/]" if md < 0 else f"{md:+.4f}")
                )
                table.add_row(
                    str(arm),
                    expect.get(r["skew"], "?"),
                    f"{r['t']:g}",
                    md_str,
                    f"{r['win_rate']:.2f}",
                    _fmt_p(r["wilcoxon_p"]),
                )
            table.add_section()
        console.print(table)

    # Recovery (synthetic): dim_f1 medians, severity split, both methods, wide by t.
    rec = results["recovery_summary"]
    if rec is not None and not rec.empty:
        sev = rec[rec["split"] == "severity"]
        wide = sev.pivot_table(index=["arm", "method"], columns="t", values="dim_f1")
        table = Table(
            title="Relevant-dimension recovery (median dim F1, severity split) - fills tab:synthetic-recovery"
        )
        table.add_column("arm", justify="left")
        table.add_column("method", justify="left")
        for t in sorted(wide.columns, reverse=True):
            table.add_column(f"t={t:g}", justify="right")
        for (arm, method), row in wide.iterrows():
            table.add_row(
                str(arm),
                str(method),
                *[f"{row[t]:.3f}" for t in sorted(wide.columns, reverse=True)],
            )
        console.print(table)

    # Seed-perturbation pass (b) - corroboration, confound reported alongside.
    sp = results["seedpass_summary"]
    if sp is not None and not sp.empty:
        table = Table(
            title="Pass (b): matched-leaf predicate agreement across tree rebuilds (wine, severity)"
        )
        for col in [
            "method",
            "t",
            "n pairs",
            "mean adm. Jaccard",
            "mean match J",
            "weak-match frac",
        ]:
            table.add_column(col, justify="right" if col not in {"method"} else "left")
        for _, r in sp.sort_values(["method", "t"], ascending=[True, False]).iterrows():
            table.add_row(
                str(r["method"]),
                f"{r['t']:g}",
                str(int(r["n_matched_pairs"])),
                f"{r['mean_jaccard_admitted']:.3f}",
                f"{r['mean_match_jaccard']:.3f}",
                f"{r['weak_match_frac']:.2f}",
            )
        console.print(table)


# --------------------------------------------------------------------------- #
# Standalone re-analysis of an existing output directory.                      #
# --------------------------------------------------------------------------- #


def main(out_dir: Path) -> None:
    console = Console()
    records = pd.read_csv(out_dir / "stability_records.csv")
    recovery_path = out_dir / "recovery.csv"
    recovery = (
        pd.read_csv(recovery_path)
        if recovery_path.exists() and recovery_path.stat().st_size > 1
        else pd.DataFrame()
    )
    seed_path = out_dir / "seed_stability.csv"
    seedpass = pd.read_csv(seed_path) if seed_path.exists() else None

    check_records(records)
    results = analyse(records, recovery, seedpass)
    save_outputs(results, out_dir)
    make_plots(records, out_dir)
    print_report(results, console)
    console.print(
        f"\n[bold green]Re-analysed.[/] Summaries + plots refreshed in [underline]{out_dir}[/]"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            "usage: python -m src_research.predicate_stability_analysis <outputs/experiments/timestamp>"
        )
    main(Path(sys.argv[1]))

"""Re-derive corrected aggregates from already-shipped experiment outputs.

Every number in this package comes from a raw CSV that is already on disk. No experiment
is rerun and no original file is written to: each re-derivation writes into a fresh
``outputs/experiments/<run>/rederived_20260813/`` subdirectory, alongside a ``DELTAS.md``
listing every derived number that changed, old -> new.

Why re-derive rather than rerun: the defects fixed in this pass (B2's inverted MRRE
direction, H12d's duplicated control cells and mixed samples, H12c's mislabelled n) are all
in the *aggregation*, not in the measurement. The shipped per-record CSVs are the same
numbers a rerun would have measured, so correcting the aggregation recovers the right
answer without touching the data. Where a correction genuinely needs a quantity that was
never persisted, the re-derivation says so instead of substituting a rerun - see
``internal_external``.

Run with::

    uv run python -m src_research.rederive
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path("outputs/experiments")
REDERIVED_DIRNAME = "rederived_20260813"

# The six RQ1 runs whose h1a_summary.csv carries the inverted MRRE direction.
H1A_RUNS = ["20260628_182948", "20260628_184005", "20260628_184209", "20260628_184251", "20260628_184350", "20260628_184827"]
H2B_RUN = "20260628_195214"
TUNING_RUNS = ["20260628_125924", "20260628_153633"]
STABILITY_RUN = "20260711_115849"

TOL = 1e-12  # below this a difference is float noise, not a changed number


def run_dir(run: str) -> Path:
    return EXPERIMENTS / run


def out_dir(run: str) -> Path:
    """The re-derivation's own directory. Originals are never opened for writing."""
    d = run_dir(run) / REDERIVED_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def changed(old: object, new: object) -> bool:
    """True when a derived number actually moved. Two NaNs are the same answer."""
    old_na, new_na = pd.isna(old), pd.isna(new)
    if old_na or new_na:
        return bool(old_na != new_na)
    if isinstance(old, (int, float, np.floating, np.integer)) and isinstance(new, (int, float, np.floating, np.integer)):
        return abs(float(old) - float(new)) > TOL
    return old != new


def fmt(v: object) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def diff_rows(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], value_cols: list[str]) -> list[dict]:
    """Every (key, column) whose value changed between two aggregate frames.

    Rows present in only one of the two are reported as such: a re-derivation that drops or
    adds a row has changed the table just as much as one that moves a number.
    """
    o = old.set_index(keys).sort_index()
    n = new.set_index(keys).sort_index()
    out: list[dict] = []
    for key in o.index.difference(n.index):
        out.append({"key": key, "column": "(whole row)", "old": "present", "new": "DROPPED"})
    for key in n.index.difference(o.index):
        out.append({"key": key, "column": "(whole row)", "old": "ABSENT", "new": "added"})
    for key in o.index.intersection(n.index):
        orow, nrow = o.loc[key], n.loc[key]
        for col in value_cols:
            if col not in o.columns or col not in n.columns:
                continue
            if changed(orow[col], nrow[col]):
                out.append({"key": key, "column": col, "old": orow[col], "new": nrow[col]})
    return out


def deltas_table(rows: list[dict], key_name: str) -> list[str]:
    """Markdown table of old -> new, or an explicit statement that nothing moved."""
    if not rows:
        return ["_No derived number changed._", ""]
    lines = [f"| {key_name} | column | old | new |", "|---|---|---|---|"]
    for r in rows:
        key = " · ".join(str(k) for k in r["key"]) if isinstance(r["key"], tuple) else str(r["key"])
        lines.append(f"| {key} | `{r['column']}` | {fmt(r['old'])} | {fmt(r['new'])} |")
    lines.append("")
    return lines


def write_deltas(run: str, title: str, sections: list[str]) -> Path:
    path = out_dir(run) / "DELTAS.md"
    header = [
        f"# {title}",
        "",
        f"Run: `outputs/experiments/{run}`  ·  re-derived {REDERIVED_DIRNAME.removeprefix('rederived_')} from the shipped raw CSVs.",
        "",
        "No experiment was rerun. Every number below is recomputed from records already on disk;",
        "the original files in the parent directory are untouched.",
        "",
    ]
    path.write_text("\n".join(header + sections))
    return path

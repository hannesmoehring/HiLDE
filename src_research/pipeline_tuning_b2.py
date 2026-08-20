"""Deviation analysis: re-judge the EQ1b candidates against a NON-DEGENERATE baseline.

Why this exists (disclosed deviation, written after seeing the main run's Breast
cancer result, before running):

The pre-registered baseline `B` (design section 2) is the *app-effective* default,
i.e. `config_defaults.py` with `hclust_umap_n_components = n_features`, because
`App.tsx` overwrites that key. On Breast cancer `B` produced **one leaf in 10/10
builds**, so `dbcv_leaf` is undefined and A1 ("complete separation") is not
evaluable; A5's noise clause is likewise vacuous, because a baseline that finds
nothing also discards nothing. The pre-registered verdict for that dataset is
therefore "defaults retained" for a reason that says nothing about the preset.

The acceptance *rule* is not touched — changing a threshold after seeing results is
exactly what pre-registration exists to prevent. What is changed is the factual
error in the design: `B` was assumed to be a meaningful comparator and on at least
one dataset it is not. This script re-runs the baseline arm with
`hclust_umap_n_components = 2` (the value `config_defaults.py` actually ships,
called `B2` in the design) and applies A1-A6 unchanged, for **every** dataset, not
only the one that failed. Both tables are reported side by side.

The candidate configs are read from the main run's verdict files and are NOT
re-selected; the 5 preset builds are re-used, so nothing about the candidate
depends on this second baseline.

Run with::

    python -m src_research.pipeline_tuning_b2 --run 20260729_101836

``design section N`` marks a rule fixed by the pre-registered design, which is
recorded in the thesis and no longer kept in this repository.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src_research.pipeline_tuning import (
    N_BASELINE_BUILDS,
    OUTPUT_ROOT,
    DatasetRun,
    _log,
    _slug,
    build,
    judge,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = OUTPUT_ROOT / args.run
    verdicts = []

    for path in sorted(run_dir.glob("verdict_*.json")):
        v = json.loads(path.read_text())
        dataset = v["dataset"]
        cfg = v.get("config")
        if not cfg:
            _log(f"{dataset}: no candidate in the main run, nothing to re-judge")
            continue

        _log(f"=== {dataset}: B2 baseline (hclust_umap_n_components=2) ===")
        run = DatasetRun(dataset=dataset)
        for i in range(N_BASELINE_BUILDS):
            m = build(dataset, {"hclust_umap_n_components": 2})
            m |= {
                "dataset": dataset,
                "arm": "baseline_b2",
                "build_index": i,
                "split": "select" if i < 5 else "test",
            }
            run.baseline.append(m)
            _log(
                f"  b2 {i + 1}/{N_BASELINE_BUILDS}: dbcv={m.get('dbcv_leaf')} tnc={m.get('tnc_mean')} leaves={m.get('n_leaves')} ari={m.get('ari')} {m.get('build_seconds', 0):.0f}s"
            )

        val = pd.read_csv(run_dir / f"validation_{_slug(dataset)}.csv")
        pre = val[val["arm"] == "preset"]
        # `.where(..., None)` cannot put None in a float64 column - it leaves NaN - so the
        # rehydrated `exception` was NaN, `not nan` is False, and A4_reliable was False for
        # every dataset regardless of the data. `_vals` was poisoned the same way: NaN is
        # `is not None`, so a missing dbcv_leaf survived the filter and NaN'd the mean.
        # Casting to object first is what actually lets None land in the records.
        run.validation = (
            pre.astype(object).where(pd.notna(pre), None).to_dict("records")
        )

        verdict = judge(dataset, run, cfg, v["feature_cols"])
        verdict["baseline_variant"] = "B2 (hclust_umap_n_components=2)"
        verdicts.append(verdict)
        (run_dir / f"verdict_b2_{_slug(dataset)}.json").write_text(
            json.dumps(verdict, indent=2, default=str)
        )
        pd.DataFrame(run.baseline).to_csv(
            run_dir / f"baseline_b2_{_slug(dataset)}.csv", index=False
        )
        _log(
            f"  VERDICT(B2) {dataset}: {'ADOPTED' if verdict['adopted'] else 'defaults retained'}"
        )

    pd.DataFrame(verdicts).to_csv(run_dir / "verdicts_b2.csv", index=False)
    _log(
        f"done. adopted under B2: {[v['dataset'] for v in verdicts if v['adopted']] or 'none'}"
    )


if __name__ == "__main__":
    main()

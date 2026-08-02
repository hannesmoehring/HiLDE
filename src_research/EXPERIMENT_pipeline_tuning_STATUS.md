# EQ1b preset tuning — STATUS as of 2026-07-29 13:38 UTC (halted mid-run)

Run halted by request during the **wine** dataset. Nothing has been integrated: no frontend
change, no thesis change, no git operation. This file records the state so the work can be
resumed without re-deriving anything.

Design: `EXPERIMENT_pipeline_tuning.md` (pre-registered, revision 2).
Harness: `pipeline_tuning.py`. Deviation script (written, **never run**): `pipeline_tuning_b2.py`.
Run directory: `outputs/experiments/20260729_101836/`.

## 1. Where exactly it stopped

| Dataset | n | d | state | verdict |
|---|---|---|---|---|
| Iris (Low) | 150 | 4 | complete (10 baseline + 40 trials + 5 validation) | **ADOPTED** |
| Breast cancer (Low) | 569 | 30 | complete | defaults retained — *but see §3, the verdict is not evaluable* |
| Concentric rings (Low) | 1800 | 2 | complete | defaults retained (correctly vetoed) |
| Digits (Low) | 1797 | 64 | complete | defaults retained (see §3 caveat) |
| Wine quality (Low) | 6497 | 11 | **INCOMPLETE** — 10 baseline builds + 36/40 trials; no selection, no validation | none |

Wine: `trials_wine_quality_low.csv` holds 36 trials (25 non-degenerate, 4 lost to the 180 s
abort). Baseline is complete and identical across 10/10 builds (`dbcv_leaf = -0.101530`,
because with pre-reduction skipped HDBSCAN is deterministic; only the view varies). Best wine
trial so far is `dbcv = -0.2173` — **no trial has beaten the baseline**, so on current evidence
wine is heading for "defaults retained" as well, but this is not yet a verdict: 4 trials and
the whole selection + validation stage are missing.

To resume wine only:

```bash
cd /path/to/SHD && PYTHONPATH=. python -m src_research.pipeline_tuning \
    --datasets "Wine quality (Low)" --trials 40 --out 20260729_101836_wine
```

(The harness has no resume; it would redo wine's 10 baseline builds, ~11 min, plus 40 trials.
Total ~90 min on 2 vCPU.)

## 2. Results that are final

**Iris — adopted.** Independently recomputed from `validation_iris_low.csv`; all six criteria
confirmed.

| | baseline (5 test builds) | preset (5 builds) |
|---|---|---|
| `dbcv_leaf` | 0.296698 (all 5 identical) | 0.589997 (all 5 identical) |
| `tnc_mean` | 0.966601 | 0.981622 |
| ARI | 0.5484 | 0.5682 |
| build seconds (median) | 13.08 | 1.94 |
| leaves | 2 | 2 |

Preset config: `hierarchical_layers 3, hclust_umap_n_components 4, hclust_min_cluster_size 22,
hclust_min_samples 21, umap_n_neighbors 32, umap_min_dist 0.0106, method PCA`
(exact values in `verdict_iris_low.json`). The win is mostly *method*: PCA instead of UMAP on a
4-dimensional dataset — better on both objectives and ~7x faster.

**Concentric rings — retained, and this is the experiment's best single result.** The baseline
recovers all three rings perfectly (ARI **1.0**, noise 0.0, 3 leaves). The DBCV-optimal
candidate scores *higher* DBCV (0.3235 vs 0.1997) by splitting into 6 leaves and discarding
19.7 % of points, dropping ARI to 0.768. A5 (granularity/noise) and A6 (ARI) vetoed it; A1–A4
all passed. This is a clean, quantified demonstration that maximising DBCV degrades a perfect
solution — the thesis's "classes are not clusters" claim turned into a shipping decision.
Note the reference line: DBCV(ground truth) on rings is +0.1997, i.e. exactly the baseline —
and the tuner found +0.3235, *above* the truth.

## 3. Two verdicts that are NOT clean, and why

**Breast cancer — the pre-registered rule is not evaluable here.** The baseline produced
**1 leaf in 10/10 builds**, so `dbcv_leaf` is undefined, A1 (complete separation) cannot be
computed, and A5's noise clause is vacuous (a baseline that finds nothing also discards
nothing). Meanwhile the candidate produced 2 leaves in 5/5 builds with **ARI 0.873** against a
baseline ARI of 0.0. Reporting this as "defaults retained" would be actively misleading: the
defaults do not work on this dataset at all.

This is a factual error in the design (the baseline was assumed to be a meaningful
comparator), not a threshold that wants relaxing. The planned handling, written but **not
run**, is `pipeline_tuning_b2.py`: re-run the baseline arm with `hclust_umap_n_components = 2`
(the value `config_defaults.py` actually ships, which does produce 3 leaves on breast cancer),
apply A1–A6 **unchanged**, for **all** datasets rather than only the failing one, and report
both tables side by side. Cost ~35 min. The acceptance rule itself must not be touched.

**Digits — the comparison is confounded and must be reported as such.** The baseline reaches
`dbcv_leaf = 0.1020` by discarding **52.8 % of all points as noise**. ARI is computed over
retained points only, so the baseline's ARI 0.904 is measured on 47 % of the data while the
candidate's 0.842 is measured on 82 %. A6 failed on that comparison. Additionally the baseline
would itself be scored degenerate by the harness's own trial-level guard (`noise_frac >= 0.5`),
which is applied to trials but not to the baseline — the asymmetry the pre-run review flagged
as M4 and which A5 only partly closes. Decision taken and to be kept: report the confound,
report `noise_frac` beside every ARI, and do **not** invent a post-hoc metric that could flip
the verdict.

## 4. Findings about the app itself (independent of presets, nothing changed yet)

1. `backend/datasets.py:27` `default_feature_cols` returns every column except `row_id`, so the
   shipped default passes `target_*`, `is_red`, `quality` into the clustering — the demo
   datasets are clustered partly on their own labels. This experiment therefore pinned a
   label-free feature set, and any preset must ship with its feature list.
2. `App.tsx` overwrites `hclust_umap_n_components` with the feature count, skipping UMAP
   pre-reduction (`analysis_routine.py:113`). Measured: breast cancer -> 1 leaf (no hierarchy),
   wine -> 2 leaves, iris -> 2. `config_defaults.py` ships 2, which gives 3 leaves on breast
   cancer. **The frontend override, not the backend default, is what degrades the hierarchy.**
   On present evidence this is the single highest-value fix in this whole thread — larger than
   any preset.
3. Dead knobs: `umap_random_state` / `tsne_random_state` are exposed in the config panel but
   never threaded into `_umap` / `_tsne` (`mds_random_state` *is*), so those controls do
   nothing and every build is stochastic. `hclust_normalize` is never read anywhere;
   `compute_analysis_tree` reads `normalize`.
4. `compute_cluster_kde` is not wrapped in the `try/except` that guards `_embed_original`, so a
   small cluster can kill an entire build with `TypeError` from `umap/spectral.py`. Stochastic,
   therefore intermittent in the UI.

## 5. Open decisions for the next session

1. **Finish wine?** (~90 min) On current evidence it will retain defaults. Cheap to skip, but
   then the run covers 4/5 datasets and the thesis must say so.
2. **Run the B2 deviation analysis?** (~35 min) Needed before breast cancer can be reported
   honestly either way.
3. **Fix `default_feature_cols` and the `App.tsx` pre-reduction override?** These are app bugs
   surfaced by the experiment; fixing them changes what "the default" means and would
   invalidate the baseline arm of this run — so decide *before* re-running anything.
4. **Ship the Iris preset alone?** One adopted preset out of four decided datasets is a thin
   basis for a UI feature. The defensible framing may be that the experiment's answer to the
   `\gap{}` is *"no — do not ship tuned configs as defaults"*, with iris as the single
   exception and the rings result as the reason why.
5. Nothing has been written to the thesis. `04_methodology.tex` still carries the original
   `\gap{}`.

## 6. Environment note (costly to rediscover)

The device `.venv` is a **macOS** venv and the device VM has no network, so this cannot run
there. It was run in the cloud container at `/tmp/shd`: `uv venv --python 3.13`, core deps from
PyPI, `uv pip install --no-deps kdbcv` plus the `np.float_`/`np.int_` shim carried in
`pipeline_tuning.py` (the shim `pyproject.toml` points at, `src_research/dbcv_tuning.py`, does
not exist in the current tree). Pinned hardware for the time guard: 2 vCPU, 7 GB RAM,
CPython 3.13.13. Wall clock for the completed part: ~3 h 20 min.

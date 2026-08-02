# EXPERIMENT: Pipeline-level configuration tuning for dataset presets (EQ1b)

**Status:** pre-registered design, revision 2. Revision 1 was adversarially reviewed against
the code before any run; the review found one design-invalidating defect and eight further
material ones, all resolved below (Section 9 records what changed and why). Not to be edited
after the run except in a dated "Deviations" section.

**Harness:** `src_research/pipeline_tuning.py`
**Closes:** the open decision in `04_methodology.tex`, `\gap{... whether the tuned per-dataset
configurations are fed back into the default UI configuration is a decision still to make.}`

---

## 1. Why a second tuning experiment

The completed tuning-effectiveness experiment (Section~\ref{sec:tuning-experiment}, run
`20260628_153633`) answers *how effective tuning is*. Its per-cell `best_params` cannot be
shipped as UI presets:

| # | Defect of the 06-28 run | Fix here |
|---|---|---|
| D1 | Tuned on a 500-row seeded subsample; the app builds the full dataset (wine: 6497 rows). `min_cluster_size`/`min_samples` are absolute counts and do not transfer across `n`. | Tune at **full `n`**; parameterise the count knobs as fractions of `n` and report both forms. |
| D2 | Tuned a **flat** `DR -> cluster` pipeline. The app is **recursive**, reuses one config at every node, and skips pre-reduction under its own default — a configuration the 06-28 grid never evaluated. | Objective computed on the **real** artefact, `compute_analysis_tree`, the same code path the backend serves. |
| D3 | One 100-trial study per cell, UMAP unseeded; on wine's UMAP+HDBSCAN cell random search beat TPE (`+0.014` vs `-0.068`). | A **validation stage** (Section 6) rebuilds preset and baseline 5 times each and requires *complete separation*, not a point comparison. |
| D4 | Objective was DBCV alone; the thesis's own result is `r ~ 0.17` between DBCV and ARI. | Two objectives from one build, a Pareto front, a pre-registered selection rule, and **ARI as a veto gate** (A6), not merely a reported number. |

> **EQ1b.** For a given dataset, does a tuned configuration of the *deployed* pipeline beat
> the shipped defaults by more than build-to-build noise, on structure and view at once,
> without degrading agreement with known structure — and is it therefore worth shipping as a
> starting-point preset?

## 2. Datasets, features, and the baseline

Scope: the "light" tier. MNIST / Fashion-MNIST / Olivetti are excluded by decision — they are
not interactively buildable, so a preset for them would be untestable under A3.

**Feature columns.** At design time `backend/datasets.py::default_feature_cols` returned *every*
column except `row_id`, so the shipped default passed the label columns (`target_*`, and wine's
then-unprefixed `is_red` / `quality`) into the clustering. Tuning under that selection would tune
the pipeline to cluster on the answer. This experiment therefore pins a **label-free feature
set**, enumerated literally, and a preset ships *with* its feature list:

> Fixed 2026-08-02: the default now excludes `target_*`, and the unprefixed label columns were
> renamed (`target_is_red`, `target_quality`, `target_manifold_position`), so the app default and
> the pinned sets below now coincide. The pinning stays — the feature space a preset was tuned on
> should be reproducible without depending on the app default of the day.

| Dataset | n | d | features | labels (reporting only) |
|---|---|---|---|---|
| Wine quality (Low) | 6497 | 11 | the 11 physicochemical columns (excl. `target_quality`, `target_is_red`) | `target_is_red` |
| Digits (Low) | 1797 | 64 | `px_0..px_63` | 10 classes |
| Breast cancer (Low) | 569 | 30 | the 30 sklearn features | 2 classes |
| Concentric rings (Low) | 1800 | 2 | `x`, `y` | 3 rings |
| Iris (Low) | 150 | 4 | the 4 sklearn features | 3 species |

**Baseline `B`** — the config a preset must beat — is the **app-effective default**: the
values in `src/config_defaults.py::default_config()` with `hclust_umap_n_components = d`,
because `App.tsx` overwrites that key with the feature count on every dataset change. Its
literal contents are written to `baseline.json` at run time. The `hclust_umap_n_components = 2`
value that `config_defaults.py` ships is recorded as a secondary reference (`B2`) but is not
what any user gets.

Measured behaviour of `B` before the run (full n, label-free features, one build): Breast
cancer yields **1 leaf — no hierarchy at all**; Wine yields 2 leaves; Iris 2; rings 3;
digits 8. `B` is therefore a weak baseline on some datasets, which is itself part of the
answer and must be stated plainly rather than used to inflate the preset's win.

## 3. Objectives

One build yields both. Both maximised. Both computed on the *same* tree, so the clustering
config's effect on which views exist is inside the measurement, not outside it.

**O1 — structure (`dbcv_leaf`).** DBCV (`kDBCV`, root-scaled original feature space) of the
**leaf partition**: each point takes its leaf's id; points dropped as HDBSCAN noise at any
level take `-1`. Verified property of the implementation, stated here because it changes the
interpretation: kDBCV excludes the noise block from cluster scoring but keeps noise in the
denominator (`weighted_score = sum (|C_i| / n) * validity_i`), so

> **O1 = coverage x mean cluster validity, with a hard ceiling of `1 - noise_frac`.**

It is a *composite* of "found good clusters" and "discarded few rows". `dbcv_leaf_noisefree`
(same score over non-noise rows only) and `noise_frac` are recorded separately so the two
halves can be read apart.

**O2 — view (`tnc_mean`).** Unweighted mean over scored nodes of `(trustworthiness +
continuity) / 2`, at a **fixed `k = 10`**, over nodes with `n >= 21`. The app's own
`_score_node` uses `k = min(20, (n-1)//2)`, which makes `k` vary per node, so the app-native
mean averages metrics computed at different `k` and is not comparable across configs with
different tree shapes. Fixed `k` is a deliberate deviation from the displayed number, recorded
as such; `tnc_mean_app` is not computed (it would double the ZADU cost per build).

Recorded, never optimised: `ari`, `ami`, `ari_depth1`, `n_leaves`, `median_leaf_size`,
`n_scored_nodes`, `scored_coverage`, `noise_frac`, `zero_embed_nodes`, `build_seconds`,
`preclustering_skipped`, `exception`.

**Reference line (pre-registered).** DBCV of the *ground-truth* partition, same space, same
implementation:

| Iris | Rings | Digits | Wine | Breast cancer |
|---|---|---|---|---|
| −0.144 | +0.200 | −0.454 | −0.678 | −0.723 |

On four of five datasets the correct answer scores negative. **Maximising O1 therefore moves
away from the labelled structure on those datasets by construction.** This is why A6 exists,
and why no claim of the form "the preset finds better clusters" may be made from O1 alone.

## 4. Search space and guards

Not tuned, with reasons: `normalize = True` (standardisation is a data-semantics decision);
clustering method (the recursive path calls `compute_clusters(..., method="HDBSCAN")`
unconditionally, so DBSCAN/K-Means/GMM are **unreachable** in the deployed pipeline whatever
the config says); the pre-clustering reducer (hardcoded UMAP).

| Parameter | Range | Note |
|---|---|---|
| `hierarchical_layers` | {1, 2, 3} | app default 1 |
| `hclust_umap_n_components` | int [2, d] | `== d` skips pre-reduction; for rings (d=2) the axis is degenerate and only that value exists |
| `f_min_cluster_size` | float [21/n, 0.25] | `hclust_min_cluster_size = max(21, round(f*n))` |
| `r_min_samples` | float [0.05, 1.0] | `hclust_min_samples = max(1, round(r * min_cluster_size))` — sampled *relative* so `min_samples > min_cluster_size` (24% of the naive box, all degenerate) is unreachable |
| `umap_n_neighbors` | int [5, 50] | **shared** by pre-reduction, the UMAP view, and the KDE (L1) |
| `umap_min_dist` | float [0.0, 0.5] | shared, as above |
| `method` | {PCA, UMAP, t-SNE} ∪ {MDS if n ≤ 2000} | view embedding only |
| `tsne_perplexity`, `tsne_learning_rate` | [5, 50], [10, 1000] | conditional on `method == t-SNE` |
| `mds_n_init`, `mds_max_iter` | [1, 4], [50, 300] | conditional on `method == MDS`; note MDS also runs at every internal node for `rel_position` regardless of `method`, so these affect `build_seconds` on every trial |

The `min_cluster_size >= 21` floor is the fix for the review's critical finding: without it,
a config that shatters the data into 2-point leaves wins **both** objectives, because leaves
with `n < 10` get `None` scores and drop out of O2 — a 171-leaf breast-cancer config scored
`tnc_mean = 0.915` computed from **one** node, the root. With the floor, every leaf HDBSCAN
can emit is large enough to be scored, so O2 is always measured on the views the config
actually creates.

**Degeneracy → worst value `(-1.0, 0.0)`:** fewer than 2 leaves; `noise_frac >= 0.5`;
`scored_coverage < 0.8`; any exception (recorded in an `exception` column — when these runs were
performed the deployed path raised `TypeError` from `umap/spectral.py` on very small clusters via
the unguarded `compute_cluster_kde`, and this is stochastic, so it must be a first-class outcome,
not a crash); or a build exceeding the time guard.

> 2026-08-02: `compute_cluster_kde` has been removed, so that specific crash source is gone. The
> pre-clustering UMAP and the MDS over centroids are still unguarded, so the rule stands as
> written; expect a lower exception rate on any re-run than the recorded trials show.

**Time guard.** Hard abort at **180 s** wall clock per build, implemented as a spawned
subprocess that is terminated on timeout — a post-hoc measurement would still pay the cost.
The guard covers tree construction *and* scoring. Hardware is pinned for reproducibility:
2 vCPU, 7 GB RAM, CPython 3.13.13, no GPU; the baseline build on wine takes ~95 s on this
machine, so the effective search space for wine is materially narrower than for the other
datasets. The count of trials lost to the guard is reported per dataset. The guard is part of
the objective: a configuration that cannot be built interactively is not a usable preset.

**Sampler.** Optuna multi-objective TPE, 40 trials (10 random startup), seed 42. UMAP and
t-SNE are not seedable through this config (L2), so trials are not exactly reproducible.

## 5. Selection rule (applied without discretion)

Baseline builds are split so that selection and testing never share data: **10 baseline
builds**, 1–5 (`B_select`) for this rule, 6–10 (`B_test`) for Section 6.

From the Pareto front of the 40 trials, the candidate is

> the trial with the highest `dbcv_leaf` among those with
> `tnc_mean >= mean(tnc_mean | B_select) - 0.01`.

Ties: lower `build_seconds`, then lower `hierarchical_layers`. If the filtered set is empty,
**no preset is proposed** and the defaults are retained.

## 6. Acceptance criterion (pre-registered)

5 fresh builds of the candidate preset, compared against `B_test`. **All six must hold:**

- **A1 — separation.** `min(dbcv_leaf | preset) > max(dbcv_leaf | B_test)`: the two sets of 5
  builds must not overlap at all. Under the null this is the exact one-sided permutation
  outcome `1/C(10,5) = 1/252 ≈ 0.004`; across 5 datasets the expected number of spurious
  adoptions is `0.02`. Revision 1's rule (`gain > sd(baseline)`) was measured by the review to
  fire with probability 0.094 under the null and 0.187 when the preset merely had twice the
  baseline's variance — it is replaced, not softened.
- **A2 — view not degraded.** `mean(tnc_mean | preset) >= mean(tnc_mean | B_test) - 0.01`.
- **A3 — still interactive.** `median(build_seconds | preset) <= min(180, 3 x median(build_seconds | B_test))`.
- **A4 — reliable.** 5/5 preset builds complete without exception.
- **A5 — no shattering, no mass discarding.** `median_leaf_size >= 21` in 5/5 builds and
  `mean(noise_frac | preset) <= mean(noise_frac | B_test) + 0.05`.
- **A6 — label agreement not sacrificed.** `mean(ari | preset) >= mean(ari | B_test) - 0.05`.
  Justified by the reference line in Section 3: O1's optimum lies away from the labelled
  structure on 4/5 datasets, so an internal-index-only criterion would license a preset that
  is measurably worse at the one thing that can be checked against ground truth.

Failing any of these, the dataset **keeps its defaults**, and that is the reported result. No
re-rolling, no per-dataset relaxation. "Defaults retained everywhere" is a publishable answer
to the `\gap{}` and must not be treated as a failed run.

## 7. Outputs

`outputs/experiments/<timestamp>/`: `baseline.json`, `trials.csv`, `pareto.csv`,
`validation.csv`, `presets.json` (adopted only, in `AnalysisConfig` key form, annotated with
run id, feature list, and the A1–A6 margins), `verdicts.md`, `reference_dbcv.csv`.

## 8. Limitations stated in advance

- **L1 — coupled knobs.** `umap_n_neighbors`/`umap_min_dist` are read by the pre-clustering
  UMAP and the UMAP view. A preset cannot set them independently. Property of the config
  schema; reported, not worked around. (These runs had a third consumer, `compute_cluster_kde`,
  removed 2026-08-02; the coupling itself is unaffected.)
- **L2 — dead seeds.** `_umap()`/`_tsne()` never receive `umap_random_state`/
  `tsne_random_state`, so those UI controls do nothing and every build is stochastic
  (`mds_random_state` *is* threaded). The validation stage measures the spread instead.
- **L3 — coarse search.** 40 trials of a stochastic pipeline over ~8 mixed dimensions. A
  preset is a *starting point*, not an optimum. A re-run may select different parameters
  (L5).
- **L4 — internal indices.** A1–A6 establish a denser, no-worse-viewed, no-less-label-aligned
  tree; not that the explanations are more *useful*. That requires the expert study.
- **L5 — one run.** The validation stage bounds noise on the *selected* config, not on the
  selection itself.
- **L6 — granularity confound.** Both O1 (via coverage) and O2 (T&C rises with node size:
  measured 0.69 at n=12 to 0.97 at n=1797 on digits) reward few large leaves, so the Pareto
  front is not a clean trade-off — the two objectives partly share a latent granularity axis.
  `n_leaves` and `median_leaf_size` are reported for every trial so this is visible, and A5
  bounds the degenerate end only.
- **L7 — measurement-space confound.** Trials with `hclust_umap_n_components == d` cluster in
  exactly the space O1 measures, mechanically aligning HDBSCAN's density notion with DBCV's;
  trials that pre-reduce do not. `preclustering_skipped` is reported per trial and for any
  adopted preset.
- **L8 — hardware-bound guard.** A3 and the 180 s abort are facts about the pinned machine,
  not about the config.

## 9. Changes from revision 1 (post-review)

Critical: min-cluster-size floor of 21 added (shattering configs won both objectives);
baseline pinned literally and switched to the app-effective one (revision 1 was ambiguous, and
one candidate gave `sd = 0`, making the old A1 vacuous); exceptions made a first-class outcome
(A4). Major: A1 replaced by complete separation; fixed `k` in O2; O1 documented as a composite
with `noise_frac`/`noise-free` columns; the ground-truth DBCV reference line added with A6 as
its consequence; A5 added; time guard made a real abort with pinned hardware and a
baseline-relative A3; selection and test split across 10 baseline builds; feature set made
label-free after finding that the shipped default feeds label columns into the clustering.
Minor: `min_samples` sampled relative to `min_cluster_size`; AMI added; zero-embedding
fallbacks detected and counted; fresh `default_config()` per build and params read from
`trial.params`; kDBCV status derived from the label vector before scoring, since kDBCV
returns a legitimate-looking `-1.0` for "not enough clusters"; `np.float_` shim carried in.

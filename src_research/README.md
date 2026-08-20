# `src_research/` — offline experiment harnesses

The evidence behind the thesis's quantitative chapters. Nothing here is part of the running
application: no harness is imported by `backend/` or `frontend/`, and the app never imports
this package. Each harness is a batch script that drives the same calculation layer
([`src/README.md`](../src/README.md)) the server drives, then writes CSVs and figures to
`outputs/experiments/<timestamp>/`.

> ### Where the design record lives
>
> Every harness here was written against a design fixed **before** the harness was
> implemented and its run executed. That is why the code says things like "fixed by the
> design, §4" and "must not be tuned to taste": those constants were chosen ahead of any
> result, not after one.
>
> The design record itself is part of the thesis, not of this repository. The
> `design §N` / `design SSN` pointers in the module docstrings and `CONFIG` blocks resolve
> against it — they mark a constant as pre-registered rather than picked after the fact.
>
> **The thesis is authoritative for the design; the code in this directory is what produced
> the numbers.**

---

## Why the harnesses are shaped this way

Two conventions run through all of them, and both exist to make the results checkable:

**Pre-registration.** The guards, objectives, metrics and acceptance criteria in the code
were fixed before the run and must not be tuned to taste afterwards. Several harnesses say
so in their module docstring. If a check fails, *the failure is the finding* — a degenerate
build gets reported, never re-rolled.

**Failures are named, never absorbed.** A region that could not be projected is recorded
and dropped from paired tests rather than scored on fabricated coordinates; the count that
dropped out is printed and written to a CSV. A shrinking denominator is not a neutral
event, so it is made loud.

---

## Harnesses

| Harness | Question | Run directory |
|---|---|---|
| `hierarchical_vs_flat.py` | RQ1 — H1a/H1b | `20260628_182948` … `184827` (6) |
| `planted_subspace_recovery.py` | RQ1-S — clean H1b | `20260628_195214` |
| `predicate_stability.py` + `…_analysis.py` | RQ2 — H2 | `20260711_115849` |
| `benchmark_workflow.py` | §6.5 census + C1–C4 | `20260728_185329` |
| `hyperparameter_tuning.py` | tuning effectiveness | `20260628_125924`, `20260628_153633` |
| `pipeline_tuning.py` + `…_b2.py` | EQ1b presets | `20260729_101836` (**halted**) |
| `dbcv_tuning.py` | standalone Optuna DBCV study | — |
| `rederive/` | post-hoc corrections | writes into the runs above |

§6's qualitative case study has **no harness here**. It carries the one claim the offline
experiments structurally cannot test — that the hierarchical lens lets an analyst *reach
and inspect* structure a global view would hide — and it is narrated in the thesis, against
a path and a set of reported numbers fixed in advance.

### RQ1 — `hierarchical_vs_flat.py`

Splits RQ1 into two claims that need different baselines and can come apart:

- **H1a — faithfulness.** For each hierarchy region, is the leaf's *local* re-embedding
  more faithful to that region's internal structure than the *global* embedding restricted
  to the same rows? Needs no ground truth. Paired per region.
- **H1b — label recovery.** Does the hierarchical leaf partition recover class labels
  better than one flat HDBSCAN?

Grid: 5 datasets × 4 DR methods × 5 replicate seeds, at depth 2, subsampled to 1000 rows
so t-SNE/UMAP and the O(n²) measures stay tractable. Writes `h1a_regions.csv`,
`h1a_summary.csv`, `h1b_recovery.csv`, `h1a_skipped_regions.csv` and paired-delta plots.

**H1b came out negative** — the flat partition recovered class labels better (mean ARI
≈ 0.40 vs 0.86). The diagnosis was over-segmentation plus the standing caveat that classes
are not clusters and certainly not subspaces, which is what motivated the next experiment.

> **Only UMAP and MDS carry replicate variance.** `_tsne` passes no `init`, and
> scikit-learn's default `init="pca"` never consults `random_state`; PCA takes no seed at
> all. For those two methods the `SEEDS` levels are identical repeats, not replicates —
> a property of the frozen `src/analysis/dim_reducer`, not of this harness. It matters for
> the paired test's `n`; `scripts/checks/05_h1a_replicate_collapse.py` demonstrates it.

### RQ1-S — `planted_subspace_recovery.py`

The clean synthetic counterpart to that negative result. Real class labels structurally
cannot express the case the hierarchy should win — **nested, multi-scale subspace clusters
that are globally hidden but conditionally visible** — so this plants exactly that and asks
whether recursion recovers the *fine* sub-clusters better than a flat clustering, and where
the crossover in `rho` falls.

Four conditions including an `oracle_conditional` upper bound (HDBSCAN within each true
coarse group), which bounds the hierarchy's available headroom. Because ARI punishes
over-segmentation — the lesson from H1b — homogeneity / completeness / V are reported
alongside ARI/NMI and read together, with within-group ARI leading.

Synthetic data is cheap, so this uses many seeds, and each cell's seed is written into
`umap_random_state` / `tsne_random_state` / `mds_random_state` as well as the generator —
capturing both generator and embedding variance.

### RQ2 — `predicate_stability.py` + `predicate_stability_analysis.py`

Tests H2: as the relaxation threshold `t` drops from 1.0, does the predicate describing a
selection get more stable under small perturbations of that selection, while its
specificity degrades only gradually — so that a favourable operating point exists?

Uses a purpose-built **axis-parallel** generator: `C` clusters each owning `r` relevant
dimensions, disjoint across clusters. Deliberately axis-aligned, because range predicates
can express these boxes — the mirror image of the RQ1 generator being fair to *that*
method, and a stated threat to validity rather than a hidden one. Knobs sweep the
within-cluster skew (including an adversarial bimodal shape where the median-based severity
split is misleading) and the separation margin.

**The unit of analysis is the selection, never the replicate pair.** Metrics are aggregated
to one number per (selection, method, split, t, delta) *before* any test — pseudo-replication
designed out rather than corrected for.

The analysis half is a separate module so statistics, verdicts and figures can be recomputed
from an existing `stability_records.csv` without repeating the sweep:

```bash
uv run python -m src_research.predicate_stability_analysis outputs/experiments/<timestamp>
```

### §6.5 — `benchmark_workflow.py`

The only harness that tests no hypothesis: a descriptive census of what the **shipped
defaults** produce on the three real benchmark datasets. A purely descriptive section
invites post-hoc storytelling, so the containment is (a) every reporting and aggregation
rule fixed in the design before any code, and (b) four **consistency checks** C1–C4 —
out-of-sample predictions derived from the four completed experiments, so the section has
falsifiable content after all.

It deliberately reuses the other harnesses' helpers (`prepare_dataset`, `collect_leaves`,
`build_predicate`, `admitted_mask`) rather than reimplementing them.

### Tuning — `hyperparameter_tuning.py`, `pipeline_tuning.py`

`hyperparameter_tuning.py` measures how effective automated tuning is on a **flat**
DR → cluster pipeline over a 500-row subsample, along two tracks (internal cluster quality,
DR faithfulness) and four effectiveness axes. Everything is seeded, so a rerun reproduces
the grid — with the flip side, stated in its own docstring, that it measures **no embedding
variance at all**: a "gain" here is a gain at one seed, not an expected gain.

`pipeline_tuning.py` (EQ1b) optimises the **deployed recursive pipeline** at full `n`
instead: every trial calls the same `compute_analysis_tree` the backend serves. It is the
only harness with CLI flags:

```bash
uv run python -m src_research.pipeline_tuning --datasets "Iris (Low)" --trials 40 --out my_run
```

> **This run was halted mid-run**, during the wine dataset, and nothing was integrated —
> no frontend change, no thesis change.
> `pipeline_tuning_b2.py` is a **disclosed deviation** written after seeing the Breast
> cancer result and before running: the pre-registered baseline produced one leaf in 10/10
> builds there, leaving the acceptance criterion not evaluable. It has never been run.

### `dbcv_tuning.py`

A small standalone 200-trial Optuna DBCV study, not tied to a thesis section. It patches
`np.float_` back in at import because kDBCV's type hints reference an alias NumPy 2.0
removed, while its actual computation is NumPy-2 compatible.

---

## `rederive/` — corrected aggregates, not reruns

The pre-release code review found four defects in the **aggregation** of already-run
experiments. The per-record CSVs on disk were sound; only the summaries derived from them
were wrong. So this package recomputes those summaries from the raw records:

| Module | Defect corrected |
|---|---|
| `h1a.py` | `HIGHER_IS_BETTER` declared both MRRE terms lower-is-better, but they arrive already inverted into a similarity — so `win_rate` and `rank_biserial` were computed against the wrong direction on every MRRE row, while `median_delta` never consulted the map. That is why the shipped summaries contain rows whose two effect sizes contradict each other. |
| `h2b.py` | The `non_nested` control forces `eff_rho = 1`, so its six rho levels are six recomputations of one condition — and the nesting gap averaged those six duplicates against six genuinely distinct ones. Plus a row whose two halves came from differently-filtered pivots. |
| `internal_external.py` | The figure joined `ari_base` from rows that never exist for Track A, then `.fillna(0)` — so an axis labelled "ARI change vs baseline" actually carried raw tuned ARI, and a cell whose tuning *lowered* ARI plotted as a gain. |
| `verdicts.py` | `n_selections` reported seeds × selections. On the wine arm the seeds are rebuilds of one dataset — the same leaves, five times over — so the column overstated the sample fivefold. The test statistic is unchanged; the sample is now reported as distinct selections. |

**No experiment was re-executed and no original file was modified.** Each re-derivation
writes into a fresh `rederived_20260813/` subdirectory of the run it corrects, next to a
`DELTAS.md` listing every derived number that moved, old → new. Ten of the thirteen run
directories have one.

Where a corrected quantity is **not recoverable** from what was persisted, the driver says
so rather than approximating it — `internal_external.py` refuses the part of its correction
that would need a baseline ARI that was computed at run time and then discarded.

```bash
uv run python -m src_research.rederive     # needs outputs/ present; regenerates byte-identically
```

---

## Running a harness

From the repository root, with the environment from `scripts/prepare_env.sh`:

```bash
uv run python -m src_research.hierarchical_vs_flat
uv run python -m src_research.planted_subspace_recovery
uv run python -m src_research.predicate_stability
uv run python -m src_research.benchmark_workflow
uv run python -m src_research.hyperparameter_tuning
```

Each stamps a fresh `outputs/experiments/<UTC timestamp>/` and writes its CSVs, a
`plots/` subdirectory (matplotlib runs headless — PNGs only, never a window) and, for some,
a `run_meta.json`. Grid constants live in a marked `CONFIG` block near the top of each file.
Most parallelise across grid cells with joblib at `PARALLEL_JOBS = -1`; expect hours, not
minutes, for the full RQ1 grid.

Importing a harness is side-effect-safe — the work runs under `__main__`.

> ### `outputs/` is read-only
>
> The directories under `outputs/experiments/` back specific thesis sections. **Never rerun
> or overwrite one.** They were produced before the release freeze, under the environment
> recorded in each run directory; a rerun on a later environment is a separate measurement
> and must be reported as one, never mixed into a table with pre-freeze numbers.
>
> `outputs/` is not tracked in git, so a fresh clone has none of it — and `rederive/`
> cannot run there.

`20260728_190741_depth2_diag` is an ad-hoc depth-2 noise diagnostic whose producing script
was removed before release; it has no harness in this directory.

---

## Related

- [`scripts/checks/`](../scripts/checks/README.md) — four regression checks retained from
  the same review, including the two that prove `src/evaluation/neighbor_metrics.py` is
  numerically equivalent to ZADU. The thesis's faithfulness numbers rest on that.
- [root README](../README.md#experiment-outputs) — the full run-directory → thesis-section
  table.

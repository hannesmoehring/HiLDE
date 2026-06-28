# Experiment Design: Hierarchical vs. Flat Embeddings (RQ1)

Design document. The implementation will live next to `hyperparameter_tuning.py` in
`src_research/`. This file specifies *what* to measure and *why* before any code is
written, so the experiment is fair by construction rather than by accident.

---

## 1. Research question and hypotheses

**RQ1 (primary).** Can a recursive, density-based hierarchical decomposition (HDBSCAN +
a local DR per region) surface and isolate interesting subspace structure more effectively
than a single global ("flat") projection?

RQ1 actually bundles two distinct claims. Keeping them separate is essential, because they
need different baselines, different metrics, and different data, and because one can hold
while the other fails.

- **H1a — faithfulness of local views.** *For a given region, re-embedding that region on
  its own (the leaf's local projection) preserves the region's internal neighbourhood and
  distance structure better than reading the same points off a single global projection.*
  Needs **no ground truth**; runnable on every dataset today.

- **H1b — discovery of nested structure.** *The recursive decomposition separates
  sub-structure that a single, non-recursive (flat) clustering merges, so it recovers a
  known ground-truth partition more accurately.* Needs **ground-truth labels** (planted
  synthetic structure, or benchmark class labels read with the "classes are not clusters"
  caveat).

The null hypotheses are the ones we genuinely try to reject: H1a-null = local re-embedding
is no more faithful than the global view (deltas centred on zero); H1b-null = recursion
recovers the ground truth no better than a single flat clustering.

---

## 2. The crux: what makes the comparison *fair*

A sloppy version of this experiment is easy to "win" and proves nothing. The two failure
modes to design against:

1. **Comparing different things.** If the hierarchical condition re-projects 80-point
   regions while the flat condition is scored on the whole 500-point dataset, any
   difference is dominated by region size, not by the method. Faithfulness measures
   (trustworthiness/continuity, MRRE) depend on `n` and on the neighbourhood size `k`, so
   these must be held identical between conditions.

2. **Changing two variables at once.** The hierarchy differs from a flat baseline in *two*
   ways: (a) it re-projects each region locally, and (b) it discovers regions recursively.
   H1a isolates (a); H1b isolates (b). Do not conflate them in a single number.

### H1a comparison (the clean, paired core)

Hold the *region* and its *points* fixed; vary only the projection.

For each region `r` with original-space points `X_r` (the node's standardised features):

| Condition       | 2D embedding scored                                              |
|-----------------|------------------------------------------------------------------|
| **Hierarchical**| the leaf's own local embedding `node["embedding_original"]`       |
| **Flat**        | a single global embedding `E_global`, sliced to `r`'s rows        |

Both are scored against the **same** `X_r` with the **same** `k`. The only difference is
that the hierarchical embedding was optimised for `X_r` alone, while the flat embedding was
optimised for the whole dataset and then restricted to `r`. This is a *paired* comparison
(same region, same points, same original distances), which makes it statistically powerful
and immune to the size confound.

> **Conditioning caveat (state this in the thesis).** In the primary H1a design the regions
> are the hierarchy's own leaves, so the comparison answers: *"given the regions the
> hierarchy identifies, are they more faithfully shown locally than globally?"* It does not
> claim the regions themselves are the right ones — that is H1b. A region-definition that is
> neutral to both methods is given as a robustness variant in §8.

### H1b comparison (structure recovery)

Hold the *clustering algorithm* fixed (HDBSCAN with identical hyperparameters); vary only
*recursion*.

| Condition       | Partition produced                                                       |
|-----------------|---------------------------------------------------------------------------|
| **Hierarchical**| leaf membership of the full tree (`compute_analysis_tree`)                 |
| **Flat**        | one non-recursive HDBSCAN run on the same pre-processed space             |

Both partitions are compared to the ground-truth labels with ARI/NMI. Using the *same*
algorithm with and without recursion isolates recursion as the variable (rather than
confounding it with "HDBSCAN vs. K-Means").

---

## 3. Conditions, defined against the actual code

Grounding (verified against `src/analysis/analysis_routine.py` and
`src/evaluation/evaluate.py`):

- `compute_analysis_tree(df, feature_cols, config)` standardises features once
  (`StandardScaler`, reused), optionally UMAP-pre-reduces for **clustering**, and recurses
  with `_build_next`. Stop conditions for a leaf: `depth >= hierarchical_layers`, or
  `len(X) < 2 * hclust_min_cluster_size`, or `< 2` non-noise clusters.
- **Every node stores `embedding_original`** — a 2D projection of that node's *original
  standardised features* `X_orig` via `config["method"]` (PCA / t-SNE / UMAP / MDS),
  computed by `_embed_original`. This is the local view the UI shows and the scorer uses.
- `evaluate._score_node(X, emb, labels)` returns the ZADU scores (stress, trustworthiness,
  continuity, MRRE; CADI when labels given), with `k = min(EVAL_K=20, (n-1)//2)` and
  neighbourhood metrics only when `n >= 10`. **Reuse this function unchanged** for both
  conditions so scoring is identical.
- `node["row_indices"]` maps a node's points back to original rows — this is what lets us
  slice the global embedding to a region.

**Flat global embedding `E_global`.** Use the *same* DR method and parameters the hierarchy
uses for leaves (`config["method"]`), fit on the whole dataset's standardised features
`X_all` to 2D. Concretely this is one call to `_embed_original(X_all, config)`, so the flat
and hierarchical embeddings come from identical code paths and differ only in their input
scope.

> Do **not** use the UMAP-pre-reduced clustering space as the flat embedding. The leaf
> `embedding_original` is computed on original features, so the flat baseline must be too,
> or the methods are not comparable.

---

## 4. Metrics

### H1a (per region, paired)

Reuse `_score_node`. Primary and secondary roles:

- **Primary: trustworthiness, continuity** — local neighbourhood preservation, the direct
  operationalisation of "is the local view faithful?". Bounded `[0,1]`, higher better.
- **Secondary: MRRE (false / missing)** — rank-weighted refinement of the same idea.
- **Reported with caution: stress** — a *global* distance-preservation measure. A local
  re-embedding will usually win on local metrics; stress checks it does not do so by
  wrecking global distances. ZADU stress is scale-sensitive, so compare only *within* a
  region (the pairing handles this) and never across regions of different size.
- **Derived: per-region delta** `Δ = score_hierarchical − score_flat`, and the **win rate**
  = fraction of regions with `Δ > 0`.

### H1b (per dataset, vs. ground truth)

- **ARI** and **NMI** of {hierarchical leaf partition, flat partition} vs. ground-truth
  labels. Higher = better recovery.
- **Merge diagnostic:** number of ground-truth classes that the *flat* clustering lumps
  into one cluster but the hierarchy separates across different leaves (a direct,
  interpretable count of "structure the flat view merged"). Defined as: for each flat
  cluster, the number of distinct ground-truth classes it contains that end up in
  different hierarchical leaves.
- **Noise accounting:** report the noise fraction (`label == -1`) for both, since HDBSCAN
  noise inflates/deflates ARI; exclude noise consistently and state how.

---

## 5. Datasets

Use the existing `DATASETS` registry (`src/ui/data.py`), subsampled to ≤ 500 rows (seeded)
to keep t-SNE/UMAP and the O(n²) measures tractable — matching the tuning experiment.

| Dataset            | Ground truth?         | Used for     | Why |
|--------------------|-----------------------|--------------|-----|
| Concentric rings   | yes (`ring_*`)        | H1a + H1b    | non-convex nested density; the case the hierarchy *should* win |
| Wine quality       | yes (`is_red`)        | H1a + H1b    | interpretable, low-dim tabular |
| Breast cancer      | yes (2 classes)       | H1a + H1b    | moderate dim |
| Digits             | yes (10 classes)      | H1a + H1b    | manifold structure, higher dim |
| Swiss roll         | no discrete classes   | H1a only     | continuous manifold; H1b not meaningful |
| (planted subspace) | yes (planted)         | H1a + H1b    | **pending the generator** — the only source of *subspace* ground truth |

Labels for H1b are reconstructed from the one-hot `target_<name>` columns
(`argmax` over `target_*`), as the registry already encodes them that way.

> **Gap.** Real benchmark labels are *classes*, not density clusters or subspaces, so H1b on
> them is reported under the "classes are not clusters" caveat. The clean H1b result needs
> the planted-subspace generator (separate gap, priority 2 in the thesis). H1a does not.

---

## 6. Procedure (pseudocode)

```
for dataset in DATASETS_TO_RUN:
    df = prepare_dataset(dataset)            # reuse the tuning harness loader/subsampler
    X_all = standardise(df[features])        # same StandardScaler the tree uses
    y = labels_from_target_columns(df)       # None if no ground truth

    for seed in SEEDS:                        # e.g. 5 seeds — embeddings are unseeded
        config = base_config(method=DR, seed=seed)

        # ---- hierarchical condition ----
        tree   = start_evaluation(df, features, config)   # builds tree + attaches scores
        leaves = collect_leaves(tree)                      # walk next_object_layer

        # ---- flat condition ----
        E_global, _ = embed_original(X_all, config)        # one global 2D embedding

        # ---- H1a: paired per-region faithfulness ----
        for leaf in leaves:
            idx   = leaf["row_indices"]
            X_r   = X_all[idx]
            emb_h = leaf["embedding_original"]             # local view (already scored too)
            emb_f = E_global[idx]                          # global view, same points
            s_h   = score_node(X_r, emb_h, None)           # reuse evaluate._score_node
            s_f   = score_node(X_r, emb_f, None)
            record_region_row(dataset, seed, region_id, n=len(idx),
                              k=s_h["k"], cond="hier", **s_h)
            record_region_row(..., cond="flat", **s_f)     # k forced equal to s_h["k"]

        # ---- H1b: structure recovery ----
        if y is not None:
            part_h = leaf_partition(leaves)                # point -> leaf id
            part_f = hdbscan_once(clustering_space(X_all, config))   # no recursion
            record_recovery_row(dataset, seed,
                                ari_h=ARI(y, part_h), nmi_h=NMI(y, part_h),
                                ari_f=ARI(y, part_f), nmi_f=NMI(y, part_f),
                                merges=merge_diagnostic(part_f, part_h, y),
                                noise_h=..., noise_f=...)
```

Notes for the implementer:

- `score_node` for the flat condition must be called with **the same `k`** the hierarchical
  call chose for that region (`s_h["k"]`), or the metrics are not comparable. `_score_node`
  derives `k` from `n`, and `n` is identical between conditions, so this is automatic — but
  assert it.
- `clustering_space(X_all, config)` = the optional UMAP pre-reduction the tree applies
  before HDBSCAN (mirror the `hclust_umap_n_components` branch in `compute_analysis_tree`),
  so the flat HDBSCAN sees the same representation the recursive one starts from.
- Skip regions with `n < 10` for the neighbourhood metrics (they return `None`); still
  record them with `stress` where available and an explicit `k = None`.

---

## 7. Confounds and how each is controlled

| Confound | Risk | Control |
|----------|------|---------|
| Region size | bigger/smaller `n` changes T&C/MRRE | pairing fixes `n` per region exactly |
| Neighbourhood `k` | different `k` ⇒ different metric | identical `n` ⇒ identical `k`; assert it |
| Stochastic embeddings | one lucky/unlucky seed | ≥ 5 seeds; report mean ± std; paired tests across (region × seed) |
| DR method | "win" might be t-SNE vs PCA, not local vs global | same `config["method"]` both conditions; optionally repeat over {PCA, t-SNE, UMAP} as a factor |
| Pre-reduction space | flat HDBSCAN seeing a different space | mirror the tree's pre-reduction for the flat clustering |
| Region selection bias | leaves are hierarchy-chosen (H1a) | acknowledged in §2; robustness variant in §8 uses method-neutral regions |
| Noise handling (H1b) | HDBSCAN `-1` skews ARI | exclude noise identically in both partitions; report noise fraction |

---

## 8. Robustness / secondary variants

1. **Method-neutral regions for H1a.** Re-run H1a with regions defined *independently* of
   the hierarchy — e.g. the ground-truth classes, or fixed equal-size spatial cells — and
   score local-re-embedding-of-region vs global-embedding-of-region. If the hierarchical
   advantage survives method-neutral regions, the result is much stronger and answers the
   conditioning caveat head-on.
2. **DR method as a factor.** Repeat across `method ∈ {PCA, t-SNE, UMAP}`. Expect the local
   advantage to be largest for t-SNE/UMAP (neighbourhood-optimising) and smallest for PCA.
3. **Depth sensitivity.** Sweep `hierarchical_layers ∈ {1, 2, 3}`; the H1a advantage should
   grow then plateau, and shallow trees (few leaves) should show little effect — a useful
   negative control.

---

## 9. Statistical analysis

- **H1a:** the unit is a (region × seed) pair. Use the **Wilcoxon signed-rank test** on the
  paired deltas `Δ` per dataset (non-parametric, paired, no normality assumption). Report
  the median `Δ`, an effect size (matched-pairs rank-biserial correlation or Cliff's delta),
  the **win rate**, and a paired-delta distribution plot. Aggregate per dataset and then
  across datasets; do **not** pool raw regions across datasets (scales differ).
- **H1b:** report ARI/NMI as mean ± std over seeds, with a Wilcoxon signed-rank test across
  seeds per dataset on `ari_h − ari_f`. With few datasets this is descriptive more than
  inferential — say so.
- **Multiple comparisons:** several datasets × several metrics ⇒ correct (Holm/BH) or, more
  honestly, pre-register T&C as primary and treat the rest as secondary/exploratory.

---

## 10. Outputs (match the tuning harness conventions)

Write to `outputs/experiments/<timestamp>/` (as `hyperparameter_tuning.py` does):

- `h1a_regions.csv` — one row per (dataset, seed, region, condition): `n, k,
  trustworthiness, continuity, mrre_false, mrre_missing, stress`.
- `h1a_summary.csv` — per (dataset, metric): median Δ, win rate, Wilcoxon p, effect size.
- `h1b_recovery.csv` — per (dataset, seed): `ari_h, nmi_h, ari_f, nmi_f, merges, noise_h,
  noise_f`.
- Figures: paired-delta distributions (violin/box per dataset) for T&C → `fig:rq1-h1a`;
  ARI hierarchical-vs-flat bars → `fig:rq1-h1b`. These slot into
  `sections/06_evaluation.tex` (the synthetic + benchmark sections) and replace the
  placeholder tables.

---

## 11. Threats to validity / how this could mislead

- **Faithfulness ≠ usefulness.** A locally faithful view of a region the analyst does not
  care about still scores well. H1a measures view fidelity, not analytic value; the case
  study and any expert feedback are what connect fidelity to usefulness. State this.
- **Local metrics structurally favour small embeddings.** Re-projecting fewer points is an
  *easier* embedding problem, so some H1a advantage is expected almost by construction. The
  honest claim is therefore not "hierarchy wins" but "*by how much*, and does global
  distance (stress) degrade as the price?" Report stress alongside, and treat a large T&C
  gain with flat/improved stress as the strong result, a T&C gain bought by wrecked stress
  as the weak one.
- **Self-selected regions (H1a primary).** Addressed by the §8.1 method-neutral variant; if
  that variant is not run, scope the claim explicitly to hierarchy-defined regions.
- **Classes ≠ clusters (H1b on real data).** Benchmark labels need not be density clusters,
  so a flat clustering can "lose" on ARI for reasons unrelated to merging real structure.
  The planted-subspace generator is the only fully clean H1b source.
- **Few datasets.** External validity is limited; do not over-generalise from 4–5 datasets.

---

## 12. Open design decisions (confirm before implementing)

1. **Primary region definition for H1a** — hierarchy leaves (paired, simplest, but
   conditioned) vs. method-neutral regions (stronger, more work). *Recommendation:* ship
   leaves as primary, add the method-neutral variant as the key robustness check.
2. **Seed count** — 5 is a reasonable default given runtime; 10 if cheap. The embeddings are
   unseeded (documented), so this directly bounds the variance we can report.
3. **DR method scope** — single method (match the UI default) for the headline, or the full
   {PCA, t-SNE, UMAP} factor. *Recommendation:* headline on the UI default, factor as
   secondary.
4. **H1b before or after the planted generator** — run H1b now on labelled benchmarks (with
   caveats) to de-risk the pipeline, then re-run on planted data once the generator exists.

---

## 13. Implementation checklist

Already exists (reuse, do not reimplement):

- `compute_analysis_tree` / `start_evaluation` — hierarchy + per-node scores.
- `_embed_original` — global flat embedding (call with `X_all`).
- `evaluate._score_node` — identical scoring for both conditions.
- `prepare_dataset` (tuning harness) — loading + seeded subsampling.
- `DATASETS`, one-hot `target_*` label convention.

To build:

- `collect_leaves(tree)` — walk `next_object_layer` to a flat list of `ExplorationObject`s.
- `labels_from_target_columns(df)` — `argmax` over `target_*` → integer labels (or `None`).
- flat HDBSCAN-once on the mirrored clustering space + ARI/NMI + `merge_diagnostic`.
- the paired H1a loop, the recovery loop, CSV writers, and the two figures.
- a `main()` with the same `OUTPUT_ROOT`/timestamp/`Parallel` pattern as
  `hyperparameter_tuning.py`.

Suggested filename: `src_research/hierarchical_vs_flat.py`.

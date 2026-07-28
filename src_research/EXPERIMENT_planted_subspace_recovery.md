# Experiment Design: Planted-Subspace Recovery (RQ1, clean H1b)

Design document. The implementation will live next to `hierarchical_vs_flat.py` in
`src_research/`. This file specifies *what* to measure and *why* before any code is
written, so the result is fair by construction rather than by accident.

This experiment exists because of a specific negative result. In
`EXPERIMENT_hierarchical_vs_flat.md` (H1b) the hierarchical leaf partition recovered
benchmark **class labels** *worse* than a single flat HDBSCAN (mean ARI ≈ 0.40 vs 0.86;
flat won ~99% of cells; concentric rings 0.33 vs 1.00). The diagnosis was over-segmentation
plus the standing caveat that **classes are not clusters, and certainly not subspaces**. The
honest conclusion was that the hierarchy's benefit is *representational, not partitional*.

The idea this experiment tests is the one piece that the benchmark data structurally cannot
test: *the hierarchy should earn its keep precisely when the interesting structure lives in
a **subspace** and is **globally hidden but conditionally visible**.* Real class labels never
isolate that case. A planted generator can. This is the "priority-2 gap" flagged in the prior
doc (§5, §11) made concrete.

> One-line thesis framing: the prior experiment showed the hierarchy is not a better
> *clustering of class labels*; this experiment asks whether it is a better *recoverer of
> nested subspace structure* — a different and, for this project, more load-bearing claim.

---

## 1. Research question and hypotheses

**RQ1-S (subspace specialisation of RQ1).** When ground-truth structure is organised as
*nested subspace clusters* — coarse groups separable in one feature subspace, and finer
sub-clusters separable only in a **different** subspace whose signal is globally smeared —
does the recursive density-based decomposition recover that structure better than a single
flat clustering?

The decomposition of the claim, kept deliberately falsifiable:

- **H2a — conditional recovery.** *The hierarchy recovers the fine (level-2) sub-clusters
  better than a flat full-space clustering, because recursion conditions on the coarse group
  before resolving the fine structure.* Measured against planted fine labels.

- **H2b — it is the nesting that matters, not subspaces in general.** *The advantage appears
  for **nested, multi-scale** subspace structure and **disappears** for non-nested
  (single-level) subspace structure, where the current method has no mechanism to help.*
  This is a **negative control on ourselves**: if the hierarchy "wins everywhere," the win
  is an artefact of the generator, not evidence for the mechanism.

- **H2c — crossover.** *There is a controllable scale-separation parameter `ρ` at which flat
  clustering transitions from adequate to failing while the hierarchy stays adequate.* The
  result we want is **not** a single rigged data point but a **crossover curve**.

The nulls we genuinely try to reject: H2a-null = hierarchy recovers fine structure no better
than flat (Δ centred on zero across `ρ`); H2b is *confirmed* (not rejected) if the
non-nested advantage is ~zero — confirming it strengthens the claim.

> **Why a new generator instead of more real datasets.** The benefit under test is
> conditional visibility. Benchmark labels (digits, wine, breast cancer) are coarse class
> partitions with no planted subspace nesting, so they cannot exhibit it; the prior H1b on
> them is the relevant "classes ≠ clusters" negative. Only planted structure controls the
> independent variable (`ρ`, subspace rotation) we need to sweep.

---

## 2. What the method actually does — and why the generator must match it

This is the most important section. Grounding (verified against
`src/analysis/analysis_routine.py`):

- `compute_analysis_tree` recurses by **partitioning samples**: it standardises features once
  (`StandardScaler`), optionally UMAP-pre-reduces for *clustering*, runs HDBSCAN, and recurses
  into each cluster with `_build_next`. **It does not select or weight features per node.**
- Therefore the only way recursion can surface subspace structure is if **conditioning on a
  sample subset changes which subspace dominates the local density / scale**. The method has
  no axis-selection mechanism (unlike classical subspace clustering — PROCLUS, SUBCLU,
  CLIQUE). Designing a generator whose subspaces are revealed by *feature selection* would
  test a method the project does not have, and would be unfair in the opposite direction.

So the planted structure must be **nested and multi-scale**, because that is the regime where
*sample conditioning* (what the method does) genuinely helps and a *single global density
threshold* (what flat HDBSCAN does) genuinely struggles:

> A single HDBSCAN pass commits to one mutual-reachability scale / one `min_cluster_size`. If
> the coarse separation is far larger than the fine separation, the global pass resolves the
> coarse groups and treats the fine sub-structure as within-cluster noise — or, pushed finer,
> fragments everything. After conditioning on a group, the local spread is set by the fine
> subspace at its native scale, so the sub-clusters resolve. This is the classic multi-scale
> argument for hierarchical over single-shot density clustering, made into a subspace problem.

The "subspace" character (not merely "multi-scale") comes from **per-group rotation** of the
fine structure: each coarse group's sub-clusters are separated along a *different* direction
in the level-2 block, so the fine signal **cancels when pooled across groups**. This defeats
the obvious shortcut "just cluster the level-2 block globally" — see the `flat-oracle-B`
baseline (§5).

> **Stated honestly up front (threat, see §11):** because the generator is built around the
> mechanism the method has, a hierarchy win is *partly by construction*. The experiment is
> therefore framed not as "does this structure exist in the wild" but as "**given** nested
> multi-scale subspace structure, can the method exploit it, by how much, and where is the
> crossover." H2b (non-nested control) and the oracle bounds (§5) are what keep this honest.

---

## 3. The planted-subspace generator

A single generator with knobs, so the "nested" and "non-nested" conditions and the `ρ` sweep
all come from one code path (fairness by shared construction, as in the prior doc).

**Dimensions.** Total `D = d_A + d_B + d_noise`.
- Block **A** (`d_A` dims): level-1 (coarse) relevant subspace.
- Block **B** (`d_B` dims): level-2 (fine) relevant subspace.
- `d_noise` dims: isotropic Gaussian noise, irrelevant — buries the signal and makes the
  problem high-dimensional.

**Hierarchy.** `G` coarse groups, each split into `K` fine sub-clusters → `G·K` planted
fine clusters; `n_per` points each.

**Construction for point in group `g`, sub-cluster `k`:**

```
x_A  =  ρ · μ_A[g]      +  N(0, σ_A^2 I_{d_A})        # coarse separation, scaled by ρ
x_B  =       R[g] · c[k] +  N(0, σ_B^2 I_{d_B})        # fine separation, per-group rotation
x_noise =                  N(0, σ_noise^2 I_{d_noise})
x = [x_A, x_B, x_noise]   then optionally a global random rotation Q (so blocks aren't axis-aligned)
```

- `μ_A[g]`: `G` well-separated coarse centroids in block A (e.g. simplex / scaled one-hot).
- `c[k]`: `K` base fine centroids in block B, **shared** across groups *before* rotation.
- `R[g]`: a per-group orthonormal rotation of block B (identity for `g=0`). Controls the
  **subspace-smearing** knob: with diverse `R[g]`, the fine constellation points in a
  different direction in each group, so the block-B marginal pooled over groups is unimodal —
  globally hidden, locally clean.
- **`ρ` (scale-separation, the headline knob):** multiplies the coarse separation while fine
  separation stays fixed. Small `ρ` → coarse and fine live at similar scales → a flat pass can
  resolve all `G·K` at once (flat fine). Large `ρ` → coarse dwarfs fine → flat resolves only
  groups, hierarchy must do the rest (hierarchy should pull ahead). The crossover lives here.

**Knobs and their roles:**

| Knob | Symbol | Effect | Swept? |
|------|--------|--------|--------|
| Scale separation | `ρ` | coarse-vs-fine scale ratio; **drives the crossover** | **yes (primary)** |
| Subspace rotation diversity | spread of `R[g]` | how globally smeared the fine subspace is | yes (secondary) |
| Noise dimensions | `d_noise` | burial / curse of dimensionality | yes (robustness) |
| Fine separation / spread | `‖c‖ / σ_B` | intrinsic detectability of fine clusters | fixed, reported |
| Groups / sub-clusters | `G, K` | partition granularity | fixed headline, varied in robustness |

**Nesting switch (for H2b).**
- **Nested (primary):** as above.
- **Non-nested control:** set `ρ = 1` and put *all* `G·K` clusters at a single level —
  every cluster gets its own subspace but there is no coarse→fine hierarchy. Prediction: flat
  and hierarchical perform *similarly*; the hierarchy advantage should **vanish**. (If it
  doesn't, our mechanism story is wrong — report that loudly.)

**Ground-truth outputs.** The generator returns, aligned by row:
- `y_fine` ∈ `0..G·K-1` (the planted sub-cluster; the primary target),
- `y_coarse = y_fine // K` (the planted group; derivable, used for within-group scoring),
- the block index sets `(idx_A, idx_B, idx_noise)` (the planted relevant subspaces, for the
  faithfulness analysis and oracle baselines).

**Registry integration (mirrors `src/ui/data.py` conventions).** Add a loader that wraps the
generator with `_one_hot_df`, encoding `y_fine` as the `target_*` columns so the existing
`prepare_dataset` recovers it via `argmax` over `target_*` with **no harness change**:

```python
# src/ui/data.py  (sketch — follows _concentric_rings / _one_hot_df)
def _nested_subspace(rho=8.0, G=3, K=4, d_A=3, d_B=3, d_noise=10,
                     n_per=60, sigma_A=0.4, sigma_B=0.4, sigma_noise=1.0,
                     rotate_blocks=True, rotation_diversity=1.0, seed=0): ...
def load_nested_subspace_dataframe() -> pd.DataFrame:
    X, y_fine, *_ = _nested_subspace()
    names = [f"c{g}_{k}" for g in range(G) for k in range(K)]
    return _one_hot_df(X, y_fine, [f"f{i}" for i in range(X.shape[1])], names)
# DATASETS["Nested subspace (Synthetic)"] = load_nested_subspace_dataframe
```

The experiment harness itself should call the **generator directly** (not the registry) so it
can sweep `ρ` and retrieve `y_coarse` / block indices, which the one-hot registry format drops.
The registry entry is for the UI / qualitative inspection and for `prepare_dataset` parity.

---

## 4. Conditions (defined against the actual code)

Hold the clustering algorithm and all hyperparameters fixed; vary only **recursion** and
**subspace scope**. All conditions consume the *same* standardised matrix `X_all` (the root
`StandardScaler`, as `standardised_X` in `hierarchical_vs_flat.py` already extracts).

| Condition | What it clusters | Role |
|-----------|------------------|------|
| **Flat-full** | one HDBSCAN run on `X_all` (full space, mirror the tree's optional UMAP pre-reduction via `clustering_space`) | the real competitor; identical to prior H1b flat |
| **Hierarchical** | leaf membership of `compute_analysis_tree` (`collect_leaves`) | the method under test |
| **Flat-oracle-B** | one HDBSCAN run on **block B only** (`X_all[:, idx_B]`) | diagnostic: does *knowing the fine subspace globally* suffice? Should fail when rotation diversity is high — isolating the "subspace smearing" effect |
| **Oracle-conditional** *(upper bound)* | HDBSCAN on block B **within each true coarse group** | best the recursive idea could achieve given a perfect level-1 split; bounds Hierarchical's headroom |

Using the **same HDBSCAN with identical hyperparameters** across Flat-full, Hierarchical, and
the oracles isolates recursion / subspace-scope as the variable (not "HDBSCAN vs anything
else"), exactly as the prior doc isolated recursion for benchmark H1b.

> Reuse, do not reimplement: `compute_clusters` (clustering), `clustering_space` and
> `collect_leaves` (already in `hierarchical_vs_flat.py`), `standardised_X`, `prepare_dataset`
> parity. The oracle conditions are the only genuinely new clustering calls.

---

## 5. Metrics

The prior experiment's lesson — **ARI punishes over-segmentation**, and the hierarchy
over-segments — is designed *into* the metric set here, not discovered after.

### Primary: fine-structure recovery (vs `y_fine`)

- **Homogeneity `h`, Completeness `c`, V-measure `v`** (sklearn). This decomposition is the
  point: a fragmenting hierarchy should score **high `h`** (leaves are pure) and **lower `c`**
  (one true cluster spread over several leaves). Reporting `h` and `c` separately *distinguishes
  "found the structure but split it" from "missed the structure"* — precisely the distinction
  ARI alone collapsed last time. `v` is the balanced summary.
- **ARI / NMI** — reported for continuity with prior H1b, but explicitly **expected to
  under-credit** the hierarchy; read alongside `h`/`c`, never alone.

### Primary: conditional (level-2) recovery — the actual hypothesis

- **Within-group ARI:** for each true coarse group `g`, restrict to its rows and compute
  ARI(predicted labels | g, `y_fine` | g), then average over `g`. This asks *"given the group,
  did the method resolve the K fine sub-clusters?"* — isolating H2a from the coarse split that
  both methods get for free. This is the headline number.

### Secondary / diagnostic

- **Coarse recovery (vs `y_coarse`):** sanity check that both methods get the easy level-1
  split (they should), so any difference is genuinely at level 2.
- **Recovery vs the oracle bound:** Hierarchical-within-group-ARI ÷ Oracle-conditional-ARI →
  fraction of achievable fine structure recovered, controlling for intrinsic difficulty.
- **Noise fraction** (`label == -1`) per condition; exclude noise consistently and report how
  (prior-doc rule).
- **n-leaves / fragmentation ratio:** `#predicted clusters ÷ G·K`, to quantify over-segmentation
  directly rather than letting it hide inside ARI.

### Faithfulness add-on (closes the prior doc's H1a gap)

With planted subspace ground truth we can finally run the **method-neutral faithfulness**
check the prior doc could only flag (its §8.1): for each true coarse group as a region, is the
**local leaf 2D view** more faithful to the **planted fine structure** than the global view?
Operationalise as fine-label kNN agreement (or silhouette of `y_fine`) in the 2D embedding,
local vs global, paired per region. This reuses `_score_node`'s neighbourhood machinery and
the planted labels, giving the clean H1a the prior design lacked.

---

## 6. Procedure (pseudocode)

```
for rho in RHO_GRID:                              # the crossover sweep (H2c)
  for nesting in {nested, non_nested}:            # H2b control
    for seed in SEEDS:                            # replicates: generator seed + embedding stochasticity
      X, y_fine, y_coarse, blocks = make_nested_subspace(rho, nesting, seed=seed, **PARAMS)
      df = to_app_df(X, y_fine)                   # _one_hot_df-format, so prepare_dataset parity holds
      X_all = standardise(X)                      # same StandardScaler the tree uses

      config = base_config(method=DR, seed=seed)  # DR = UI default for headline

      # --- conditions ---
      tree   = start_evaluation(df, features, config)      # builds tree (+ per-node scores, reused for §5 add-on)
      part_hier = leaf_partition(collect_leaves(tree))
      part_flat = hdbscan_once(clustering_space(X_all, config))      # full space
      part_oracleB = hdbscan_once(X_all[:, blocks.B])               # fine subspace, global
      part_cond  = within_group_hdbscan(X_all[:, blocks.B], y_coarse) # upper bound

      # --- metrics ---
      for name, part in conditions.items():
          record(rho, nesting, seed, name,
                 h=homogeneity(y_fine, part), c=completeness(y_fine, part), v=vmeasure(...),
                 ari=ARI(y_fine, part), nmi=NMI(y_fine, part),
                 within_g_ari=mean_within_group_ari(part, y_fine, y_coarse),
                 coarse_ari=ARI(y_coarse, part_to_coarse(part)),
                 noise_frac=..., n_clusters=...)

      # --- faithfulness add-on (planted-label H1a) ---
      if RUN_FAITHFULNESS:
          E_global = embed_original(X_all, config)
          for g in groups:
              idx = where(y_coarse == g)
              record_region(rho, seed, g,
                  local =label_knn_agree(leaf_view_for(g, tree), y_fine[idx]),
                  global=label_knn_agree(E_global[idx],          y_fine[idx]))
```

Implementer notes:
- `make_nested_subspace` is the one new generator (§3). Everything else reuses existing code
  (`start_evaluation`, `collect_leaves`, `clustering_space`, `compute_clusters`, `_score_node`,
  `_embed_original`, `prepare_dataset` conventions).
- HDBSCAN hyperparameters **must be identical** across Flat-full, Hierarchical, and both
  oracles (assert it). `min_cluster_size` interacts with scale (§7) so it is a documented,
  swept robustness factor, not a free tuning knob.
- Replicates: `make_nested_subspace` is seeded; the DR embeddings are stochastic for UMAP/t-SNE
  (unseeded in `reduce_dimensionality`, as the prior doc documents), so the `SEEDS` loop again
  captures both generator and embedding variance.

---

## 7. Confounds and controls

| Confound | Risk | Control |
|----------|------|---------|
| **Generator ↔ method circularity** | structure built to suit the method ⇒ guaranteed win | H2b non-nested control (predict **no** advantage); sweep `ρ` for a *crossover* not a point; report oracle headroom |
| Over-segmentation | ARI under-credits the hierarchy (prior finding) | report `h`/`c`/V and fragmentation ratio; lead with within-group ARI |
| "Easier problem" | re-embedding/clustering fewer points is intrinsically easier | within-group ARI and oracle-relative recovery control for difficulty; the faithfulness add-on reports stress (prior §11) |
| `min_cluster_size` × scale | one choice may rig flat vs hier | identical hyperparameters both conditions; sweep `min_cluster_size` as robustness; document sensitivity |
| Subspace shortcut | "just cluster block B" might trivially win | `flat-oracle-B` baseline + rotation-diversity knob: high diversity must break the global-B shortcut while hierarchy survives |
| DR method | a "win" might be t-SNE-vs-PCA | same `config["method"]` across conditions; optionally repeat over {PCA, UMAP, t-SNE} |
| Pre-reduction space | flat HDBSCAN seeing a different space than the tree's first split | mirror the tree's UMAP pre-reduction via `clustering_space` (prior-doc rule) |
| Noise handling | HDBSCAN `-1` skews ARI | exclude noise identically across conditions; report noise fraction |
| Coarse split not shared | difference might be at level 1, not level 2 | report coarse recovery; both methods must get groups before level-2 comparison is meaningful |

---

## 8. Robustness / secondary variants

1. **Rotation-diversity sweep.** From `R[g] = I` (fine subspace shared & globally visible) to
   maximally diverse rotations. Predict: `flat-oracle-B` degrades sharply with diversity while
   Hierarchical is roughly flat — the cleanest single demonstration that the effect is
   *subspace* smearing, not mere scale.
2. **Noise-dimension sweep.** `d_noise ∈ {0, 10, 50, 200}`. Tests robustness to burial; expect
   all methods to degrade, hierarchy's relative advantage to persist or grow.
3. **`min_cluster_size` sensitivity.** Show the crossover `ρ*` is not an artefact of one
   density threshold.
4. **`G, K` granularity.** Deeper/wider trees; checks the depth-sensitivity intuition from the
   prior doc (advantage grows then plateaus).
5. **DR-method factor.** {PCA, UMAP, t-SNE} as in the prior doc — and connect to that doc's
   finding that t-SNE leaf views gave little faithfulness benefit.

---

## 9. Statistical analysis

- **H2a / H2c:** unit = (`ρ`, seed). Per `ρ`, paired Wilcoxon signed-rank on
  `within_g_ari_hier − within_g_ari_flat` across seeds; report median Δ, rank-biserial effect
  size, and a **crossover plot** (mean ± CI of each condition's within-group ARI vs `ρ`, with
  the two oracle reference curves). Estimate the crossover `ρ*` where the flat curve drops
  below a chosen adequacy threshold (state the threshold a priori).
- **H2b:** the key comparison is *between* nested and non-nested of the Hierarchical−Flat gap.
  Report the gap with CI in both regimes; the claim is supported only if the gap is clearly
  positive for nested and ~zero for non-nested. Treat a non-zero non-nested gap as evidence
  *against* our mechanism story, not noise to explain away.
- **Multiplicity:** several `ρ` × metrics → pre-register within-group ARI as primary; `h`/`c`,
  ARI/NMI, faithfulness are secondary/exploratory (Holm/BH if claimed inferentially).
- With a synthetic generator we *can* afford many seeds, so report real CIs, not just means.

---

## 10. Outputs (match the harness conventions)

Write to `outputs/experiments/<timestamp>/` (as `hyperparameter_tuning.py` /
`hierarchical_vs_flat.py` do):

- `subspace_recovery.csv` — one row per (`ρ`, nesting, seed, condition): `h, c, v, ari, nmi,
  within_g_ari, coarse_ari, noise_frac, n_clusters`.
- `subspace_summary.csv` — per (`ρ`, nesting, metric): median Δ (hier−flat), win rate,
  Wilcoxon p, effect size, oracle-relative recovery.
- `subspace_faithfulness.csv` — per (`ρ`, seed, region): local vs global fine-label agreement
  (the planted-label H1a).
- Figures: **crossover curve** (within-group ARI vs `ρ`, four conditions) → `fig:rq1s-crossover`;
  `h`-vs-`c` scatter (over-segmentation signature) → `fig:rq1s-homcomp`; rotation-diversity
  diagnostic → `fig:rq1s-rotation`. These slot into `sections/06_evaluation.tex` as the clean
  synthetic counterpart to the benchmark H1b tables.

---

## 11. Threats to validity / how this could mislead

- **Circularity (the big one).** The generator is built around the method's mechanism
  (multi-scale, sample-recursive, no feature selection). A hierarchy win on nested data is
  therefore partly *by construction*. The defensible claim is bounded: *"the method can exploit
  nested multi-scale subspace structure where it exists, and here is the crossover and the
  headroom"* — **not** "such structure is common in real data." H2b and the oracle bounds are
  the guardrails; the case study / real data remain the only evidence for prevalence.
- **No feature selection in the method.** The current tree cannot do *axis-aligned* subspace
  clustering (PROCLUS/SUBCLU-style). If the planted subspaces required feature selection to
  recover, the method would fail for reasons unrelated to recursion. The generator deliberately
  uses *sample-conditioned scale separation* so the test is fair to the method that exists; a
  feature-selecting node-splitter is **future work**, and worth naming as such in the thesis.
- **Over-segmentation persists.** Even on planted data the hierarchy will fragment; within-group
  ARI and `h`/`c` are designed to surface this honestly rather than let a high `h` masquerade as
  full recovery.
- **Parameter arbitrariness.** Planted `σ`, separations, `G/K` are choices. Sweep ranges and
  report sensitivity; do not cherry-pick the `ρ` that flatters the method.
- **Easier-problem confound (carried from the prior doc).** Re-clustering fewer points is
  intrinsically easier; oracle-relative recovery and the faithfulness stress check (per prior
  §11) keep "wins by how much, at what cost" in view.
- **Synthetic ≠ real.** External validity is the standing limit; this experiment is the
  *internal-validity* complement to the benchmark H1b, not a replacement for real evidence.

---

## 12. Open design decisions (confirm before implementing)

1. **Primary structure = nested multi-scale** (recommended) vs classical single-level subspace
   clusters. *Recommendation:* nested as primary (matches the method's mechanism and yields a
   predictable crossover); single-level as the H2b negative control, **not** the headline.
2. **Whether to also implement a feature-selecting node-splitter** to test classical subspace
   clustering. *Recommendation:* out of scope here; name as future work. Doing it would change
   the method, not just the experiment.
3. **Ground-truth target = fine labels** (recommended, with `y_coarse` derived) vs scoring the
   full planted hierarchy with a tree-edit/dendrogram metric. *Recommendation:* fine labels +
   within-group ARI for the headline; a hierarchy-aware metric is a nice-to-have extension.
4. **`ρ` grid and adequacy threshold** — pick a priori (e.g. `ρ ∈ {1,2,4,8,16,32}`, adequacy =
   within-group ARI ≥ 0.7) and pre-register, so the crossover claim is not retrofitted.
5. **Seed count** — synthetic is cheap; ≥ 20 seeds for real CIs (vs 5 in the benchmark harness).

---

## 13. Implementation checklist

Already exists (reuse, do not reimplement):

- `compute_analysis_tree` / `start_evaluation` — hierarchy + per-node scores.
- `compute_clusters` — HDBSCAN for the flat and oracle conditions.
- `clustering_space`, `collect_leaves`, `standardised_X`, `prepare_dataset` — already in
  `hierarchical_vs_flat.py`; lift them or import.
- `_embed_original`, `_score_node` — for the faithfulness add-on.
- `_one_hot_df` / `DATASETS` registry convention — for the UI loader.

To build:

- `make_nested_subspace(...)` — the generator (§3), returning `X, y_fine, y_coarse, blocks`.
- `load_nested_subspace_dataframe` + a `DATASETS` entry (UI parity; `_one_hot_df` on `y_fine`).
- `within_group_hdbscan` and `mean_within_group_ari` — the level-2 condition + metric.
- the `flat-oracle-B` and `oracle-conditional` clustering calls.
- the `ρ` × nesting × seed loop, CSV writers, and the three figures.
- a `main()` reusing the `OUTPUT_ROOT` / timestamp / `Parallel` pattern from
  `hierarchical_vs_flat.py`.

Suggested filename: `src_research/planted_subspace_recovery.py`.

---

## 14. Relationship to the prior experiment (one paragraph for the thesis)

`hierarchical_vs_flat.py` established a clean negative: the hierarchy is not a better
*clustering of benchmark class labels* (H1b), and its value is representational rather than
partitional. That result is honest but incomplete, because class labels cannot express the one
structure the hierarchy is built for — nested subspace structure that is globally hidden. This
experiment supplies exactly that structure under controlled conditions and asks whether
recursion recovers it. A positive, crossover-shaped result would reframe RQ1 from "partially
and conditionally yes" to a *specified* condition: **the hierarchy wins when structure is
nested and multi-scale in different subspaces, and the prior negative is what you correctly get
when it is not.** A null result here would be the stronger finding still: it would mean even the
best case for the method does not materialise, and the representational-only conclusion stands
unqualified.

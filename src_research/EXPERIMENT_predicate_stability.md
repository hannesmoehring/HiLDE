# Experiment Design: Predicate Stability under Relaxation (RQ2, H2)

Design document. The implementation will live next to `hierarchical_vs_flat.py` and
`planted_subspace_recovery.py` in `src_research/`. This file specifies *what* to measure
and *why* before any code is written, so the result is fair by construction rather than by
accident.

This experiment exists because RQ2 — the original premise of the project — currently has
**zero empirical content**. The thesis states H2, specifies the metrics (Ch. 6, "Metrics
and Their Justification"), specifies the threshold grid, and flags the simulated-user
driver and stability metrics as the most important missing piece. Everything below turns
those existing specifications into a runnable harness; almost nothing here is a new design
decision, which is deliberate given the 2026-07-13 feature freeze.

> One-line thesis framing: RQ1's experiments established *where* the hierarchy earns its
> keep; this experiment asks whether the *descriptions* of what the user finds there can be
> made robust to how exactly they selected it, and at what cost in specificity.

---

## 1. Research question and hypotheses

**H2 (verbatim from the thesis, §6.1).** As the relaxation threshold is lowered from
`t = 1.0`, the predicate describing a selection becomes more stable under small
perturbations of that selection (higher Jaccard/ARI of the admitted sets, lower variance
of F1), while its ability to separate the selection (F1) degrades only gradually, so that
a favourable operating point exists.

Decomposed into falsifiable pieces:

- **H2a — stability gain.** Mean pairwise Jaccard of the admitted sets across perturbed
  re-selections increases monotonically as `t` decreases from 1.0 through the grid
  {0.95, 0.9, 0.8}; F1 variance across perturbations decreases.
- **H2b — graceful specificity cost.** Median F1 against the unperturbed target selection
  degrades but stays above a pre-specified floor (see operating-point criterion, §5) for
  at least one `t < 1.0`.
- **H2c — the severity split earns its place (ablation of the novel element).** The
  thesis's stated novel element is *how* the trim is divided between the tails
  (severity-proportional, `_tail_removal_shares`), not that trimming happens. Prediction:
  on **skewed** within-cluster distributions the severity split retains higher F1 at
  matched `t` than a naive 50/50 split; on **symmetric** distributions the two are
  indistinguishable. The symmetric case is a negative control on ourselves: if
  severity-split wins there too, something else is going on and the mechanism story is
  wrong.

Nulls we genuinely try to reject: H2a-null = Jaccard(t<1) − Jaccard(1.0) centred on zero;
H2c-null = severity − symmetric F1 difference centred on zero on skewed data. H2b is a
bounded claim, not a test: if no `t` satisfies the operating-point criterion, **H2 is
refuted and that is the reported result.**

> **Stated honestly up front (threat, §8):** relaxation trims tail values, and tail values
> are exactly what small selection perturbations move. A stability gain is therefore
> partly *by construction*. What is **not** by construction: the size of the F1 cost,
> whether a favourable operating point exists on the pre-specified grid, and whether the
> severity split beats the naive split. Those three are the actual findings; the
> monotone stability trend alone would be nearly tautological.

---

## 2. What the method actually does — grounding in code

Verified against `src/analysis/predicate_generator.py`:

- `_predicate_threshold` builds, for **every** feature, an interval clause
  `[quantile(left_trim), quantile(1 − right_trim)]` over the *selected* values, with the
  total trim `1 − t` split across tails by `_tail_removal_shares` (severity = distance
  from selection median to that side's extreme). At `t = 1.0` the clause is the
  selection's min/max — the strict bounding box. The predicate is the conjunction over
  all features; **there is no clause selection**, so predicate length is constant `d`.
- `_predicate_db` (DimBridge-style) builds the same per-feature ranges, then greedily
  conjoins the clauses that most improve F1 against the selection labels; it yields a
  *sparse* clause set (`in_predicate`, `predicate_step`) and needs `selected_indices`.
- Admitted set: membership masks over the full standardised matrix are already computed
  clause-wise in `_predicate_db`; the harness needs a small `admitted_mask(rows, X_all)`
  helper for the threshold method (AND over per-feature interval masks).

Consequences for the design:

1. Stability of the **admitted set** and of the **bounds** is meaningful for both methods;
   stability of the **clause set** (which features appear) is only meaningful for `db` —
   report it there as clause-set Jaccard, a metric the greedy step makes genuinely at
   risk: one moved point can reroute the greedy path.
2. **Relevant-dimension recovery** (synthetic only) needs different readouts per method:
   for `db`, the `in_predicate` feature set directly; for `threshold`, rank features by
   selectivity `sel_range / (global_max − global_min)` and score **precision@r** where `r`
   is the planted number of relevant dims. Pre-specified here to avoid inventing a cutoff
   after seeing results.
3. The harness calls `generate_predicate` from the calc layer directly (as the other
   harnesses call `compute_analysis_tree` directly), not through `backend/app.py`.

---

## 3. Data

Two sources, both required, each for what only it can provide.

**Wine quality (real, interpretable).** The thesis's own requirement: predicates must be
readable to be checked. Selections come from actual tree leaves, so stability is measured
where the tool would really produce descriptions. Wine gives ecological validity; it
cannot give recovery scores.

**Axis-parallel planted generator (synthetic) — the one new generator.** This is the
generator the thesis's §6.2 gap box explicitly requires and explicitly distinguishes from
the RQ1 generator: *axis-aligned*, because axis-parallel range predicates cannot express
the RQ1 generator's per-group rotation. Do **not** reuse `make_nested_subspace` — its
rotation makes every predicate unrecoverable by construction, which would be unfair in
the opposite direction of the RQ1 circularity.

Construction (single level, no nesting — RQ2 needs recoverable boxes, not hierarchy):

```
C clusters, n_per points each, D = d_rel·C? No — shared feature space:
  each cluster c has a relevant-dimension set R_c (|R_c| = r, sampled without replacement)
  x_j ~ within-cluster distribution centred at μ_cj with spread σ_rel   for j ∈ R_c
  x_j ~ N(0, 1)                                                          for j ∉ R_c
  membership ground truth = the planted assignment; relevant-dim ground truth = R_c
```

Knobs:

| Knob | Role | Setting |
|------|------|---------|
| `skew` | within-cluster distribution: symmetric (Gaussian) vs skewed (e.g. lognormal, standardised) | **swept: {symmetric, skewed}** — drives H2c |
| `margin` | separation between cluster box and background along relevant dims | swept coarse {wide, tight}: prediction — with wide margins strict is already stable and relaxation buys ~nothing (secondary negative control) |
| `r`, `C`, `d_noise` | relevant dims per cluster, cluster count, noise burial | fixed headline (e.g. r=3, C=4, d_noise=15), documented |

Registry integration follows `_one_hot_df` conventions in `src/datasets.py` for UI parity;
the harness calls the generator directly to keep `R_c` (which the one-hot format drops).

---

## 4. Selections, perturbations, conditions

### Simulated user (the missing driver, thesis §6.2/§6.3)

Head-less, exactly as the thesis specifies: build the tree with the existing
`compute_analysis_tree` path, take **leaf clusters** as selections (wine), or take the
points of a **planted cluster** as the selection (synthetic; the tree is not needed to
define the target there, which keeps recovery scoring independent of tree quality).
Minimum selection size 20 (quantiles on fewer points are noise); report sizes.

### Perturbation operators — the thesis names two families

**(a) Selection perturbation (primary).** For a base selection S of size n, a perturbed
replicate drops a fraction δ of S uniformly at random and adds the same number of points
from the k-nearest non-selected neighbours of S's members in the standardised full space
(boundary jitter — mimics a user lassoing slightly differently, which moves the boundary,
not the core). δ = 0.1 headline, δ ∈ {0.05, 0.2} robustness. m = 20 replicates per base
selection.

**(b) Seed perturbation (secondary).** Rebuild the tree per replicate id (the embeddings
are stochastic and unseeded, as documented in both prior experiment docs), match leaves
across rebuilds by maximum member-set Jaccard, and compare predicates of matched leaves.
**Known confound, stated in advance:** this conflates tree instability with predicate
instability. Report the matching Jaccard alongside, and treat weakly matched leaves
(match < 0.5) as tree-level instability — excluded from the predicate claim, counted and
reported. If time runs out, (b) is the first thing cut (see §9): H2's wording is about
selection perturbation; (b) is corroboration.

### Conditions

| Factor | Levels | Role |
|--------|--------|------|
| Threshold `t` | **{1.0, 0.95, 0.9, 0.8}** | the H2 sweep; grid pre-specified in the thesis — do not add levels after seeing results |
| Method | `threshold` (primary), `db` (secondary) | `db` is the DimBridge-style baseline the thesis compares against |
| Tail split | severity (`_tail_removal_shares`), symmetric (0.5/0.5) | H2c ablation; implement as a parameter, not a monkeypatch |
| Data | wine, synthetic × {symmetric, skewed} × {wide, tight} | ecological validity + recovery + H2c |

All conditions consume the same `StandardScaler` matrix (`standardised_X`, lift from
`hierarchical_vs_flat.py`). t = 1.0 **is** the strict baseline — no separate condition.

---

## 5. Metrics

All are already specified in the thesis's metrics section; the harness implements them.

**Stability (primary, the H2 claim):**
- `jaccard_admitted`: mean pairwise Jaccard of admitted sets over the m perturbation
  replicates, **aggregated to one number per (selection, t, method, split) before any
  test** — the unit of analysis is the selection, never the replicate pair
  (pseudo-replication was the §6.6 lesson; build the aggregation in, don't bolt it on).
- `f1_sd`: standard deviation of F1 (vs the *base* selection) across replicates.
- `bound_sd`: mean per-feature sd of (sel_min, sel_max) across replicates — diagnostic
  for *why* the admitted set moves.
- `db` only: `jaccard_clauses`: mean pairwise Jaccard of the `in_predicate` feature sets.

**Predicate quality (the cost axis):**
- `f1`, `precision`, `recall` of the *unperturbed* predicate vs the base selection
  (reuse `_f1`); `coverage` = admitted fraction of the dataset; `length` = d for
  threshold (constant, reported once), #clauses for `db`.

**Recovery (synthetic only — fills the thesis's placeholder Table `tab:synthetic-recovery`):**
- relevant-dimension precision/recall/F1 (`db`: `in_predicate` set; `threshold`:
  precision@r by selectivity rank, per §2),
- membership F1 of the admitted set vs planted membership.

**Trivial-stability guard.** A predicate admitting the whole dataset is perfectly stable.
Stability is therefore never reported alone: every stability figure carries the matched
F1/coverage, and the operating point is a **joint** criterion:

> **Operating point (pre-specified).** t* = the largest t < 1.0 with (i) median
> per-selection Jaccard gain over strict > 0 with Wilcoxon p < 0.05 (Holm-corrected over
> the three thresholds) and (ii) median F1 ≥ 0.9 × median strict F1. H2 is supported iff
> such t* exists; refuted otherwise.

The 0.9 floor is a judgement call and is stated here, before any run, so the operating
point cannot be retrofitted around the data.

---

## 6. Procedure (pseudocode)

```
for dataset in {wine} ∪ synthetic_variants:            # §3
  for seed in SEEDS:                                   # generator + tree replicate id
    df, X_all, selections = simulate_user(dataset, seed)   # leaves (wine) / planted (synth)
    for sel in selections:                             # |sel| ≥ 20
      replicates = [perturb(sel, delta=0.1, rng) for _ in range(M)]   # §4a
      for method in {threshold, db}:
        for split in {severity, symmetric}:            # H2c
          for t in {1.0, 0.95, 0.9, 0.8}:
            base = predicate(method, split, t, sel)
            reps = [predicate(method, split, t, r) for r in replicates]
            record(dataset, seed, sel_id, method, split, t,
                   jaccard_admitted = mean_pairwise_jaccard(admitted(reps)),
                   f1_sd            = sd(f1(rep, sel) for rep in reps),
                   f1, precision, recall, coverage, length = quality(base, sel),
                   bound_sd, jaccard_clauses,
                   recovery metrics if synthetic)
# seed-perturbation pass (b): matched-leaf predicates across SEEDS, wine only, if time
```

Implementer notes:
- Reuse, do not reimplement: `prepare_dataset`, `standardised_X`, `collect_leaves`,
  `clustering_space` (all in `hierarchical_vs_flat.py`), `generate_predicate` / `_f1`
  (calc layer), `OUTPUT_ROOT`/timestamp/`Parallel` main() pattern (both prior harnesses).
- To build: the axis-parallel generator, `perturb`, `admitted_mask`, the `split`
  parameter on `_tail_removal_shares` (touches `predicate_generator.py` — smallest
  possible diff: one keyword argument defaulting to current behaviour), metric
  aggregation, CSV writers, figures.
- Cost model: predicates are O(n·d) — the sweep is cheap. Tree builds dominate; wine ≤
  500 rows × 5 seeds is minutes. Synthetic selections don't need trees at all (§4).
  SEEDS = 5 for wine (tree-bound), 20 for synthetic (cheap, matches the RQ1-S standard).

---

## 7. Statistical analysis

- Unit of analysis: **selection** (per dataset, seed). Replicate pairs are aggregated
  first (§5). This is the §6.6 pseudo-replication lesson applied prospectively.
- **H2a:** per t < 1.0, paired Wilcoxon signed-rank of per-selection
  `jaccard_admitted(t) − jaccard_admitted(1.0)`; report median Δ, rank-biserial effect
  size. Holm across the three thresholds. Same for `f1_sd` (secondary).
- **H2b/operating point:** the joint criterion of §5; report the full trade-off curve
  (Jaccard and F1 vs t), not just t*.
- **H2c:** paired Wilcoxon of severity − symmetric F1 at matched t on skewed synthetic;
  the symmetric-data arm must come out null. Both results reported regardless of
  direction.
- Primary metric pre-registered here: `jaccard_admitted` on the threshold method at
  δ = 0.1. Everything else (ARI variant, `db`, δ sweep, bound_sd, seed-perturbation) is
  secondary/exploratory and labelled as such in the thesis.

---

## 8. Threats to validity / how this could mislead

- **Partly-by-construction stability (the big one, §1).** Trimming tails mechanically
  reduces sensitivity to boundary jitter. Bounded claim: *"relaxation buys measurable
  stability at a quantified F1 cost, with a favourable operating point at t*"* — not
  "relaxed predicates are better descriptions." H2c and the wide-margin control are the
  guardrails; the case study remains the only evidence the trade matters to a user.
- **Perturbation model arbitrariness.** Real users don't perturb selections uniformly at
  random from kNN boundaries. δ and the jitter mechanism are choices; sweep δ, describe
  the operator precisely, and claim stability *under this operator* only.
- **Generator favours the method (RQ2 edition).** Axis-parallel boxes are exactly what
  range predicates can express — deliberately, per the thesis's own gap note, and the
  mirror image of the RQ1 generator being fair to *that* method's mechanism. State the
  symmetry explicitly in the thesis; wine is the check that the effect survives off the
  generator.
- **Skew knob chosen to flatter H2c.** Lognormal skew is the friendly case for
  severity-splitting. The symmetric arm is the control; additionally report one
  adversarial shape if time allows (e.g. bimodal within-cluster, where the median-based
  severity is misleading).
- **`db` greedy instability is not a bug of relaxation.** The greedy path can reroute
  under perturbation independent of t; `jaccard_clauses` isolates this so it doesn't
  contaminate the t-effect read-off.
- **Small selections.** Quantile bounds on n = 20 are noisy in themselves; min-size
  filter plus reporting `n` per selection; no claims stratified below n = 30.
- **Multiplicity.** 3 thresholds × 2 methods × 2 splits × several data arms — the Holm
  correction and the single pre-registered primary are the containment; everything else
  is labelled exploratory. This addresses, prospectively, the same issue the thesis still
  owes §6.7 retrospectively.

---

## 9. Scope control (feature freeze 2026-07-13)

Build order, with the cut line explicit:

1. **Day 1:** axis-parallel generator; `perturb`; `admitted_mask`; split parameter;
   selection-perturbation sweep on synthetic (both skew arms) + wine leaves; records CSV.
2. **Day 2 morning:** aggregation, Wilcoxon/Holm, summary CSV, the trade-off figure
   (`fig:rq2-tradeoff`: Jaccard and F1 vs t, one panel per dataset arm) and the H2c
   figure (severity vs symmetric F1 at matched t, skewed vs symmetric data).
3. **Day 2 afternoon:** seed-perturbation pass (b); δ robustness; adversarial skew shape.

Anything in (3) that doesn't finish is documented as not-run — **do not** let (3) eat
(2). A complete (1)+(2) fully answers H2 as worded; (3) is corroboration. If even (2) is
at risk, drop the `db` method arm before dropping the H2c ablation: H2c tests the novel
contribution, `db` is context.

## 10. Outputs

`outputs/experiments/<timestamp>/` (harness convention):
- `stability_records.csv` — one row per (dataset, seed, selection, method, split, t, δ).
- `stability_summary.csv` — per (dataset-arm, method, split, t): median Δ vs strict, win
  rate, Wilcoxon p (Holm), effect size, median F1/coverage; the t* verdict line.
- `recovery.csv` — synthetic arms: relevant-dim P/R/F1, membership F1 per t → fills
  thesis Table `tab:synthetic-recovery` (strict vs relaxed rows, exactly its columns).
- Figures: `fig:rq2-tradeoff`, `fig:rq2-ablation`, coverage-vs-t diagnostic.

Thesis integration: results slot into the RQ2 half of the synthetic-data section and
resolve the two §6.3/§6.4 gap boxes (simulated-user driver, stability metrics) and the
`tab:synthetic-recovery` placeholder; Ch. 4's relaxation section gains the H2c ablation
as evidence for the severity split specifically. If H2 is refuted, the thesis reports the
refutation with the same prominence H1b's negative received — that precedent is set and
is worth keeping.

Suggested filename: `src_research/predicate_stability.py`.

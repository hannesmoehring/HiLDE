# Experiment Design: Benchmark Workflow Walk (RQ1, RQ3) — thesis §6.5

Design document. The implementation will live next to the four existing harnesses in
`src_research/`; suggested filename `benchmark_workflow.py`. This file specifies *what* to
report and *why* before any code is written.

This experiment exists because §6.5 is the last quantitative section with **zero content**:
`tab:benchmark-results` is a placeholder and the gap box says the driver "is not yet
scripted" (roadmap 2026-07-12, item 12a). Unlike the other four experiments it tests no
hypothesis — the thesis's own wording is "descriptive and faithfulness-based ... not to
claim a recovery result the data cannot support". That wording is honest but dangerous:
a purely descriptive section invites post-hoc storytelling ("the hierarchy behaves
sensibly"), which the 2026-07-12 writing review flagged as this thesis's besetting vice.
The containment here is to pre-register (a) every reporting and aggregation rule, so no
number in Table 6.2 involves a choice made after seeing results, and (b) four
**consistency checks** — out-of-sample predictions derived from the four completed
experiments — so the section has falsifiable content after all.

> One-line thesis framing: the controlled experiments established *where* the hierarchy
> and the relaxation earn their keep; this run shows what the **shipped defaults** actually
> produce on the real benchmarks, and checks that it is the behaviour those experiments
> predict.

---

## 1. Purpose, and what may not be claimed

**Deliverables.** Fill `tab:benchmark-results` (per dataset: depth, #leaves, mean
trustworthiness, mean continuity, example leaf predicate), replace the §6.5 placeholder
prose, delete the gap box.

**Descriptive readouts (D) and pre-registered consistency checks (C):**

- **D1 — hierarchy shape.** Realized depth, leaf count, leaf sizes, and — first-class,
  not a footnote — the **noise fraction**: points dropped mid-recursion (HDBSCAN label
  −1) belong to no leaf, so a leaf count without coverage misleads.
- **D2 — per-node faithfulness.** The shipped per-node ZADU scores (trustworthiness,
  continuity, MRRE, stress; CADI on internal nodes), i.e. exactly what the UI shows.
- **D3 — predicate readability.** Length, F1, coverage of leaf predicates, strict vs
  relaxed, both methods, over **all** eligible leaves — the example in the table is
  drawn by a fixed rule (§5), not chosen for looking good.
- **D4 — RQ3 divergence readout.** Where labels exist, leaf class purity/enrichment as
  the *secondary* signal, and the association between leaf faithfulness and leaf purity.
  Expectation, from the tuning experiment's DBCV–ARI $r \approx 0.17$ and
  classes≠clusters: **weak**. A strong correlation would be the surprise to explain.
- **C1 — compounding law, out of sample.** RQ2 established dense-predicate recall
  $\approx t^{d_{\mathrm{eff}}}$ on $d=11$ (wine) and $d=27$ (synthetic). Breast cancer
  ($d=30$) and digits ($d=64$) extend the curve to dimensionalities never fit. Prediction:
  dense recall at $t=0.95$ lies **at or above** the independence bound $t^{d}$ (correlated
  features trim overlapping mass) and **far below** 1 — concretely, digits dense recall at
  $t=0.95$ below $0.5$ ($0.95^{64} \approx 0.037$; wine sat above its bound at every $t$,
  by 1.05× at $t=0.95$ up to 1.9× at $t=0.8$).
- **C2 — H1a replication arm (paired, UMAP).** For each leaf, score the leaf's own local
  embedding against the root embedding sliced to the same rows, identical scoring path —
  the exact H1a contrast. Prediction from run 20260628_184827 (UMAP, hier_leaf): small
  positive median trustworthiness delta, $|\tilde{\Delta}| \le 0.05$, win rate in
  $[0.5, 0.8]$. A large positive delta would contradict H1a's "UMAP is roughly a wash".
- **C3 — noise fractions in the H1b range.** Same datasets, same defaults, same depth
  cap as H1b's UMAP cells ⇒ predicted per-build noise fractions: wine $0.00$–$0.19$,
  breast cancer $0.15$–$0.32$, digits $0.12$–$0.17$ (observed ranges over 5 seeds in
  `h1b_recovery.csv`). Modest excursions are expected (different depth cap); factor-level
  disagreement means a harness bug or an undocumented config drift — investigate, don't
  average away.
- **C4 — sparse brevity replicates.** RQ2 found relaxation buys the sparse (`db`)
  predicate brevity at no F1 cost on wine (median length 10 → 6 at $t=0.95$). Prediction:
  wine reproduces this within ±2 clauses; breast cancer and digits show the same
  *direction* (median length at $t=0.95$ < strict length ≪ $d$), magnitude unconstrained.

**Explicitly out of scope — claims this section must not make.** No recovery claims (no
ground-truth subspaces); no "hierarchy beats flat" claims (H1b answered the partition
question negatively and §6.5 must not relitigate it); no stability claims (RQ2's domain);
no implication that high purity = good clustering (classes are not clusters,
`jeon_classes_2023` — an impure leaf is not a failure). If a check C1–C4 fails, the
failure **is the finding** and is reported with the same prominence the H1b and H2
negatives received; that precedent is set and is worth keeping.

---

## 2. What the code actually does — grounding, verified 2026-07-15

Verified against `src/analysis/analysis_routine.py`, `src/evaluation/evaluate.py`,
`src/analysis/predicate_generator.py`:

- `start_evaluation(df, feature_cols, config)` = `compute_analysis_tree` + per-node ZADU
  scores. This is the entire driver core; the harness mostly walks the returned tree.
- **Stopping rules** (determine realized depth, D1): a node becomes a leaf when
  depth ≥ `hierarchical_layers`, or $n < 2\cdot$`hclust_min_cluster_size`, or HDBSCAN
  finds < 2 non-noise clusters. Children are HDBSCAN clusters, so every child has
  ≥ `hclust_min_cluster_size` (25) points.
- **Noise is dropped at every split** (label −1 points get no child): leaves do not
  partition the dataset. D1's noise fraction = $1 - \sum_\text{leaves} n_\ell / n$.
- **Clustering space:** the UMAP pre-reduction to `hclust_umap_n_components` (2) happens
  **once at the root**; recursion clusters *slices of the root's reduced space*. The
  per-node `embedding_original` (what the UI shows and what D2/C2 score) is re-fit per
  node on that node's original standardised features. The design must describe the system
  as it is, not as Ch. 5's stale Streamlit text describes it.
- **Scoring:** `_score_node` uses $k = \min(20, (n-1)/2)$, T&C/MRRE only for $n \ge 10$,
  CADI only on internal nodes (needs child labels), root scaler applied throughout. So
  **k varies with node size below $n = 41$** — a leaf's T&C at $k=12$ is not strictly
  comparable to the root's at $k=20$. Consequences in §6 (C2 restriction) and §8.
- **Embedding fallback:** nodes too small to embed get a zero embedding and `None`
  scores; the harness reports scored-leaf coverage rather than silently averaging over
  a shrunken denominator.
- `generate_predicate(method ∈ {threshold, db}, df_sel, X_scaled_full, threshold,
  selected_indices, tail_split)` plus the RQ2 harness's `admitted_mask` and `_f1` give
  every predicate number. UMAP/t-SNE are unseeded (only MDS takes a random state), so
  tree builds are stochastic → repeated builds, not single runs.
- Reuse, don't reimplement: `prepare_dataset`, `standardised_X`, `collect_leaves`
  (`hierarchical_vs_flat.py`); `admitted_mask`, output/timestamp/`Parallel` conventions
  (`predicate_stability.py`). New code: the tree walk/tabulation, purity, C1–C4
  computations, CSV/figure writers. Registry is `src/datasets.py` (the thesis's
  `src/ui/data.py` citation is stale — fix in the same edit pass).

---

## 3. Data

The three real datasets of the placeholder table, prepared exactly as in the RQ1 harness
(`prepare_dataset`: seeded 1000-row subsample, seed 42, index reset; labels: `is_red`,
one-hot `target_*` argmax):

| Dataset | $d$ | labels | role |
|---|---|---|---|
| Wine quality (Low) | 11 | 2 (is_red) | anchor: overlaps RQ2's wine arm → C4 directly comparable |
| Breast cancer (Low) | 30 | 2 | C1 extension point between wine (11) and synthetic (27→30) |
| Digits (Low) | 64 | 10 | C1 stress point; 10-class purity makes D4 non-trivial |

Exclusions, stated so they aren't quiet choices: **concentric rings / swiss roll** are
synthetic — §6.5 is explicitly the no-ground-truth arm; **Iris** (150 rows) collides with
`hclust_min_cluster_size` = 25 conventions and has no row in the placeholder table;
**MNIST/Fashion-MNIST** (784-d) is an optional stress arm behind the cut line (§9) — if
run, it extends C1 by another order of magnitude ($0.95^{784} \approx 10^{-18}$: the dense
predicate must admit essentially nothing).

---

## 4. Configuration

Shipped defaults (`default_config()`), because the point is ecological validity — with
every deviation named and justified here:

| Knob | Value | Status |
|---|---|---|
| method (DR) | UMAP | default; matches all prior harnesses |
| normalize, scaler | True, root StandardScaler | default |
| hclust_umap_n_components | 2 | default |
| hclust_min_cluster_size / min_samples | 25 / 5 | default |
| `hierarchical_layers` | **4** | **deviation** — UI default 1, prior harnesses 2. Table 6.2 treats depth as *discovered*; with the cap at the harness convention (2) the depth column is vacuous (every build realizes 2). Cap 4 lets the stopping rules (§2) terminate branches naturally; prediction: realized depth 2–3, cap never binding. If any branch hits 4, the cell is reported as cap-censored ("4†"). |
| subsample | 1000 rows, seed 42 | RQ1/RQ2 harness convention (thesis §6.3) |
| builds per dataset | **5** (rebuild, unseeded UMAP) | SEEDS_WINE convention; tree variability across builds is itself reported (H1b saw wine ARI 0.09–0.85 across seeds — expect leaf counts to vary too) |
| predicate `t` grid | {1.0, 0.95, 0.9, 0.8} | pre-specified in the thesis; no new levels |
| predicate methods | threshold (dense) + db (sparse) | dense needed by C1, sparse by C4/the table's example |
| tail split | **severity** (shipped default) | RQ2 found the splits indistinguishable on wine ($|\Delta| \le 0.03$, $p \ge 0.05$); using the shipped default keeps §6.5 descriptive of the tool as it is, while Ch. 7 still recommends symmetric going forward. Not swept here — the ablation belongs to RQ2 and is done. |
| min selection size | 20 (leaves below it get no predicates; counted) | RQ2 convention |

Cost model: 5 builds × 3 datasets; per build one root UMAP (1000×d), ≤ ~40 node UMAPs on
≤ 1000 points, ZADU per node ($O(n^2)$, $n \le 1000$), predicates $O(n·d)$ per leaf ×
t-grid. Minutes per build on a laptop; the whole run well under an hour, parallel over
(dataset, build) cells.

---

## 5. Measured quantities and fixed reporting rules

Aggregation is pre-specified per cell of Table 6.2. Unit of analysis: the **build**
(median over 5 builds, min–max range in the cell); within a build, leaves aggregate
unweighted (every leaf is one view the user can reach; size-weighting would let one big
leaf dominate — the size-weighted variant goes in the CSV, not the table).

- **depth** = realized max leaf depth per build → median (range) over builds; "†" if
  cap-censored.
- **#leaves** = leaf count per build → median (range). Reported alongside (prose, not a
  table column): noise fraction per build → median (range), and median leaf size.
- **mean trust. / mean cont.** = unweighted mean over *scored* leaves (shipped scores,
  D2) per build → median over builds. Scored-leaf coverage reported; if any leaf lacks
  scores ($n < 10$ — impossible given child-size ≥ 25, but the rule is stated anyway),
  the denominator says so.
- **example leaf predicate** = from the build with the **median leaf count** (tie →
  earlier build id), the leaf of **median size** (tie → larger): the sparse `db`
  predicate, strict ($t=1.0$) and relaxed ($t=0.95$), verbatim clauses with F1, coverage
  and length for both. Rule fixed here precisely so the example cannot be shopped for.
- **purity/enrichment (D4)** per leaf: majority-class share of leaf members (noise
  excluded — it belongs to no leaf; reported separately via D1), and enrichment =
  purity / global share of that class. Association readout: Spearman ρ between leaf mean
  T&C and leaf purity, pooled per dataset over builds, reported with $n_\text{leaves}$.
- **C1** per dataset: dense-predicate recall of leaf members at each $t$, median over
  leaves and builds, plotted against the $t^d$ bound (the `rq2_compounding` axes, two
  new points per $t$).
- **C2** per leaf: paired `_score_node`(leaf X, leaf embedding) vs `_score_node`(leaf X,
  root embedding rows) — **restricted to leaves with $n \ge 41$** so $k = 20$ on both
  sides (the H1a parity condition; smaller leaves are recorded but excluded from the
  check). Readout: median paired Δtrust/Δcont, win rate; Wilcoxon signed-rank, the
  **only** p-value in this experiment (single test, no multiplicity apparatus needed).
- **C3/C4** as defined in §1, computed from D1/D3 records.

---

## 6. Procedure (pseudocode)

```
for dataset in {wine, breast_cancer, digits}:
  df, feature_cols, y = prepare_dataset(dataset)            # 1000 rows, seed 42
  for build in range(5):                                    # unseeded UMAP → replicates
    cfg = default_config(); cfg["hierarchical_layers"] = 4
    tree = start_evaluation(df, feature_cols, cfg)          # tree + shipped node scores
    X_all = standardised_X(df, feature_cols, tree)
    record shape: realized depth, leaves, sizes, noise fraction          # D1
    for node in walk(tree): record node scores (+ CADI if internal)      # D2
    for leaf in collect_leaves(tree) with n >= 20:
      for method in {threshold, db}, t in {1.0, .95, .9, .8}:
        pred = generate_predicate(method, df_leaf, X_all, t, tail_split="severity")
        record F1/precision/recall/coverage/length vs leaf membership    # D3, C1, C4
      if n >= 41: record paired leaf-vs-root _score_node deltas          # C2
      if y is not None: record purity, enrichment                        # D4
aggregate per §5; emit Table 6.2 cells + consistency verdict lines C1–C4
```

---

## 7. Analysis

Deliberately thin — this is a census, not a trial. Medians and ranges throughout;
the single Wilcoxon for C2; no Holm ladder because there is no hypothesis family, and
inventing one would dress description up as inference (the reverse of the §6.6 lesson:
there the danger was too little correction, here it is fake rigor). Each C-check gets a
one-line verdict in the run summary: prediction, observed, pass/fail.

---

## 8. Threats to validity / how this could mislead

- **Post-hoc storytelling (the big one).** "Behaves sensibly" is defined *here, before
  the run*, as: C1–C4 pass and no degenerate builds (a build whose root finds < 2
  clusters, i.e. a single-leaf tree, is degenerate). Degenerate builds are reported and
  counted, never re-rolled — with unseeded UMAP, re-rolling is silent p-hacking.
- **Consistency checks are not independent replication.** C2/C3 reuse H1b/H1a's datasets
  and defaults; passing them shows internal coherence of the project's harnesses, not
  external validity. Only C1 and C4-direction on breast cancer/digits are genuinely
  out-of-sample. Say so in the thesis prose.
- **k-dependence of shipped scores.** Table 6.2's leaf means mix $k = 12$–$20$ with the
  root's $k = 20$; leaf-vs-root comparisons from the *table* are therefore soft. The
  inference-grade contrast is C2's paired, k-matched arm. The prose must route the claim
  through C2, and the table's role is descriptive.
- **Purity temptation.** With labels on all three datasets, high-purity leaves will
  exist. The classes≠clusters caveat is already in §6.5's text; keep enrichment next to
  purity (10-class digits makes raw purity look bad, enrichment fixes the base rate) and
  keep D4's weak-correlation expectation visible so purity cannot quietly become the
  headline.
- **Noise fraction vs leaf count.** A tree with many leaves and 25% noise describes a
  different tool experience than the same leaves at 3% noise; the table footnote must
  carry the noise range (D1), or the #leaves column overstates coverage.
- **Example predicate representativeness.** The fixed rule (§5) prevents cherry-picking
  but not unluckiness: the median-size leaf may be unusually messy. Tolerated — one
  verbatim example plus distributional stats (D3) over all leaves is exactly the split
  between illustration and evidence; the stats carry the claim, the example carries
  readability.
- **Severity split as shipped default.** Ch. 7 recommends symmetric; this run uses
  severity (§4). On wine RQ2 measured the difference as null at these $t$; if a reader
  objects, the D3 records contain enough to re-run with `symmetric` in minutes. Not
  swept, to keep §6.5 out of RQ2's ablation business.
- **UMAP-only scope.** Everything here is within-UMAP (the shipped default); H1a showed
  DR-method dependence is the dominant axis. §6.5 must not generalize across DR methods
  — one sentence of scope, pointing at Table 6.3.

---

## 9. Scope control

Feature freeze applies to `src/`; this harness, like the other four, is `src_research/`
scaffolding and touches no shipped code path. Build order, cut line explicit:

1. **Core (must ship):** driver over 3 datasets × 5 builds; D1/D2 records; `db`
   predicates (D3 minimum: lengths + F1 at $t \in \{1.0, 0.95\}$); Table 6.2 emission;
   C3, C4.
2. **Second:** threshold-method sweep (C1), paired C2 arm, D4 purity/ρ, figures.
3. **Optional:** MNIST stress arm; depth-2 comparability arm (same run at
   `hierarchical_layers = 2` to bridge exactly to H1b); full $t$-grid on `db`.

A complete (1) fills the table and removes the placeholder — the submission blocker.
(2) is what makes the section scientific rather than decorative; cut (3) freely.

## 10. Outputs

`outputs/experiments/<timestamp>/` (harness convention):

- `tree_shape.csv` — one row per (dataset, build): realized depth, #leaves, leaf-size
  stats, noise fraction, degenerate flag.
- `node_scores.csv` — one row per node: depth, is_leaf, n, k, T/C/MRRE/stress/CADI.
- `leaf_predicates.csv` — one row per (dataset, build, leaf, method, t): length, F1,
  precision, recall, coverage; verbatim clause dump for the example-rule leaf.
- `consistency.csv` + summary print — C1–C4 verdict lines (prediction / observed /
  pass-fail); C2 paired records.
- Figures: `benchmark_shape_faithfulness.png` (per dataset: leaf T&C distributions with
  root reference marks); `benchmark_compounding.png` (C1 points on the $t^d$ axes —
  extends `rq2_compounding`).

**Thesis integration.** Table 6.2 cells per §5 (add a noise-fraction note to the
caption); replace the §6.5 body: one paragraph of shape + faithfulness description, one
paragraph of predicate readability with the verbatim example, one paragraph of C1–C4
consistency verdicts (this is the section's actual contribution — phrase C-failures, if
any, as findings); delete the gap box; update the §6.10 summary sentence "the case-study
results remain to be filled in" to reflect that all quantitative sections are complete;
fix the stale `src/ui/data.py` registry citation in §6.2 while touching the chapter. If
MNIST runs, it enters the Datasets table's usage note, not new prose.

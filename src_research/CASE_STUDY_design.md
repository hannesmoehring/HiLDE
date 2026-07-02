# Case-Study Design: The Lens, Evidentially (RQ1, RQ2)

Design document. The scripted run will live in `src_research/` next to the experiment
harnesses; the captured figures and the narrative go into `sections/06_evaluation.tex`
(the "Qualitative Case Study" section, currently a placeholder). This file specifies *how
to make a single walk-through count as evidence rather than anecdote* before any figure is
captured.

The case study carries a specific burden in this thesis. The offline experiments answer the
*representational* and *partitional* questions; what they cannot show is the project's actual
aim — that the hierarchical **lens** lets an analyst *reach and inspect* locally relevant
structure that a single global view would hide. "Interesting" is analyst-dependent and not
formally definable, so this claim is irreducibly about use. The risk is that a case study
becomes a cherry-picked story. The whole design below exists to remove that degree of
freedom: every narrated "the global view merged this, the local view separated it" must come
with a number computed at that exact step, on a path chosen by a rule fixed in advance.

---

## 1. What is actually being claimed (and what would falsify it)

The walk-through illustrates two claims, each tied to an offline result so the qualitative
and quantitative evidence corroborate rather than substitute:

- **C1 (lens / reachability).** Drilling into a region and re-projecting it locally surfaces
  sub-structure that the global projection of the whole dataset merges. This is the
  *qualitative counterpart of H1a* — and it must be shown with the **same faithfulness
  measures** (`evaluate._score_node`: trustworthiness, continuity, stress) plus a direct
  **structure-visibility** measure at each step, not by visual assertion.
- **C2 (predicate readability + stability).** A user selection inside a leaf is described by a
  strict predicate and a relaxed one; the relaxed predicate is shorter / more readable and
  more stable under selection perturbation, at a stated cost in specificity. This is the
  *qualitative counterpart of H2*.

**Falsification conditions, stated up front (a case study that cannot fail is not evidence):**
- C1 fails at a step if the local re-projection's faithfulness delta over the global view is
  ~0 *and* no sub-structure is visible locally that is hidden globally. **These steps will be
  reported, not hidden.** A run in which most steps are null is a negative result for C1.
- C2 fails if relaxation does not reduce predicate length / variance, or if it inflates
  coverage to near-trivial (the predicate admits most of the data). The coverage/length/
  stability numbers decide this, not the prose.

---

## 2. Why a case study can be evidence at all: four anti-anecdote controls

1. **Pre-registered path (no cherry-picking the route).** The drill-down is driven by a
   **deterministic policy fixed before looking at outcomes** (§4), not by the analyst chasing
   the prettiest split. The faithfulness gain is the *outcome*, never the selection criterion.
2. **A number on every claim.** Each step records the local-vs-global faithfulness delta and a
   structure-visibility score (§5). The narrative quotes these.
3. **External corroboration of "interesting."** The dataset carries labels / semantically
   meaningful features, so when the lens separates a sub-group we can *name what it is* (a
   chemical regime, a digit substyle). This grounds "interesting" without pretending to define
   it — labels corroborate, they do not adjudicate (the "classes are not clusters" caveat
   still holds).
4. **The counterfactual is shown, not asserted.** Every reveal step is a *side-by-side panel*:
   the same points read off the global projection vs the local re-projection, with the
   faithfulness numbers under each (§6).

---

## 3. Dataset choice

The dataset must let *both* claims be shown: a real two-level structure (coarse split, then
finer within-group structure in a different subspace — mirroring the planted-subspace finding)
**and** semantically meaningful features so predicates are readable.

| Candidate | Coarse structure | Fine (conditional) structure | Predicate readability |
|-----------|------------------|------------------------------|-----------------------|
| **Wine quality (recommended)** | red vs white (large global separation) | within-type style/quality regimes living in *different* chemical dimensions (e.g.\ whites by residual sugar / SO₂, reds by volatile acidity / sulphates) | **high** — named chemical features, interpretable ranges |
| Digits (visual alternative) | digit identity | substyles within a digit (e.g.\ 1 with/without serif, 7 with/without crossbar) | low — pixel predicates are not readable, but the *images* make the reveal visually striking |
| Breast cancer | malignant / benign | severity gradients within class | moderate |

**Recommendation:** Wine quality as the primary case study — its red/white → within-type
structure is the real-data analogue of the nested-subspace generator (coarse split, then fine
structure in a *different* feature subspace per group), and its named features make C2 land.
Use **Digits as a one-figure visual companion** for C1, where the "global merged / local
separated" panel is immediately legible to a reader without reading axis labels. Decide one
primary; do not dilute across many (external validity is already the standing limit, §10).

---

## 4. The pre-registered drill-down policy

Fix this *before* inspecting any embedding, and state it verbatim in the thesis so the route
is auditable.

- **Start:** root density overview (the KDE topography the UI shows), full dataset, the UI's
  default DR method and HDBSCAN config (record them).
- **Descent rule (neutral, outcome-independent):** at each level descend into the child region
  with the **largest point mass**. Largest mass is chosen precisely because it is *not* a
  proxy for the effect under test (it does not favour regions whose local re-projection
  happens to improve faithfulness). Record ties and break them by lowest region index.
- **Depth:** stop at depth 2 (matching the experiments) or earlier if a node becomes a leaf
  (`< 2` non-noise clusters). Capture every level on the way down, not just the final leaf.
- **Selection within the terminal leaf:** the **densest connected component** in the leaf's
  local 2D embedding (operationalised as the largest cluster from one HDBSCAN pass on the
  leaf's local 2D), simulating an analyst lassoing the dominant sub-mode. This is reproducible
  and not hand-drawn.
- **Robustness:** re-run the whole policy under 2–3 alternative neutral descent rules
  (largest mass; highest mean GLOSH-core density; largest child by count) and report whether
  the qualitative story is rule-dependent. If a reveal only appears under one route, say so.

> The point of the policy is that a sceptical reader can object to the *rule* but not to
> *cherry-picking within* it. The narrative then honestly reports whatever that rule produced.

---

## 5. Per-step instrumentation (the numbers behind the narrative)

At each drill-down step into region `r` (points `X_r`, the node's standardised features),
record, reusing existing code:

- **Faithfulness delta (C1, = H1a at this step).** `local = _score_node(X_r, leaf_local_2D)`
  vs `global = _score_node(X_r, E_global[idx_r])`, identical `k` (forced by equal `n`). Report
  `Δtrust, Δcont, Δstress`. Reuse `_score_node` and `_embed_original` unchanged; `E_global` is
  one global embedding of the full standardised data via the same DR method (as in
  `hierarchical_vs_flat.py`).
- **Structure-visibility score (C1, the direct "reveal").** Quantify that sub-structure exists
  *locally* but not *globally* for the same points. Two complementary measures:
  - *label-based (corroboration):* fine-label silhouette (or kNN-label agreement) of the
    region's points in the local 2D vs in the global-restricted 2D. A positive local−global
    gap means the lens makes a known sub-grouping separable that the global view flattened.
  - *label-free (honest):* number of HDBSCAN clusters recovered in the local 2D vs the
    global-restricted 2D, and their DBCV. More resolvable modes locally = the reveal, with no
    appeal to labels.
- **What the sub-structure *is* (naming "interesting").** For the revealed sub-groups, report
  the feature(s) / label(s) that distinguish them (top differentiating dimensions, or class
  composition). This is the sentence "...and these turn out to be the high-volatile-acidity
  reds," which is what makes the reveal *interesting* rather than merely *present*.

All three are cheap and reuse the experiment harness; a step's row is directly comparable to
the H1a table, so the case study is anchored to the aggregate result rather than floating free.

---

## 6. The counterfactual panel (the core figure)

For each reveal step, one figure with two sub-panels on the **same point set**:

- **(a) Global view:** `E_global` restricted to `idx_r`, coloured by the revealed sub-grouping.
- **(b) Local view:** the leaf's `embedding_original`, same colouring.

Caption carries the §5 numbers: `Δtrust`, `Δstress`, and the local−global structure-visibility
gap. The reader *sees* the merge-vs-separate and *reads* the measure that it is real. This is
the figure that turns the narrative into evidence; it is the case-study analogue of
`fig:hierarchy-benefit` but on real data with metrics attached.

Thesis figure slots (replace the placeholder `\gap`): density overview → drill-down sequence
→ **counterfactual panel(s)** → leaf scatter with the selected component → predicate range
chart.

---

## 7. Predicate evidence (C2)

On the terminal selection, run both predicate methods (`predicate_generator._predicate_db` for
strict/DimBridge-style, `_predicate_threshold` for relaxed with skew-aware asymmetric trimming)
and report, as a small table beside the range chart:

- **Readability/specificity:** predicate **length** (number of clauses), **coverage** (fraction
  of data admitted), **precision/recall** of the selected points.
- **Stability (the H2 point, qualitatively):** perturb the selection (drop/add a small random
  fraction of points, a few times) and report the variance of the resulting predicate
  (clause-set Jaccard, or range-endpoint spread) for strict vs relaxed. The claim "relaxed is
  more stable" must be a number here, not an adjective.
- **Dimension match (grounds C2 in meaning):** do the predicate's named dimensions coincide
  with the features that actually distinguish the region (from §5)? On wine this is checkable
  and makes the predicate *trustworthy*, not just short.

---

## 8. Honest reporting rules

- **Report null and negative steps.** If a drill-down step yields ~0 faithfulness gain and no
  extra resolvable structure, it goes in the table and the narrative says so. A walk-through
  where the lens helps at some steps and not others is more credible — and more useful to a
  reader deciding when to drill — than an unbroken string of wins.
- **No post-hoc route changes.** If the pre-registered policy lands on a boring region, that is
  the result; do not re-pick the path to find a better story (the robustness rules in §4 are
  the only sanctioned alternatives, and all are reported).
- **Labels corroborate, never adjudicate.** A leaf that is label-impure is not a failure
  (classes ≠ clusters); a leaf that is label-pure is corroboration that the reveal is
  meaningful.
- **Scope it.** One dataset is an existence demonstration of the lens in use, not evidence of
  prevalence. State that the expert-feedback session (§ next) is what extends a single
  analyst's path toward external validity.

---

## 9. Outputs and reproducibility

Match the harness conventions (`outputs/experiments/<timestamp>/` style, scripted, seeded):

- `case_study_steps.csv` — one row per drill-down step: `level, region_id, n, Δtrust, Δcont,
  Δstress, vis_local, vis_global, vis_gap, top_dims, label_composition, descent_rule`.
- `case_study_predicates.csv` — strict vs relaxed: `length, coverage, precision, recall,
  perturbation_variance`.
- `plots/` — density overview, drill-down sequence, the §6 counterfactual panel(s), leaf
  scatter + selection, predicate range chart. Saved with the names the thesis `\includegraphics`
  calls expect, so figures regenerate with the rest.
- A `main()` driving the prototype's analysis path head-less from a fixed seed and config, so
  the entire walk-through is reproducible from one command (no manual clicking). This is also
  what the benchmark-driver placeholder needs, so build it once and share it.

---

## 10. Threats to validity

- **Single dataset / single path.** The deepest limit. Mitigated by the deterministic policy +
  robustness routes (§4) and by framing the result as "the lens demonstrably surfaces hidden
  structure *here*," not "everywhere."
- **Confirmation bias.** Mitigated by pre-registration, the falsification conditions (§1), and
  the rule to report null steps (§8).
- **"Interesting" is grounded by labels.** Labels are corroboration, not the definition; the
  expert-feedback session is the human check that the revealed structure is *interesting to a
  domain analyst*, which labels cannot certify.
- **Easier-problem confound (carried from H1a).** Re-projecting fewer points is intrinsically
  easier; reporting `Δstress` alongside `Δtrust` (a large trust gain with non-degraded stress
  is the strong reveal) keeps this visible, exactly as in the offline experiment.
- **Prototype-path fidelity.** The scripted run must use the *same* code path as the
  interactive UI (`compute_analysis_tree`, `predicate_generator`), or the case study describes
  a different system than the one a user touches. Assert shared code paths.

---

## 11. Open decisions (confirm before scripting)

1. **Primary dataset** — wine quality (readable predicates, nested red/white→within-type story)
   vs digits (visually striking reveal). *Recommendation:* wine primary, one digits panel for
   C1.
2. **Descent rule** — largest mass (recommended, outcome-neutral) vs density/GLOSH-based.
   Whichever is primary, the others are the §4 robustness routes.
3. **How many reveal steps to feature** — recommend 2–3 down to one leaf, plus at least one
   honestly-null step if the policy produces one.
4. **Expert session coupling** — the same scripted walk-through should be what the domain
   expert is shown, so C1/C2 evidence and the expert's think-aloud are about the identical
   artefact.

---

## 12. Implementation checklist

Reuse (do not reimplement):

- `compute_analysis_tree` / `start_evaluation`, `collect_leaves`, `clustering_space`,
  `standardised_X` — the tree, leaves, and clustering space (already in
  `hierarchical_vs_flat.py`).
- `_embed_original`, `_score_node` — global embedding and identical faithfulness scoring.
- `predicate_generator._predicate_db`, `_predicate_threshold`, `_tail_removal_shares` — strict
  and relaxed predicates.
- `characteristics.py` (per-region stats, GLOSH) and the KDE overview — for the density figure.

To build:

- the deterministic descent driver (§4) over the prototype's analysis path, head-less + seeded.
- the per-step instrumentation (§5) and the perturbation loop for predicate stability (§7).
- the counterfactual-panel plotter (§6) and the CSV writers.
- a `main()` that emits all figures under the names the thesis expects.

Suggested filename: `src_research/case_study.py` (and it doubles as the benchmark-driver the
other placeholder needs).

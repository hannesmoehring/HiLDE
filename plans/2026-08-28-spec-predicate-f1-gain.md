# Spec: Marginal F1 Gain for Predicate Clauses

Status: source plan (Hannes) + binding amendments from the code-verified review of 26.08.
Blocks marked **AMENDMENT (binding)** override the surrounding original text where they
conflict. Scope decision (Hannes, 28.08): sequential ΔF1 is kept and is defined ONLY for
greedy/sequential construction algorithms; it is a property of the construction order,
not of a predicate as such.

## Goal

Expose how much each selected clause improves the greedy predicate's F1 score at the
moment it is added. This should help users understand how the predicate was constructed
and which clauses contributed most to its quality.

## Metric semantics

For a clause selected at greedy step k:

```text
predicate_f1_gain = predicate_f1_after - predicate_f1_before
```

Where `predicate_f1_before` is the conjunction's F1 before adding the clause,
`predicate_f1_after` after, and `predicate_step` remains the zero-based selection order.
The initial baseline is the F1 of the empty conjunction, which matches every background
point. Therefore `baseline F1 + sum(selected clause gains) = final predicate F1` within
floating-point tolerance.

> **AMENDMENT (binding) — record counts and P/R, not only ΔF1.** At RCM 1.0 every clause
> interval is the selection's exact min/max, so every clause has recall 1 and
> `F1_k = 2|S| / (|M_k| + |S|)` — a pure function of the match count `|M_k|`. The
> interpretable per-step quantities are therefore `n_matched`, `precision`, and `recall`
> after each step; `_f1` already computes precision/recall and the greedy loop discards
> them via `[0]`. Record all three per selected clause alongside the gain. At RCM 0.9 a
> clause can trade recall for precision — ΔF1 alone hides this; the P/R pair
> disambiguates.

### Interpretation

Call this metric "F1 gain when added" or "greedy contribution". Do not describe it as
general feature importance because it is: dependent on clause selection order; affected
by redundant or correlated features; conditional on previously selected clauses; and
specific to the selected RCM and predicate scope. The existing `clause_f1` is presented
as **Standalone F1** (the clause alone, without the rest of the conjunction).

> **AMENDMENT (binding):** add to the caveat list: the gain is conditional on the FIXED
> interval produced by the RCM trim, not on the feature per se — a discriminative
> feature with an unlucky 0.9 trim scores low.

## Data contract

### Predicate row

Add to every `"db"` predicate row:

```python
predicate_f1_before: float | None
predicate_f1_after: float | None
predicate_f1_gain: float | None
predicate_n_matched: int | None  # |M_k| after this clause was added   (AMENDED)
predicate_precision: float | None  # precision after this clause         (AMENDED)
predicate_recall: float | None  # recall after this clause            (AMENDED)
predicate_baseline_f1: float  # identical on every row              (AMENDED)
```

For selected clauses all step fields are populated; for unselected clauses and for the
no-labels path (`selected_indices is None`) they are `None`. Keep the existing
`predicate_f1` field (final conjunction F1, identical on every row) for backward
compatibility.

> **AMENDMENT (binding) — baseline plumbing.** Do NOT derive the baseline in
> `backend/predicate.py` from the first selected clause's `predicate_f1_before` with a
> zero-clause special case (original §4 below). `_predicate_db` already holds the
> baseline as the initial `best_f1`; stamp `predicate_baseline_f1` on every row exactly
> as `predicate_f1` is stamped (pattern documented at `src/README.md:167`). The
> zero-clause case (baseline == final) is then correct automatically.

### Predicate summary

Add `predicate_baseline_f1: float` (read from any row). The summary describes the
trajectory, e.g. `F1 0.02 → 1.00`.

> **AMENDMENT (binding):** the baseline is `2|S|/(N+|S|)` — a function of selection and
> background size only, not of the data. Label it "no clauses (all N points)" wherever
> it is displayed, so it is not read as an informative score. In local scope on a small
> node it is large (39 of 200 → 0.33), which is exactly where mislabeling would mislead.

## Backend implementation

### 1. Record the F1 transition during greedy selection

In `src/analysis/predicate_generator.py`: retain `previous_f1 = best_f1` before
accepting a candidate; on acceptance record `predicate_f1_before`, `predicate_f1_after`,
`predicate_f1_gain` (and, per amendment, `predicate_n_matched`, `predicate_precision`,
`predicate_recall` from the accepted candidate's `_f1` result) on the accepted row.
Avoid computing the gain after overwriting `best_f1`.

> **AMENDMENT (advisory, forward-compatibility):** other predicate generators are
> planned. Prefer recording only `predicate_step` in the loop and computing
> before/after/gain/P/R by replaying the recorded order over the clause masks in a small
> helper — the same helper then serves any future sequential generator. Either
> implementation is acceptable for this track; the data contract is what is binding.

### 2. Initialize nullable fields

Initialize all new fields to `None` where the standalone clause metrics are initialized,
and in the no-labels branch.

### 3. Make tie-breaking deterministic

Resolve equal-F1 candidates by original feature-column order, e.g.
`max(scored, key=lambda c: (c[0], -c[1]))`. This must not otherwise alter the greedy
selection rule or stopping tolerance.

> **AMENDMENT (binding) — verified fact + required regression check.** The current
> behaviour is ALREADY deterministic on CPython: small ints hash to themselves,
> `set(range(n))` iterates ascending, and `discard` does not reorder, so `max` already
> returns the lowest index on ties. The explicit key is worth adding because it makes
> this implementation-independent, but it is a behaviour-preserving change and must be
> VERIFIED, not asserted: re-run `SHD/misc/predicate_baselines_2026-08-26/run_hilde.py`
> before and after and diff clause lists and order. Expected, unchanged: F1 1.000 with
> 10 clauses at t=1.0 and F1 0.861 with 5 clauses at t=0.9, identical clause order.

### 4. Expose the baseline through the backend response

Original: read the baseline from the first selected clause's `predicate_f1_before`; if
no clause is selected, the baseline equals the final predicate F1.

> **AMENDMENT (binding):** superseded — read `predicate_baseline_f1` from any row (see
> Data contract). Add it to the summary dict. Continue using the RCM 1.0 predicate for
> the main summary; the RCM 0.9 run calculates its own gains independently and those do
> not replace the main summary values.

## Frontend contract

Extend `PredicateRow` in `frontend/src/types.ts` with the new nullable fields (incl. the
amended count/P/R fields and `predicate_baseline_f1`), and `PredicateSummary` with
`predicate_baseline_f1: number`.

## Frontend presentation

### 1. Sort selected clauses by construction order

In `frontend/src/charts/PredicateBands.tsx`: selected clauses before unselected;
selected by ascending `predicate_step`; unselected by descending standalone `clause_f1`.

> **AMENDMENT (binding) — the RCM 0.9 run must not be invisible.** `PredicateBands`
> currently uses `trimmed` only for core-band geometry and the tooltip row is always a
> `full` row, so the 0.9 gains would be computed and never shown — yet the 0.9 predicate
> is typically the interesting one (E2: 5 clauses at 0.861 vs 10 at 1.000). The 0.9 run
> can differ in clause membership AND order (the `LayerSide` comment already says so).
> Required: the tooltip carries a second block for the core (RCM 0.9) run when the
> feature is in that run's predicate (its step, gain, F1 transition), and the summary
> shows both trajectories. Note in a code comment that row order follows the 1.0
> construction; the 0.9 order can differ.

### 2. Display the marginal gain visibly

For each selected clause show a right-aligned `+0.214` (three decimals), labelled
"ΔF1 when added". Do not encode the gain in band width or opacity (those channels
already carry the interval and membership).

> **AMENDMENT (binding):** three decimals do NOT prevent a displayed `+0.000` — the
> stopping tolerance is 1e-6, and a clause excluding one background point of 4898 gains
> ≈ 3e-6, is accepted, and rounds to +0.000. Do not claim otherwise, and do not change
> the tolerance. Mitigation: show the per-step match count (`4898 → 611 → … → 39`) next
> to or instead of tiny ΔF1 values; counts are exact and never degenerate.

### 3. Expand the tooltip

Selected clause: step, F1 gain when added, predicate F1 transition, standalone F1 — plus
(amended) matched-count and precision/recall after the step, and the core-run block per
the amendment above. Unselected clause: "Not selected" + standalone F1. Rename the
existing label `Clause F1` → `Standalone F1`.

### 4. Improve the predicate summary

In `LayerSide.tsx` and `ExplorationPanel.tsx`, change `F1 0.77` to a trajectory
(`F1 0.02 → 0.77`), with the baseline labelled "no clauses (all N points)" (amended) via
tooltip or accessible label. Show both the full and trimmed trajectories (amended).

## Testing

Create `backend/tests/test_predicate.py`, runnable standalone and via pytest (copy the
pattern of `backend/tests/test_serialize.py`).

Required cases (original list, as amended):

1. Selected-clause gain consistency: gain > 0, before/after populated,
   before + gain ≈ after.
2. Telescoping invariant: baseline + sum(gains) ≈ final F1.
3. Step continuity: steps are 0..n_selected−1.
4. ~~Transition continuity (after_k ≈ before_{k+1})~~ — **AMENDMENT (binding):**
   tautological (it tests the recording code against itself). Replace with the step-0
   invariant: the first selected clause has the maximum standalone `clause_f1`, and
   `gain_0 == clause_f1 − baseline`. This catches mis-recorded transitions; the original
   test cannot.
5. Unselected clauses: all step fields `None`.
6. No-label path: fields present and `None`.
7. Deterministic ties: two identical feature columns → the earlier column is selected
   first (the second is never selected: zero gain given the first).
8. No selected clause: baseline == final, `n_features_used == 0`.
9. JSON safety: full response encodes with `json.dumps(..., allow_nan=False)`.
10. RCM independence: full and trimmed each internally consistent even when their
    selected clauses differ.
11. **AMENDMENT (binding), added:** regression — the tie-break change is
    behaviour-preserving on the E2 stand-in selection (see Backend §3).

## Fixtures and documentation

Update `frontend/src/fixtures/predicate_iris.json` (note: it has NO consumer in the
codebase — update for documentation value only), `src/README.md`, `frontend/README.md`.
Document all fields incl. the amended ones; state explicitly that the marginal gain is
order-dependent and conditional on the RCM-trimmed interval.
`src_research/predicate_stability.py` is the only other `generate_predicate` caller —
additive keys should not break it; run it once to confirm.

## Verification commands

```bash
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
cd frontend && npm run build
uv run ruff format --check .
```

Manual: cluster Predicate tab + lasso selection in the exploration panel; local and
global scope; selected clauses in step order; gains reconstruct final F1 from baseline;
unselected features show standalone F1 only; full + trimmed bands still render; the
trimmed tooltip block appears where the 0.9 run differs.

## Non-goals

No Shapley values, no order-independent importance, no leave-one-clause-out necessity,
no change to the greedy selection rule or the 1e-6 stopping tolerance, no merging of the
RCM 1.0/0.9 runs, no clustering or DR changes. (Generator-agnostic attribution for
non-sequential algorithms is explicitly deferred; see project notes of 26.08.)

## Acceptance criteria

Original list, minus the rounding claim (a displayed `+0.000` is acceptable per the
amendment), plus: per-step `n_matched`/precision/recall populated for selected clauses;
`predicate_baseline_f1` on every row and in the summary; both RCM trajectories visible
in the UI; tie-break regression (E2 stand-in) unchanged; backend tests pass; production
frontend build passes; docs and fixture match the contract.

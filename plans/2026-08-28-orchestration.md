# Orchestration Plan: Predicate ΔF1 Gain + Cluster Movement

Date: 2026-08-28 · Repo: `SHD/dev/SHD` · For: the implementing (orchestrator) coding agent

You are the ORCHESTRATOR. This folder (`plans/`) is self-contained. You implement TWO
features by delegating bounded tasks to subagents and integrating their results:

- **Track P — Marginal F1 gain for predicate clauses.**
  Spec: `plans/2026-08-28-spec-predicate-f1-gain.md`
- **Track M — Cluster-to-point / cluster-to-cluster movement (counterfactual preview).**
  Spec: `plans/2026-08-28-spec-cluster-movement.md`

Both specs contain the original plans WITH inline blocks marked **AMENDMENT (binding)**
from a code-verified review. The amendments override the surrounding original text
wherever they conflict. Read both specs in full before spawning anything.

---

## 1. Ground rules

- Work on a new branch `feat/predicate-gain-and-movement` off the current HEAD of
  `presentation_additions`. HEAD currently sits exactly on the tag `final-thesis`
  (thesis reproducibility anchor). NEVER move, delete, or retag `final-thesis`.
- Commit at the end of each stage (§3) with a message naming the stage. Track P gets its
  own commit(s) before movement work lands, so it stays cherry-pickable. Do not push
  unless Hannes asks.
- Python via the repo venv: `PYTHONPATH=. .venv/bin/python …`; formatting via
  `uv run ruff format .` (check with `--check`). Frontend: `cd frontend && npm run build`.
  All commands run natively on this macOS machine.
- New backend tests must be runnable BOTH standalone
  (`PYTHONPATH=. .venv/bin/python -m backend.tests.<name>`) and under pytest — copy the
  pattern at the top of `backend/tests/test_serialize.py`.
- Do not reformat, rename, or "improve" code outside the files a task owns.
- Do not mutate the source dataframe or the analysis tree anywhere in either feature.
- Interface-freeze workflow (this repo has used it before — see the header comment of
  `frontend/src/charts/props.ts`): contracts are written and committed FIRST, then
  subagents implement against them in parallel. Subagents never edit a contract file;
  contract changes go through you.

### File ownership map (conflict avoidance)

| Owner | Files |
|---|---|
| Orchestrator only | `frontend/src/types.ts`, `frontend/src/api.ts`, `frontend/src/charts/props.ts`, `backend/app.py` (route wiring), `frontend/src/App.tsx` (final integration), all README updates |
| P1 (Track P) | `src/analysis/predicate_generator.py`, `backend/predicate.py`, `frontend/src/charts/PredicateBands.tsx`, `frontend/src/components/LayerSide.tsx`, `frontend/src/components/ExplorationPanel.tsx`, new `backend/tests/test_predicate.py`, `frontend/src/fixtures/predicate_iris.json` |
| A1 (M foundations) | `backend/movement.py` (skeleton: artifact cache, alignment, validation), `src/analysis/analysis_routine.py` + `backend/serialize.py` (effective_config + analysis_id + schema version, additive only) |
| B1 (M PCA) | `backend/movement.py` (PCA section), `backend/tests/test_movement.py` (PCA + alignment + validation tests) |
| U1 (M UMAP) | `backend/movement.py` (UMAP section), `backend/movement_jobs.py`, `backend/tests/test_movement.py` (UMAP tests appended — do not touch B1's) |
| F1 (M interaction) | `frontend/src/charts/ClusterScatter.tsx`, new `frontend/src/movement.ts` (state hook) |
| F2 (M panel) | new `frontend/src/components/ClusterMovementPanel.tsx`, `frontend/src/styles.css` (additions only) |
| R1 (reviewer) | read-only + scratch verification scripts |

Two agents never own the same file at the same time. `backend/movement.py` passes
sequentially A1 → B1 → U1. **Sequencing note:** `LayerSide.tsx` and
`ExplorationPanel.tsx` belong to P1 during stages A–B; the movement "Move C2…" action
may also touch `LayerSide.tsx` — that edit is done by the orchestrator during stage C
integration, strictly AFTER P1 has finished and committed.

---

## 2. Binding amendments — summary index

Full text lives inline in the specs; this is the orchestrator's checklist.

Track P (spec §§ as marked): per-step `n_matched`/precision/recall recorded next to the
gain; RCM 0.9 trajectory shown (tooltip core-run block + both trajectories in summary);
`predicate_baseline_f1` stamped on every row (no first-clause special case); explicit
tie-break `(f1, -index)` verified behaviour-preserving by re-running
`SHD/misc/predicate_baselines_2026-08-26/run_hilde.py` (expect 1.000/10 at t=1.0,
0.861/5 at t=0.9, identical order); a displayed `+0.000` is acceptable — never widen the
1e-6 tolerance; baseline labelled "no clauses (all N points)"; tautological
transition-continuity test replaced by the step-0 invariant.

Track M: (1) movement refits read a serialized `effective_config` (post-build mutated
config) only — never the raw request config; schema version bump; missing → 409.
(2) reducers built exclusively via `fit_dimensionality_reducer`/`_embed_original`.
(3) same-process refit→identity alignment canary test; large RMSE is surfaced, never
tolerated away. (4) ALL response 2D quantities in visible coordinates; PCA client-side
slider uses the returned visible-space displacement. (5) free-point min-norm sentence in
the panel + norm-space caveat in docs. (6) UMAP `strength_trajectory` returned and
plotted; non-monotonicity caveat. (7) `out_of_range_fraction` = fraction of moved points
with ≥1 mutable feature outside the full dataset's raw range; warn > 0.05.
(8) movement state lives in a `useMovement` hook (`frontend/src/movement.ts`) — there is
no `Navigation` struct in `App.tsx`. (9) validation additions (child-count minimums,
non-null parent embedding, MDS named in the gate/UI). (10) substantive PCA tests
(Δ ∈ rowspace(P), immutable zeros, centroid-reaches-target, raw-delta unscaling) over
the tautological pairwise-distance check. (11) vertical-slice checkpoint after
PCA + interaction, before UMAP.

---

## 3. Orchestration stages

### Stage A — Baseline, contracts, and parallel start

Orchestrator first: verify baseline green (`test_serialize`, `test_targets`, frontend
build), create the branch, read both specs and:
`src/analysis/analysis_routine.py`, `src/analysis/dim_reducer.py`,
`src/analysis/predicate_generator.py`, `backend/app.py`, `backend/predicate.py`,
`backend/run_cache.py`, `backend/jobs.py`, `backend/serialize.py`,
`frontend/src/App.tsx`, `frontend/src/charts/ClusterScatter.tsx`,
`frontend/src/charts/PredicateBands.tsx`, `frontend/src/api.ts`,
`frontend/src/types.ts`.

Then write and commit the CONTRACTS yourself (no subagent):
- `frontend/src/types.ts`: Track P — extended `PredicateRow` and `PredicateSummary`
  (per the P spec's data contract, amended fields included). Track M —
  `MovementTarget`, `MovementRequest`, `MovementResponse` (incl. `strength_trajectory`,
  `out_of_range_fraction`, `alignment_rmse`, `preview_sampled`, warnings;
  visible-coordinate semantics documented in comments), `MovementState`.
- `frontend/src/api.ts`: `runMovement()` mirroring `runAnalysis` polling
  (same `POLL_RETRIES` behaviour).
- A fixture `frontend/src/fixtures/movement_iris.json` — hand-written, contract-valid,
  so F1/F2 can build before the movement backend exists.
- The HTTP contract (request/response JSON, 400/409 semantics) as a docstring header in
  a stub `backend/movement.py`.

Then spawn in parallel:
- **P1 (Track P, complete feature)**: implement its spec end to end — generator
  recording, `backend/predicate.py` summary, PredicateBands sort/gain/tooltip (incl. the
  core-run block), both panel summaries, `test_predicate.py`, the `run_hilde.py`
  tie-break regression, fixture + README notes. Acceptance: `test_predicate` green;
  regression numbers unchanged; frontend builds; orchestrator spot-checks the tooltip
  and both trajectories in the dev server. Commit as its own commit when green.
- **A1 (Track M foundations)**: `effective_config` + `analysis_id` + schema version in
  the build/serialize path (additive fields only — update `test_serialize`'s
  `_NODE_KEYS` additively if needed); node lookup from serialized ids; request
  validation incl. amendment 9; parent reconstruction (root scaler → parent rows)
  reusing `_embed_original`; bounded LRU `ProjectionArtifact` cache; similarity
  alignment (Procrustes: centering, uniform scale, rotation/reflection) with both
  directions stored; alignment canary tests. Acceptance: canary green for PCA and UMAP
  on Iris; `test_serialize` still green.

### Stage B — Parallel: PCA backend ∥ frontend interaction

- **B1 (PCA movement)**: free-point pinv displacement, cluster-centroid displacement,
  mutable masking, strength scaling, raw-unit reporting, metrics incl. the amended
  `out_of_range_fraction`, all responses in visible coordinates; PCA + validation tests
  per amendment 10. Inline responses (no job) for PCA. Acceptance: `test_movement`
  PCA/alignment/validation sections green.
- **F1 (interaction, against the fixture)**: `useMovement` hook; `ClusterScatter`
  movement mode — target resolution precedence (centroid label / legend / other-cluster
  point / 10 px snap / free point), noise hit targets, screen→embedding inversion
  through the d3 zoom transform and the equal-aspect projection, drag/zoom suppression,
  Escape/cancel, stale-response guards; ghost points, target marker, arrow. Normal
  drill-down untouched when movement mode is off. Acceptance: build green; source-plan
  manual checklist items 1–13 pass against the fixture (orchestrator spot-checks).

### Stage C — Integration (orchestrator) + panel

- **F2 (panel)**: `ClusterMovementPanel.tsx` + CSS — source/destination, strength slider
  (PCA local update via the returned visible-space displacement; debounced backend calls
  reserved for UMAP), feature-change table sorted by |standardized change|, min-norm
  sentence, distance metrics, `strength_trajectory` mini-chart placeholder, warnings,
  frozen-projection note, JSON/CSV export, no Apply button.
- Orchestrator: wire routes in `app.py`, add the "Move C2…" action (LayerSide — after
  P1's commit), mount panel + hook in `App.tsx`, replace fixture with live API, run PCA
  end-to-end on Iris (point + cluster targets, zoomed and panned). Commit.
  **Vertical-slice checkpoint: STOP and report to Hannes before Stage D.**

### Stage D — UMAP (after the checkpoint)

- **U1**: inverse-transform point targets; neighbor-interpolation fallback
  (`target_estimator`); observed-centroid cluster targets; deterministic strength grid +
  trajectory; support ratio + warnings; deterministic sampling
  (`MAX_MOVEMENT_PREVIEW_POINTS=5000`, `MAX_STRENGTH_SEARCH_POINTS=1000`); background
  jobs via the `backend/jobs.py` pattern (`/api/movement/jobs/{id}`); UMAP tests (no
  exact-destination assertions).
- Orchestrator: wire jobs polling in `api.ts`/panel, debounce slider requests, verify on
  Iris and one wide image dataset (sampled preview visibly badged). Commit.

### Stage E — Light review + docs

- **R1 (reviewer, read-only + scratch scripts)**. Scope — exactly this, not a full
  audit:
  1. Independently re-derive the PCA movement math on a toy dataset (small script):
     pinv solution reaches the target, row-space membership, immutable zeros, raw-unit
     inversion.
  2. Track P math check on live Iris output: telescoping invariant
     (baseline + Σ gains ≈ final), step-0 invariant, and the `run_hilde.py` regression
     numbers.
  3. Field-by-field parity: `backend/movement.py` and `backend/predicate.py` responses
     ↔ `types.ts` ↔ component usage.
  4. Grep-level safety pass: no writes to the source dataframe / analysis tree / cached
     payloads; LRU actually bounded; artifacts keyed by analysis_id + node_id.
  5. State hygiene: every clear-condition in the movement spec actually clears; the
     stale-response guard checks analysis id, node, source, target, and generation.
  6. Run the full suite + production build.
  R1 reports findings as file:line + failure scenario. Orchestrator fixes only
  trivial/typo-level findings directly; everything else goes verbatim into the final
  report. Do not iterate review rounds.
- Docs (orchestrator, last): README.md, backend/README.md, frontend/README.md,
  src/README.md per both specs' documentation sections. Final commit.

---

## 4. Verification gates (run at every stage boundary)

```bash
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
cd frontend && npm run build
uv run ruff format --check .
```

(`test_predicate` exists from Stage A onward; `test_movement` from Stage B onward.)

## 5. Stop and ask Hannes (do not improvise past these)

- The `run_hilde.py` tie-break regression changes ANY number or clause order.
- The same-process alignment canary cannot be made green (a real parity gap — do NOT
  widen tolerances to pass).
- The installed `umap-learn` lacks a working `inverse_transform` for the configured
  parameters (fallback-only mode is a product decision).
- Any change to existing serialized payload fields beyond additive ones seems required.
- The vertical-slice checkpoint (end of Stage C).
- Anything touching the `final-thesis` tag or thesis-cited outputs.

## 6. Final deliverable

A summary report: what was built per stage and track, test/build status, R1's unresolved
findings, deviations from the specs (each justified against the amendments), and the
commit list on `feat/predicate-gain-and-movement`.

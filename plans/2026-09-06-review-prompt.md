# Review task: verify the implementation of `plans/2026-08-28-*.md`

Repo: `SHD/dev/SHD` (macOS, native). Branch: `feat/predicate-gain-and-movement`.
You are an INDEPENDENT REVIEWER, not the implementer. You never fix, refactor, commit,
stash, or format. You may write scratch scripts under `/tmp/review/` only. Every claim
in your report must be backed by something you ran or read (file:line, command output).
Code comments, docstrings, and READMEs are claims to be checked, not evidence.

## State you are starting from (verified facts — do not re-derive, do confirm)

- `HEAD` = `aa1ee92`, which IS the tag `final-thesis`. The branch has ZERO commits:
  all work is in the working tree — 21 modified tracked files (+1378/−86) and 8
  untracked paths (`backend/movement.py`, `backend/movement_jobs.py`,
  `backend/tests/test_movement.py`, `backend/tests/test_predicate.py`,
  `frontend/src/components/ClusterMovementPanel.tsx`, `frontend/src/movement.ts`,
  `frontend/src/fixtures/movement_iris.json`, `.claude/settings.local.json`).
- `plans/` is untracked (only just un-ignored via `.gitignore`), so there is NO committed
  baseline of the specs. `plans/2026-08-28-spec-predicate-f1-gain.md` has a later mtime
  (15:46) than the other two (15:22), and `pyproject.toml` gained
  `[tool.ruff] extend-exclude = ["plans"]` with a comment admitting `ruff format`
  "silently rewrote a design document under plans/".
- No final report (§6 of the orchestration plan) exists anywhere in the repo.
- `backend/movement_jobs.py` exists → Stage D (UMAP) was apparently implemented even
  though §3 Stage C says "STOP and report to Hannes before Stage D".

## Inputs

Read in full, in this order: `plans/2026-08-28-orchestration.md`,
`plans/2026-08-28-spec-predicate-f1-gain.md`, `plans/2026-08-28-spec-cluster-movement.md`.
Blocks marked **AMENDMENT (binding)** override surrounding text. Then `git diff` and
every untracked file listed above.

## What to review

### 0. Process compliance (report as facts, no editorializing)
Against orchestration §1, §3, §5, §6: stage commits (none), Track P cherry-pickability
(currently impossible — P and M edits are interleaved in the same working tree; say which
files contain BOTH tracks' edits), the Stage C checkpoint, the file-ownership map (any
file edited that no track owns?), the stop-and-ask list (was any trigger hit and
improvised past? e.g. tolerance widened, `inverse_transform` fallback-only mode,
non-additive payload changes), and the missing final report.

### 1. Verification gates (orchestration §4) — run all, paste outputs verbatim
```
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q
(cd frontend && npm run build)
uv run ruff format --check .
uv run ruff check .
```
Never run `ruff format` without `--check`. Also confirm the two new test files run under
BOTH invocation styles (standalone + pytest), as §1 requires.

### 2. Track P — spec + amendment compliance
For each amendment in orchestration §2 (Track P paragraph), give ✓ / ✗ / partial with
file:line evidence:
per-step `n_matched`/precision/recall stored next to the gain; RCM 0.9 trajectory in
tooltip core-run block and both trajectories in the summary; `predicate_baseline_f1` on
every row with no first-clause special case; explicit tie-break `(f1, -index)`; 1e-6
tolerance unchanged (`+0.000` allowed); baseline labelled "no clauses (all N points)";
step-0 invariant test present and the tautological continuity test gone.
Then run the regression yourself:
`SHD/misc/predicate_baselines_2026-08-26/run_hilde.py` — expected 1.000 / 10 clauses at
t=1.0, 0.861 / 5 at t=0.9, identical clause order to before. Any change = BLOCKER.
On live Iris output check the telescoping invariant (baseline + Σ gains ≈ final F1 within
1e-6) and the step-0 invariant numerically, not by reading the test.

### 3. Track M — amendments 1–11, each with evidence
Pay particular attention to:
(1) every refit path reads `effective_config` (the post-build, mutated copy) — grep every
place `config` reaches `fit_dimensionality_reducer`/`_embed_original` from movement code;
missing `effective_config` → HTTP 409; schema version bumped. (2) no reducer is
constructed by any other route. (3) the alignment canary: what tolerance does the test
use, and is there any code path that swallows or caps `alignment_rmse`? (4) EVERY 2D
quantity in the response is in visible coordinates, and the PCA slider in the panel uses
the returned visible-space displacement, not `Y + α(PΔ)`. (5) min-norm sentence present
in the panel. (6) `strength_trajectory` has 11 points, is plotted, and non-monotonicity
triggers a warning. (7) `out_of_range_fraction` = fraction of moved points with ≥1
mutable feature outside the FULL dataset's raw range (not the node's range; not
per-feature average), warning threshold 0.05. (8) state lives in `useMovement`
(`frontend/src/movement.ts`); no `Navigation` struct. (9) validation: child-count
minimums, non-null parent embedding, MDS named in the gate AND the UI. (10) substantive
PCA tests exist (Δ ∈ rowspace(P), immutable zeros, centroid reaches target, raw-delta
unscaling) — and the tautological pairwise-distance check is not the only test.
(11) sampling constants `MAX_MOVEMENT_PREVIEW_POINTS=5000`, `MAX_STRENGTH_SEARCH_POINTS=1000`,
deterministic; `preview_sampled` surfaced in the UI.

### 4. Independent math re-derivation (write your own script, do not import theirs)
On a toy dataset (e.g. 60×5, 3 blobs) with your own PCA: pinv displacement reaches the
target exactly for a free point; Δ lies in rowspace(P); immutable features are exactly 0;
raw-unit unscaling matches `scaler.inverse_transform` differences; cluster Δ = μt − μs.
Then call the repo's functions with the same inputs and compare numerically.

### 5. Contract parity
Field-by-field: `backend/movement.py` and `backend/predicate.py` response dicts ↔
`frontend/src/types.ts` ↔ what `ClusterMovementPanel.tsx`, `PredicateBands.tsx`,
`LayerSide.tsx`, `ExplorationPanel.tsx`, `movement.ts` actually read. List every field
that is produced but never consumed, consumed but never produced, or typed differently
(optional vs required, null vs undefined). Check the hand-written
`movement_iris.json` fixture still validates against the final types.

### 6. Safety / mutation
Grep-level and read-level: no writes to the source dataframe, the analysis tree, node
dicts, or cached payloads from movement or predicate code (look for in-place ops on
`ctx.X_orig`, `config`, node dicts, `run_cache` entries). `ProjectionArtifact` LRU is
actually bounded and keyed by `analysis_id` + node id. Serialized payload changes are
additive only — diff `_NODE_KEYS` in `test_serialize.py` and check old cached payloads
(without `effective_config`) are still loadable and yield 409 on movement, not 500.

### 7. Frontend state hygiene
In `movement.ts`/`ClusterScatter.tsx`/`App.tsx`: every clear-condition in the movement
spec actually clears movement state (dataset change, re-run, node navigation, mode
toggle, Escape); the stale-response guard compares analysis id, node, source, target
AND generation; drill-down behaviour with movement mode OFF is byte-for-byte the old
behaviour (diff `ClusterScatter.tsx` and reason about the off-mode path).

### 8. End-to-end smoke (backend only; you have no browser)
Start the backend, run Iris analysis, then via curl: PCA movement with a point target and
a cluster target on a child node (200, visible coords, warnings list), a request with the
raw config instead of `effective_config` (expect 409), a malformed target (400), a UMAP
movement job through `/api/movement/jobs/{id}` to completion. Paste the responses
(truncated to relevant fields).

### 9. Housekeeping
`.claude/settings.local.json` must not be committed; `.gitignore` `!plans/*` and the
`pyproject.toml` ruff exclude — sensible or not; and whether the predicate spec was
altered by `ruff format` (inspect its Python code blocks for formatter-style changes and
check all AMENDMENT blocks are intact and readable).

## Deliverable

Write `plans/2026-09-06-review-report.md` containing, in this order:
1. GO / NO-GO for committing the working tree as-is, one sentence of justification.
2. Findings ranked by severity (BLOCKER / MAJOR / MINOR), each as
   `file:line — claim — failure scenario — how verified`. No praise, no filler.
3. Compliance matrix: one row per amendment (P: 7 items, M: 11 items) with ✓/✗/partial
   and the evidence pointer.
4. Verbatim gate outputs (§1) and the regression numbers (§2).
5. Process deviations (§0) as a plain list.
6. What you could NOT verify and why.
7. A recommended commit split that keeps Track P cherry-pickable (file list per commit).
   Do NOT execute it.

Be scientific and critical of the implementation and of your own checks. If a test
passes, ask whether it could have passed with a wrong implementation. If something
looks right, say what evidence would have shown it wrong and whether you looked for it.

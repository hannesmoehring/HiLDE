# Shipping plan: predicate ΔF1 + cluster movement → hardened, split, compact UI, Apply

Date: 2026-09-06 · Repo: `SHD/dev/SHD` · Branch: `feat/predicate-gain-and-movement`
For: the implementing (orchestrator) coding agent. Read first:
`plans/2026-09-06-review-report.md` (findings, §7 commit split), then the two specs
`plans/2026-08-28-spec-*.md` (their AMENDMENT blocks stay binding), then this file.
This file overrides both where they conflict.

Starting state: HEAD `aa1ee92` = tag `final-thesis`, zero commits on the branch, the whole
implementation is an uncommitted working tree (21 modified + 8 untracked). All gates green,
no correctness defect found by review. Two independent reviews agree: splitting commits
fixes delivery, not the testing/runtime gaps below — those come first.

---

## 0. Decisions taken by Hannes (06.09) — binding

| # | Question | Decision |
|---|---|---|
| D1 | Runtime alignment gate `ALIGNMENT_RMSE_MAX = 0.05` | KEEP the value as a PRODUCT decision, stated as such. What it bounds, exactly: the root-mean-square residual of the similarity alignment over the node's EXISTING points, normalized by the cloud's RMS radius. It does NOT bound individual residuals (single points can exceed 5 %) and it says nothing about the error of NEW counterfactual points. Write that sentence, verbatim in meaning, in the constant's comment and backend/README; also report `alignment_max_residual` alongside the RMSE in the response (additive field) so the panel can show both. Boundary tests (synthetic alignment at 0.049 → 200, 0.051 → 409) verify enforcement, not suitability. Never widen. |
| D2 | Movement preview when drilling DEEPER | KEEP current behaviour (preview persists while its node is on the path; cancelled when the node leaves the path). Update the movement spec text + frontend/README to say so. |
| D3 | `feature_centroid_distance_after` for POINT targets | OMIT: backend returns `null` for point targets (it is (1−α)‖Δ‖ by construction), panel hides the tile. Keep it for cluster targets. |
| D4 | Apply | YES — full rebuild on a counterfactual copy of the dataset (§3). Original spec's "no Apply button" is superseded. |
| D5 | Apply stacking | Stack of immutable edits with Undo-last and Reset-to-original; badge everywhere. |
| D6 | Panel text | Drastic reduction (§2): numbers, table, sparkline, one-line warnings; every explanation moves into a tooltip or a collapsed `<details>`. |

---

## 1. Hardening pass (before any commit) — closes the review findings

Do these on the working tree as-is; each item names the finding it closes.

H1 · **Lock the artifact cache** (MINOR-6/7). One `threading.Lock` in `backend/movement.py`
guarding get/move_to_end/put/evict of `_artifact_cache` as a single critical section in
`artifact_for` and `clear_artifact_cache`; `movement_jobs.py` calls only through those
functions. Dedupe cold refits: an in-flight table `{(analysis_id,node_id): Future}` so a
second cold request waits on the first instead of refitting. The lock is held ONLY for
dictionary operations — release it before fitting and before waiting on a Future
(pattern: under lock, look up cache or in-flight; if absent, register a new Future and
release; fit outside the lock; under lock, store the artifact, pop the Future; set its
result after release). A fit that raises must pop its Future and propagate to all waiters. Test: two threads request the
same cold node concurrently → `fit_node_projection` called exactly once (spy).

H2 · **Replace the mislabelled canary with a plumbing test** (MAJOR-1). Do NOT rely on a
small-row fixture: umap-learn truncates `n_neighbors` to `n−1` internally, so raw and
effective configs can still produce identical fits — a data-driven canary cannot
discriminate. Instead, in `test_movement.py`: build a payload whose `meta.effective_config`
is deliberately different from `meta.config` in a refit-relevant key (e.g. `umap_n_neighbors`
7 vs 15, `dim_reduction` unchanged), spy on `fit_node_projection`/`refit_config`, run a
movement through `compute_movement`, and assert the spy received the EFFECTIVE values.
Negative control in the same test: monkeypatch the pipeline to read `meta.config` and
assert the test fails. Rename the existing canaries to `test_*_refit_reproduces_the_serialized_embedding`
(they prove parity — keep them, fix their docstrings).

H3 · **409 ordering** (MINOR-2): schema/`effective_config` check BEFORE the `analysis_id`
comparison, so a legacy payload gets "rebuild required". Test with a stripped payload.

H4 · **Baseline label with the real count** (MINOR-3): `PredicateSummary` gains
`n_background` (backend/predicate.py); label reads `no clauses (all 50 points)` in both
panels; fixture + `types.ts` updated (P-only hunk).

H5 · **pytest** (MINOR-5): `uv add --dev pytest`; keep the dual-invocation docstrings.

H6 · **READMEs** (MINOR-9): the two new test commands in every runnable block; the
test-suite sentence names four scripts.

H7 · **D1, D3** as specified above.

H8 · Small: test build banner prints "Wine quality (Low)" while feeding Iris — set
`dataset_choice: "Iris"` in the test config (cosmetic, unreported). Drop
`trimmed_n_features_used` from `PredicateSummary` + `types.ts` (MINOR-4) unless a consumer
is added. Regenerate `frontend/src/fixtures/movement_iris.json` from a LIVE backend response
(PCA point target on root/1 of Iris) so the fixture satisfies the shared-translation
invariant (MINOR-10); add a script `frontend/scripts/capture_movement_fixture.py` or note
the curl in frontend/README. `.gitignore`: `!plans/*.md` instead of `!plans/*` (MINOR-12).

Gate after H1–H8: the full §5 gate list green; `run_hilde.py` regression unchanged.

---

## 2. Panel compaction + relocation (D6)

Current: `ClusterMovementPanel` is stacked under `LayerSide` in the narrow right column
(`.layer__side`, ~40 % width) and carries ~12 paragraphs of prose.

Target layout — `frontend/src/App.tsx`, `styles.css`, `ClusterMovementPanel.tsx`:
- **Idle** (no movement started): only a single button row stays in the right column
  under LayerSide: `[Move C{k}…]` (disabled tooltip explains why when no child selected)
  and, for t-SNE/MDS runs, the same button disabled with the one-line tooltip
  "Not available for {method}: no feature→2D mapping. Rebuild with PCA or UMAP." Nothing else.
- **Active** (started, loading, or ready): the panel MOVES to a new full-width row
  `.layer__movement` rendered directly below `.layer__cols` (i.e. under the plot AND the
  characteristics), inside the same layer card. Three-column grid
  `minmax(220px,1fr) minmax(260px,1.2fr) minmax(320px,1.6fr)`, collapsing to one column
  under 900 px:
  1. **Controls**: "C{src} → {destination}" one line; strength slider with the numeric value
     and a `rec.` tick at the recommended strength; buttons `[Apply]` (§3) `[Cancel]`
     `[CSV] [JSON]`.
  2. **Metrics**: stat tiles only — projected distance before → after, feature-space distance
     before → after (cluster targets only, D3), alignment RMSE, out-of-range %, support ratio
     (UMAP), `sampled` badge; plus the strength sparkline. Each tile label carries an
     `(i)` tooltip (`title=`) with the current explanatory sentence; no paragraph.
  3. **Feature changes** table (as now), sorted by |standardized change|; held-fixed features
     shown as a footer line "{n} held fixed (ids/labels)".
- **Warnings** become one-line chips below the grid, each with the long text in `title`:
  non-monotone trajectory, out-of-range > 5 %, support ratio > 2, under-determined
  destination (amended min-norm sentence — keep ONE short visible line, keyed on the
  backend's norm space exactly as the long sentence is: "Smallest standardized change
  reaching this point; larger ones exist." when the run normalized, "Smallest raw-unit
  change …" when it did not; the full backend sentence in the tooltip), preview sampled.
- **FROZEN_NOTE** and the projected/feature-space units explanation move into a single
  collapsed `<details><summary>About this preview</summary>` at the bottom. Nothing else
  explanatory is visible by default.
- Prompt state ("click a destination…") is one line above the plot area, not in the panel.
Accessibility: every tooltip text also exists in the `<details>` block, so nothing is
hover-only.

Binding-amendment check: min-norm sentence still displayed (short form + tooltip),
non-monotone caveat still displayed (chip), `preview_sampled` badge, out-of-range warning,
MDS/t-SNE named. The spec's "no Apply button" line is struck by D4.

---

## 3. Apply — counterfactual session (D4, D5)

Semantics: **Apply writes the currently previewed feature deltas (at the current strength,
mutable features only, one rigid translation per moved row) into a server-side
counterfactual copy of the dataset and rebuilds the whole analysis on it with the same
request config.** The original dataframe and every existing payload stay untouched.
Ancestors, this node, children, characteristics, predicates and DR-quality scores all
reflect the moved rows afterwards. The hierarchy may change; cluster indices need not
correspond to the previous run — that is the honest result and the UI says so once.

### Backend (`backend/counterfactual.py`, new)
Two separate stores, deliberately:
- **Edit history** (small, unbounded within a session lifetime): `cf_id → CounterfactualEdit
  {base_dataset, parent: cf_id|None, node_id, source_child_index, target, strength,
  row_positions: list[int], row_ids: list[int], deltas_raw: {feature: float}, created_at}`.
  Never evicted by the frame cache; expires only by explicit session lifetime
  (`COUNTERFACTUAL_TTL = 24 h` since last access, sweep on insert) so Undo across a long
  session and across tabs keeps working. Snapshot identity:
  `cf_id = sha256(base_dataset | parent_id | canonical-json(edit fields))[:12]` — the base
  dataset is part of the identity, so the same edit on two datasets never collides.
- **Materialized frames** (large, bounded): LRU of 8 `df` copies keyed by `cf_id`; on a miss
  the frame is REBUILT by replaying the edit chain from the base loader — possible because
  history is never evicted. README notes the per-copy memory cost on wide image datasets.

Dataset key convention: `"{base}@cf:{cf_id}"`. `backend/datasets.load(key)` resolves it
(history lookup → materialized frame or replay). Plain keys unchanged; cf keys are NOT
listed by `/api/datasets`; `/api/datasets/{key}/columns` and `/image/{row_id}` resolve
through the same function. Unknown cf id → 404 with "counterfactual expired or unknown".

**Which rows change — exact rule.** Apply uses ALL members of the source cluster, i.e. the
source child's serialized `row_indices`, never `preview_points` (which are a ≤5000 sample).
`row_indices` are POSITIONS in the dataframe (see `_child_positions` docstring in
`backend/movement.py`), and the movement code fills `preview_points[].row_id` from those
positions, so the two coincide only while the loader's index is a `RangeIndex`. The
endpoint therefore: (1) takes positions from the tree, (2) records BOTH `row_positions`
and `row_ids = df["row_id"].iloc[positions]` in the edit, (3) applies via
`df.iloc[positions, col_positions] += δ` on the copy (positional, because the tree is
positional), (4) uses `row_ids` only for the export and the UI. Mutable features only
(`mutable_flags`); one rigid translation `applied_delta_raw` per row at the applied strength.
Tests: a frame with a non-contiguous, non-Range index; a source cluster larger than
`MAX_MOVEMENT_PREVIEW_POINTS` (lower the cap in the test) — every member changes, not the
sample.

- `POST /api/counterfactual/apply` `{analysis_id, dataset, feature_cols, config, node_id,
  source_child_index, target, strength}` → server RECOMPUTES the movement at that strength
  (never trusts client deltas; a UMAP source goes through the artifact cache and may take
  the job path — reuse `movement_jobs`, answer `{status:"running", job_id}` and let the
  client poll `/api/counterfactual/jobs/{id}`) → registers the edit on top of `dataset`'s
  chain (base or cf key) → returns `{dataset_key, cf_id, edits: [summary…],
  n_rows_changed}`. 409/400 semantics as `/api/movement`; strength ≤ 0 → 400. Idempotent:
  the same edit on the same parent yields the same `cf_id` (identity is content-derived),
  so a duplicate click cannot create two snapshots.
- Undo/Reset need no endpoint: keys are immutable snapshots, the client switches back.
- `run_cache.store` is SKIPPED for cf keys (hosting mode must never persist counterfactual
  runs to disk); `_tree_cache` in memory is fine. `analysis_id` already includes the
  dataset key, so movement on a cf run works unchanged.
- `GET /api/counterfactual/{cf_id}/rows.csv` → the changed rows (row_id + original and
  counterfactual values of the mutable features) for export; `GET /api/counterfactual/{cf_id}`
  → the edit chain (the banner's chip list is rendered from this, not from client memory).

### Frontend
- `frontend/src/counterfactual.ts`: `useCounterfactual({ baseDataset })` holding
  `stack: {datasetKey, cfId, summary}[]`, `activeDatasetKey`, `phase:
  "idle"|"applying"|"rebuilding"|"error"`; `apply(...)`, `undo()`, `reset()`. App passes
  `activeDatasetKey` as `dataset` to everything (analysis, predicate, characteristics,
  targets, rows, movement) — no other component changes.
- **Apply enablement — exact condition.** `state.phase === "ready"` is NOT sufficient: the
  hook stays `ready` while a UMAP strength refinement is pending (`pendingRef`) or after a
  failed refinement. Apply is enabled only when ALL hold: phase `ready`; no pending
  refinement timer and no in-flight movement request; `response.applied_strength ===
  state.strength` (PCA: local preview is exact, so equality is by construction; UMAP: the
  last response must be for the current slider value); the response identity (analysis id,
  node, source, target, generation) equals the hook's current request identity; and the
  counterfactual hook is `idle`. Otherwise the button is disabled with a tooltip naming the
  reason ("waiting for the preview at this strength").
- **Transition.** Click → `applying` (button disabled, no double-submit; a second click is
  a no-op) → `/api/counterfactual/apply` (poll if job) → on success `rebuilding`: run
  `/api/analysis` on the NEW key while the CURRENT analysis and layer stack stay on screen
  and interactive (read-only banner "rebuilding counterfactual…") → on success, in ONE state
  update: push the stack entry, swap `activeDatasetKey`, replace the displayed analysis,
  cancel the movement preview, navigate to ROOT. On any failure (apply 4xx/409, job error,
  rebuild error): phase `error` with the message in the banner, stack and displayed run
  unchanged, movement preview kept, button re-enabled. Stale-response guard: every response
  is dropped if the base dataset, the stack head, or the generation changed meanwhile.
- **Navigation after rebuild is always ROOT.** Node ids like `root/1` are generated from
  child positions and can name entirely different rows after a rebuild, so "the same path
  still exists" preserves nothing. The banner shows one line: "Hierarchy rebuilt on
  counterfactual data — cluster numbering does not correspond to the previous run."
  (Membership-based correspondence — Jaccard of row sets between old and new nodes — is
  a possible later feature, not part of this plan.)
- Banner above the layer stack whenever `stack.length > 0`: `Counterfactual data · {k}
  edit(s) applied · [Undo] [Reset] [Export rows]`, listing edits as "C{src} → … @ {α}"
  chips. Undo/Reset re-run `/api/analysis` on the previous/base key (cache hit → inline)
  and also navigate to root. Dataset selector shows the base name with a `cf` badge;
  changing the base dataset resets the stack. On page reload the stack is gone (session
  state is client-side); the URL is not made to carry it in this plan.
- Types in `types.ts`: `CounterfactualEdit`, `CounterfactualApplyResponse`,
  `CounterfactualPhase`.

### Tests (`backend/tests/test_counterfactual.py`, dual-invocation)
- resolved cf frame equals base except exactly `row_positions × mutable features`, differing
  by `applied_delta_raw` to 1e-12; base frame identical before/after (values and `id()` of
  the memoized base object untouched); immutable features and `target_*` untouched.
- non-Range index frame: positions vs `row_id` differ, the correct rows change, the export
  lists the right `row_id`s; source larger than the preview cap: all members change.
- history survives frame eviction: create 9 edits (LRU 8), resolve the oldest key → replayed
  correctly from the base; same edit twice on the same parent → same `cf_id`, one entry;
  same edit on two base datasets → two different ids.
- chain of two edits applies in order; undoing to the parent key still resolves.
- `/api/analysis` on the cf key produces a payload with a different `analysis_id` and
  (on Iris, a large move) a different tree; hosting mode writes nothing to `cache_dir`.
- cf keys absent from `/api/datasets`; LRU bounded; unknown cf id → 404.
- Movement on a cf run: 200 (artifact keyed by the new analysis_id).

---

## 4. Stages, commits, and what must be true at each

Commit only when the stage's gates are green; **each commit must build and test on its
own, with the gates that exist AT that commit** (a commit cannot run tests a later commit
introduces). Verify in a disposable worktree, never by stashing this large mixed tree:
`git worktree add /tmp/wt-<n> <hash>`, run that commit's gate set there (Python via the
repo `.venv` with `PYTHONPATH=/tmp/wt-<n>`; frontend needs its own `npm ci` in the
worktree once — or symlink `node_modules`), then `git worktree remove`. Gate sets per
commit: 0 → `ruff format --check`, `ruff check`; 1 → + `test_predicate`, `test_serialize`,
`test_targets`, `npm run build`, `run_hilde.py` regression; 2 → + `test_movement` (the
foundations/plumbing subset that exists then — split the file if needed so the PCA/UMAP
sections arrive with commit 3); 3 → full `test_movement`; 4 → + `npm run build` with the
live panel; 5 → all. Track P must be cherry-pickable onto `presentation_additions`
standalone — prove it by cherry-picking commit 1 onto a scratch branch at `final-thesis`
in a worktree and running commit 1's gate set.

| Stage | Content | Commit(s) |
|---|---|---|
| S1 | Hardening H1–H8 on the working tree | none yet |
| S2 | Split per report §7, with the hardening hunks folded into the right commit: 0 plans/chore · 1 Track P (+H4, H8-trimmed field, P hunks of H6) · 2 M foundations (+H2 test, H3) · 3 M backend (+H1, D1 gate+tests, D3) · 4 M frontend (+D2 text, fixture regen) · 5 docs (+H6) | 0–5 |
| S3 | **CHECKPOINT — STOP.** Report the commit list + gate outputs. Hannes does the first browser pass on the reviewed (old) layout (spec manual checklist 1–13: destination under zoom/pan, drag vs select, strength, cancel pending UMAP job, navigate during a request, Escape). Do not continue until he answers. | — |
| S4 | Panel compaction + relocation (§2) | 6 |
| S5 | Apply backend (§3, registry + endpoints + tests) | 7 |
| S6 | Apply frontend (hook, banner, button, navigation) | 8 |
| S6½ | **CHECKPOINT — STOP.** Hannes's second browser pass on the NEW layout and Apply: compact panel at full width and under 900 px; Apply on PCA and on UMAP (job path); stacked edits; Undo; Reset; a failed rebuild (e.g. kill the backend mid-job) leaves the previous run on screen; Apply disabled while a strength refinement is pending; movement on a counterfactual run. | — |
| S7 | Light review (R1 scope: parity backend↔types↔components for the new endpoints; mutation safety of the base frame; LRU bound; state hygiene of the cf stack) + docs (README, backend/README, frontend/README, src/README: counterfactual semantics, memory note, "hierarchy may change") | 9 |

Branch stays `feat/predicate-gain-and-movement`; do not push; never touch `final-thesis`.

## 5. Gates (every stage boundary, and per commit in S2)
```
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
PYTHONPATH=. .venv/bin/python -m backend.tests.test_counterfactual   # from S5
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q
(cd frontend && npm run build)
uv run ruff format --check .      # never plain `ruff format` — it rewrites Markdown fences
uv run ruff check .
```
Plus, at S1 and S2: `misc/predicate_baselines_2026-08-26/run_hilde.py` on a scratch copy —
1.000/10 at t=1.0, 0.861/5 at t=0.9, identical order.

## 6. Stop and ask Hannes
- Any regression number or clause order changes.
- A commit in S2 cannot be made independently green without moving hunks across tracks.
- H2's negative control cannot be made to fail (means the plumbing test is vacuous).
- Apply on a wide dataset needs more than one `df.copy()` per edit (memory), or the
  rebuild on a cf key exceeds the job timeout.
- Any non-additive change to serialized node fields.
- The S3 and S6½ checkpoints — mandatory stops.

## 7. Final deliverable
Report with: commit list with hashes, per-commit gate result, H1–H8 each mapped to its
finding and its test, the three §0 UI/Apply decisions as implemented, deviations (each
justified), open items. Write it to `plans/2026-09-XX-shipping-report.md`.

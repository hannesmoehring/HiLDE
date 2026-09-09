# Shipping report — S1–S6 + Numba fix · CHECKPOINT S6½

Date: 2026-09-06 · Branch: `feat/predicate-gain-and-movement` · Base: `aa1ee92` = `final-thesis`
Executed per `plans/2026-09-06-shipping-plan.md`. §1–§5 below are the S3 checkpoint as
reported; §6 onward covers S4 (panel), S5 (Apply backend), S6 (Apply frontend) and the
Numba threading fix Hannes reported, done after his S3 pass. **Stopped at S6½ as
instructed.** Nothing pushed; `final-thesis` untouched; S7 (light review + final docs) not
started.

---

## 1. Commits (each independently green in a disposable worktree)

| # | Hash | Subject | Gate set run at that commit | Result |
|---|---|---|---|---|
| 0 | `816c443` | chore: design specs, review and shipping plan; keep ruff out of plans/; pytest as a dev dependency | `ruff format --check` · `ruff check` | GREEN (33 findings, pre-existing) |
| 1 | `39edfb7` | feat(predicate): marginal ΔF1 per clause, per-step counts/P/R, baseline, RCM 0.9 trajectory | + `test_serialize` · `test_targets` · `test_predicate` · `pytest` (16) · `npm run build` · `run_hilde.py` regression | GREEN; regression IDENTICAL |
| 2 | `abba5a5` | feat(movement): analysis_id, effective_config, schema v2, fit_node_projection (foundations) | + `test_movement` (foundations subset, 2 tests) · `pytest` (18) | GREEN |
| 3 | `d99154f` | feat(movement): backend PCA/UMAP cluster movement, background jobs, tests | + full `test_movement` (49 tests) · `pytest` (65) | GREEN |
| 4 | `4682368` | feat(movement): scatter interaction, useMovement hook, movement panel, fixtures | all, `npm run build` with the live panel | GREEN |
| 5 | `6318696` | docs(movement): READMEs for cluster movement | all | GREEN |

Verification method: `git worktree add <scratch>/wt-<n> <hash>`, `PYTHONPATH=<wt>`, the repo
`.venv` python/ruff, `node_modules` symlinked, then `git worktree remove`. Per-commit output
(commit 5 shown; 1–4 identical in shape):

```
PASS  ruff format --check
INFO  ruff check: 33 findings (33 pre-existing expected)
PASS  test_serialize
PASS  test_targets
PASS  test_predicate
PASS  npm run build
PASS  test_movement
PASS  pytest: 65 passed, 23 warnings in 21.85s
GATE GREEN for 6318696 level 2
```

**Track P cherry-pick proof.** Commit 1 cherry-picked onto `final-thesis` in a worktree
(`git cherry-pick 39edfb7` → `2c4df57`, no conflicts, 12 files) and commit 1's gate set run
there: GREEN (16 pytest, build clean). The scratch branch `scratch/track-p-on-final-thesis`
is kept locally for inspection; delete with `git branch -D scratch/track-p-on-final-thesis`.

**Regression** (`misc/predicate_baselines_2026-08-26/run_hilde.py`, scratch copy, generator
from commit 1 and from the final tree — both):
```
strict t=1.0: 10 clauses  F1 1.0000  P 1.0000  R 1.0000  matched 39  covers 39/39
relaxed t=0.9: 5 clauses  F1 0.8611  P 0.9394  R 0.7949  matched 33  covers 31/39
diff hilde_result.json <stored 2026-08-26>  → IDENTICAL
```

Sum check: every one of the 37 files of the hardened working tree is byte-identical to its
content at `6318696`; `git status` is clean apart from this report.

---

## 2. Hardening H1–H8 → finding → where → test

| Item | Closes | Change | Test |
|---|---|---|---|
| H1 | MINOR-6, MINOR-7 | `backend/movement.py`: `_artifact_lock` guards every get / move_to_end / put / evict of `_artifact_cache` (`artifact_for`, `clear_artifact_cache`, `artifact_is_cached`); `_artifact_in_flight: {(analysis_id, node_id): Future}` dedupes cold refits. Lock held only for dict ops; the owner fits outside it, waiters block on the Future outside it; a failing fit pops its Future and propagates to all waiters. `movement_jobs.py` never touched the cache directly and still does not. | `test_concurrent_cold_requests_share_one_refit` (second thread verified to be *waiting* while the first is inside the fit; `fit_node_projection` spy called exactly once; same artifact object for both), `test_failed_refit_reaches_every_waiter_and_leaves_nothing_cached` |
| H2 | MAJOR-1 | Plumbing test: payload with `meta.config.umap_n_neighbors = 15` and `meta.effective_config.umap_n_neighbors = 7` (PCA run, so the knob does not touch the fit and the alignment gate stays out of the picture); spy on `fit_node_projection` asserts the fit received **7**. Negative control in the same test: `refit_config` monkeypatched to build from the request config → the fit receives **15** (i.e. the assertion *would* fail) — confirmed, the test is not vacuous. Canaries renamed `test_{pca,umap}_refit_reproduces_the_serialized_embedding`, docstring states they prove parity, not the amendment. | `test_refit_reads_effective_config_not_the_request_config` |
| H3 | MINOR-2 | `_require_compatible_run`: schema/`effective_config` check now precedes the `analysis_id` comparison. | `test_payload_predating_effective_config_is_409` — stripped payload (no `analysis_id`, `effective_config`, `schema_version`) gets "before movement support…", not "different analysis run" |
| H4 | MINOR-3 | `PredicateSummary.n_background` (backend + `types.ts` + fixture); `LayerSide.tsx` / `ExplorationPanel.tsx` titles read `no clauses (all 150 points)` with the real count. P-only hunks. | `test_summary_reports_the_background_size_the_baseline_is_scored_on` (local N = node size, global N = 150, baseline = 2|S|/(N+|S|)) |
| H5 | MINOR-5 | `uv add --dev pytest` (pyproject + uv.lock, commit 0); dual-invocation docstrings kept, pytest line corrected to `.venv/bin/python -m pytest`. `backend/requirements.txt` untouched (production image, no dev deps). | `pytest backend/tests -q` → 65 passed |
| H6 | MINOR-9 | Both runnable blocks (README.md, backend/README.md) list all four scripts plus the pytest one-liner; the limitations bullet names four scripts (three at commit 1). Every command was run. | — |
| H7 | D1, D3 | see §3 | see §3 |
| H8 | cosmetic, MINOR-4, MINOR-10, MINOR-12 | test build banner: `dataset_choice = "Iris (Low)"` (banner now prints Iris). `trimmed_n_features_used` dropped from `PredicateSummary`, `types.ts`, fixture, test. `movement_iris.json` regenerated from a LIVE backend (uvicorn on a scratch port) by `frontend/scripts/capture_movement_fixture.py`, which checks the shared-translation invariant before writing — deviation 8.9e-16, alignment RMSE 2.8e-16. `.gitignore`: `!plans/*.md`. | script asserts the invariant; `test_alignment_gate_is_enforced_at_its_boundary` |

---

## 3. §0 decisions as implemented

**D1 — alignment gate.** `ALIGNMENT_RMSE_MAX = 0.05` kept; its comment (and backend/README
"The alignment gate") states verbatim in meaning: it bounds the root-mean-square residual of
the similarity alignment over the node's EXISTING points, normalized by the visible cloud's
RMS radius; it does NOT bound individual residuals (single points can exceed 5 %); it says
nothing about the error of NEW counterfactual points; never widen. `Alignment.max_residual`
added and reported as `metrics.alignment_max_residual` (additive; `types.ts`; the panel shows
it beside the RMSE). Boundary test: a visible cloud bisected to normalized RMSE
0.049 → `require_alignment` passes (and its max residual is *above* 0.05 — the point of
reporting it); 0.051 → 409 "rebuild". Enforcement, not suitability.

**D2 — drilling deeper.** Behaviour unchanged (`nodeIsOnPath`). Documented in
frontend/README ("A movement survives drilling deeper, not stepping sideways") and as a
DECISION block under "User interaction → 8. Cancel" in the movement spec (commit 0).

**D3 — `feature_centroid_distance_after`.** `null` for point targets in both the PCA and the
UMAP builder (comment explains the (1−α)‖Δ‖ tautology); kept for cluster targets. Panel
hides the tile when null. Asserted in the PCA and UMAP free-point tests; cluster tests keep
`< 1e-9`.

---

## 4. Deviations from the plan (each justified)

1. **H2 test and H3 live in commit 3, not commit 2.** Both exercise `backend/movement.py`
   (`compute_movement`, `_require_compatible_run`), which only exists from commit 3; the
   plan's own escape hatch ("the foundations/plumbing subset that exists then — split the
   file") was used: commit 2's `test_movement.py` carries the run-identity test and a new
   effective-config-recording test that need only `serialize.py` / `app.py`.
2. **All frontend/README movement hunks travel with commit 4**, not commit 5, because the
   plan puts the D2 text in commit 4 and a three-way split of that file (P / D2 / rest) bought
   nothing. Commit 5 is README.md, backend/README.md, src/README.md.
3. **`analysis_iris.json` regenerated together with `movement_iris.json`.** Not listed in H8.
   The tracked analysis fixture predated the current tree shape (root children 100/17/33 —
   a 17-point cluster is impossible at min-cluster-size 25 — and a `kde` field the serializer
   no longer emits), so `root/1` was a *leaf* there while the plan's "PCA point target on
   root/1" needs the live 100-point internal node. The invariant MINOR-10 asks for is only
   meaningful against the embedding the ghosts were computed from, so both come from one
   capture and the script refuses to write if they disagree.
4. **The negative control for H2 is asserted positively** (the helper returns 15 under the
   bug) rather than via `pytest.raises` on an AssertionError, so the file stays runnable
   without pytest. Equivalent discriminating power; confirmed by running it.
5. **pyproject/uv.lock in commit 0** (pytest dev dep) rather than a separate commit — it is
   tooling, and commit 0 is the tooling/plans commit.

No hunk crossed tracks; no serialized node field changed; no regression number or clause
order changed; the H2 negative control fails as required. None of the §6 stop conditions
fired.

---

## 5. Open items after S3 (as reported then)

- S3 browser pass — done by Hannes (06.09). Outcome: panel to use the full width (→ S4),
  Apply wanted (→ S5/S6), Numba workqueue abort seen (→ §7).
- `.cache/hilde_runs/22d6…json.gz` (hosting-mode wine UMAP run from this morning, schema 2)
  was left in place: the payload shape did not change in S1 and the entry is already
  schema 2, so it is not stale — but the project convention says to clear the cache after
  backend changes; your call.
- `backend/requirements.txt` still pins the 2026-08-14 freeze (no runtime dep changed).
- `scratch/track-p-on-final-thesis` (local branch) — delete when no longer useful.
- S4–S7 per the plan: panel compaction (§2), Apply backend (§3), Apply frontend, S6½
  checkpoint, light review + docs.

---

## 6. S4–S6: commits 6–9 (each independently green in a worktree, gate level 3)

| # | Hash | Subject | Result |
|---|---|---|---|
| 6 | `c4f4358` | feat(movement): compact panel, full width under the plot and characteristics | GREEN (65 pytest) |
| 7 | `36d72a2` | feat(counterfactual): Apply — edits on a copy of the dataset, rebuilt on a new key | GREEN (74 pytest, `test_counterfactual` 9 cases) |
| 8 | `ba21027` | feat(counterfactual): Apply button, session stack, banner with Undo/Reset/Export | GREEN (74 pytest, build clean) |
| 9 | `0f2c994` | fix(umap): serialize every Numba-parallel call — the workqueue layer aborts on concurrency | GREEN (75 pytest) |
| 10 | `728fe09` | fix(predicate): band tooltips flip at the right edge; say what "matched" counts | GREEN (75 pytest, build clean) |

Gate at commit 9:
```
PASS  ruff format --check
INFO  ruff check: 33 findings (33 pre-existing expected)
PASS  test_serialize · test_targets · test_predicate · npm run build
PASS  test_movement · test_counterfactual
PASS  pytest: 75 passed, 31 warnings in 23.25s
GATE GREEN for 0f2c994 level 3
```

**HTTP smoke** (scratch uvicorn, stdlib client, before commit 8), PCA and UMAP: analysis →
movement → apply at strength 0 is a 400 → apply at 0.6 → the same apply again yields the
same `cf_id` (idempotent) → `/api/counterfactual/{id}` chain → `rows.csv` has exactly
`n_rows_changed` rows → `/columns` on the cf key → cf key absent from `/api/datasets` →
`/api/analysis` on the cf key rebuilds (`cached: false`, new `analysis_id`) → predicate,
rows and movement on the cf run → `/api/analysis` on the base is an inline cache hit
(`cached: true`, what Undo relies on) → unknown cf id is a 404 "expired or unknown". The
UMAP apply went through the job path (cold artifact). Both methods produced the same
`cf_id` for the cluster target, correctly: Δ = μ_t − μ_s does not depend on the reducer.

### D4/D5/D6 as implemented

**D6 — panel (S4, commit 6).** `MovementStartRow` is the only thing left in the side column
(one button; disabled with a one-line tooltip when nothing is selected or the run is
t-SNE/MDS, naming the method). A running movement renders `ClusterMovementPanel` in a new
`.layer__movement` row of the layer card under `.layer__cols`: grid
`minmax(220px,1fr) minmax(260px,1.2fr) minmax(320px,1.6fr)`, one column under 900 px.
Column 1: `C{src} → destination`, strength slider with a `rec.` tick, `[Apply] [Cancel]
[CSV] [JSON]`. Column 2: stat tiles at the server's strength — projected before → after,
feature-space before → after (cluster targets only, D3), alignment RMSE · worst point,
out-of-range %, support ratio (UMAP point targets), `sampled` badge, the sparkline. Column
3: the feature table sorted by |standardized change|, mutable rows only, held-fixed
columns as a footer line. Warnings are one-line chips below the grid (min-norm sentence in
the short form keyed on the backend's norm space, non-monotone trajectory, out-of-range >
5 %, support > 2, sampled, and every other backend warning mapped to a short form), each
with the full sentence in `title`. FROZEN_NOTE, the units, and every tooltip sentence live
in a collapsed `<details>About this preview</details>`. The prompt stays in the chart's
status line above the plot. Binding-amendment check: min-norm sentence displayed (chip +
tooltip + About), non-monotone caveat (chip), `preview_sampled` badge, out-of-range
warning, MDS/t-SNE named — all present.

**D4 — Apply (S5, commit 7; S6, commit 8).** Backend as §3 of the plan:
`backend/counterfactual.py` (edit history + frame LRU of 8 + `{base}@cf:{id}` keys +
replay + changed-rows export), `backend/counterfactual_apply.py` (server recomputes the
movement at the requested strength through the movement pipeline, then registers the
edit), `backend/datasets.py::load` resolves cf keys for every endpoint, routes
`POST /api/counterfactual/apply` (job path reused from `movement_jobs` for a cold UMAP
node), `GET …/jobs/{id}`, `GET …/{cf_id}`, `GET …/{cf_id}/rows.csv`. Rows changed = all
members of the source child by POSITION (`iloc`), `row_ids` recorded beside them; mutable
features only; identity `sha256(base | parent | canonical edit)[:12]` with deltas rounded
to 12 significant digits. `run_cache.store` skipped for cf keys. Strength ≤ 0 → 400.
Frontend: `useCounterfactual` (stack, `activeDatasetKey`, `idle/applying/rebuilding/error`,
generation + base + stack-head stale guard), App passes `activeDatasetKey` as `dataset` to
everything and keys the layer stack on it (a switch remounts it, so the movement machine
restarts idle in the same render); Apply enablement is exactly `movement.settled`
(phase ready, no pending timer, no in-flight request, `applied_strength === strength`,
response id = run id) and no apply/rebuild out — otherwise disabled with the reason as
tooltip; click → `applying` → `rebuilding` (current run stays on screen with a banner) →
one update: stack push, key swap, run replaced, path to root; any failure → `error` banner,
stack and run unchanged, button re-enabled.

**D5 — stacking.** Banner above the layers whenever the stack is non-empty:
`Counterfactual data · k edit(s) applied · [C0 → C1 @ 60%] chips (tooltip: node, rows,
features) · "Hierarchy rebuilt on counterfactual data — cluster numbering does not
correspond to the previous run." · [Undo] [Reset] [Export rows]`. Undo/Reset rebuild on
the parent/base key (tree-cache hit → inline) and go to root. Dataset selector shows the
base with a `cf` badge and is locked while an apply/rebuild is out; changing the dataset
resets the stack. The rail's Build rebuilds the counterfactual data (the edits are the data
now). Session state is client-side and gone on reload; the server keeps the history 24 h.

### Deviations in S4–S6 (each justified)

1. **The rail's "Build & Apply" on a session builds the counterfactual key**, not the base.
   The plan did not say; the edits are the data of the session, and a rebuild with other
   knobs on the original data would silently drop them while the badge still says `cf`.
2. **`edits` in the apply response is the whole chain**, not only the new edit, so the
   banner renders from the server's chain (the plan's "rendered from `/api/counterfactual/
   {cf_id}`, not from client memory") without a second request; the chain endpoint exists
   as specified for anyone else.
3. **The Numba fix (commit 9) was not in the plan.** Reported by Hannes at S3; it is
   process-killing and the Apply rebuild makes the race more likely, so it went in before
   the S6½ pass rather than being deferred. See §7.
4. Commit 5's docs said "there is no Apply"; commit 8 rewrote those sentences. Reading the
   history linearly, the statement was true at commit 5.

None of the §6 stop conditions of the plan fired: no wide-dataset memory issue was hit
(Iris only in this environment), no rebuild exceeded a timeout, no serialized node field
changed.

---

## 7. The Numba abort (commit 9)

Diagnosis on this machine: `numba 0.67.0`, threading layer `workqueue`; `tbbpool` absent
(the `tbb` wheel exists for manylinux x86_64 and win_amd64 only); `omppool` present but
fails to load (`@rpath/libomp.dylib` not found). The workqueue layer aborts the process
when two threads enter a parallel region at once. Numba parallel regions in this process:
UMAP (`umap_.py` 151, 359, 1044; `layouts.py` 227) and pynndescent; ZADU's CADI is plain
`@njit` and does not touch the threading layer; HDBSCAN is Cython.

Fix: `NUMBA_PARALLEL_LOCK` (re-entrant) in `src/analysis/dim_reducer.py` around the UMAP
fit, and around `transform` / `inverse_transform` in `backend/movement.py`. UMAP work
queues instead of overlapping (a slider re-request issued during a UMAP build waits for
it); PCA is unaffected. Test: four UMAP movements at once through a reducer proxy that
counts overlapping calls (zero), and two cold refits in parallel that both complete. Not
reproducible as a unit test in the negative direction — the failure is an `abort()`.

---

## 8. For the S6½ browser pass (Hannes)

- Compact panel: full width under the plot and characteristics; three columns; under 900 px
  one column; tiles' `(i)` tooltips; chips' tooltips; the About block lists every sentence.
- Idle: only the one button in the side column; t-SNE/MDS: same button disabled with the
  method named.
- Apply on PCA (inline) and on UMAP (job path — the "Applying" banner, then "Rebuilding").
- Apply disabled while a UMAP strength refinement is pending (drag the slider, click within
  350 ms) — tooltip "waiting for the preview at this strength".
- Stacked edits: chips in the banner; Undo; Reset; Export rows (CSV download).
- A failed rebuild (kill the backend mid-job) leaves the previous run on screen and the
  button re-enabled; the error banner names the failure.
- Movement on a counterfactual run (the `cf` badge showing).
- Navigation after Apply is the root; the banner's "numbering does not correspond" line.
- Changing the dataset resets the session; the rail's Build on a session rebuilds the
  counterfactual data.

### Commit 10 — Hannes's questions on the ΔF1 gutter (06.09, after S6½)

*What is "matched"?* The gutter's second number is `predicate_n_matched`: how many
background points the greedy conjunction (every clause up to and including this one)
still matches after the clause was added. The empty predicate matches all N; each clause
narrows it; the selection is a subset of the matched set, so precision = selected ÷
matched, and at RCM 1.0 (recall 1) the count alone determines the F1. Now labelled
"ΔF1 when added · points matched" with the definition as the header's tooltip, and the
row tooltip reads "Matches n of N background points" (N = `summary.n_background`, passed
as an optional prop). *Tooltip cut off on the right:* the predicate and target band charts
opened the tooltip only to the right of the cursor inside the clipped side column; both
now flip to the left at the edge, as `CharacteristicsBar` already did.

## 9. S6½ checkpoint — PASSED (Hannes, 09.09)

Browser pass done on the new layout and Apply. Verdict: the features look right, release
them. No defect reported, no rework requested. S7 released to run.

---

## 10. S7 — light review and the last docs

**Docs.** `README.md`, `backend/README.md` and `frontend/README.md` already carried the
counterfactual semantics, the per-copy memory note and the "hierarchy may change" line
from commits 8–10. `src/README.md` had nothing: it now states, under `datasets.py`, that
counterfactual data reaches the calc layer as an ordinary frame, that the copy is made in
`backend/counterfactual.py` and resolved from its key by `backend/datasets.py::load`, and
that a counterfactual build is a normal build — so the rebuilt hierarchy need not
correspond to the run the edit was previewed on, and the moved cluster is not guaranteed
to merge with its destination.

**Light review (R1 scope).** Four areas, one real defect.

| Area | Verdict |
|---|---|
| Parity backend ↔ `types.ts` ↔ components | **clean.** All four endpoints match field for field: the 12 keys of `CounterfactualEdit`, the apply response, the job envelope (its done arm is the apply dict carrying `"status": "ok"`, which is what `api.ts` narrows on) and the chain response. `_edit_target` produces exactly `MovementTarget`. `tsc --noEmit` clean. |
| Mutation safety of the base frame | **clean.** The only write path is `apply_edit`, reachable only through `replay`, which always starts from `base.copy(deep=True)`. The write is one `frame.iloc[positions, col] = ndarray` on a parentless copy, no chained assignment. Every other consumer of the frame is read-only. pandas 3.0.5 with copy-on-write makes write-through structurally impossible on top of that. |
| LRU bound | **clean.** `_FRAME_CACHE_MAX = 8`; the single insert site runs under the lock and does `move_to_end` before `popitem(last=False)`, so an entry can never evict itself and `len <= 8` holds at every lock release. History lives in a separate dict that LRU never touches, so replay after eviction is exact — the edit carries its own `deltas_raw` and edits change values only, never rows, order or index. |
| State hygiene of the cf stack | **one defect, fixed below.** The stale check (generation + base + stack head), the dataset-change reset, the three independent double-submit guards and both error branches are all clean. |

**The defect — the Build button was not gated on `cfBusy`** (`frontend/src/App.tsx`). The
dataset selector, Undo and Reset all carry `cfBusy`; Build did not, and it is the one
control that can move App state while an apply or rebuild is out. `useCounterfactual`'s
`stale()` cannot see it, because Build changes neither the base dataset, nor the
generation, nor the stack head. Two interleavings:

- Build during `applying` nulls `analysis`, so the pending `rebuild` reads
  `analysisRef.current?.meta` as null and rejects with "there is no run on screen to
  rebuild" — an error message that misdescribes what happened, with the edit already
  registered on the server.
- If the counterfactual rebuild resolves first, `onSwitched` sets the cf run and then
  `build()`'s own awaited `setAnalysis` overwrites it with a run built on the key it
  captured before the swap. `analysis.meta.dataset !== cf.activeDatasetKey`, so
  `shownAnalysis` is null: an empty canvas under a "1 edit applied" banner.

Neither ever displays or sends a mismatched tree/dataset pair — `shownAnalysis` is the
guard that holds. Both are confusing dead states, both recoverable by pressing Build
again, and the apply is content-addressed so no duplicate snapshot is created. Fixed by
gating the button exactly as its three neighbours are gated, with the reason as its
`title`.

Two non-defects worth recording: `GET /api/counterfactual/{cf_id}` and its client
`fetchCounterfactualChain` have no consumer — the banner renders from the apply
response's `edits`, which is the identical chain (deviation 2 of §6), so nothing
diverges — and `n_rows_changed` is sent, declared and unused. Two theoretical-only
observations, neither actionable now: `resolve_frame` hands back the cached frame object
itself rather than a copy, which is safe only while every consumer stays read-only; and a
TTL sweep landing between the chain lookup and the frame store would strand one cache
slot until LRU pressure reclaims it.

---

## 11. Open items

- **S7 is done** (§10). The plan is fully executed; nothing is pushed and `main` is
  untouched, per Hannes's decision of 09.09 — the release step is his to take.
- `resolve_frame` returning the shared frame object is safe only while every consumer is
  read-only. Worth a `.copy()` the first time one is not.
- `.cache/hilde_runs/22d6…json.gz` left in place (see §5 of the S3 report).
- `scratch/track-p-on-final-thesis` (local branch) — delete when no longer useful.
- `frontend/src/fixtures/movement_iris.json` predates commit 6's compaction only in the
  sense that no fixture is imported anywhere; the wire shape is unchanged.

# Review report: `feat/predicate-gain-and-movement` working tree

Date: 2026-09-06 · Reviewer: independent (R1 role, read-only) · Base: `HEAD` = `aa1ee92` = tag `final-thesis`, 0 commits on the branch, 21 modified tracked files (+1378/−86) and 8 untracked paths. Scratch scripts and captured outputs live under `/tmp/review/` (referenced below as `/tmp/review/<name>`). No tracked file was modified, nothing was committed, stashed or formatted; `git status --porcelain` has the same 29 lines before and after this review.

---

## 1. GO / NO-GO

**NO-GO for committing the working tree as-is in one commit; GO for the same content once split as in §7.** Every verification gate that can run in this environment passes, the E2 tie-break regression is byte-identical before and after, and my own re-derivations of the PCA movement math and the predicate bookkeeping agree with the implementation to 1e-15 — no correctness defect was found — but one commit of this tree forfeits the Track P cherry-pickability that orchestration §1 makes a requirement, and the alignment "canary" does not test the guarantee its name and docstring advertise (finding MAJOR-1).

---

## 2. Findings (ranked)

No BLOCKER.

### MAJOR

**MAJOR-1 · `backend/tests/test_movement.py:389-439` — the alignment canary cannot detect a refit that reads the raw request config instead of `effective_config`.**
Claim: `_canary`'s docstring says it "fails the moment a refit reads a configuration other than the one the build used", and the spec calls it "the regression canary for the effective-config amendment". On the dataset it uses (Iris, 150 rows) the in-place clamp in `src/analysis/analysis_routine.py:146-147` sets `umap_n_neighbors = min(15, 149) = 15`, i.e. changes nothing, so the raw merged request config and the post-build config have identical refit-relevant knobs.
Failure scenario: replace `refit_config(effective_config)` at `backend/movement.py:471` with a merge of `MovementRequestData.config`; both canaries (`test_pca_refit_aligns…`, `test_umap_refit_aligns…`) stay green at RMSE ≈ 1e-16, and `test_payload_predating_effective_config_is_409` (274-287) still passes because it only checks the 409 path, not what the refit reads. The regression would surface only on a dataset with fewer than 16 rows, where the clamp fires — which is also why the practical exposure of the amendment itself is small.
How verified: `/tmp/review/check_canary_sensitivity.py` builds Iris with PCA and UMAP and diffs `effective_config(raw)` against `effective_config(built)`: `keys that differ between raw and effective: NONE` for both. Separately confirmed that `start_evaluation` does hand the same dict object to `compute_analysis_tree` (spy: `same object: True`), so the recorded `effective_config` in `app.py:222` is genuinely the mutated one. What the canary *does* prove (same rows, same root scaler, same `init="pca"`, same seed) is real and valuable; the label is what is wrong. A canary that actually tests the amendment needs a fixture where the clamp fires (e.g. a 12-row frame with `umap_n_neighbors=15`) and an assertion that the refit from the raw config is *rejected*.

### MINOR

**MINOR-1 · `backend/movement.py:135` — production alignment gate `ALIGNMENT_RMSE_MAX = 0.05` is unrelated to the spec's canary tolerances.**
Claim: the spec fixes 1e-6 (PCA) / 1e-3 (UMAP) for the same-process canary and says a large RMSE is "never a tolerance to widen"; the runtime gate is 50× looser than the UMAP canary and 5·10⁴× looser than the PCA one. Failure scenario: a refit with normalized RMSE 0.04 (ghosts displaced by 4 % of the cloud's RMS radius) is drawn, not refused with 409. This is not a widening-to-pass (measured same-process RMSE is 3.5e-16 / 6.9e-16) and it is documented as a "tripwire" at 128-135, but the number is an unstated product decision that belongs on the stop-and-ask list. How verified: read `require_alignment` (631-660); no code path caps or swallows `alignment.rmse` (`_finite` at 1084/1500 only maps non-finite values, which `require_alignment` has already rejected).

**MINOR-2 · `backend/movement.py:264-283` — 409 checks are ordered so a legacy payload gets the wrong explanation.**
Claim: the `analysis_id` comparison (264-270) runs before the schema/`effective_config` check (275-283); a pre-feature payload has no `meta.analysis_id`, so the message is "requested against a different analysis run" rather than "built before movement support was added… Rebuild". Failure scenario: hosting mode with an old on-disk entry and any client that computes the id itself. How verified: live on port 8124 with `HILDE_HOSTING=1` and a scratch cache holding a payload stripped to `{dataset, feature_cols, config, n_total}`: `/api/analysis` served it (`cached: true`), `/api/movement` answered HTTP 409 with the "different analysis run" text for both a correct and a bogus id (§8 appendix). Status code is right; only the diagnosis is misleading. The shipped frontend never reaches this branch (`frontend/src/movement.ts:388-400` refuses client-side when `meta.analysis_id` is absent).

**MINOR-3 · `frontend/src/components/LayerSide.tsx:146,152`, `ExplorationPanel.tsx:597,604` — baseline label is the literal string "no clauses (all N points)".**
Claim: the amendment wants the baseline read as a function of selection and background size; the `title` tooltip reproduces the spec's placeholder letter "N" instead of the count, and `PredicateSummary` carries no background size to render (`backend/predicate.py:92-109` has `n_selected` but not N). Failure scenario: on a 50-point node with 16 selected the user sees "F1 0.48 → 1.00" and a tooltip saying "all N points" — the number the label exists to explain is still missing. How verified: grep of the three READMEs and both components; `frontend/README.md:149` documents the literal label.

**MINOR-4 · `backend/predicate.py:101-105` / `frontend/src/types.ts:146` — `trimmed_n_features_used` is produced, typed, and never read.**
How verified: grep of `frontend/src` finds no consumer outside `types.ts`; only `trimmed_predicate_f1` and `trimmed_baseline_f1` are rendered.

**MINOR-5 · `backend/tests/test_predicate.py:6`, `backend/tests/test_movement.py:6` — the documented pytest invocation fails from a fresh clone.**
Claim: `PYTHONPATH=. pytest backend/tests/…` is advertised, but pytest is in neither `.venv`, `pyproject.toml` nor `uv.lock` (gate 5 output below). Orchestration §1's "runnable under pytest" holds structurally — 60 tests pass through an ephemeral `uv run --with pytest` overlay — but the command as written does not run.

**MINOR-6 · `backend/movement.py:450-519` — `_artifact_cache` is shared between request threads and `movement_jobs` worker threads without a lock.**
Failure scenario: a job thread's `popitem(last=False)` (519) racing a request thread's `move_to_end` (505) on the same key raises `KeyError` inside the request → HTTP 500. Same unlocked pattern as the pre-existing `_tree_cache` (`backend/app.py:63-77`), but new exposure because `movement_jobs.py:110` now writes it from a thread. Reasoned from the code; not reproduced.

**MINOR-7 · `backend/movement_jobs.py:71-88` vs `backend/movement.py:1515-1526` — two different first UMAP requests on one cold node refit the same reducer twice.**
The job key includes target and strength; `needs_background` keys on the artifact. Both jobs call `artifact_for` and the second overwrites the first. Deterministic, so not incorrect; wasteful on a large node. Reasoned from the code.

**MINOR-8 · `backend/movement.py:773,1272` with `1014-1019,1450-1456` — `feature_centroid_distance_after` is content-free for point targets.**
`target_input := source_mean_input + delta`, so `feature_centroid_distance_after = (1−α)·‖Δ‖` by construction and is exactly 0 at the recommended strength whether or not the click was reached; the panel shows it as "feature-space after" next to the genuine projected distances. How verified: smoke UMAP point target reports `feature_centroid_distance_after: 0.0` while `projected_distance_after: 0.522`. Cluster targets are fine (`target_input = μ_t`).

**MINOR-9 · `README.md:115-116`, `backend/README.md:245-246` — runnable test blocks still list two scripts; the prose says four.**
`README.md:364-366` and `backend/README.md:197` name `test_predicate.py` / `test_movement.py`, but no README contains the two new commands. CLAUDE.md requires every README command to have been run; conversely the new gates are not documented anywhere a user would run them.

**MINOR-10 · `frontend/src/fixtures/movement_iris.json` — contract-valid, but not a shared translation.**
Type-checks against `MovementResponse` (§5 appendix), yet per-point displacement against the `analysis_iris.json` root embedding has std (0.57, 0.92) while `visible_displacement` is (3.55, −0.62); moved centroid (1.2018, −0.32) vs target (1.2, −0.32). A reference sample that violates the feature's central invariant (every ghost = source + d_vis, which the live backend satisfies to 1.2e-15). Documented as hand-written; no importer.

**MINOR-11 · `frontend/src/movement.ts:496-505` — navigation clear-condition is narrower than the spec's list.**
The spec says cancel on "navigating to another node"; the hook cancels only when the movement's node leaves the path (`nodeIsOnPath`), deliberately keeping the movement while drilling *deeper*, because the parent projection stays on screen in the layer stack. Reasonable, documented in code, but a deviation to confirm with Hannes. Verified by reading; no browser.

**MINOR-12 · `.gitignore:17` — `!plans/*` re-includes every file type under `plans/`**, including `*.png` / `*.log` that the patterns above exclude. Harmless today.

---

## 3. Compliance matrix

### Track P (7 amendments)

| # | Amendment | Status | Evidence |
|---|---|---|---|
| P1 | per-step `n_matched` / precision / recall recorded next to the gain | ✓ | `src/analysis/predicate_generator.py:193-207` (recomputed from the accepted mask); `/tmp/review/check_track_p.py` recomputes all three from clause masks on 20 live Iris cases — all equal to 1e-9 |
| P2 | RCM 0.9 trajectory in the tooltip core-run block; both trajectories in the summary | ✓ | tooltip core block `frontend/src/charts/PredicateBands.tsx:294-306` (step, gain, before→after); summaries `LayerSide.tsx:146-157`, `ExplorationPanel.tsx:597-610`; backend supplies `trimmed_predicate_f1`/`trimmed_baseline_f1` at `backend/predicate.py:95-100`; order comment at `PredicateBands.tsx:49-52` |
| P3 | `predicate_baseline_f1` on every row, no first-clause special case | ✓ | stamped in the final loop `predicate_generator.py:212-214`, no-labels path 153; `backend/predicate.py:88` reads row 0 as "any row"; `test_predicate.py:126` asserts identical on every row; my script asserts `2|S|/(N+|S|)` on 20 cases |
| P4 | explicit tie-break `(f1, −index)` | ✓ | `predicate_generator.py:190`; regression identical (§4) |
| P5 | 1e-6 tolerance unchanged; `+0.000` allowed | ✓ | `predicate_generator.py:191` is not in the diff; `PredicateBands.tsx:40-42` `gain()` = `toFixed(3)` with the match count beside it (188-203), header "ΔF1 when added · matched" (120-131) |
| P6 | baseline labelled "no clauses (all N points)" wherever displayed | partial | present as `title` tooltips at `LayerSide.tsx:146,152`, `ExplorationPanel.tsx:597,604` — literal "N", count never rendered (MINOR-3) |
| P7 | step-0 invariant test present; tautological transition-continuity test gone | ✓ | `test_predicate.py:134-146` compares `clause_f1` (independent loop, 160-174 of the generator) with `gain₀ + baseline`; no `after_k ≈ before_{k+1}` assertion anywhere in the file |

Also verified on Track P: `src_research/predicate_stability.py` (the only other caller) reads rows by `in_predicate`/`feature` only (`:271-390`), so the additive keys cannot break it (read, not run — see §6).

### Track M (11 amendments)

| # | Amendment | Status | Evidence |
|---|---|---|---|
| M1 | refits read only `effective_config`; missing → 409; schema version bumped | ✓ (test caveat MAJOR-1) | only fit path: `backend/movement.py:471-481` `refit_config(effective_config)` → `fit_node_projection`; request `config` used solely for the id at 264; 409s at 264-283; `SCHEMA_VERSION = 2` `backend/serialize.py:144`; written at `backend/app.py:221-223` after `start_evaluation` returned (spy confirms same dict object mutated); client sends `meta.config`, not the rail (`movement.ts:409-418`); live: cache-miss config → 409, wrong id → 409, legacy disk payload → 409 |
| M2 | reducers built exclusively via `fit_dimensionality_reducer` / `_embed_original` | ✓ | grep: `PCA(`/`umap.UMAP(`/`TSNE(`/`MDS(` occur only in `src/analysis/dim_reducer.py:82,98,108,114`; `_embed_original` now delegates to `fit_node_projection` (`analysis_routine.py:79-117`), which movement imports (`movement.py:113`) |
| M3 | same-process alignment canary at 1e-6 / 1e-3; large RMSE surfaced, never tolerated | ✓ / MAJOR-1, MINOR-1 | tolerances `test_movement.py:48-49`; measured max 3.549e-16 (PCA), 6.887e-16 (UMAP) over 5 nodes each; `require_alignment` 631-660 raises 409, no cap; runtime gate 0.05 (135) |
| M4 | every 2D response quantity in visible coordinates; PCA slider uses returned visible displacement | ✓ | click → fitted before solving (958, 1335); centroid/target/displacement/preview mapped `to_visible` (966-985, 1379-1396); `previewAtStrength` uses `visible_displacement` only (`movement.ts:222-239`); my reflected-frame check: det(R) = −1, RMSE 6.6e-16, target/ghosts/displacement all come back in *my* frame; smoke: ghosts − serialized source == `visible_displacement` to 1.2e-15 |
| M5 | min-norm sentence in the panel; norm-space caveat in docs | ✓ | emitted `movement.py:702-722` (standardized vs raw wording keyed on `normalize`); panel `ClusterMovementPanel.tsx:230-234, 385-389` (backend sentence, fallback 72-76); docs `README.md:220-222`, `backend/README.md:110-115`; smoke shows the sentence for the PCA point target and not for the cluster target |
| M6 | `strength_trajectory` returned (11 points), plotted, non-monotone warning | ✓ | `STRENGTH_GRID = linspace(0,1,11)` (1096), returned 1492-1495; sparkline 147-198, mounted 461-464; `isMonotone` 104-117 drives the caveat 465-471; smoke: 11 entries, both live Iris trajectories monotone; unit test prints a root trajectory that is monotone but jumps 4.4 units in one step |
| M7 | `out_of_range_fraction` = fraction of moved points with ≥1 mutable feature outside the FULL dataset's raw range; warn > 0.05 | ✓ | `_out_of_range_fraction` 885-905 (`df[feature_cols].min()/max()`, mutable columns, `outside.any(axis=1).mean()`, at the applied strength); threshold 674; warnings 1023-1028 / 1440-1445; panel 56, 486-492; my independent recompute on the toy set equals the response (0.1000) |
| M8 | state lives in `useMovement` (`frontend/src/movement.ts`); no `Navigation` struct | ✓ | hook 324-525; `App.tsx:388` mounts it; `Navigation` at `App.tsx:355` is the pre-existing component, holds no movement state |
| M9 | validation: child-count minimums, non-null parent embedding, MDS named in gate AND UI | ✓ | 375-384 (≥1 / ≥2), 309-314 (`embedding_original` null → 400), 126-127 + 328-335 (message names t-SNE and MDS); panel 274-286 names both; tests 335-352, 355-367; live t-SNE run → 400 naming both |
| M10 | substantive PCA tests (Δ ∈ rowspace(P), immutable zeros, centroid reaches target, raw-delta unscaling); pairwise check not the only one | ✓ | `test_movement.py:615-636` (row space + null-space detour grows the norm), 808-830 (label column exactly 0.0), 638-702 (free point and cluster, residual < 1e-9 through the frozen reducer), 751-789 (raw = standardized × root scale); pairwise smoke kept and labelled as such (597-612) |
| M11 | `MAX_MOVEMENT_PREVIEW_POINTS=5000`, `MAX_STRENGTH_SEARCH_POINTS=1000`, deterministic; `preview_sampled` surfaced | ✓ | 670, 1104; `_preview_indices` / `_strength_search_indices` are `linspace` strides, no RNG (863-875, 1116-1125); tests 865-878, 1118-1128, 1147-1167 (cap lowered to 12); panel badge 327-331 and warning 507-513 |

---

## 4. Verbatim gate outputs (§1) and regression numbers (§2)

Environment: Python 3.13.11 (`.venv`), numpy 2.4.6, scikit-learn 1.9.0, umap-learn 0.5.12, ruff 0.16.3, vite 5.4.21.

### Gate 1
```
$ PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
OK — predicate step bookkeeping passed
exit=0
```

### Gate 2
```
$ PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
OK — test_alignment_identity_is_identity
OK — test_alignment_recovers_rotation
OK — test_alignment_allows_reflection
OK — test_alignment_recovers_translation_and_scale
OK — test_alignment_ignores_non_finite_serialized_points
OK — test_unalignable_pair_is_rejected
OK — test_alignment_shape_mismatch_is_rejected

╭── Starting build with selected config ──╮
│             Dataset  Wine quality (Low) │
│            Features  4 selected         │
│           DR method  PCA                │
│           Normalize  True               │
│ Hierarchical layers  2                  │
│      Cluster method  HDBSCAN            │
│    Min cluster size  25                 │
│         Min samples  5                  │
│    hclust UMAP dims  2                  │
╰─────────────────────────────────────────╯
▶ Computing analysis tree
  · Pre-clustering UMAP: 4D → 2D  (150 points)
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (150 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x4 -> 2D
  · Clustering: HDBSCAN  (100 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Dim reduction: PCA  64x4 -> 2D
  · Dim reduction: PCA  31x4 -> 2D
  · Dim reduction: PCA  100x4 -> 2D
  · Dim reduction: PCA  150x4 -> 2D
▶ Evaluating embedding quality (ZADU)
✓ Build complete   nodes=2  leaves=3
OK — test_find_node_resolves_serialized_path_ids
OK — test_valid_requests_are_accepted
OK — test_analysis_identity_mismatches_are_409
OK — test_payload_predating_effective_config_is_409
OK — test_bad_requests_are_400
OK — test_unsupported_reducers_are_400_and_name_tsne_and_mds
OK — test_parent_without_projection_is_400
OK — test_analysis_id_matches_the_tree_cache_key_shape
  · Dim reduction: PCA  150x4 -> 2D
  · Dim reduction: PCA  50x4 -> 2D
  · Dim reduction: PCA  100x4 -> 2D
  · Dim reduction: PCA  64x4 -> 2D
  · Dim reduction: PCA  31x4 -> 2D
   PCA canary RMSE: max=3.549e-16  {'root': 2.5535646704060835e-16, 'root/0': 2.7621225207980145e-16, 'root/1': 2.7751205944859956e-16, 'root/1/0': 2.129903712664479e-16, 'root/1/1': 3.548598344588104e-16}
OK — test_pca_refit_aligns_with_the_serialized_embedding

╭── Starting build with selected config ──╮
│             Dataset  Wine quality (Low) │
│            Features  4 selected         │
│           DR method  UMAP               │
│           Normalize  True               │
│ Hierarchical layers  2                  │
│      Cluster method  HDBSCAN            │
│    Min cluster size  25                 │
│         Min samples  5                  │
│    hclust UMAP dims  2                  │
╰─────────────────────────────────────────╯
▶ Computing analysis tree
  · Pre-clustering UMAP: 4D → 2D  (150 points)
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (150 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: UMAP  50x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (100 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Dim reduction: UMAP  64x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  31x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  100x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
▶ Evaluating embedding quality (ZADU)
✓ Build complete   nodes=2  leaves=3
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  50x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  100x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  64x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: UMAP  31x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
   UMAP canary RMSE: max=6.887e-16  {'root': 6.887146715466416e-16, 'root/0': 3.7088192736540773e-16, 'root/1': 5.033413181851614e-16, 'root/1/0': 5.563100649027903e-16, 'root/1/1': 3.940045763224794e-19}
OK — test_umap_refit_aligns_with_the_serialized_embedding
  · Dim reduction: PCA  100x4 -> 2D
OK — test_refit_does_not_mutate_the_source_dataframe
  · Dim reduction: PCA  150x4 -> 2D
OK — test_pca_free_point_movement_is_finite_and_shared
OK — test_pca_movement_preserves_pairwise_distances
OK — test_pca_delta_is_the_minimum_norm_row_space_solution
   free-point centroid residual: 2.483e-16 (visible units)
OK — test_pca_free_point_reaches_the_requested_target
   cluster centroid residual:    1.196e-15 (visible units)
OK — test_pca_cluster_movement_reaches_the_target_centroid
OK — test_pca_strength_scales_the_displacement_exactly
OK — test_pca_raw_deltas_reverse_the_root_standardization
OK — test_out_of_range_fraction_counts_points_leaving_the_dataset_range

╭── Starting build with selected config ──╮
│             Dataset  Wine quality (Low) │
│            Features  3 selected         │
│           DR method  PCA                │
│           Normalize  True               │
│ Hierarchical layers  2                  │
│      Cluster method  HDBSCAN            │
│    Min cluster size  25                 │
│         Min samples  5                  │
│    hclust UMAP dims  2                  │
╰─────────────────────────────────────────╯
▶ Computing analysis tree
  · Pre-clustering UMAP: 3D → 2D  (150 points)
  · Dim reduction: UMAP  150x3 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (150 points)
  · Dim reduction: MDS  3x2 -> 2D
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x3 -> 2D
  · Clustering: HDBSCAN  (52 points)
  · Dim reduction: PCA  52x3 -> 2D
  · Dim reduction: PCA  48x3 -> 2D
  · Dim reduction: PCA  150x3 -> 2D
▶ Evaluating embedding quality (ZADU)
✓ Build complete   nodes=1  leaves=3
  · Dim reduction: PCA  150x3 -> 2D
OK — test_pca_holds_label_features_at_exactly_zero

╭── Starting build with selected config ──╮
│             Dataset  Wine quality (Low) │
│            Features  3 selected         │
│           DR method  PCA                │
│           Normalize  True               │
│ Hierarchical layers  2                  │
│      Cluster method  HDBSCAN            │
│    Min cluster size  25                 │
│         Min samples  5                  │
│    hclust UMAP dims  2                  │
╰─────────────────────────────────────────╯
▶ Computing analysis tree
  · Pre-clustering UMAP: 3D → 2D  (150 points)
  · Dim reduction: UMAP  150x3 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (150 points)
  · Dim reduction: MDS  3x2 -> 2D
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x3 -> 2D
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x3 -> 2D
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x3 -> 2D
  · Dim reduction: PCA  150x3 -> 2D
▶ Evaluating embedding quality (ZADU)
✓ Build complete   nodes=1  leaves=3
  · Dim reduction: PCA  150x3 -> 2D
OK — test_pca_rank_deficient_components_return_the_closest_reachable_point
OK — test_preview_sampling_is_deterministic_and_bounded
OK — test_pca_movement_does_not_mutate_the_source_dataframe
OK — test_pca_movement_response_is_json_safe
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
   UMAP strength trajectory: [13.5271, 13.3757, 12.9804, 12.6755, 8.2604, 3.3918, 2.2252, 1.7976, 1.4559, 0.7181, 0.2392]
   monotone non-increasing:  True
OK — test_umap_free_point_movement_inverts_and_previews_finitely
OK — test_umap_cluster_target_uses_the_observed_centroid
OK — test_umap_inverse_failure_falls_back_to_neighbour_interpolation
OK — test_umap_recommended_strength_is_never_worse_than_standing_still
OK — test_umap_strength_search_is_deterministic
OK — test_umap_strength_search_sample_is_deterministic_and_bounded
OK — test_umap_unsupported_target_warns_instead_of_being_rejected
OK — test_umap_preview_sampling_is_flagged_and_deterministic
OK — test_umap_movement_does_not_mutate_the_source_dataframe
OK — test_umap_movement_response_is_json_safe
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Dim reduction: PCA  150x4 -> 2D
OK — test_only_an_uncached_umap_refit_needs_a_background_job
OK — test_movement_job_runs_polls_and_returns_the_response
OK — test_movement_job_reattaches_instead_of_starting_a_second_run
OK — test_movement_job_reports_a_failure_through_the_poll
OK — movement alignment + canary passed
exit=0
```

### Gates 3 and 4
```
$ PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize

╭── Starting build with selected config ──╮
│             Dataset  Wine quality (Low) │
│            Features  4 selected         │
│           DR method  PCA                │
│           Normalize  True               │
│ Hierarchical layers  2                  │
│      Cluster method  HDBSCAN            │
│    Min cluster size  25                 │
│         Min samples  5                  │
│    hclust UMAP dims  2                  │
╰─────────────────────────────────────────╯
▶ Computing analysis tree
  · Pre-clustering UMAP: 4D → 2D  (150 points)
  · Dim reduction: UMAP  150x4 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
  · Clustering: HDBSCAN  (150 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Clustering: HDBSCAN  (50 points)
  · Dim reduction: PCA  50x4 -> 2D
  · Clustering: HDBSCAN  (100 points)
  · Dim reduction: MDS  2x2 -> 2D
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/lib/python3.13/site-packages/sklearn/manifold/_mds.py:775: UserWarning: The provided input is a square matrix. Note that ``fit`` constructs a dissimilarity matrix from data and will treat rows as samples and columns as features. To use a pre-computed dissimilarity matrix, set ``metric='precomputed'``.
  warnings.warn(
  · Dim reduction: PCA  64x4 -> 2D
  · Dim reduction: PCA  31x4 -> 2D
  · Dim reduction: PCA  100x4 -> 2D
  · Dim reduction: PCA  150x4 -> 2D
▶ Evaluating embedding quality (ZADU)
✓ Build complete   nodes=2  leaves=3
OK — serialize round-trip passed
exit=0

$ PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
OK — target stats passed
exit=0
```

### Gate 5 (as written in the review prompt)
```
$ PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q
/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/.venv/bin/python: No module named pytest
exit=1

$ .venv/bin/python -m pip list 2>/dev/null | grep -i pytest ; grep -n pytest pyproject.toml uv.lock | head
(no output above = pytest is not in the venv, pyproject, or lock)
```

Supplementary, not a gate of the repo's own environment — the same tests through an ephemeral overlay, to establish that the pytest invocation style works once pytest exists (`-p no:cacheprovider` so no `.pytest_cache` is written; build/warning noise filtered):
```
$ PYTHONPATH=. uv run --with pytest python -m pytest backend/tests -q   (ephemeral overlay; pytest is NOT in the repo venv)
............................................................             [100%]
=============================== warnings summary ===============================
backend/tests/test_movement.py: 16 warnings
backend/tests/test_serialize.py: 1 warning
backend/tests/test_movement.py::test_find_node_resolves_serialized_path_ids
backend/tests/test_movement.py::test_find_node_resolves_serialized_path_ids
backend/tests/test_movement.py::test_umap_refit_aligns_with_the_serialized_embedding
backend/tests/test_movement.py::test_umap_refit_aligns_with_the_serialized_embedding
backend/tests/test_serialize.py::test_serialized_tree_is_json_safe_and_well_formed
backend/tests/test_serialize.py::test_serialized_tree_is_json_safe_and_well_formed
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
60 passed, 23 warnings in 18.39s
```

### Gate 6
```
$ (cd frontend && npm run build)

> shd-frontend@0.1.0 build
> tsc -b && vite build

vite v5.4.21 building for production...
transforming...
✓ 625 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                                              0.74 kB │ gzip:   0.45 kB
dist/assets/ibm-plex-mono-latin-400-normal-CvHOgSBP.woff    13.14 kB
dist/assets/ibm-plex-mono-latin-500-normal-CB9ihrfo.woff    13.16 kB
dist/assets/ibm-plex-mono-latin-400-normal-DMJ8VG8y.woff2   14.71 kB
dist/assets/ibm-plex-mono-latin-500-normal-DSY6xOcd.woff2   14.89 kB
dist/assets/ibm-plex-sans-latin-400-normal-CYLoc0-x.woff    22.10 kB
dist/assets/ibm-plex-sans-latin-400-normal-CDDApCn2.woff2   22.59 kB
dist/assets/ibm-plex-sans-latin-600-normal-Cu4Hd6ag.woff    23.88 kB
dist/assets/ibm-plex-sans-latin-500-normal-BgVn5rGT.woff    23.92 kB
dist/assets/ibm-plex-sans-latin-500-normal-6ng42L7E.woff2   24.18 kB
dist/assets/ibm-plex-sans-latin-600-normal-CuJfVYMP.woff2   24.25 kB
dist/assets/index-DokMPimx.css                              20.82 kB │ gzip:   4.53 kB
dist/assets/index-ClaTn3OI.js                              309.20 kB │ gzip: 100.39 kB
✓ built in 697ms
npm notice
npm notice New major version of npm available! 10.9.2 -> 12.0.2
npm notice Changelog: https://github.com/npm/cli/releases/tag/v12.0.2
npm notice To update run: npm install -g npm@12.0.2
npm notice
exit=0
```

### Gates 7 and 8
```
$ uv run ruff format --check .
58 files already formatted
exit=0

$ uv run ruff check .
BLE001 Do not catch blind exception: `Exception`
  --> backend/jobs.py:52:16
   |
50 |         try:
51 |             work()
52 |         except Exception as exc:  # the failure reaches the client through the poll
   |                ^^^^^^^^^
53 |             _finish(job, "error", str(exc))
54 |         else:
   |

PLC0206 Extracting value from dictionary without calling `.items()`
  --> scripts/checks/02_zadu_equivalence.py:50:9
   |
48 |         "mrre_missing": z[2]["mrre_missing"],
49 |     }
50 |     for m in ref:
   |         ^^^^^^^^
51 |         a, b = float(repo[m]), float(ref[m])
   |                                      ------
52 |         diff = abs(a - b)
53 |         worst = max(worst, diff)
   |
help: Use `for m, value in ref.items()` instead

BLE001 Do not catch blind exception: `Exception`
   --> src/analysis/analysis_routine.py:98:12
    |
 96 |             method=config["method"], X=X_orig, n_components=2, config=config
 97 |         )
 98 |     except Exception as exc:
    |            ^^^^^^^^^
 99 |         clog.warn(
100 |             f"Projection failed for a node of {n} points ({type(exc).__name__}: {exc}) — left unembedded"
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/evaluation/evaluate.py:111:12
    |
109 |         # which a root node of tens of thousands of points cannot afford.
110 |         scores.update(neighbor_scores(X, emb, k))  # type: ignore[typeddict-item]
111 |     except Exception as exc:
    |            ^^^^^^^^^
112 |         # `k` stays None so the node reads as unscored rather than advertising a
113 |         # neighbourhood size for five missing numbers. MemoryError lands here too —
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/evaluation/evaluate.py:149:12
    |
147 |         ).measure(emb[valid], label=lab)[0]
148 |         return float(result["Class Angular Distortion Index"])
149 |     except Exception:
    |            ^^^^^^^^^
150 |         return None
    |

F821 Undefined name `PCA`
 --> src/types.py:8:18
  |
6 | import pandas as pd
7 |
8 | type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"
  |                  ^^^

F821 Undefined name `t`
 --> src/types.py:8:26
  |
6 | import pandas as pd
7 |
8 | type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"
  |                          ^

F821 Undefined name `SNE`
 --> src/types.py:8:28
  |
6 | import pandas as pd
7 |
8 | type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"
  |                            ^^^

F821 Undefined name `UMAP`
 --> src/types.py:8:36
  |
6 | import pandas as pd
7 |
8 | type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"
  |                                    ^^^^

F821 Undefined name `MDS`
 --> src/types.py:8:45
  |
6 | import pandas as pd
7 |
8 | type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS"
  |                                             ^^^

F841 Local variable `n_labels` is assigned to but never used
  --> src/util/datasets.py:60:9
   |
58 |             raise ValueError(f"Invalid magic number {magic} in {path}")
59 |
60 |         n_labels = int.from_bytes(f.read(4), "big")
   |         ^^^^^^^^
61 |         labels = np.frombuffer(f.read(), dtype=np.uint8)
   |
help: Remove assignment to unused variable `n_labels`

F821 Undefined name `root_emb`
   --> src_research/benchmark_workflow.py:260:29
    |
258 |         "n_leaves_predicated": int(sum(1 for s in leaf_sizes if s >= MIN_SELECTION)),
259 |         # Projections that failed outright, so the census says how much of the tree it saw.
260 |         "root_unprojected": root_emb is None,
    |                             ^^^^^^^^
261 |         "n_leaves_unprojected": int(
262 |             sum(1 for leaf in leaves if leaf["embedding_original"] is None)
    |

RUF059 Unpacked variable `labels` is never used
   --> src_research/benchmark_workflow.py:662:14
    |
660 |         ax.set_xticks([0, 1], ["trust.", "cont."])
661 |         ax.set_ylabel("score" if ds == DATASETS_TO_RUN[0] else "")
662 |     handles, labels = axes[0].get_legend_handles_labels()
    |              ^^^^^^
663 |     fig.legend(
664 |         handles[:1],
    |
help: Prefix it with an underscore or any other dummy variable pattern

PLC0206 Extracting value from dictionary without calling `.items()`
   --> src_research/benchmark_workflow.py:771:21
    |
769 |         with parallel_config(backend="loky", inner_max_num_threads=1):
770 |             for result in Parallel(n_jobs=PARALLEL_JOBS, return_as="generator")(jobs):
771 |                 for key in buckets:
    |                     ^^^^^^^^^^^^^^
772 |                     buckets[key].extend(result[key])
    |                     ------------
773 |                 progress.advance(task)
    |
help: Use `for key, value in buckets.items()` instead

BLE001 Do not catch blind exception: `Exception`
   --> src_research/hyperparameter_tuning.py:344:12
    |
342 |         with contextlib.redirect_stdout(io.StringIO()):
343 |             return float(DBCV_score(data, labels)[0])
344 |     except Exception:
    |            ^^^^^^^^^
345 |         return None
    |

S110 `try`-`except`-`pass` detected, consider logging the exception
   --> src_research/hyperparameter_tuning.py:376:9
    |
374 |           try:
375 |               out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
376 | /         except Exception:
377 | |             pass
    | |________________^
378 |
379 |       if y is not None:
    |

BLE001 Do not catch blind exception: `Exception`
   --> src_research/hyperparameter_tuning.py:376:16
    |
374 |         try:
375 |             out["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
376 |         except Exception:
    |                ^^^^^^^^^
377 |             pass
    |

S110 `try`-`except`-`pass` detected, consider logging the exception
   --> src_research/hyperparameter_tuning.py:399:5
    |
397 |               out["trustworthiness"] = float(results[1]["trustworthiness"])
398 |               out["continuity"] = float(results[1]["continuity"])
399 | /     except Exception:
400 | |         pass
    | |____________^
401 |       return out
    |

BLE001 Do not catch blind exception: `Exception`
   --> src_research/hyperparameter_tuning.py:399:12
    |
397 |             out["trustworthiness"] = float(results[1]["trustworthiness"])
398 |             out["continuity"] = float(results[1]["continuity"])
399 |     except Exception:
    |            ^^^^^^^^^
400 |         pass
401 |     return out
    |

BLE001 Do not catch blind exception: `Exception`
   --> src_research/hyperparameter_tuning.py:468:16
    |
466 |             _record(trial, **dm, **ext)
467 |             return finite_or_worst(score, metric)
468 |         except Exception:
    |                ^^^^^^^^^
469 |             return metric.worst
    |

BLE001 Do not catch blind exception: `Exception`
   --> src_research/hyperparameter_tuning.py:510:12
    |
508 |         dm = dr_metrics(ds.X, embedding)
509 |         return {"baseline": finite_or_worst(tnc_mean(dm), TNC_METRIC), **dm}
510 |     except Exception:
    |            ^^^^^^^^^
511 |         return {"baseline": (DBCV_METRIC if track == "A" else TNC_METRIC).worst}
    |

S112 `try`-`except`-`continue` detected, consider logging the exception
  --> src_research/rederive/h1a.py:87:9
   |
85 |           try:
86 |               p = float(wilcoxon(nz, method=branch).pvalue)
87 | /         except Exception:  # noqa: BLE001 - a branch that cannot run is simply not a match
88 | |             continue
   | |____________________^
89 |           rel = abs(p - old) / scale
90 |           if rel < best_rel:
   |

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:126:9
    |
124 |           "## 1. h1a_summary.csv — MRRE direction (B2)",
125 |           "",
126 | /         f"Source: `h1a_regions.csv` ({len(regions)} region records) -> `{out_dir(run).name}/h1a_summary.csv` "
127 | |         f"({len(corrected)} summary rows, shipped had {len(old)}).",
    | |___________________________________________________________________^
128 |           "",
129 |           f"`HIGHER_IS_BETTER` now reads `{HIGHER_IS_BETTER}`.",
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:134:9
    |
132 |           "",
133 |           f"- **{len(b2)}** are the B2 fix — `win_rate` / `rank_biserial` on an MRRE metric.",
134 | /         f"- **{len(env)}** are `wilcoxon_p`, and they appear on every metric alike, MRRE or not — the "
135 | |         "fingerprint of an environment difference rather than a direction fix. Each was re-tested against "
136 | |         "SciPy's two branches on the shipped deltas:",
    | |_____________________________________________________^
137 |           f"    - **{len(exact_hits)}** are reproduced to within {EXACT_TOL:g} relative by naming a branch "
138 |           'explicitly, i.e. `method="auto"` simply picks a different branch now than it did in June;',
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:137:9
    |
135 |           "fingerprint of an environment difference rather than a direction fix. Each was re-tested against "
136 |           "SciPy's two branches on the shipped deltas:",
137 | /         f"    - **{len(exact_hits)}** are reproduced to within {EXACT_TOL:g} relative by naming a branch "
138 | |         'explicitly, i.e. `method="auto"` simply picks a different branch now than it did in June;',
    | |___________________________________________________________________________________________________^
139 |           f"    - **{len(near_hits)}** match a branch to within {NEAR_TOL:g} but not {EXACT_TOL:g} "
140 |           f"(worst {worst_near:.1e} relative) — the same branch computing slightly differently, i.e. SciPy's "
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:139:9
    |
137 |           f"    - **{len(exact_hits)}** are reproduced to within {EXACT_TOL:g} relative by naming a branch "
138 |           'explicitly, i.e. `method="auto"` simply picks a different branch now than it did in June;',
139 | /         f"    - **{len(near_hits)}** match a branch to within {NEAR_TOL:g} but not {EXACT_TOL:g} "
140 | |         f"(worst {worst_near:.1e} relative) — the same branch computing slightly differently, i.e. SciPy's "
141 | |         "internals, not the data;",
    | |__________________________________^
142 |           f"    - **{len(unmatched)}** match neither branch"
143 |           + (
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:155:9
    |
153 |           ),
154 |           "",
155 | /         "`n_pairs` and `median_delta` are identical on every row, which is what rules out any change to "
156 | |         "the measurements or to the pairing.",
    | |_____________________________________________^
157 |           "",
158 |           f"No conclusion moves: **{len(flips)}** rows change significance at alpha = {ALPHA}.",
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/h1a.py:165:9
    |
163 |           "### 1b. Environment differences (SciPy `wilcoxon`)",
164 |           "",
165 | /         f"Listed for completeness; none of these is a fix, and none crosses alpha = {ALPHA}. "
166 | |         "The `branch` column names the SciPy branch that reproduces the shipped value from the shipped "
167 | |         "deltas, and `rel` how closely.",
    | |________________________________________^
168 |           "",
169 |       ]
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
  --> src_research/rederive/h2b.py:79:9
   |
77 |           "## 2. subspace_summary.csv — duplicated control + mixed samples (H12d)",
78 |           "",
79 | /         f"Source: `subspace_recovery.csv` ({len(rec)} condition records) -> "
80 | |         f"`{out_dir(run).name}/subspace_summary.csv` ({len(corrected)} rows, was {len(old)}).",
   | |______________________________________________________________________________________________^
81 |           "",
82 |           "### 2a. The duplicated control",
   |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
  --> src_research/rederive/h2b.py:84:9
   |
82 |           "### 2a. The duplicated control",
83 |           "",
84 | /         f"`non_nested` was enumerated at {ctrl['rho'].nunique()} rho levels "
85 | |         f"({sorted(ctrl['rho'].unique().tolist())}), each carrying {int(per_condition.iloc[0])} records, "
86 | |         f"but `run_cell` forces `eff_rho = 1` for that arm — so they are "
87 | |         f"{ctrl['rho'].nunique()} recomputations of one condition. "
88 | |         f"{n_dropped} of {len(rec)} recovery records ({100 * n_dropped / len(rec):.1f}%) are dropped as duplicates, "
89 | |         f"leaving {len(deduped)}.",
   | |__________________________________^
90 |           "",
91 |           "### 2b. The H2b nesting gap (the number the control exists to produce)",
   |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/internal_external.py:112:9
    |
110 |           "### 3a. The join returned nothing, and fillna(0) hid that",
111 |           "",
112 | /         f'- `sampler == "none"` rows in this run: **{n_base_rows}**, of which **{n_base_trackA}** are Track A '
113 | |         f"and **{n_base_ari}** carry a non-null `ari`.",
    | |_______________________________________________________^
114 |           f"- Track-A TPE rows entering the figure: **{len(merged)}**.",
115 |           f"- `ari_base` after the join is entirely null: **{all_nan}**.",
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
   --> src_research/rederive/internal_external.py:116:9
    |
114 |           f"- Track-A TPE rows entering the figure: **{len(merged)}**.",
115 |           f"- `ari_base` after the join is entirely null: **{all_nan}**.",
116 | /         f"- Therefore the plotted y equals raw tuned ARI exactly: **{identical}** "
117 | |         "(`ari - ari_base.fillna(0)` == `ari`).",
    | |________________________________________________^
118 |           "",
119 |           "A cell whose tuning *lowered* ARI plotted above zero, because zero was never the baseline.",
    |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

ISC004 Unparenthesized implicit string concatenation in collection
  --> src_research/rederive/verdicts.py:57:9
   |
55 |           "## 4. verdicts.csv — n_selections was n_pairs (H12c)",
56 |           "",
57 | /         f"Source: `stability_records.csv` ({len(records)} records) -> `{out_dir(run).name}/verdicts.csv` "
58 | |         f"({len(corrected)} verdict rows) and `{out_dir(run).name}/stability_summary.csv` ({len(summary)} rows).",
   | |_________________________________________________________________________________________________________________^
59 |           "",
60 |           "### 4a. What the sample actually was",
   |
help: Did you forget a comma?
help: Wrap implicitly concatenated strings in parentheses

Found 33 errors.
No fixes available (13 hidden fixes can be enabled with the `--unsafe-fixes` option).
exit=1
```
The 33 `ruff check` findings are the count CLAUDE.md documents as pre-existing; the one at `src/analysis/analysis_routine.py:98` is the `except Exception` that already existed in `_embed_original` at HEAD (line 94 there), now inside `fit_node_projection`. No new finding was introduced.

### Regression: `misc/predicate_baselines_2026-08-26/run_hilde.py`

Run on scratch copies (`/tmp/review/regress/{before,after}/`) with `common.py`, `run_hilde.py` and `winequality-white.csv` copied beside a `predicate_generator.py` taken from `git show HEAD:src/analysis/predicate_generator.py` (before) and from the working tree (after). The stored `misc/…/hilde_result.json` (mtime Aug 26 12:04:54) was not touched.
```
=== BEFORE: predicate_generator.py from 'git show HEAD:src/analysis/predicate_generator.py' ===
strict t=1.0: 10 clauses  F1 1.0000  P 1.0000  R 1.0000  matched 39  covers 39/39
    density                [0.9977, 1]
    total sulfur dioxide   [117, 174]
    pH                     [3.04, 3.3]
    citric acid            [0.23, 0.42]
    sulphates              [0.41, 0.54]
    fixed acidity          [6.2, 7.7]
    free sulfur dioxide    [22, 45]
    alcohol                [8.7, 9.6]
    residual sugar         [11, 18.3]
    volatile acidity       [0.21, 0.33]
relaxed t=0.9: 5 clauses  F1 0.8611  P 0.9394  R 0.7949  matched 33  covers 31/39
    residual sugar         [14.4, 18.3]
    total sulfur dioxide   [117, 154.2]
    free sulfur dioxide    [31, 43]
    density                [0.998, 1]
    citric acid            [0.23, 0.408]
=== AFTER: predicate_generator.py from the working tree ===
strict t=1.0: 10 clauses  F1 1.0000  P 1.0000  R 1.0000  matched 39  covers 39/39
    density                [0.9977, 1]
    total sulfur dioxide   [117, 174]
    pH                     [3.04, 3.3]
    citric acid            [0.23, 0.42]
    sulphates              [0.41, 0.54]
    fixed acidity          [6.2, 7.7]
    free sulfur dioxide    [22, 45]
    alcohol                [8.7, 9.6]
    residual sugar         [11, 18.3]
    volatile acidity       [0.21, 0.33]
relaxed t=0.9: 5 clauses  F1 0.8611  P 0.9394  R 0.7949  matched 33  covers 31/39
    residual sugar         [14.4, 18.3]
    total sulfur dioxide   [117, 154.2]
    free sulfur dioxide    [31, 43]
    density                [0.998, 1]
    citric acid            [0.23, 0.408]
=== diff before/after hilde_result.json ===
IDENTICAL
=== diff after vs stored misc/.../hilde_result.json (Aug 26) ===
IDENTICAL
```
Expected 1.000 / 10 clauses at t = 1.0 and 0.861 / 5 at t = 0.9 with identical order: met, and the two JSON outputs are byte-identical to each other and to the stored result.

### Track P invariants on live Iris output (`/tmp/review/check_track_p.py`)

Independent recomputation from the clause masks (background scaled per scope exactly as `compute_predicate` does): baseline = 2|S|/(N+|S|); every standalone clause F1/P/R; at every step the recorded clause is the greedy argmax with ties to the lowest column index; before/after/gain/n_matched/P/R after each step; the stopping rule; telescoping; step-0 invariant; `None`s on unselected rows and on the no-labels path.
```
$ PYTHONPATH=. .venv/bin/python /tmp/review/check_track_p.py
  target_setosa/local/full         N= 150 |S|= 50 baseline=0.5000 final=1.0000 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_setosa/local/trimmed      N= 150 |S|= 50 baseline=0.5000 final=0.9796 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_setosa/global/full        N= 150 |S|= 50 baseline=0.5000 final=1.0000 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_setosa/global/trimmed     N= 150 |S|= 50 baseline=0.5000 final=0.9796 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_versicolor/local/full     N= 150 |S|= 50 baseline=0.5000 final=0.9259 clauses=2 order=['petal', 'petal'] telescope_err=0.0e+00
  target_versicolor/local/trimmed  N= 150 |S|= 50 baseline=0.5000 final=0.9412 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_versicolor/global/full    N= 150 |S|= 50 baseline=0.5000 final=0.9259 clauses=2 order=['petal', 'petal'] telescope_err=0.0e+00
  target_versicolor/global/trimmed N= 150 |S|= 50 baseline=0.5000 final=0.9412 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_virginica/local/full      N= 150 |S|= 50 baseline=0.5000 final=0.8475 clauses=2 order=['petal', 'petal'] telescope_err=0.0e+00
  target_virginica/local/trimmed   N= 150 |S|= 50 baseline=0.5000 final=0.8980 clauses=1 order=['petal'] telescope_err=0.0e+00
  target_virginica/global/full     N= 150 |S|= 50 baseline=0.5000 final=0.8475 clauses=2 order=['petal', 'petal'] telescope_err=0.0e+00
  target_virginica/global/trimmed  N= 150 |S|= 50 baseline=0.5000 final=0.8980 clauses=1 order=['petal'] telescope_err=0.0e+00
  lasso in sub-node/local/full     N=  90 |S|= 36 baseline=0.5714 final=1.0000 clauses=1 order=['petal'] telescope_err=0.0e+00
  lasso in sub-node/local/trimmed  N=  90 |S|= 36 baseline=0.5714 final=0.9714 clauses=1 order=['petal'] telescope_err=0.0e+00
  lasso in sub-node/global/full    N= 150 |S|= 36 baseline=0.3871 final=1.0000 clauses=1 order=['petal'] telescope_err=0.0e+00
  lasso in sub-node/global/trimmed N= 150 |S|= 36 baseline=0.3871 final=0.9714 clauses=1 order=['petal'] telescope_err=0.0e+00
  small node, large baseline/local/full N=  50 |S|= 16 baseline=0.4848 final=1.0000 clauses=1 order=['sepal'] telescope_err=0.0e+00
  small node, large baseline/local/trimmed N=  50 |S|= 16 baseline=0.4848 final=0.9677 clauses=1 order=['sepal'] telescope_err=0.0e+00
  small node, large baseline/global/full N= 150 |S|= 16 baseline=0.1928 final=1.0000 clauses=3 order=['sepal', 'petal', 'petal'] telescope_err=0.0e+00
  small node, large baseline/global/trimmed N= 150 |S|= 16 baseline=0.1928 final=0.8667 clauses=2 order=['sepal', 'petal'] telescope_err=1.1e-16
  no-labels path: baseline 0.0, step fields None  OK
ALL TRACK P INVARIANTS HOLD (recomputed independently from clause masks)
```

---

## 5. Process deviations (orchestration §1, §3, §5, §6) — facts

- **Zero commits.** `HEAD` is still `aa1ee92` = `final-thesis`; `git log main..HEAD` is empty. §1 requires a commit at the end of every stage and §3 requires Track P "as its own commit when green". Nothing — not even the Stage A contracts (`types.ts`, `api.ts`, `props.ts`, the `movement.py` stub) — was committed first, so the interface-freeze workflow was not followed.
- **Track P is not cherry-pickable as the tree stands.** Files carrying hunks from BOTH tracks (`git diff -U0`): `frontend/src/types.ts` (P: `PredicateRow` 109-129, `PredicateSummary` 139-148; M: `AnalysisMeta` 42-55, movement types 209-355), `README.md` (P: hunk at 191; M: 208-229; the limitations hunk at 364-380 contains one *sentence* that names both tracks' tests), `frontend/README.md` (P: rows/paragraph at 130 and 136-151; M: 46-51, 99, 110, 162-165, 184-196), `src/README.md` (P: 184-199; M: 138-158), `backend/README.md` (M throughout; the `tests/` row at 197 names `test_predicate.py`). All other changed files are single-track (see §7).
- **Stage C checkpoint skipped.** "STOP and report to Hannes before Stage D" — Stage D artefacts exist: `backend/movement_jobs.py`, the UMAP section of `backend/movement.py` (1088-1508), the UMAP and job tests (`test_movement.py:893-1294`), the debounced slider (`movement.ts:464-487`). No checkpoint report exists.
- **Final report (§6) absent.** No summary of stages, deviations, R1 findings or commit list exists anywhere in the repo; this document is the first review.
- **Stage E (R1 light review) was not run by the orchestrator** before this review.
- **File-ownership map:** `pyproject.toml` and `.gitignore` were edited; neither is in the map. `plans/2026-08-28-spec-predicate-f1-gain.md` was rewritten by `ruff format` (mtime 17:46 vs 17:22 for the other two) — a spec is an input owned by no track.
- **Stop-and-ask list:** (a) `run_hilde.py` — no change, trigger not hit. (b) Canary green at the spec tolerances, no widening; the separate runtime gate 0.05 was chosen without asking (MINOR-1). (c) `umap-learn 0.5.12` `inverse_transform` works for the configured parameters (`estimator: "umap_inverse"` on live Iris) — fallback-only mode was not needed. (d) Payload changes are additive only: `_NODE_KEYS` (`test_serialize.py:18-31`) and `serialize_node` are untouched, `meta` gains `analysis_id`, `effective_config`, `schema_version`. (e) The vertical-slice checkpoint — hit and improvised past. (f) `final-thesis` untouched.
- **Spec alteration.** `ruff format` genuinely reformats Python fences in Markdown (ruff 0.16.3's default `include` contains `*.md`; probe at `/tmp/review/ruffmd/probe.md` was rewritten). The predicate spec's data-contract block now shows ruff's two-space inline-comment style with residual column padding before each `(AMENDED)` marker — consistent with a whitespace-only rewrite; the pre-format text is unrecoverable (no baseline). All `AMENDMENT (binding)` blocks are prose and intact. The `extend-exclude = ["plans"]` in `pyproject.toml` is therefore justified.
- **`.claude/settings.local.json`** is not untracked but *ignored*, by the user's global ignore file (`~/.github_hm/.gitignore_global:6`); mtime Aug 14, contents `{"permissions":{"allow":["Skill(dataviz)"]}}` — predates this work and cannot be committed by accident.
- `.cache/hilde_runs/22d6…json.gz` (Sep 6 10:03, wine UMAP, schema 2) shows a hosting-mode run this morning; gitignored, no consequence.

---

## 6. What could NOT be verified, and why

- **Any browser interaction.** Target resolution precedence, the 10 px snap after zoom/pan, screen→embedding inversion, drag/zoom suppression, Escape, the crosshair and ghost rendering, the PCA slider feel and the UMAP debounce were verified only by reading `movement.ts` (pure functions) and `ClusterScatter.tsx`, plus `npm run build`. No frontend test exists and none was written.
- **The Stage B/C manual checklist and the "wide image dataset with a visibly badged sampled preview".** `datasets/` is gitignored; sampling was exercised only by lowering the cap to 12 in the unit test and by `_preview_indices` directly.
- **The pytest gate in the repo's own environment.** Not installable without editing `pyproject.toml`; verified only through an ephemeral overlay.
- **What `ruff format` changed in the predicate spec.** No committed or backed-up pre-format copy exists.
- **A genuinely old on-disk payload.** I synthesized one by stripping the three new meta keys from a live payload; node-level shape is unchanged by this work, so it is representative, but it is not a file written by the pre-feature code.
- **`src_research/predicate_stability.py` "run it once to confirm".** Read (rows consumed by `in_predicate`/`feature` only), not executed — it is a research harness with its own runtime.
- **Thread-safety of `_artifact_cache` (MINOR-6) and the double refit (MINOR-7).** Reasoned from the code, not reproduced under load.
- **Behaviour of the clamp path itself** (`umap_n_neighbors` > rows−1): no bundled dataset has fewer than 16 rows, so the case `effective_config` exists for was never exercised end-to-end — only its plumbing.

---

## 7. Recommended commit split (keeps Track P cherry-pickable) — NOT executed

Four files need hunk-level staging (`git add -p`) and one needs a manual sentence split; everything else stages whole. Suggested order and contents:

**Commit 0 — `chore(plans): design specs, review prompt and report; keep ruff out of plans/`**
`.gitignore`, `pyproject.toml`, `plans/2026-08-28-orchestration.md`, `plans/2026-08-28-spec-predicate-f1-gain.md`, `plans/2026-08-28-spec-cluster-movement.md`, `plans/2026-09-06-review-prompt.md`, `plans/2026-09-06-review-report.md`.

**Commit 1 — `feat(predicate): marginal ΔF1, per-step counts/P/R, baseline, RCM 0.9 trajectory (Track P)`** — the cherry-pickable one
- whole files: `src/analysis/predicate_generator.py`, `backend/predicate.py`, `backend/tests/test_predicate.py`, `frontend/src/charts/PredicateBands.tsx`, `frontend/src/components/LayerSide.tsx`, `frontend/src/components/ExplorationPanel.tsx`, `frontend/src/fixtures/predicate_iris.json`
- hunks only: `frontend/src/types.ts` (the `PredicateRow` and `PredicateSummary` hunks, lines 109-129 and 139-148 — nothing from `AnalysisMeta` or below `AnalysisConfig`), `README.md` (the exploration-panel paragraph hunk at 191; from the limitations hunk only the test-suite bullet, reworded to name three scripts), `frontend/README.md` (the `PredicateBands` table row at 130 and the "How `PredicateBands` shows the construction" paragraph 136-151), `src/README.md` (the predicate table rows and the gain paragraph, 184-199), `backend/README.md` (the `tests/` row at 197 with only `test_predicate.py` added).
- Standalone check after staging: `test_predicate`, `test_serialize`, `test_targets`, `npm run build` — P's `types.ts` hunks are self-contained interfaces and `backend/predicate.py` depends on nothing new in `serialize.py`.

**Commit 2 — `feat(movement): effective_config, analysis_id, schema v2, fit_node_projection (Stage A foundations)`**
`src/analysis/analysis_routine.py`, `backend/serialize.py`, `backend/app.py` (the `_build` meta hunk and the imports it needs — or the whole file if not splitting further), `frontend/src/types.ts` (`AnalysisMeta` hunk 42-55).

**Commit 3 — `feat(movement): backend PCA + UMAP movement, jobs, tests (Stages B, D)`**
`backend/movement.py`, `backend/movement_jobs.py`, `backend/tests/test_movement.py`, remaining `backend/app.py` hunks (routes).

**Commit 4 — `feat(movement): scatter interaction, useMovement, panel, styles (Stages B, C)`**
`frontend/src/types.ts` (movement types hunk 209-355), `frontend/src/api.ts`, `frontend/src/charts/props.ts`, `frontend/src/charts/ClusterScatter.tsx`, `frontend/src/movement.ts`, `frontend/src/components/ClusterMovementPanel.tsx`, `frontend/src/styles.css`, `frontend/src/App.tsx`, `frontend/src/fixtures/movement_iris.json`.

**Commit 5 — `docs(movement): READMEs`**
Remaining hunks of `README.md`, `backend/README.md`, `frontend/README.md`, `src/README.md` (and the test-suite bullet updated to four scripts, plus — if MINOR-9 is fixed — the two new test commands in the runnable blocks).

Commit 1 then cherry-picks onto `presentation_additions` (or any base at `final-thesis`) without Commit 0 if `plans/` is not wanted there.

---

## Appendix A — Independent math re-derivation (§4)

`/tmp/review/check_pca_math.py`: 60×5 toy set (three Gaussian blobs, one `target_*` column so an immutable feature exists), my own standardization and SVD-based PCA with the second component **sign-flipped** relative to what sklearn would produce, so my "visible" frame is a reflection of the fitted one.
```
$ PYTHONPATH=. .venv/bin/python /tmp/review/check_pca_math.py
rank(P_mutable)      = 2
pinv reaches target  : residual 3.3306690738754696e-16
rowspace residual    : 5.438959822042073e-16
raw unscaling error  : 4.440892098500626e-16
repo free-point delta vs mine: max|diff| = 0.0
repo cluster delta vs mine   : max|diff| = 0.0
alignment: rmse=6.63e-16 scale=1.000000 det(R)=-1.000
pipeline raw delta vs mine   : max|diff| = 8.881784197001252e-15
out_of_range mine=0.1000 repo=0.1000
cluster target: projected_distance_after=0.365506 expected gap (immutable label)=0.365506
ALL INDEPENDENT PCA CHECKS PASS (toy 60x5, 3 blobs, reflected visible frame)
```
Read: the repo's `free_point_displacement` / `cluster_displacement` reproduce my pinv and centroid deltas exactly (max |diff| = 0.0); the full `compute_movement` on a synthetic serialized payload aligns my reflected frame with det(R) = −1 at RMSE 6.6e-16 and returns target, ghosts, displacement and half-strength ghosts in *my* coordinates; raw deltas equal Δ·σ to 9e-15; the label column is exactly 0; `out_of_range_fraction` equals my recomputation; with the label immutable the cluster target is missed by exactly ‖P[:,immutable]·(μ_t−μ_s)[immutable]‖ = 0.365506, as it must be; neither `df` nor the payload was mutated.

## Appendix B — Contract parity (§5)

Field-by-field, `backend/movement.py` (PCA 1039-1091, UMAP 1462-1508) and `backend/predicate.py` (92-109 + generator rows) ↔ `frontend/src/types.ts` ↔ consumers:

- **Produced but never consumed:** `PredicateSummary.trimmed_n_features_used` (MINOR-4); `PredicateRow.predicate_baseline_f1` at row level (only the summary copy is read — by design, "identical on every row"); `MovementResponse.target.x/y` for **cluster** targets (`ClusterScatter.tsx` marks the destination with its own plot-space centroid, the panel prints coordinates only for `kind === "point"`); `AnalysisMeta.effective_config` and `schema_version` (typed optional, server-side only — by design).
- **Consumed but never produced:** none found. Every `metrics.*`, `target.*`, `feature_changes[*].*`, `preview_points[*].*`, `strength_trajectory`, `visible_displacement`, `recommended_strength`, `applied_strength`, `source_size`, `target_size`, `warnings`, `status`, and the three ids are emitted by both backend paths.
- **Typed differently:** `MovementRequest.target` is `dict[str, Any]` in pydantic (`app.py:133`) vs the discriminated union in TS — looser server side, validated by hand at `movement.py:342-372`; `MovementResponse.method: string` vs backend literals `"PCA" | "UMAP"` (canonicalized at 320-322); `target.estimator` is a 4-literal union in TS and a plain `str` in Python (values match: 726-777, 779-800, 1250-1262); all nullable wire fields are `X | null` on both sides — no `undefined` on the wire; `AnalysisMeta.analysis_id/effective_config/schema_version` optional in TS and always present in new payloads (correct for legacy payloads, and `movement.ts:388-400` handles the absence). `MovementJob`'s finished variant is the response itself (no `"done"` wrapper) consistently in `movement_jobs.py:119-131` and `api.ts:123-139`.
- **Fixtures:** both `movement_iris.json` and `predicate_iris.json` type-check when inlined as object literals with `satisfies MovementResponse` / `satisfies PredicateResponse` under `strict` (`/tmp/review/tscheck/`, tsc exit 0); a negative control with `estimator: "bogus"` and `preview_sampled: "no"` fails with the expected TS2322 errors, so the check has teeth. `predicate_iris.json` is internally consistent (baseline + Σgains = final for both runs, exact). `movement_iris.json`: MINOR-10.

## Appendix C — Safety / mutation (§6)

- Grep of assignment/in-place operators in `movement.py`, `movement_jobs.py`, `predicate.py`, `predicate_generator.py`: every subscript write lands on a freshly allocated array (`delta` at 743/761-762/795/1268-1270), on module-owned caches, or on the generator's own row dicts; `df[feature_cols].to_numpy()` feeds `StandardScaler().fit_transform` (copy) or is fancy-indexed (`X_root[row_indices]`, 476 — copy); `meta = dict(...)` (271), `effective = dict(...)` (296); the serialized node dicts are only read (`find_node`, `_child_positions`, `require_alignment`); `predicate_generator.py:113` `df = df.copy()`. Tests `test_movement.py:441-449, 880-886, 1169-1175` and my toy check confirm `df` and the tree are byte-unchanged.
- `ProjectionArtifact` LRU: `_ARTIFACT_CACHE_MAX = 4` (142), `OrderedDict` keyed `(analysis_id, node_id)` (450, 502), `move_to_end` + `popitem(last=False)` (516-519) — bounded and keyed as required; unlocked (MINOR-6).
- Payload: `serialize_node` and `_NODE_KEYS` untouched (`git diff` of `serialize.py` is additions after line 134 only; `test_serialize.py` not in the diff); `meta` gains three keys. Legacy payload: loads through `run_cache.load`, is served by `/api/analysis` with `cached: true`, and `/api/movement` answers 409, not 500 (hosting-mode run on port 8124; MINOR-2 for the message).
- Movement results are never persisted: `movement_jobs.py` holds them in memory (`_MAX_JOBS = 16`, 54); `run_cache.store` is called only from `_build` (`app.py:229`). No `HILDE_*` env was set during the dev smoke, and `.cache/hilde_runs` kept its single 10:03 entry.

## Appendix D — Frontend state hygiene (§7)

- Clear conditions (`frontend/src/movement.ts`): dataset change and new analysis → effect on `[datasetKey, analysis]` (492-494) calls `cancel`; `App.tsx:129-130` derives `shownAnalysis` from the state object itself (no per-render clone), so this fires only on a real new run. Navigation → 500-505, clears when the movement's node leaves the path (MINOR-11 on the drill-deeper exception). Source change / re-start → `start` bumps the generation and replaces the state (358-361). Cancel button → `cancel` (349-353, generation bump, pending timer cleared, IDLE). Escape → window `keydown` listener registered while `phase !== "idle"` (510-517). "Explore entire layer" goes through `navigate`, hence the path effect. Unmount clears the debounce timer (356).
- Stale guard (`isCurrentResponse`, 256-292): generation (263), phase must still be `loading`/`ready` (270), node (272), source child (274), target via `sameTarget` (276), analysis id of the run on screen (278), and the response's own `analysis_id`/`node_id`/`source_child_index`/`target.kind`/`child_index` (283-289). Rejections are checked the same way minus the response comparisons (441); a failed strength *refinement* deliberately keeps the existing preview (448).
- Drill-down with movement mode off (diff of `ClusterScatter.tsx` against `/tmp/review/ClusterScatter.HEAD.tsx`): every `inMovement`-conditional evaluates to the HEAD value when false — circle `fillOpacity`, `stroke`/`strokeWidth`/`vectorEffect` (`undefined`), cursor `pointer`, `<title>` text, noise `strokeOpacity` (`dim ? 0.18 : 0.5`), legend/centroid handlers (`clickRef` → `handleClick` → `onSelectRef.current(child)` for `child >= 0`, which is the only kind of circle drawn), SVG cursor `grab`, no `onPointerDown`, no background `<rect>`, no `noiseHits`, no ghosts/marker/arrow, no status line. Residual non-behavioural differences: the zoom handler also sets `zoomFiredRef` (297), `pointsLayer`'s memo deps gained `inMovement, sourceChild` (261), a `useId()` call (357), and `hasChart`/`highlighted`/zoom refs moved up in the file. The `[hasChart]` and `[node]` effect dependencies are unchanged (306, 313 vs HEAD 167, 174).

## Appendix E — Backend smoke (§8), `/tmp/review/smoke.py` against `uvicorn backend.app:app --port 8123` (dev mode, no run cache)

Responses truncated to the relevant fields (`preview_points` and `feature_changes` abbreviated; trajectory as `(strength, distance)` pairs).
```
PCA analysis: cached=False analysis_id=27acbdd190f8c555… schema_version=2 effective_config={'normalize': True, 'method': 'PCA', 'umap_n_neighbors': 15, 'umap_min_dist': 0.1, 'umap_random_state': 42, 'tsne_perplexity': 30.0, 'tsne_learning_rate': 200.0, 'tsne_random_state': 42, 'mds_metric': True, 'mds_n_init': 2, 'mds_max_iter': 100, 'mds_random_state': 42}
internal child node: root/1 children: [64, 31] embedding rows: 100

### PCA point target on root/1 (click (1.572, -0.302))
HTTP 200
{
 "status": "ok",
 "analysis_id": "27acbdd190f8c55505190db34695f71f17d4f5795f07857c903668d04bc094a0",
 "node_id": "root/1",
 "method": "PCA",
 "source_child_index": 0,
 "target": {
  "kind": "point",
  "child_index": null,
  "x": 1.5718,
  "y": -0.3024,
  "estimator": "pca_pinv"
 },
 "recommended_strength": 1.0,
 "applied_strength": 1.0,
 "source_size": 64,
 "target_size": null,
 "visible_displacement": [
  2.1538,
  -0.4043
 ],
 "metrics": {
  "projected_distance_before": 2.191398,
  "projected_distance_after": 0.0,
  "projected_distance_reduction": 2.191398,
  "feature_centroid_distance_before": 2.191398,
  "feature_centroid_distance_after": 0.0,
  "out_of_range_fraction": 0.265625,
  "support_ratio": null,
  "alignment_rmse": 0.0,
  "preview_sampled": false
 },
 "warnings": [
  "Many different feature changes reach this position; this is the smallest standardized change that reaches this point (smallest measured in root-standardized units \u2014 one unit is one standard deviation over the whole dataset).",
  "27% of the moved points fall outside the range this dataset covers in at least one changed feature \u2014 the counterfactual is extrapolating beyond the observed data."
 ],
 "preview_points": "<64 points, first [{'row_id': 50, 'x': 2.954051246373466, 'y': -0.20976820656957468}, {'row_id': 51, 'x': 2.4998870282070245, 'y': 0.19243171629636047}]>",
 "feature_changes": [
  {
   "feature": "sepal length (cm)",
   "mutable": true,
   "source_mean_raw": 5.9406,
   "target_value_raw": 7.2499,
   "recommended_delta_raw": 1.3093,
   "applied_delta_raw": 1.3093,
   "standardized_magnitude": 1.5865
  },
  {
   "feature": "sepal width (cm)",
   "mutable": true,
   "source_mean_raw": 2.775,
   "target_value_raw": 3.1468,
   "recommended_delta_raw": 0.3718,
   "applied_delta_raw": 0.3718,
   "standardized_magnitude": 0.856
  },
  "\u2026"
 ],
 "strength_trajectory": []
}
check: ghosts == serialized source + visible_displacement, max err 1.22e-15; moved centroid (1.5718, -0.3024) vs click (1.5718, -0.3024)

### PCA cluster target C0 -> C1 on root/1, strength 0.5
HTTP 200
{
 "status": "ok",
 "analysis_id": "27acbdd190f8c55505190db34695f71f17d4f5795f07857c903668d04bc094a0",
 "node_id": "root/1",
 "method": "PCA",
 "source_child_index": 0,
 "target": {
  "kind": "cluster",
  "child_index": 1,
  "x": 1.2718,
  "y": -0.1024,
  "estimator": "observed_centroid"
 },
 "recommended_strength": 1.0,
 "applied_strength": 0.5,
 "source_size": 64,
 "target_size": 31,
 "visible_displacement": [
  1.8538,
  -0.2043
 ],
 "metrics": {
  "projected_distance_before": 1.865003,
  "projected_distance_after": 0.932502,
  "projected_distance_reduction": 0.932502,
  "feature_centroid_distance_before": 1.87602,
  "feature_centroid_distance_after": 0.93801,
  "out_of_range_fraction": 0.015625,
  "support_ratio": null,
  "alignment_rmse": 0.0,
  "preview_sampled": false
 },
 "warnings": [],
 "preview_points": "<64 points, first [{'row_id': 50, 'x': 1.7271612965786725, 'y': 0.09238007700771558}, {'row_id': 51, 'x': 1.2729970784122313, 'y': 0.4945799998736507}]>",
 "feature_changes": [
  {
   "feature": "sepal length (cm)",
   "mutable": true,
   "source_mean_raw": 5.9406,
   "target_value_raw": 6.9258,
   "recommended_delta_raw": 0.9852,
   "applied_delta_raw": 0.4926,
   "standardized_magnitude": 1.1937
  },
  {
   "feature": "sepal width (cm)",
   "mutable": true,
   "source_mean_raw": 2.775,
   "target_value_raw": 3.129,
   "recommended_delta_raw": 0.354,
   "applied_delta_raw": 0.177,
   "standardized_magnitude": 0.815
  },
  "\u2026"
 ],
 "strength_trajectory": []
}

### config = effective_config instead of the requested config
HTTP 409
{
 "detail": "The analysis this movement refers to is no longer on the server. Rebuild it and try the movement again."
}

### wrong analysis_id
HTTP 409
{
 "detail": "This movement was requested against a different analysis run than the one the server holds. Rebuild the analysis and try again."
}

### malformed target {'kind': 'elsewhere'}
HTTP 400
{
 "detail": "Unknown target kind 'elsewhere'; expected 'point' or 'cluster'."
}

### malformed target {'kind': 'point', 'x': 'abc', 'y': 0.0}
HTTP 400
{
 "detail": "Target coordinates must be finite numbers; got 'abc', 0.0."
}

### malformed target {'kind': 'cluster', 'child_index': 0}
HTTP 400
{
 "detail": "Source and destination clusters must differ."
}

### malformed target {'kind': 'cluster', 'child_index': 99}
HTTP 400
{
 "detail": "Destination child index 99 is out of range for 'root/1' (2 children)."
}

### strength 1.5
HTTP 400
{
 "detail": "Strength must be null or a number in [0, 1]; got 1.5."
}

### leaf node root/0
HTTP 400
{
 "detail": "Node 'root/0' is a leaf: it has no child clusters to move."
}

UMAP analysis: analysis_id=7f74286e2ae992f1… effective_config={'normalize': True, 'method': 'UMAP', 'umap_n_neighbors': 15, 'umap_min_dist': 0.1, 'umap_random_state': 42, 'tsne_perplexity': 30.0, 'tsne_learning_rate': 200.0, 'tsne_random_state': 42, 'mds_metric': True, 'mds_n_init': 2, 'mds_max_iter': 100, 'mds_random_state': 42}; movement node root/1 children [64, 31]

### UMAP point target, first request -> HTTP 200 {"status": "running", "job_id": "28c1c813f0af4db9976b1379fb8023c5"}
polled 4x over 2.1s -> HTTP 200 status=ok

### UMAP point target (job result)
HTTP 200
{
 "status": "ok",
 "analysis_id": "7f74286e2ae992f1ef73ccf01f0a4ca9155aa4d05c3d1d014c8d511c5be305ad",
 "node_id": "root/1",
 "method": "UMAP",
 "source_child_index": 0,
 "target": {
  "kind": "point",
  "child_index": null,
  "x": 9.8197,
  "y": 1.1821,
  "estimator": "umap_inverse"
 },
 "recommended_strength": 1.0,
 "applied_strength": 1.0,
 "source_size": 64,
 "target_size": null,
 "visible_displacement": [1.1521, 4.0988],
 "metrics": {
  "projected_distance_before": 4.343257,
  "projected_distance_after": 0.522107,
  "projected_distance_reduction": 3.82115,
  "feature_centroid_distance_before": 1.895161,
  "feature_centroid_distance_after": 0.0,
  "out_of_range_fraction": 0.296875,
  "support_ratio": 1.084988,
  "alignment_rmse": 0.0,
  "preview_sampled": false
 },
 "warnings": [
  "UMAP's inverse projection is approximate, so this recommendation is not guaranteed to land the cluster's centre exactly on the position you clicked. The preview shows where the frozen projection actually puts the moved points.",
  "30% of the moved points fall outside the range this dataset covers in at least one changed feature \u2014 the counterfactual is extrapolating beyond the observed data."
 ],
 "preview_points": "<64 points, first [{'row_id': 50, 'x': 9.743560791015623, 'y': 2.1734030246734615}, {'row_id': 51, 'x': 9.117239952087402, 'y': 1.8276752233505245}]>",
 "feature_changes": [
  {
   "feature": "sepal length (cm)",
   "mutable": true,
   "source_mean_raw": 5.9406,
   "target_value_raw": 6.8293,
   "recommended_delta_raw": 0.8886,
   "applied_delta_raw": 0.8886,
   "standardized_magnitude": 1.0767
  },
  {
   "feature": "sepal width (cm)",
   "mutable": true,
   "source_mean_raw": 2.775,
   "target_value_raw": 3.1021,
   "recommended_delta_raw": 0.3271,
   "applied_delta_raw": 0.3271,
   "standardized_magnitude": 0.753
  },
  "\u2026"
 ],
 "strength_trajectory": [[0.0, 4.602], [0.1, 4.341], [0.2, 3.814], [0.30000000000000004, 3.244], [0.4, 2.61], [0.5, 2.23], [0.6000000000000001, 1.801], [0.7000000000000001, 1.402], [0.8, 0.931], [0.9, 0.693], [1.0, 0.522]]
}

same UMAP request again (artifact warm) -> HTTP 200 status=ok inline=True; identical to job result: True

### UMAP cluster target C0 -> C1
HTTP 200
{
 "status": "ok",
 "analysis_id": "7f74286e2ae992f1ef73ccf01f0a4ca9155aa4d05c3d1d014c8d511c5be305ad",
 "node_id": "root/1",
 "method": "UMAP",
 "source_child_index": 0,
 "target": {
  "kind": "cluster",
  "child_index": 1,
  "x": 9.8197,
  "y": 1.1821,
  "estimator": "observed_centroid"
 },
 "recommended_strength": 1.0,
 "applied_strength": 1.0,
 "source_size": 64,
 "target_size": 31,
 "visible_displacement": [1.4506, 4.0037],
 "metrics": {
  "projected_distance_before": 4.343257,
  "projected_distance_after": 0.35006,
  "projected_distance_reduction": 3.993197,
  "feature_centroid_distance_before": 1.87602,
  "feature_centroid_distance_after": 0.0,
  "out_of_range_fraction": 0.15625,
  "support_ratio": null,
  "alignment_rmse": 0.0,
  "preview_sampled": false
 },
 "warnings": [
  "16% of the moved points fall outside the range this dataset covers in at least one changed feature \u2014 the counterfactual is extrapolating beyond the observed data."
 ],
 "preview_points": "<64 points, first [{'row_id': 50, 'x': 9.837458610534666, 'y': 2.190407514572143}, {'row_id': 51, 'x': 9.498414039611815, 'y': 2.2434875965118404}]>",
 "feature_changes": [
  {
   "feature": "sepal length (cm)",
   "mutable": true,
   "source_mean_raw": 5.9406,
   "target_value_raw": 6.9258,
   "recommended_delta_raw": 0.9852,
   "applied_delta_raw": 0.9852,
   "standardized_magnitude": 1.1937
  },
  {
   "feature": "sepal width (cm)",
   "mutable": true,
   "source_mean_raw": 2.775,
   "target_value_raw": 3.129,
   "recommended_delta_raw": 0.354,
   "applied_delta_raw": 0.354,
   "standardized_magnitude": 0.815
  },
  "\u2026"
 ],
 "strength_trajectory": [[0.0, 4.602], [0.1, 4.292], [0.2, 3.866], [0.30000000000000004, 3.333], [0.4, 2.712], [0.5, 2.28], [0.6000000000000001, 1.798], [0.7000000000000001, 1.409], [0.8, 0.922], [0.9, 0.648], [1.0, 0.35]]
}

### unknown movement job
HTTP 404
{
 "detail": "Unknown movement job"
}

### t-SNE run: movement request
HTTP 400
{
 "detail": "Movement is only available for PCA and UMAP projections. This run uses t-SNE; t-SNE and MDS are not supported (neither offers a usable inverse of the projection)."
}
```

Hosting-mode legacy payload (second server on port 8124, `HILDE_HOSTING=1 HILDE_CACHE_DIR=/tmp/review/legacy_cache`, payload = live PCA payload with `analysis_id`/`effective_config`/`schema_version` removed):
```
$ curl -s http://127.0.0.1:8124/api/mode
{"hosting":true,"cache_dir":"/tmp/review/legacy_cache"}
analysis from hosting cache: HTTP 200 status done cached True meta keys ['config', 'dataset', 'feature_cols', 'n_total']
movement vs legacy payload (correct id): HTTP 409 {'detail': 'This movement was requested against a different analysis run than the one the server holds. Rebuild the analysis and try again.'}
movement vs legacy payload (bogus id):   HTTP 409 {'detail': 'This movement was requested against a different analysis run than the one the server holds. Rebuild the analysis and try again.'}
server log: POST /api/analysis 200 OK · POST /api/movement 409 Conflict · POST /api/movement 409 Conflict (no 500)
```

## Appendix F — Housekeeping (§9)

- `.claude/settings.local.json`: ignored globally, not at risk (see §5).
- `.gitignore` `!plans/*`: needed to commit the specs and this report because `*.md` is ignored globally in the repo; over-broad (MINOR-12) but sensible.
- `pyproject.toml` `extend-exclude = ["plans"]`: justified — proven that ruff 0.16.3 rewrites Python fences in `.md` files; without the exclude the documented `uv run ruff format .` would keep editing the specs.
- Predicate spec: rewritten by ruff (whitespace in the one Python fence; all AMENDMENT blocks intact and readable). The spec should be treated as authoritative in its current form since no earlier copy exists.

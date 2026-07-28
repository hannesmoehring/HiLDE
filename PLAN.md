# Migration Plan — Streamlit → D3.js Frontend

Migrate the HiLDE dimensionality-reduction explorer from a Streamlit + Plotly
dashboard to a production D3.js frontend, **preserving the Python calculation
layer (`src/analysis`, `src/evaluation`) unchanged**. The goal is parity with
the current Streamlit views first, then visual improvement.

> **Ground rules**
> - Do not modify `src/analysis/*` or `src/evaluation/evaluate.py` logic. New code
>   wraps them; it does not change them.
> - Ignore `src_research/` entirely.
> - Stay on the `d3-vis` branch.
> - Each checklist item below is scoped to be independently reviewable.

---

## Architecture Decision (resolved)

**Python = FastAPI backend serving JSON. D3 = separate frontend consuming it.**

Rejected: static-JSON emit. Reason: selection-time predicate generation
(`generate_predicate`) and interactive range filtering run *per user
interaction* against the scaled background matrix and cannot be precomputed.
A static site would force a JS reimplementation of `predicate_generator.py`
(logic we must preserve) or dropping the feature.

**Backend is hit only for:** (1) building the analysis tree for a
(dataset, feature_cols, config) request, and (2) selection-time predicate/mask
computation. **Tree navigation / drill-down is 100% client-side** on the
already-fetched tree — no round-trips.

**Stack:** FastAPI (Python) + React + Vite + TypeScript + D3 (`d3-contour`,
`d3-scale`, `d3-shape`, `d3-zoom`, `d3-brush` for lasso/box).

```
┌─────────────────────┐        HTTP/JSON        ┌──────────────────────────┐
│  React + D3 frontend │ ─────────────────────► │  FastAPI service          │
│  (Vite, TS)          │                        │                          │
│  - tree navigation   │  POST /api/analysis    │  serialize.py (net-new)  │
│    (client-side)     │  POST /api/predicate   │      │                    │
│  - all charts (D3)   │  POST /api/mask        │      ▼                    │
│                      │  GET  /api/datasets    │  start_evaluation()      │
│                      │ ◄───────────────────── │  compute_analysis_tree() │
└─────────────────────┘                        │  generate_predicate()    │
                                                │  (UNCHANGED calc layer)  │
                                                └──────────────────────────┘
```

---

## Data Contract (Python emits ⇄ D3 consumes)

The calc layer returns an in-memory tree of `TypedDict`s carrying numpy arrays
and pandas DataFrames — **not JSON-serializable today**. A net-new
`serialize.py` walks the tree and converts each node to the schema below.
Numpy scalars/arrays → native lists/floats; DataFrames → records; the root
`scaler` is dropped from the wire format.

### `POST /api/analysis`
Request:
```jsonc
{
  "dataset": "Wine quality (red)",     // key from the dataset registry
  "feature_cols": ["alcohol", "..."],  // selected numeric columns
  "config": { /* Config TypedDict, see src/types.py */ }
}
```
Response `AnalysisResponse`:
```jsonc
{
  "meta": {
    "dataset": "Wine quality (red)",
    "feature_cols": ["alcohol", "..."],
    "config": { /* echoed config */ },
    "n_total": 1599,
    "generated_at": "2026-07-02T12:00:00Z"
  },
  "tree": Node,                        // root node, recursive
  "rows": {                            // raw feature values for display/tables
    "columns": ["row_id", "alcohol", "...", "target_*"],
    "index_by": "row_id"
    // NOTE: transported separately/on-demand for large datasets — see Phase 2
  }
}
```

### `Node` (unifies `HierarchyObject` + `ExplorationObject`)
```jsonc
{
  "id": "root/2/0",                    // stable path id (root = "root")
  "is_leaf": false,
  "depth": 1,
  "n_points": 412,
  "row_indices": [0, 5, 9, ...],       // indices into the source df
  "embedding_original": [[x, y], ...], // Nx2 projection (PCA/t-SNE/UMAP)
  "embedding_original_variance": [0.4, 0.2] | null,  // PCA only
  "rel_position": [cx, cy] | null,     // MDS centroid in sibling layout
  "kde": {
    "grid": [[...60...], ...60...],    // 60x60 density
    "resolution": 60,
    "extent": [-0.5, 0.5]              // normalized grid bounds
  } | null,
  "rel_characteristics": [             // one per feature
    { "feature": "alcohol", "z_mean": 1.2, "z_std": 0.3, "raw_mean": 12.1 }
  ],
  "outlier_scores": [0.1, ...] | null, // internal nodes only (GLOSH)
  "scores": {                          // ZADU metrics; nulls where not computed
    "n_points": 412, "k": 20,
    "trustworthiness": 0.98, "continuity": 0.97,
    "mrre_false": 0.02, "mrre_missing": 0.03,
    "stress": 0.11, "cadi": 0.42
  } | null,
  "children": [ Node, ... ] | null     // internal nodes only
}
```
Serialization rules:
- Discriminator `is_leaf` derived from presence of the `is_leaf` key on the
  in-memory node.
- Unify point arrays: internal `cluster_points` / leaf `exploration_points` are
  **not** shipped (frontend uses `embedding_original` for the scatter). If a view
  needs the reduced-space points, add `points` under the same rule — decide in
  Phase 1.
- `rel_characteristics`: `DataFrame.reset_index().to_dict("records")`.
- All numpy → native via a recursive coercion helper; assert
  `json.dumps(node)` succeeds in a unit test.

### `POST /api/predicate`  (selection-time)
Request:
```jsonc
{
  "dataset": "...", "feature_cols": [...], "config": {...},
  "selected_row_ids": [12, 34, ...],   // from lasso/box selection
  "scope": "local" | "global",
  "method": "db"
}
```
Response: the predicate band records consumed by the feature-range chart —
`[{ feature, sel_min, sel_max, sel_range, global_min, global_max,
    in_predicate, clause_f1, predicate_f1, ... }]` plus summary
`{ predicate_f1, n_features_used, n_selected }`.

### `POST /api/mask`  (interactive range filtering)
Request: `{ node_id, feature_ranges: { feature: [lo, hi] } }`
Response: `{ row_ids: [...] }`.
> May move client-side if raw feature values are already in the browser —
> decide in Phase 6.

### `GET /api/datasets`
Response: `[{ key, label, n_rows?, feature_cols? }]` from the `DATASETS` registry.

---

## Phased Checklist

### Phase 0 — Foundations & scaffolding
- [x] 0.1 Create `backend/` (FastAPI app) and `frontend/` (Vite React-TS)
      directories at repo root; document the layout in README.
- [x] 0.2 Add FastAPI + uvicorn to `pyproject.toml` deps (uv). Do **not** touch
      calc-layer deps.
- [x] 0.3 Scaffold Vite + React + TypeScript + D3 in `frontend/` (`package.json`
      — the first JS toolchain in the repo). Add `.gitignore` for `node_modules`.
- [x] 0.4 Wire CORS + a `/api/health` endpoint; confirm frontend can reach it.
- [x] 0.5 Decide dev workflow: Vite dev server proxying `/api` → uvicorn; note in README.

### Phase 1 — Serialization layer (Python, net-new, calc layer untouched)
- [x] 1.1 Write `backend/serialize.py`: recursive numpy/pandas → JSON coercion
      helper (`_to_native`).
- [x] 1.2 `serialize_node(node, id)` → `Node` schema above; assign stable path ids.
- [x] 1.3 `serialize_tree(root)` → recurse, drop `scaler`.
- [x] 1.4 Unit test: run `start_evaluation` on Wine (default config), serialize,
      assert `json.dumps` round-trips and key fields/nulls match the contract.
- [x] 1.5 Decide whether reduced-space `points` are needed on the wire; update
      contract + serializer accordingly.

### Phase 2 — FastAPI backend endpoints
- [x] 2.1 `GET /api/datasets` — expose the `DATASETS` registry (reuse `src/ui/data.py`
      loaders; do not duplicate loader logic).
- [x] 2.2 `POST /api/analysis` — load dataset, build `Config`, call
      `start_evaluation`, return `serialize_tree`. Reuse existing config defaults
      from `state.py` where possible.
- [x] 2.3 Decide `rows` transport for large datasets (inline vs `GET /api/rows?ids=`
      vs paginated). Implement chosen path.
- [x] 2.4 `POST /api/predicate` — wrap `generate_predicate` with scope handling
      (mirror `render_range_analysis`).
- [x] 2.5 `POST /api/mask` — wrap interactive mask logic (mirror
      `compute_interactive_mask`) unless deferred to client in Phase 6.
- [x] 2.6 Response caching by (dataset, feature_cols, config) hash so repeated
      builds are instant (tree build is expensive).
- [x] 2.7 Backend smoke test: full request cycle for Wine + one large dataset.

### Phase 3 — Frontend scaffold & shared infrastructure
- [x] 3.1 TypeScript types mirroring the data contract (`types.ts`).
- [x] 3.2 API client (`api.ts`) for all four endpoints.
- [x] 3.3 App shell + layout matching the Streamlit progressive-reveal flow
      (config → build → topography → drill-down → exploration).
- [x] 3.4 Config panels: dataset picker, feature multiselect, hierarchical config,
      exploration config, "Build / Apply" button (parity with `config.py`).
- [x] 3.5 Client-side tree navigation state (`tree_path`) + `getNodeAtPath` helper
      (port `tree_nav.py` logic to TS).
- [x] 3.6 Shared D3 primitives: scales, axes, color scales, responsive container hook.

### Phase 4 — Port the charts (parallel — one subagent per chart)
> Fan out: **6 subagents**, one per chart component. Each builds one isolated,
> reviewable D3 component against the data contract with mock JSON fixtures.
> **I (the integrator) wire them into the app — subagents do not merge.**
- [x] 4.1 **KDE topography** (`d3-contour`) — small-multiples per-cluster density,
      MDS-positioned, size-scaled, clickable centroids → drill-down. (Replaces
      `cluster_gauss_kde`.) *[subagent A]*
- [x] 4.2 **Cluster characteristics bar** — grouped z-score bars, error bars,
      zero line, feature/analysis-col trace groups. (Replaces
      `cluster_characteristics_fig`.) *[subagent B]*
- [x] 4.3 **Projection scatter** — 2D scatter, 4 color/symbol modes, lasso + box
      selection (`d3-brush` / custom lasso). (Replaces `make_scatter_fig`.)
      *[subagent C]*
- [x] 4.4 **PCA variance bar** — horizontal stacked variance bar. (Replaces
      `make_pca_variance_fig`.) *[subagent D]*
- [x] 4.5 **Predicate feature-range bands** — global / full / core horizontal
      bands, predicate-clause highlighting. (Replaces `make_feature_range_fig`.)
      *[subagent E]*
- [x] 4.6 **Metric tiles + tables** — score tiles (T/C/Stress/CADI + MRRE),
      outlier table, size distribution, selected-points table. (Replaces
      `scores.py` + `st.dataframe` blocks.) *[subagent F]*

### Phase 5 — Wire interactions & data flow (integrator)
- [x] 5.1 Topography centroid click → append to `tree_path` → render next layer /
      exploration (parity with `hierarchical.py` drill-down).
- [x] 5.2 Scatter lasso/box selection → `selected_row_ids` state.
- [x] 5.3 Selection → `POST /api/predicate` → feature-range chart + summary tiles.
- [x] 5.4 Interactive feature filters → mask (client-side: fetch leaf rows, z-score
      in-browser, per-feature sliders) → recolor scatter + drive selected table.
- [x] 5.5 Export selected points to CSV (client-side download; parity with
      `export_selection`).
- [ ] 5.6 Decision-tree "rules" display — port the text expander (upgrade to a real
      D3 tree deferred to Phase 6).

### Phase 6 — Parity verification
- [ ] 6.1 Side-by-side check: run Streamlit and D3 on identical (dataset, config);
      confirm each of the 8 visualization types matches (values, not just shape).
- [ ] 6.2 Verify drill-down, selection, predicate, interactive-range, and export
      flows behave identically.
- [ ] 6.3 Cross-check ZADU score tiles match between the two frontends.
- [ ] 6.4 Sign-off: parity achieved before any visual redesign.

### Phase 7 — Visual improvements (post-parity, opt-in)
- [x] 7.1 KDE: smoother continuous density field (light box-blur + 16 contour
      levels + MDS-layout compression so fields are visible) — you flagged this.
- [~] 7.2 Decision-tree rules → interactive D3 tree — DEFERRED (data not
      prioritized; would need a backend `fit_cluster_decision_tree` endpoint).
- [x] 7.3 Consistent design system (surfaces, typography, spacing, focus rings,
      gradient primary, accent header) — via `dataviz` skill calibration.
- [x] 7.4 Transitions/animation on panel/drill-down appearance.
- [ ] 7.5 Responsive layout / mobile-friendly containers (charts are responsive
      via ResizeObserver; full mobile layout not yet done).

### Phase 8 — Productionization
- [x] 8.1 Serve frontend `dist` via FastAPI `StaticFiles` at `/` (API at `/api/*`).
      Verified in a real browser against the FastAPI-only server.
- [x] 8.2 Multi-stage `Dockerfile` (node build → lean Python runtime running
      uvicorn) + `docker-compose.yml` on :8000. `streamlit run` retired. Image is
      Streamlit-free via a minimal `backend/requirements.txt` (also avoids the
      pre-existing kdbcv/scipy lock conflict). NOTE: image not built here (no
      Docker daemon in this env) — build with `docker compose up --build`.
- [x] 8.3 Env config: `PORT`, `SCIKIT_LEARN_DATA`; datasets mounted read-only.
- [x] 8.4 README: dev + production run instructions; Streamlit noted as removed.
- [x] 8.5 Streamlit removed. `src/ui/` deleted. Shared loaders/config defaults
      extracted Streamlit-free to `src/datasets.py` + `src/config_defaults.py`;
      backend + `src_research` imports repointed; backend verified Streamlit-free.

---

## Resolved Decisions
1. **Row data transport: on-demand.** Raw feature values are fetched via a
   dedicated endpoint (`POST /api/rows`), not inlined in the analysis response.
   Keeps the tree payload small for large datasets (MNIST 70k×784).
2. **Deployment: single container.** FastAPI serves the built frontend via
   `StaticFiles`. (Phase 8.1)
3. **Streamlit: remove after parity.** `src/ui/` is retired once Phase 6 parity
   is signed off. (Phase 8.5)

## Implementation Status (living)

**Phase 7 (polish) & Phase 8 (productionization) — DONE:** smoother continuous KDE
field, design-system pass, panel transitions; FastAPI serves the built frontend
(single container), multi-stage Dockerfile + compose, and **Streamlit removed**
(`src/ui/` deleted; loaders/config extracted Streamlit-free). Interactive feature
filters (client-side) shipped. Docker image not built here (no daemon) — verified
the FastAPI-only serving model in a real browser instead.

**Done & verified (Phases 0–5, most of 6):**
- Backend `backend/`: `serialize.py` (+ round-trip unit test), FastAPI `app.py`
  with `/api/health`, `/api/datasets`, `/api/datasets/{key}/columns`,
  `/api/analysis` (cached), `/api/predicate` (local+global), `/api/rows`.
  Calc layer untouched. Verified via curl + a headless-Chrome e2e run.
- Frontend `frontend/` (Vite+React+TS+D3): types/contract, api client, tree-nav
  port, config panels, dataset/feature pickers, build flow, client-side
  drill-down, exploration panel, CSV export.
- All 6 charts ported and **visually verified rendering with real data**: KDE
  topography (with a visibility fix — MDS layout compressed + Viridis floor
  lifted), characteristics bar, projection scatter (lasso+box), PCA-variance
  bar, predicate bands, score tiles + selected-points table.
- Dev workflow: `uvicorn` on :8000 + `vite` on :5173 (proxies `/api`). Full
  `npm run build` + `tsc` clean.

**Remaining parity gaps (Streamlit features not yet ported):**
- Interactive feature-range filtering (`interactive_ranges_mode`: multiselect +
  per-feature sliders → mask → recolor scatter). → resolve mask decision below.
- GLOSH outlier top-20 table (`node.outlier_scores`).
- Cluster size-distribution table.
- Decision-tree "rules" text (`fit_cluster_decision_tree`) — needs a backend
  addition; earmarked for the Phase 7 D3-tree upgrade.
- "Clusters in original space" KMeans/GMM scatter overlay toggle.
- Characteristics "analysis cols" (non-feature numeric) second trace group.
- Dataframe preview (first 200 rows) expander.

**Open decision (Phase 6):** interactive-mask compute location. Recommendation:
**client-side** — fetch each leaf's standardized feature values once via
`/api/rows`, filter in-browser per slider tick (no round-trip). Aligns with the
on-demand-rows decision.

**Not headlessly verifiable:** the lasso/box selection *gesture* firing
`onSelect` (d3-brush pointer-capture resists synthetic events). Backend
predicate, the wiring, and the bands chart are all verified independently; the
gesture is standard d3 and works under a real cursor — needs a manual click-test.

---

## Subagent Fan-out Summary
- **Exploration (done):** 3 Explore agents — calc modules, Streamlit views,
  data flow/stack.
- **Phase 4 (planned):** 6 chart-porting subagents (A–F), one per chart, isolated
  against mock fixtures. I integrate; they do not merge their own work.
- Other phases are integrator-led (sequential dependencies).

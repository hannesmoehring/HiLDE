# Frontend — React + D3 client

The interactive explorer. React owns state and layout; every visualisation is hand-drawn
D3 into SVG (no chart library). TypeScript throughout, built by Vite.

For install and run instructions see the [root README](../README.md); for the API this
client talks to, [`backend/README.md`](../backend/README.md). This document is the code
map.

---

## Stack and commands

React 18 · D3 7 · TypeScript 5 · Vite 5. IBM Plex Sans/Mono are self-hosted via
`@fontsource`, so a build has **no external origins** and works offline.

```bash
npm install
npm run dev        # Vite dev server on :5173, proxying /api -> 127.0.0.1:8000
npm run build      # tsc -b && vite build  ->  dist/
npm run typecheck  # tsc -b --noEmit
```

In production nothing runs here: `backend/app.py` mounts `dist/` at `/` and serves the UI
from the same uvicorn process. `uv run host.py` rebuilds `dist/` when it is missing or
older than `src/`.

> `npm run typecheck` reports `TS6310` on `tsconfig.node.json` — a pre-existing
> interaction between project references and `--noEmit`. `npm run build` is clean.

---

## Where state lives

`App.tsx` owns everything and passes it down; there is no store, no context, no router.

| State | Meaning |
|---|---|
| `datasetKey`, `columns`, `featureCols` | the current dataset and its selected feature columns |
| `config` | the pipeline knobs, sent verbatim as the request's `config` |
| `analysis` | the entire tree, fetched once per build |
| `treePath` | `number[]` — the drill-down path, child index per layer |
| `exploreWhole` | *Explore entire layer* is active for the current path |
| `useCache`, `mode` | the run-cache toggle and whether the server has one |

Cluster movement is the one exception to "`App.tsx` owns everything": its state machine
lives in `movement.ts::useMovement`, because it is a five-phase machine with in-flight
requests and half a dozen clear-conditions, and folding that into the flat `useState`
block above would have buried it. `App.tsx` still owns the hook — it just does not own the
transitions.

**The tree arrives once and navigation is local.** `treeNav.ts::getNodeAtPath` walks the
already-fetched tree, so clicking a cluster costs nothing. Only per-selection questions —
predicate, characteristics, targets, raw rows, image pixels — go back to the server.

`exploreWhole` is cleared by every navigation, so it can never outlive the path it was set
for. It is the only way to reach the rows HDBSCAN labelled noise, which belong to no child
cluster.

Building is a polled job, not one long request: `api.ts::runAnalysis` POSTs to
`/api/analysis`, and either gets a cached payload inline or a `job_id` it polls until done.

---

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│ topbar — brand, intro video, run meta                        │
├───────────────┬──────────────────────────────────────────────┤
│ ConfigPanel   │  Layer 1:  ClusterScatter    │  LayerSide     │
│ (collapsible  │            OutlierPanel      │  (tiles +      │
│  rail)        │  Layer 2:  ClusterScatter    │   chars OR     │
│               │            OutlierPanel      │   predicate)   │
│               ├──────────────────────────────┴────────────────┤
│               │  ExplorationPanel — ProjectionScatter +       │
│               │  [Predicate | Characteristics | Ranges]       │
└───────────────┴───────────────────────────────────────────────┘
```

One layer view per hierarchical level, then the exploration panel on whichever node the
path resolves to.

---

## Files

### Entry and contract

| File | Role |
|---|---|
| `main.tsx` | React root, font imports, `styles.css`. |
| `App.tsx` | All state, the layer loop, and the decision of which node the exploration panel opens on. |
| `ErrorBoundary.tsx` | React 18 unmounts the whole root on an uncaught render throw, which would take the topbar, rail and Build button down with it — recovery would be F5, losing the tree and the path. This keeps a failure to a message. |
| `api.ts` | Typed client for every endpoint, including the analysis-job poll. |
| `types.ts` | The data contract. **Mirrors `backend/serialize.py`** — change one and change the other. |
| `config.ts` | `DEFAULT_CONFIG`, mirroring `src/config_defaults.py`, except `method`: the app opens on UMAP while the Python default stays PCA (a research harness reads its DR method off that default). Every request carries `method`, so this is what the app actually runs. |
| `treeNav.ts` | Client-side drill-down over the fetched tree. |
| `movement.ts` | The cluster-movement state machine (`useMovement`) plus the pure geometry it needs: screen→embedding inversion, target resolution, strength re-expression, and the stale-response guard. Kept out of the chart so all of it is testable without a DOM. |

### Components

| File | Role |
|---|---|
| `ConfigPanel.tsx` | Dataset, feature checkboxes, pipeline knobs, *Build & Apply*. Collapses to a narrow strip. |
| `LayerSide.tsx` | A layer's side column: DR-quality tiles, then **one of two accounts** of the selected cluster — its characteristics, or the predicate separating it from the space it was selected out of. The choice is per layer, so different depths can show different accounts at once. |
| `ExplorationPanel.tsx` | The node's own embedding with lasso/box selection, feeding three tabs: *Predicate*, *Characteristics*, *Ranges*. Also the selected-points table and CSV export. |
| `RangeFilters.tsx` | The *Ranges* tab — the inverted interaction: pick columns, slide a `[min, max]` window over each, and points inside *every* window become the selection. Windows are in raw column units. |
| `OutlierPanel.tsx` | GLOSH scores for one internal layer, folded into a `<details>`; the closed summary carries the headline numbers. Clicking a row reveals that point's values and rings it in the projection above. |
| `ClusterMovementPanel.tsx` | Two pieces. `MovementStartRow` is the one button that sits under `LayerSide` in the side column. `ClusterMovementPanel` is the full-width row a layer card grows under the plot *and* the characteristics once a movement runs: three columns (route + strength slider with a `rec.` tick + Cancel/CSV/JSON · stat tiles at the server's strength + the UMAP sparkline · the feature-change table sorted by standardized magnitude, held-fixed columns as a footer line), then one-line warning chips, then a collapsed *About this preview*. Every tile label and chip carries its sentence as a `title`, and the same sentences are listed in the About block, so nothing is hover-only. The "click a destination" prompt is the chart's own status line above the plot. Deliberately has **no *Apply*** yet — this is a preview, not an edit. |
| `PointImage.tsx` | One row drawn as its image. The server sends raw greyscale pixels, so the canvas is drawn at native size and blown up with `image-rendering: pixelated` — an 8×8 digit stays a grid of squares. |

**Why `target_*` columns are offered in *Ranges* but never in the predicate.** A range
filter is a question the analyst asks, not an explanation the tool induces, so slicing on
a label explains nothing away — it just asks *where do the points with this label sit?*
Targets stay in their own hue throughout so the two never blur together.

### Charts

Six of these are ports of the removed Streamlit UI, labelled **A–F**; `charts/props.ts` is
their shared contract and names the Python source each replaces. Those references are
deliberate provenance — the files they name are not in this repository.

| File | Chart |
|---|---|
| `ClusterScatter.tsx` | **A** — parent node's embedding, coloured by child cluster, HDBSCAN noise as grey ×, clickable legend and centroid labels. Click to drill in. |
| `CharacteristicsBar.tsx` | **B** — z-score bars per column with ±`z_std` error bars and sign-based colouring. |
| `ProjectionScatter.tsx` | **C** — the exploration embedding (equal aspect) with lasso + box selection. |
| `PcaVarianceBar.tsx` | **D** — compact stacked explained-variance strip that rides in the projection toolbar. |
| `PredicateBands.tsx` | **E** — one row per feature, each normalised to its own global range: a faint track, a translucent full band (RCM 1.0) and a solid core band (RCM 0.9). Clause features are indigo and sort to the top in the order the greedy loop added them, each carrying its ΔF1 when added and its match count. |
| `TargetBands.tsx` | **E2** — E's row geometry for the held-out `target_*` columns, drawn entirely in the target hue so an indigo band always means "predicate clause" and a teal one never does. |
| `ScoreTiles.tsx` | **F** — trustworthiness / continuity / stress / CADI. The bar under each value is a *quality* reading, so longer and cooler is better on every tile: the two distortion measures are inverted before they reach a bar. |
| `OutlierHistogram.tsx` | GLOSH score distribution over the fixed range `[0, 1]`, so shapes are comparable between layers. New here, not a port — its props are declared in the file rather than in `props.ts`. |
| `theme.ts` | Design tokens. Neutral shell, ink type, hairline rules; **colour is reserved for data**. Keep in sync with the custom properties in `styles.css`. |

**How `PredicateBands` shows the construction.** Row order *is* the RCM 1.0 greedy order —
selected clauses by ascending `predicate_step`, everything else by standalone F1. A right
gutter gives each clause the F1 it bought at its step and the number of background points
the conjunction still matched there (`+0.214 · 47`); the gain deliberately does not touch
band width or opacity, which already carry the interval and the membership. A displayed
`+0.000` is genuine — the greedy stopping tolerance is `1e-6` — and that is exactly why
the exact match count sits beside it. The tooltip adds the F1 transition
(`before → after`), precision/recall after the step, and the renamed **Standalone F1**;
unselected features show "Not selected" and their standalone F1 only. Because the RCM 0.9
run is a *separate* greedy pass whose membership and order both differ, it gets its own
tooltip block instead of being folded into the 1.0 numbers. The predicate summaries in
`LayerSide.tsx` and `ExplorationPanel.tsx` show both runs as trajectories
(`F1 0.18 → 0.54`, `Core F1 0.18 → 0.51`); the left-hand number is the empty predicate —
labelled "no clauses (all 150 points)" in its `title`, with the real background size from
the summary's `n_background` — and is a function of the selection and background sizes,
never a score.

### Hooks

`useResize.ts` — `ResizeObserver` wrapper giving each chart its container's measured size.
`useDebounced.ts` — trails a fast-changing value; a range slider fires on every pixel of a
drag, and the work behind it (re-scan the node, refetch rows) is far too heavy to run at
that rate.

### `fixtures/`

`analysis_iris.json`, `predicate_iris.json`, `movement_iris.json` — captured `/api/analysis`,
`/api/predicate` and `/api/movement` responses for a two-layer PCA build on Iris. No source
file imports any of them; they are kept as a reference sample of the wire format.

The analysis and movement fixtures come from ONE live run, captured together by
`scripts/capture_movement_fixture.py` (standard library only) against a running backend:

```bash
PYTHONPATH=. uv run uvicorn backend.app:app --port 8000
python frontend/scripts/capture_movement_fixture.py --base-url http://127.0.0.1:8000
```

The movement sample is a PCA point target on `root/1` (source child 0, aimed at child 1's
visible centroid). The script refuses to write unless every ghost equals the serialized
source point plus `visible_displacement` — the shared-translation invariant a hand-written
sample cannot be trusted to satisfy — so the two files always agree with each other.

---

## Things worth knowing

- **Axis labels follow the run, not the knob.** The variance strip and the axis labels
  describe the coordinates on screen, so they key off the method the tree was *built*
  with. Keyed off the live Method knob instead, flipping UMAP → PCA would relabel UMAP
  coordinates "PC1/PC2" with nothing on screen to tell — the backend only emits
  `explained_variance_ratio` for PCA, so the strip is simply absent under the UMAP default.
- **A selection covering the whole node is refused** in both the predicate and the
  characteristics tabs, because the comparison would be self-referential.
- **The *Ranges* tab is capped.** It fetches every row × every offered column to compute
  bounds and histograms and refuses above 5 000 000 cells — the MNIST root would be an
  ~800 MB JSON body, past V8's string limit. Cluster-sized nodes are far inside it.
- **Requests are not cancelled.** There is no `AbortController`; a superseded response is
  discarded on arrival rather than aborted at the socket, so a slow build keeps running
  server-side after you navigate away.
- **Movement coordinates are always the *visible* ones.** The server refits the parent's
  reducer to answer a movement and aligns that refit to the embedding already on screen, so
  every 2D number the client sees — target, ghosts, displacements, distances — is in
  `embedding_original` units. Fitted-reducer coordinates never reach the frontend.
- **The strength slider costs nothing under PCA and one request under UMAP.** PCA is
  linear, so `previewAtStrength` interpolates ghosts exactly from the returned
  `visible_displacement`. UMAP's transform is not linear, so its ghosts are only correct at
  a strength the server actually projected: the slider stays instant, and a re-request
  goes out 350 ms after it settles. The re-request deliberately does not blank the panel —
  hiding the ghosts on every nudge would remove exactly what is being compared.
- **`metrics` describe the strength the *server* answered at**, not the live slider
  position; the panel labels them with it and flags the mismatch rather than silently
  showing a stale distance as current.
- **A movement survives drilling deeper, not stepping sideways.** The preview belongs to
  one parent projection. Drilling into a child keeps that projection on screen in the
  layer stack, so the movement stays; navigating to a sibling, back up, or to another
  dataset removes the projection and cancels the movement (`nodeIsOnPath` in
  `movement.ts`). Decision of 06.09.
- **There are no frontend tests.** Verification is `npm run build` plus the app itself.

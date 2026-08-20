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

### Components

| File | Role |
|---|---|
| `ConfigPanel.tsx` | Dataset, feature checkboxes, pipeline knobs, *Build & Apply*. Collapses to a narrow strip. |
| `LayerSide.tsx` | A layer's side column: DR-quality tiles, then **one of two accounts** of the selected cluster — its characteristics, or the predicate separating it from the space it was selected out of. The choice is per layer, so different depths can show different accounts at once. |
| `ExplorationPanel.tsx` | The node's own embedding with lasso/box selection, feeding three tabs: *Predicate*, *Characteristics*, *Ranges*. Also the selected-points table and CSV export. |
| `RangeFilters.tsx` | The *Ranges* tab — the inverted interaction: pick columns, slide a `[min, max]` window over each, and points inside *every* window become the selection. Windows are in raw column units. |
| `OutlierPanel.tsx` | GLOSH scores for one internal layer, folded into a `<details>`; the closed summary carries the headline numbers. Clicking a row reveals that point's values and rings it in the projection above. |
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
| `PredicateBands.tsx` | **E** — one row per feature, each normalised to its own global range: a faint track, a translucent full band (RCM 1.0) and a solid core band (RCM 0.9). Clause features are indigo and sort to the top. |
| `TargetBands.tsx` | **E2** — E's row geometry for the held-out `target_*` columns, drawn entirely in the target hue so an indigo band always means "predicate clause" and a teal one never does. |
| `ScoreTiles.tsx` | **F** — trustworthiness / continuity / stress / CADI. The bar under each value is a *quality* reading, so longer and cooler is better on every tile: the two distortion measures are inverted before they reach a bar. |
| `OutlierHistogram.tsx` | GLOSH score distribution over the fixed range `[0, 1]`, so shapes are comparable between layers. New here, not a port — its props are declared in the file rather than in `props.ts`. |
| `theme.ts` | Design tokens. Neutral shell, ink type, hairline rules; **colour is reserved for data**. Keep in sync with the custom properties in `styles.css`. |

### Hooks

`useResize.ts` — `ResizeObserver` wrapper giving each chart its container's measured size.
`useDebounced.ts` — trails a fast-changing value; a range slider fires on every pixel of a
drag, and the work behind it (re-scan the node, refetch rows) is far too heavy to run at
that rate.

### `fixtures/`

`analysis_iris.json`, `predicate_iris.json` — captured `/api/analysis` and `/api/predicate`
responses for a two-layer PCA build on Iris. No source file imports them; they are kept as a
reference sample of the wire format.

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
- **There are no frontend tests.** Verification is `npm run build` plus the app itself.

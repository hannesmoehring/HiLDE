# Backend — the FastAPI service

The HTTP layer over the calculation pipeline: routes, the JSON data contract, background
builds and the run cache.

For install and run instructions see the [root README](../README.md); for the method
itself, [`src/README.md`](../src/README.md). This document is the code map for
`backend/`: what each module is for and why it is shaped the way it is.

---

## Layering

```
frontend/  ──HTTP──▶  backend/  ──imports──▶  src/
                                                ▲
                       src_research/  ──────────┘
                       scripts/checks/  ────────┘
```

**`src/` never imports from `backend/` or `frontend/`.** It is a plain Python library:
give it a DataFrame, a list of feature columns and a `Config`, and it returns a tree of
TypedDicts. The API is a thin wrapper over it, and the research harnesses call the exact
same entry point the server calls — an experiment and a served build run identical code.

`src/` also carries no Streamlit dependency. `config_defaults.py` and `datasets.py` were
extracted from the removed Streamlit UI for that reason; their docstrings say so.

---

## Request lifecycle

A tree build (UMAP/MDS + HDBSCAN + ZADU scoring) can run for minutes, past the ~100 s
response timeout a proxy such as Cloudflare enforces. So `/api/analysis` never blocks:

```
POST /api/analysis
   │
   ├─ cache hit  ────────────────────────────────▶  {status: "done", cached: true}
   │     in-process LRU (8 payloads), then the
   │     on-disk run cache when hosting
   │
   └─ miss ─▶ jobs.submit()  ─▶ worker thread ─▶  {status: "running", job_id}
                   │                                        │
                   │                            client polls GET /api/analysis/jobs/{id}
                   └─ keyed by the same signature           │
                      as the cache, so a reload   ◀─────────┘
                      re-attaches instead of              {status: "done" | "error"}
                      launching a second build
```

The cache key is `(dataset, feature_cols, config)` — **not the code**. After changing
anything under `src/` or `backend/`, `rm -rf ../.cache/hilde_runs` or you will be served
trees built by the previous version.

Navigation is *not* a request. The whole tree ships in one payload and drill-down happens
in the browser; only per-selection questions (predicate, characteristics, targets, raw
rows, image pixels) come back to the server.

---

## HTTP API

| Method | Path | Answers |
|---|---|---|
| `GET` | `/api/health` | liveness probe |
| `GET` | `/api/mode` | whether the run cache is active (drives the "Cached" banner), and whether the server is cache-only |
| `GET` | `/api/cached-runs` | the datasets with a stored run, and how many each — the cache-only UI's dataset list |
| `GET` | `/api/cached-runs/{dataset}` | that dataset's stored runs, compact and gzipped (see `run_index.encode`) — the cache-only UI's run picker |
| `GET` | `/api/datasets` | the loader registry |
| `GET` | `/api/datasets/{key}/columns` | column names + the default feature selection |
| `GET` | `/api/datasets/{key}/image/{row_id}` | raw greyscale pixels for one row |
| `POST` | `/api/analysis` | start (or serve from cache) a tree build |
| `GET` | `/api/analysis/jobs/{job_id}` | poll a running build |
| `POST` | `/api/movement` | a counterfactual movement preview for one cluster |
| `GET` | `/api/movement/jobs/{job_id}` | poll a running movement (UMAP refits only) |
| `POST` | `/api/counterfactual/apply` | write a previewed movement into a counterfactual copy of the dataset |
| `GET` | `/api/counterfactual/jobs/{job_id}` | poll an Apply that needed a UMAP refit |
| `GET` | `/api/counterfactual/{cf_id}` | the edit chain behind a counterfactual key |
| `GET` | `/api/counterfactual/{cf_id}/rows.csv` | the changed rows, original and counterfactual values |
| `POST` | `/api/predicate` | induce an axis-aligned predicate for a selection |
| `POST` | `/api/characteristics` | z-scored column means for a selection |
| `POST` | `/api/targets` | held-out `target_*` values for a selection |
| `POST` | `/api/rows` | raw column values for a set of row ids |

Request bodies are the `BaseModel` classes at the top of `app.py`. The four
selection endpoints take `row_indices` (the node's rows, dataset positions) plus
`selected_local_indices` (offsets *into* that list) — the client already holds both from
the tree, so the server never has to remember which node is open.

> **Known gap:** `config` is a free-form dict overlaid onto the defaults with no schema.
> A misspelled key is accepted with HTTP 200 and silently ignored. See *Known limitations*
> in the root README.

---

## Cluster movement

`POST /api/movement` answers one question: *if this cluster were to sit over there
instead, what would have to change about it?* It is a *counterfactual preview* — the
dataframe, the analysis tree and the cached payload are all read-only, and nothing is
written back. Turning the preview into data is a separate, explicit step: *Apply*, below.

**Shared translation.** Every source point receives the *same* feature-space
displacement, `X' = X + strength · Δ`. One displacement means every pairwise distance
inside the cluster is preserved: the cluster moves without morphing. Fitting a separate
change per point would produce a better-looking overlap and a meaningless
recommendation.

**Targets.** A click resolves by precedence: another cluster's centroid label or legend
entry → that cluster; a point belonging to another cluster → that cluster; the nearest
other-cluster point within 10 screen pixels (measured after zoom/pan) → that cluster;
anything else → a free 2D point. Clicking the source cluster resolves to nothing.

**PCA** is linear, so a free point has an exact answer — but not a unique one. The
displacement is the minimum-norm solution `pinv(P_mutable) @ d_y`, one canonical answer
out of a `(d−2)`-dimensional family, and *minimum* is measured in the reducer's **input**
space: root-standardized units when the run normalized, raw units when it did not. That
choice changes which features the recommendation prefers, so the panel states it in
words. Cluster targets (`Δ = μ_t − μ_s`) are unique and carry no such caveat.

**UMAP** has no exact inverse. A free point is inverted with `inverse_transform`
(`estimator: "umap_inverse"`); if that raises or returns non-finite values, the
destination falls back to an inverse-distance-weighted mean of the 10 nearest displayed
points (`"neighbor_interpolation"`). Either way the result is an *estimate*, and the
response says so. A cluster target never inverts anything — it uses the destination
cluster's observed feature-space centroid.

**The strength trajectory.** Because UMAP's out-of-sample transform is neighbour-driven,
projected distance against strength can plateau and then jump; intermediate strengths are
the least trustworthy part of the preview. So the endpoint returns the whole
`strength_trajectory` grid, not just the recommended strength. A real Iris trajectory is
monotone but far from smooth — nearly flat to α≈0.3, then 4.4 units in one step. If no
positive strength improves on zero, no recommendation is returned at all.

**Frozen projection.** The preview always uses the reducer fitted for the visible parent
node, and the embedding is *never* recomputed after the movement — a fresh fit could
rotate, reflect or reshape it, and the preview would no longer be comparable to what is
on screen. A refit that fails to align with the stored embedding is refused with 409
rather than drawn.

**The alignment gate** (`ALIGNMENT_RMSE_MAX = 0.05`) is a product decision, not a derived
tolerance. What it bounds, exactly: the root-mean-square residual of the similarity
alignment over the node's *existing* points, normalized by the visible cloud's RMS
radius — how far, on average, the refit places the points already on screen from where
they are drawn, as a fraction of the plot's own scale. It does **not** bound individual
residuals (single points can exceed 5 %; the response reports the worst one as
`alignment_max_residual` next to `alignment_rmse`), and it says nothing about the error
of *new* counterfactual points, which the alignment never saw. A same-process refit lands
at ~1e-16 (PCA) and well under 1e-3 (UMAP), so the gate is a tripwire; it is never
widened.

**`feature_centroid_distance_after` is `null` for point targets.** A point target's
feature-space destination is *defined* as `μ_source + Δ`, so the remaining distance would
be `(1−α)·‖Δ‖` by construction — exactly 0 at full strength whether or not the click was
reached. Only cluster targets, whose destination is an observed centroid, report it.

**Mutable features.** Everything except `row_id` and `target_*`. A `target_*` column that
was manually selected as an analysis feature stays immutable anyway, and the response
reports that the destination may then not be reachable exactly. Immutable entries of `Δ`
are exactly zero.

**`out_of_range_fraction`** is the fraction of *moved* source points with at least one
*mutable* feature outside the **full dataset's** observed raw `[min, max]`. Raw units,
whole dataset as the reference. Above 0.05 the panel warns: the counterfactual is leaving
the region the data actually covers.

**Sampling.** Previews above `MAX_MOVEMENT_PREVIEW_POINTS` (5000) are deterministically
subsampled and badged; the strength search uses at most
`MAX_STRENGTH_SEARCH_POINTS` (1000). Feature centroids are always computed from the
*complete* cluster, never the sample, so the recommendation itself is unaffected.

**Supported reducers: PCA and UMAP only.** t-SNE has no out-of-sample transform at all,
and MDS only reconstructs from a distance matrix, so neither can say where a moved point
would land. Both are named in the rejection and in the disabled UI state.

**What movement does not promise.** It does not guarantee that a rebuild would make
HDBSCAN merge the two clusters. It moves a cluster in feature space and shows where the
frozen projection would put it; clustering is a separate, non-linear decision over the
whole dataset, and a nearby projection is not the same thing as a merged cluster. The
preview is also not causal — it says what would have to differ, not what would cause it.

**One UMAP at a time.** UMAP (and pynndescent under it) runs Numba parallel regions, and
Numba's default `workqueue` threading layer is not thread-safe: two threads entering a
parallel region at once make it *abort the process*. TBB has no macOS arm64 wheel and the
OpenMP layer needs a `libomp` that is not shipped, so every UMAP fit, transform and inverse
in the process goes through `NUMBA_PARALLEL_LOCK` (`src/analysis/dim_reducer.py`). Builds on
job threads, refits on movement-job threads and warm UMAP movements on request threads
therefore queue rather than overlap; a UMAP slider re-request issued during a build waits
for it. PCA movements are unaffected.

**Coordinates.** Every 2D quantity crossing the wire — target, preview points,
displacements, distances — is in *visible* coordinates, the units of the parent's
serialized `embedding_original`. The refitted reducer's own coordinates never leave the
server.

**Status codes.** `400` = the request cannot describe a movement (unknown/leaf/childless
node, bad child index, source == destination, non-finite coordinates, strength outside
`[0, 1]`, unsupported reducer). `409` = well-formed but unanswerable against the run the
client holds (`analysis_id` mismatch, payload gone, payload predating `effective_config`,
or a refit that will not align). Validation runs synchronously, so these are always HTTP
errors and never a job failure discovered one poll later.

> **Rebuild required for older runs.** Movement refits read `meta.effective_config` — the
> configuration *after* `compute_analysis_tree` mutated it in place, which is what the
> visible embeddings were actually fit with. `meta.config` holds only the knobs the client
> sent (often 2 of 12), so it cannot stand in. Payloads written before this field existed
> answer 409; that is what `meta.schema_version` is for.

---

## Counterfactual sessions (Apply)

`POST /api/counterfactual/apply` takes the same fields as `/api/movement` with a required
strength in `(0, 1]`, and **writes the previewed feature deltas at that strength into a
server-side counterfactual copy of the dataset**: one rigid translation per moved row,
mutable features only. The server recomputes the movement itself — client deltas are never
trusted — then records the edit and answers with a new dataset key. The client rebuilds the
whole analysis on that key with `/api/analysis`, with the same request config. The original
dataframe and every existing payload stay untouched; ancestors, the node, its children,
characteristics, predicates and DR-quality scores all reflect the moved rows afterwards.

**The hierarchy may change.** HDBSCAN decides over the whole dataset, so after a rebuild the
cluster indices need not correspond to the previous run — `root/1` may name entirely
different rows, or not exist. That is the honest result; the client navigates to the root
and says so once.

**Key convention.** `"{base}@cf:{cf_id}"`. `backend/datasets.py::load` resolves it, so every
endpoint that takes a `dataset` — analysis, predicate, characteristics, targets, rows, image
pixels, movement, and a further Apply — works on a counterfactual run unchanged. Plain keys
are unchanged; counterfactual keys are not listed by `/api/datasets`. An unknown or expired
id is a 404 ("counterfactual expired or unknown").

**Which rows change — exactly.** All members of the source cluster: the serialized child's
`row_indices`, never the ≤5000-point preview sample. Those are *positions* in the frame
(the whole pipeline is positional), so the edit is applied with `iloc`; the frame's `row_id`
values at those positions are recorded alongside for the export and the UI. The two
coincide only while the loader's index is a `RangeIndex`.

**Identity and stacking.** `cf_id = sha256(base | parent | canonical edit)[:12]` — content
derived, so the same edit on the same parent is one snapshot (a duplicate click cannot fork
the history), the same edit on two datasets never collides, and stacking is a chain of
immutable snapshots. Undo and Reset need no endpoint: the client switches back to the parent
key or the base.

**Two stores, deliberately** (`backend/counterfactual.py`). The *edit history* is small and
never evicted by the frame cache; it expires only by session lifetime (24 h since an edit or
any descendant was last touched, swept on insert), so Undo keeps working across a long
session. *Materialized frames* are an LRU of 8 full copies; on a miss the frame is replayed
from the base loader through the chain. Each copy is a full frame in memory — on MNIST
(70 000 × 784 float64) that is ~440 MB per copy, so the bound matters on the wide image
datasets.

**Never persisted.** Hosting mode skips the on-disk run cache for counterfactual keys: a
counterfactual run is session state and must not outlive the history that gives its key
meaning. The in-memory tree cache holds it like any other run.

**Export.** `GET /api/counterfactual/{cf_id}/rows.csv` lists every row any edit in the chain
moved, with the original and the counterfactual value of every feature any edit touched
(`feature`, `feature__cf`), keyed by `row_id`.

---

## `backend/` — the service

| File | Role |
|---|---|
| `app.py` | Every route, the request models, the tree-cache lookup (memory, then disk), and the static mount that serves `frontend/dist` in production. |
| `serialize.py` | Walks the calc layer's tree of TypedDicts (numpy arrays, DataFrames) and emits the JSON `Node` schema. Its TypeScript counterpart is `frontend/src/types.ts`. `_finite()` maps NaN/Inf to `null`, because Starlette encodes with `allow_nan=False`. |
| `jobs.py` | Build-on-a-worker-thread, keyed by the run-cache signature so a retry or reload re-attaches to the run already in flight. Keeps the last 64 jobs so a late poll can still read the outcome. |
| `run_cache.py` | Gzipped on-disk payload cache, **hosting mode only** (`HILDE_HOSTING=1`, which `host.py` sets). Dev runs never touch the disk. A corrupt entry is deleted rather than served. With `HILDE_CACHE_ONLY=1` (implies hosting) the stored runs are the only runs: `/api/analysis` answers a miss with 409 instead of building, and nothing is written to or deleted from the cache directory. |
| `run_listing.py` | What `/api/cached-runs` answers: every stored run a request can reach. Reads each entry's `meta` off the head of the file instead of parsing the payload, once: the result is indexed in memory and in `<cache dir>/.listing/`, and only new or changed entries are read when the cache directory changes. |
| `run_index.py` | The listing's index: what was read off each entry, its on-disk form (`<cache dir>/.listing/index.json.gz`, valid while the cache directory's mtime is unchanged) and the compact per-dataset form `/api/cached-runs/{dataset}` sends. |
| `tree_cache.py` | The in-memory LRU of parsed runs (8 entries). In cache-only mode, where only `/api/movement` parses an entry, it is also bounded by an estimate of the parsed size (12x the gzipped file, 48 MiB in all), and entries are parsed one at a time. |
| `stored_run.py` | Cache-only `/api/analysis` hits: sends the entry file itself with `Content-Encoding: gzip` (inflated chunk by chunk for a client that does not accept gzip), so serving a run never parses it. The body is the stored `{meta, tree}`, without `status`/`cached`. A truncated entry is still a 409. |
| `movement.py` | Cluster movement: one shared feature-space displacement that translates a child cluster toward a clicked point or a sibling. Refits the parent's reducer, aligns it to the stored embedding, solves for the displacement, and answers entirely in *visible* coordinates. Its module docstring is the HTTP contract. |
| `movement_jobs.py` | The same worker-thread pattern as `jobs.py`, for the one movement case that can be slow (a UMAP refit). Keyed on the whole request so a duplicate POST re-attaches; keeps 16 jobs; results live in memory only and are never written to `run_cache.py`. Also carries an Apply that needs a refit. |
| `counterfactual.py` | Counterfactual sessions: the immutable, content-addressed edit history, the bounded LRU of materialized frames (replayed from the base on a miss), the `{base}@cf:{id}` key convention, and the changed-rows export. Its module docstring is the design. |
| `counterfactual_apply.py` | Apply = recompute the movement at the requested strength through the movement pipeline, then register it as an edit on top of the base or of an existing counterfactual key. |
| `predicate.py` | Reproduces the local/global scaling for a selection and runs `generate_predicate("db", …)` twice — at RCM 1.0 (full range) and 0.9 (trimmed core). |
| `characteristics.py` | The same z-score contrast the tree stores per cluster, but for an arbitrary lasso selection: a two-label split (selected vs. rest of node) through the unchanged calc-layer function, reproducing the root scaler rather than refitting, so both frames land on one axis. |
| `targets.py` | The separate question the predicate must not answer: what are the *labels* of these points? Reported against the whole-dataset range so the selection has a scale to sit in. |
| `images.py` | Pixel lookup for the four image datasets. Returns plain numbers — the frontend draws the canvas, so there is no image library on the server. |
| `datasets.py` | Thin access to `src/datasets.py`. Also defines `default_feature_cols`: everything except `row_id` and `target_*`. |
| `requirements.txt` | Pinned to the resolved `uv.lock` for the Docker image. If `pyproject.toml` changes, re-lock and re-pin this to match. |
| `tests/` | `test_serialize.py` (tree → JSON contract), `test_targets.py` (target statistics), `test_predicate.py` (clause ΔF1 bookkeeping), `test_movement.py` (alignment, PCA and UMAP movement) and `test_counterfactual.py` (edits, keys, frames, the rebuild on a counterfactual key). Run as modules, see below. |

---

## What the service wraps

Two directories carry their own code maps rather than being summarised here:

- **[`src/README.md`](../src/README.md)** — the calculation layer. The pipeline
  (standardise → UMAP pre-reduction → HDBSCAN → per-node projection → recurse), the two
  node kinds, the predicate induction and its RCM threshold, the neighbourhood metrics and
  why they are chunked, every config knob, and the determinism settings.
- **[`src_research/README.md`](../src_research/README.md)** — the offline experiment
  harnesses, their pre-registered designs, and the `rederive/` correction driver.

`start_evaluation(df, feature_cols, config)` in `src/evaluation/evaluate.py` is the single
entry point this service calls. It builds the tree, then attaches DR-quality scores to
every node.

Two properties of the tree that the JSON contract inherits, and that client code has to
account for:

- **Noise rows are counted in a parent but appear in no child.** HDBSCAN labels points
  `-1` when they belong to no cluster and the recursion descends only into real clusters,
  so `children` sizes do not sum to `n_points`.
- **A node that was never projected has `embedding_original: null` and all-`null`
  scores.** That is the real state, not a serialization gap — a failed projection is never
  given fabricated coordinates.

---


## `scripts/checks/`

Four regression checks retained from the pre-freeze code review, each with a
one-line statement of what it proves in [`scripts/checks/README.md`](../scripts/checks/README.md).
They must run from **outside** the repository so no relative dataset path resolves back
into it.

---

## Running

```bash
# API only (frontend runs separately on :5173 in dev)
PYTHONPATH=. uv run uvicorn backend.app:app --reload --port 8000

# tests (each script also runs under pytest: PYTHONPATH=. .venv/bin/python -m pytest backend/tests -q)
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
PYTHONPATH=. .venv/bin/python -m backend.tests.test_counterfactual

# formatting
uv run ruff format --check .
```

`uv run ruff check .` is **not** clean at the release freeze — a nonzero exit there is
pre-existing, not something a change introduced.

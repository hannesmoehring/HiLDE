# HiLDE — Hierarchical Local Decomposition and Explanation
[![hilde.3m0.de](https://img.shields.io/website?url=https%3A%2F%2Fhilde.3m0.de&label=hilde.3m0.de&up_message=online&down_message=offline)](https://hilde.3m0.de)

**Live instance:** [hilde.3m0.de](https://hilde.3m0.de) · **Intro video:** [watch on YouTube](https://youtu.be/BHsBJ5zTIYA) · **Thesis:** PDF coming soon!<!-- [PDF](./thesis.pdf) -->

Interactive explorer for high-dimensional tabular data. It recursively partitions a
dataset with HDBSCAN, computes a separate dimensionality reduction inside each region,
scores every embedding for neighbourhood faithfulness, and induces axis-aligned range
predicates that describe what the analyst selected.

Prototype accompanying a bachelor thesis. Formerly *SHD*; the repository name and the
package identifier in `pyproject.toml` keep the historical acronym.

Stack: FastAPI backend (`backend/`) over an unchanged calculation layer (`src/`), with a
React + D3 frontend (`frontend/`) served from the same process in production.

---

## Quickstart

Both paths below were run verbatim from a fresh `git clone` into `/tmp` on macOS
(darwin 24.6.0 arm64, Docker 27.5.1, Node 22.14, npm 10.9, uv 0.11.9). Run them from the
repository root.

### Docker (no local Python or Node needed)

```bash
docker compose up --build
```

The image builds the frontend in a Node stage, installs `backend/requirements.txt` into a
`python:3.13-slim` stage, and serves the API and the built UI from one uvicorn process.
First build takes a few minutes (hdbscan compiles from source on arm64).

Open <http://localhost:8000>. To check it from a shell:

```bash
curl -s http://localhost:8000/api/health
```

Stop and remove the container with:

```bash
docker compose down
```

`docker-compose.yml` mounts `./datasets` (writable, so the downloading loaders can
populate it) and `./.cache/hilde_runs` from the host.

### Local

```bash
bash scripts/prepare_env.sh
uv run host.py
```

`prepare_env.sh` requires [uv](https://docs.astral.sh/uv/); it installs Python 3.13 and
syncs `uv.lock`. `host.py` runs `npm install` and `npm run build` itself when
`frontend/dist` is missing or older than `frontend/src`, then serves on
<http://0.0.0.0:8000>. Node is therefore needed for the first run.

Bind elsewhere with:

```bash
uv run host.py --port 9000 --host 127.0.0.1
```

`host.py` is the only mode with a **persistent run cache**: a completed `/api/analysis`
is stored under `.cache/hilde_runs/` keyed by (dataset, feature columns, config), so an
identical request — including after a restart — reuses the stored run. The UI shows a
"Cached" banner when it serves one, and the *Use cached results* checkbox next to
*Build & Apply* forces a recompute. Move the cache with `HILDE_CACHE_DIR`, or clear it:

```bash
rm -rf .cache/hilde_runs
```

Read the limitation on this cache below before trusting it after a code change.

### Development (two terminals, hot reload)

```bash
# terminal 1 — API on :8000
PYTHONPATH=. uv run uvicorn backend.app:app --port 8000 --reload
```

```bash
# terminal 2 — Vite dev server on :5173, proxying /api to :8000
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173> (Vite binds `[::1]`, so `localhost` resolves but a literal
`127.0.0.1` does not). `HILDE_HOSTING` is unset here, so nothing is written to disk and
no cache banner appears.

To produce a production bundle by hand instead of letting `host.py` do it:

```bash
cd frontend && npm install && npm run build
```

`backend/app.py` mounts `frontend/dist` at `/` whenever that directory exists.

### Tests and checks

The test suite is two scripts, run as modules:

```bash
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
```

`scripts/checks/` holds four regression checks retained from the pre-release code
review; `scripts/checks/README.md` states what each one proves. They must run from
outside the repository so no relative dataset path resolves into it:

```bash
cd /tmp && PYTHONPATH=<repo> <repo>/.venv/bin/python <repo>/scripts/checks/02_zadu_equivalence.py
```

Formatting:

```bash
uv run ruff format --check .
```

---

## Datasets

Ten datasets ship in the registry (`src/datasets.py`, exposed as `/api/datasets`). Label
columns are named `target_*` and are held out of the feature space by default.

**Work fully offline from a fresh clone** (no network at any point):

| Dataset | Rows | Source |
|---|---|---|
| Wine quality (Low) | 6497 | bundled CSVs, `datasets/wine_quality/wine+quality/` |
| Iris (Low) | 150 | bundled with scikit-learn |
| Digits (Low) | 1797 | bundled with scikit-learn |
| Breast cancer (Low) | 569 | bundled with scikit-learn |
| Concentric rings (Low) | 1800 | generated in-process |
| Swiss roll (Low) | 1500 | generated in-process (`make_swiss_roll`) |

**Self-provision over the network on first use**, then cache to disk:

| Dataset | Fetched from | Cached to |
|---|---|---|
| Olivetti faces (Medium) | scikit-learn download | `SCIKIT_LEARN_DATA` (the image sets it to `datasets/sklearn`) |
| QM9 molecules (Medium) | `deepchemdata.s3-us-west-1.amazonaws.com` (~30 MB, subsampled to 30 000 rows) | `datasets/QM9/qm9.csv` |
| Fashion-MNIST (High) | OpenML | `SCIKIT_LEARN_DATA` |
| MNIST (High) | local IDX files under `datasets/MNIST/raw/` if present, else OpenML | `SCIKIT_LEARN_DATA` |

The wine CSVs are the only data files tracked in git (340 KB). Everything else under
`datasets/` is ignored, so a fresh clone has no MNIST IDX files and MNIST falls back to
the OpenML download.

---

## The UI

**Configuration rail.** Dataset, feature-column checkboxes, and the pipeline knobs:
normalisation, number of hierarchical layers, the UMAP pre-reduction dimensionality used
for clustering, HDBSCAN `min_samples` / `min_cluster_size`, and the per-node DR method
(PCA, t-SNE, UMAP, MDS) with its own parameters. *Build & Apply* runs the pipeline as a
background job that the client polls. The rail collapses to a narrow strip so the canvas
can widen. Selecting a new dataset resets the feature selection to that dataset's default
(everything except `row_id` and `target_*`).

**Layer views.** One panel per hierarchical layer. Each shows the layer's cluster
projection as a scatter, coloured by HDBSCAN cluster; clicking a cluster drills into it
and opens the next layer below. The side column of a layer reports the selected cluster's
DR quality tiles and then either its characteristics (z-scored column means against the
parent) or the predicate separating it from the space it was selected out of.

**Explore entire layer.** Under each layer's projection, this button opens the exploration
panel on *all* of that layer's points instead of waiting for a drill-down into one
cluster. It is the only way to reach the rows HDBSCAN labelled noise, which belong to no
child cluster. Pressing it again returns to cluster picking; every navigation clears the
flag, so it can never outlive the path it was set for.

**Exploration panel.** The per-node embedding, with lasso and box selection. The
selection drives three tabs. *Predicate* induces an axis-aligned conjunction describing
the selection and reports its F1, precision and recall, scoped either against the whole
dataset (global) or against the explored node (local). *Characteristics* shows the
selection's per-column z-scores against the node's own baseline. A selection covering the
entire node is refused in both, because the comparison would be self-referential.

**Ranges (interactive predicate).** The third tab inverts the interaction: instead of
lassoing points and reading off a description, the analyst picks columns, slides a
`[min, max]` window over each in raw column units, and the points inside *every* window
become the selection. Both feature and `target_*` columns are offered — a range filter is
a question, not an induced explanation, so slicing on a label explains nothing away, and
targets stay marked in the target hue. The tab badge shows how many columns are filtered.

**Outliers.** Each internal layer carries a collapsible GLOSH outlier section: the score
distribution and a ranked table of the 100 most outlying points. Clicking a row reveals
that point's column values and rings it in the projection above.

**Selected points and images.** A table of the selection's raw rows, features first and
held-out labels fenced off to the right, exportable as CSV. For the four image datasets
(Digits, Olivetti, Fashion-MNIST, MNIST) a table row can be opened as the image it
encodes, drawn at native resolution and scaled with `image-rendering: pixelated`.

---

## Architecture

```
backend/       FastAPI service. app.py holds every route; serialize.py turns the
               analysis tree into the JSON contract; predicate.py, characteristics.py,
               targets.py, images.py answer per-selection queries; jobs.py runs a build
               on a worker thread; run_cache.py is the hosting-mode disk cache.
src/           Analysis pipeline, independent of any UI. analysis/ holds the recursive
               routine, clustering, dimensionality reduction, characteristics and the
               predicate generator; evaluation/ holds the ZADU-equivalent neighbourhood
               metrics; datasets.py is the loader registry; config_defaults.py the
               canonical Config.
frontend/      React + D3 (Vite, TypeScript). src/App.tsx owns navigation state;
               components/ holds the panels, charts/ the D3 visualisations.
src_research/  Offline experiment harnesses, one per thesis experiment, each next to
               its pre-registered design document. rederive/ holds the post-hoc
               correction driver.
scripts/       prepare_env.sh (environment) and checks/ (retained regression checks).
host.py        Single-process launcher: build the frontend if stale, then serve.
main.py        Minimal smoke script — loads the wine CSV and runs one PCA.
```

`src/` never imports from `backend/` or `frontend/`; the API is a wrapper over it.

Four code maps go a level deeper than this overview, one per directory:

- [`src/README.md`](./src/README.md) — the method. The pipeline end to end, the two node
  kinds, predicate induction, the neighbourhood metrics, and every config knob.
- [`backend/README.md`](./backend/README.md) — the HTTP API, the request/job lifecycle and
  the two-tier cache.
- [`frontend/README.md`](./frontend/README.md) — where state lives, the layer/exploration
  layout, and every component, chart and hook.
- [`src_research/README.md`](./src_research/README.md) — the experiment harnesses, what
  each one tests, and the `rederive/` correction driver.

### Experiment outputs

Harnesses write to `outputs/experiments/<timestamp>/`. **`outputs/` is not tracked in
git**, so a clone does not contain any of it; the directories below exist in the author's
working tree and back specific thesis sections.

| Run directory | Harness | Thesis content |
|---|---|---|
| `20260628_125924`, `20260628_153633` | `hyperparameter_tuning.py` | tuning effectiveness; internal-vs-external validity |
| `20260628_182948` … `20260628_184827` (6 runs) | `hierarchical_vs_flat.py` | RQ1 — H1a (faithfulness) and H1b (label recovery) |
| `20260628_195214` | `planted_subspace_recovery.py` | RQ1 — clean H1b on planted subspaces |
| `20260711_115849` | `predicate_stability.py` | RQ2 — H2, predicate stability under relaxation |
| `20260728_185329` | `benchmark_workflow.py` | §6.5 benchmark walk, RQ1/RQ3 consistency checks |
| `20260729_101836` | `pipeline_tuning.py` | EQ1b preset tuning — **halted mid-run**, nothing integrated |
| `20260728_190741_depth2_diag` | not in the repository | ad-hoc depth-2 noise diagnostic; the producing script was removed |

Each harness was pre-registered: its guards, objectives and acceptance criteria were fixed
before the run. The design documents themselves are no longer kept in the repository — the
thesis carries the design record, and the code in `src_research/` is what produced the
numbers. See [`src_research/README.md`](./src_research/README.md).

### `rederived_20260813/`

Several run directories carry a `rederived_20260813/` subdirectory. These are **post-hoc
corrected aggregates, not reruns**: the driver recomputes summaries from the raw records
already on disk after the pre-release review found defects in the original aggregation
(an inverted MRRE direction, a duplicated control arm, a join that silently returned
nothing, an inflated n). No experiment was re-executed and no original file was modified.
`DELTAS.md` in each subdirectory is the provenance record — what changed, by how much, and
what turned out not to be derivable at all.

The driver is `src_research/rederive/`. It needs `outputs/` present, so it cannot run from
a bare clone; with the run directories in place, `uv run python -m src_research.rederive`
regenerates every `rederived_20260813/` byte-identically.

---

## Reproducibility

**Dimensionality reduction is deterministic.** UMAP is constructed with `init="pca"` —
`random_state` alone is not enough, because the default spectral init routes a
disconnected fuzzy graph through `multi_component_layout`, outside the seeded path. PCA
uses `svd_solver="covariance_eigh"`, which consults no RNG; the default `"auto"` selects
the randomized solver on wide or small matrices and draws from the unseeded process-global
RNG.

**Pinned environment.** The Docker image installs `backend/requirements.txt`, whose
versions are pinned to the resolved `uv.lock` at this release freeze (including the
transitive `llvmlite`, `pynndescent`, `faiss-cpu` and `joblib`, which had drifted between
the lockfile and the image before). `pyproject.toml` caps the scientific core
(`numpy<2.5`, `scipy<2`, `scikit-learn<2`, `umap-learn<0.6`, `hdbscan<0.9`, `numba<0.68`,
`zadu<0.5`) and `uv.lock` fixes everything for the local path. If `pyproject.toml`
changes, re-lock and re-pin `backend/requirements.txt` to match.

**Do not mix reruns with pre-freeze outputs in one table.** The numbers under
`outputs/experiments/` were produced before this freeze, under the environment recorded
in each run directory. A rerun on a later environment is a separate measurement; report it
as one.

---

## Known limitations

- **Client config is not validated server-side.** `/api/analysis` overlays the request's
  `config` dict onto the defaults without a schema. A misspelled key (`umap_n_neighbours`
  for `umap_n_neighbors`) or an invented one is accepted with HTTP 200, silently ignored,
  and the build runs on the default value. Verified against a running server.
- **`pca_components` is dead configuration.** It is declared in `src/types.py`, defaulted
  in `src/config_defaults.py` and mirrored in the frontend, but never read: every node
  embedding is hardcoded to 2D.
- **The Ranges tab is capped on very large nodes.** It fetches every row × every offered
  column to compute bounds and histograms, and refuses above 5 000 000 cells — the MNIST
  root (70 000 × 794) would be an ~800 MB JSON body, past V8's string limit. The tab says
  so and asks you to drill deeper or filter on fewer columns. Cluster-sized nodes are far
  inside the budget.
- **In-flight requests are not cancelled.** The frontend uses no `AbortController`;
  superseded responses are discarded on arrival rather than aborted at the socket, so a
  slow build keeps running server-side after you have navigated away.
- **The run cache is keyed on the config only.** The key is (dataset, feature columns,
  config) — not the code. After *any* change to `src/` or `backend/`, delete
  `.cache/hilde_runs` or the server will serve trees built by the previous version. This
  was observed across a container and a later local process reusing the same entry.
- **HDBSCAN noise rows are counted in a parent but appear in no child.** On the default
  wine build the root holds 6497 points and its two clusters hold 833 + 3469 = 4302; the
  remaining 2195 are noise. They are not lost — the layer view, via *Explore entire
  layer*, is where they can be selected — but child cluster sizes do not sum to the
  parent's.
- **The test suite is minimal.** Two scripts (`backend/tests/`) covering tree
  serialization and target statistics, plus the four regression checks in
  `scripts/checks/`. There is no test runner configuration and no coverage of the
  frontend. `uv run ruff format --check .` passes; `uv run ruff check .` does not
  (33 findings).
- **t-SNE and PCA seed replicates are repeats, not replicates.** The experiment harnesses
  vary a seed across replicates and thread it into `tsne_random_state`, but `_tsne` passes
  no `init`, and scikit-learn's default `init="pca"` never consults `random_state`. PCA
  takes no seed at all. Only UMAP and MDS replicates carry embedding variance; for the
  other two, the `SEEDS` levels are identical repeats.
  `scripts/checks/05_h1a_replicate_collapse.py` demonstrates this and its consequence for
  the H1a paired test.

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
| `GET` | `/api/mode` | whether the run cache is active (drives the "Cached" banner) |
| `GET` | `/api/datasets` | the loader registry |
| `GET` | `/api/datasets/{key}/columns` | column names + the default feature selection |
| `GET` | `/api/datasets/{key}/image/{row_id}` | raw greyscale pixels for one row |
| `POST` | `/api/analysis` | start (or serve from cache) a tree build |
| `GET` | `/api/analysis/jobs/{job_id}` | poll a running build |
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

## `backend/` — the service

| File | Role |
|---|---|
| `app.py` | Every route, the request models, the two-tier tree cache, and the static mount that serves `frontend/dist` in production. |
| `serialize.py` | Walks the calc layer's tree of TypedDicts (numpy arrays, DataFrames) and emits the JSON `Node` schema. Its TypeScript counterpart is `frontend/src/types.ts`. `_finite()` maps NaN/Inf to `null`, because Starlette encodes with `allow_nan=False`. |
| `jobs.py` | Build-on-a-worker-thread, keyed by the run-cache signature so a retry or reload re-attaches to the run already in flight. Keeps the last 64 jobs so a late poll can still read the outcome. |
| `run_cache.py` | Gzipped on-disk payload cache, **hosting mode only** (`HILDE_HOSTING=1`, which `host.py` sets). Dev runs never touch the disk. A corrupt entry is deleted rather than served. |
| `predicate.py` | Reproduces the local/global scaling for a selection and runs `generate_predicate("db", …)` twice — at RCM 1.0 (full range) and 0.9 (trimmed core). |
| `characteristics.py` | The same z-score contrast the tree stores per cluster, but for an arbitrary lasso selection: a two-label split (selected vs. rest of node) through the unchanged calc-layer function, reproducing the root scaler rather than refitting, so both frames land on one axis. |
| `targets.py` | The separate question the predicate must not answer: what are the *labels* of these points? Reported against the whole-dataset range so the selection has a scale to sit in. |
| `images.py` | Pixel lookup for the four image datasets. Returns plain numbers — the frontend draws the canvas, so there is no image library on the server. |
| `datasets.py` | Thin access to `src/datasets.py`. Also defines `default_feature_cols`: everything except `row_id` and `target_*`. |
| `requirements.txt` | Pinned to the resolved `uv.lock` for the Docker image. If `pyproject.toml` changes, re-lock and re-pin this to match. |
| `tests/` | `test_serialize.py` (tree → JSON contract) and `test_targets.py` (target statistics). Run as modules, see below. |

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

Four regression checks retained from the pre-freeze adversarial review, each with a
one-line statement of what it proves in [`scripts/checks/README.md`](../scripts/checks/README.md).
They must run from **outside** the repository so no relative dataset path resolves back
into it.

---

## Running

```bash
# API only (frontend runs separately on :5173 in dev)
PYTHONPATH=. uv run uvicorn backend.app:app --reload --port 8000

# tests
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets

# formatting
uv run ruff format --check .
```

`uv run ruff check .` is **not** clean at the release freeze — a nonzero exit there is
pre-existing, not something a change introduced.

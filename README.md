# SHD Interactive Dimensionality Reduction

## D3 frontend (new — FastAPI backend + React/D3)

The production UI is migrating from Streamlit to a D3 frontend served by a FastAPI
backend that wraps the unchanged calc layer (`src/analysis`, `src/evaluation`).
See `PLAN.md` for the full migration plan and status.

Dev (two terminals):

```bash
# 1. Backend (FastAPI on :8000). Uses the existing venv; fastapi/uvicorn are
#    installed on top of it, so run the venv python directly (NOT `uv run`,
#    which re-syncs to the lock and would remove them).
PYTHONPATH=. .venv/bin/python -m uvicorn backend.app:app --port 8000 --reload

# 2. Frontend (Vite dev server on :5173, proxies /api -> :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. Backend tests: `PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize`.

## Production (single container)

FastAPI serves the built frontend (`frontend/dist`) at `/` and the API at `/api/*`
from one process — no separate frontend host.

```bash
# Build the frontend once, then run the backend (it auto-mounts frontend/dist):
cd frontend && npm run build && cd ..
PYTHONPATH=. .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
# -> whole app at http://localhost:8000

# Or via Docker (multi-stage: builds the frontend, then a lean Python runtime):
docker compose up --build     # -> http://localhost:8000
```

Local datasets (wine CSVs, MNIST IDX files) are mounted read-only via
`docker-compose.yml`; sklearn-provided datasets (iris, digits, …) need no mounts.

> The Streamlit UI (`src/ui/`) has been **removed**; the D3 frontend replaces it.
> `src/analysis` and `src/evaluation` (the calc layer) are unchanged. Dataset
> loaders and config defaults now live Streamlit-free in `src/datasets.py` and
> `src/config_defaults.py`.

---


This project now includes a local browser UI for interactive 2D dimensionality reduction using the hardcoded red wine quality dataset.

## Features

- Choose reduction method: PCA, t-SNE, UMAP.
- Configure method-specific parameters.
- For PCA: select x/y principal components with defaults based on highest explained variance.
- Brush or lasso points in the scatter plot.
- Access selected points in backend state and export them to disk.

## Run

Install dependencies with your preferred tool (for example `uv sync`), then launch:

```bash
streamlit run ui/app.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Workflow

1. Select method and parameters in the sidebar.
2. Click **Run reduction**.
3. Brush/lasso points in the plot.
4. Review selected rows and click **Export selected points to CSV** when needed.

Exports are written to `outputs/selections/`.

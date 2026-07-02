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

Open http://localhost:5173. Build the static bundle with `npm run build` (output
in `frontend/dist/`). Backend tests: `PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize`.

## Legacy Streamlit UI

```bash
PYTHONPATH=. streamlit run src/ui/app.py
```

(Retired once D3 parity is signed off — see `PLAN.md` Phase 8.)

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

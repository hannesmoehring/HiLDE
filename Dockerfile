# Single-container image: FastAPI backend serves the built D3 frontend.
# (Replaces the former Streamlit image.)

# ---- Stage 1: build the D3 frontend -> frontend/dist ----
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci 2>/dev/null || npm install
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime ----
FROM python:3.13-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HILDE_HOSTING=1 \
    SCIKIT_LEARN_DATA=/app/datasets/sklearn

# build-essential for any source builds (hdbscan/llvmlite fallbacks)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Calc layer + backend
COPY src ./src
COPY backend ./backend

# Built frontend from stage 1 (served by FastAPI StaticFiles at "/")
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Datasets
COPY datasets/wine_quality ./datasets/wine_quality

EXPOSE 8000
# Datasets needing local files (wine CSVs, MNIST IDX) are mounted at runtime, on a
# writable mount so the downloading loaders (QM9, sklearn's cache) can populate it;
# the sklearn-provided datasets (iris, digits, …) work with no mounts.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]

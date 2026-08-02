"""Hosting-mode launcher — `uv run host.py`.

Builds the frontend if `frontend/dist` is missing or older than `frontend/src`,
then serves the API and the built UI from a single uvicorn process with the
persistent run cache enabled (see `backend/run_cache.py`).

For development use the two-terminal setup in README.md instead; that leaves the
run cache off so every build recomputes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"

# Touching any of these invalidates a previous build.
SOURCES = [FRONTEND / "src", FRONTEND / "index.html", FRONTEND / "package.json", FRONTEND / "vite.config.ts"]


def _say(message: str) -> None:
    print(f"[host] {message}", flush=True)  # flush: stdout is block-buffered when piped


def dist_is_stale() -> bool:
    index = DIST / "index.html"
    if not index.is_file():
        return True
    built = index.stat().st_mtime
    for source in SOURCES:
        files = source.rglob("*") if source.is_dir() else [source]
        if any(f.is_file() and f.stat().st_mtime > built for f in files):
            return True
    return False


def build_frontend() -> None:
    if not (FRONTEND / "node_modules").is_dir():
        subprocess.run(["npm", "install"], cwd=FRONTEND, check=True)
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND, check=True)


def ensure_frontend() -> None:
    if not dist_is_stale():
        _say("frontend/dist is up to date")
        return
    _say("frontend/dist is missing or stale — building")
    try:
        build_frontend()
    except FileNotFoundError:
        if not (DIST / "index.html").is_file():
            sys.exit("[host] npm not found and no frontend/dist to serve — install node, or build elsewhere")
        _say("npm not found — serving the existing (stale) frontend/dist")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HiLDE in hosting mode.")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    ensure_frontend()

    os.environ["HILDE_HOSTING"] = "1"
    from backend import run_cache

    _say(f"run cache: {run_cache.cache_dir()}")
    _say(f"serving on http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run("backend.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

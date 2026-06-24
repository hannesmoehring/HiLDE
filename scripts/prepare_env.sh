#!/usr/bin/env bash

set -euo pipefail

echo "Checking for uv..."
command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is not installed."
    echo "Install it from https://docs.astral.sh/uv/"
    exit 1
}

echo "Ensuring Python 3.13 is available..."
uv python install 3.13

echo "Installing dependencies from lockfile..."
uv sync --locked --python 3.13

echo "Done."
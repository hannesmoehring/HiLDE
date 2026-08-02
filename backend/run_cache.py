"""On-disk cache of analysis runs — hosting mode only.

In hosting mode (`uv run host.py`, or `HILDE_HOSTING=1`) a completed
`/api/analysis` payload is written to disk keyed by (dataset, feature_cols,
config), so restarting the server does not recompute it. Dev runs are
unaffected: `is_hosting()` is false and nothing touches the disk.

Reusing a stored run means a re-request does not re-execute UMAP/HDBSCAN — the
frontend surfaces that with a banner, and the "Use cached results" toggle
bypasses this module entirely.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

HOSTING_ENV = "HILDE_HOSTING"
CACHE_DIR_ENV = "HILDE_CACHE_DIR"

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "hilde_runs"


def is_hosting() -> bool:
    return os.environ.get(HOSTING_ENV, "").strip().lower() not in ("", "0", "false", "no")


def cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV, "").strip()
    return Path(override) if override else _DEFAULT_DIR


def _path_for(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return cache_dir() / f"{digest}.json.gz"


def load(key: str) -> dict[str, Any] | None:
    """Return the stored payload for `key`, or None if absent/unreadable."""
    path = _path_for(key)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        # A truncated/corrupt entry must never break a request — just recompute.
        return None


def store(key: str, payload: dict[str, Any]) -> None:
    """Write `payload` for `key`. Failures are ignored (cache is an optimization)."""
    path = _path_for(key)
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)  # atomic: a reader never sees a half-written entry
    except (OSError, TypeError, ValueError):
        tmp.unlink(missing_ok=True)

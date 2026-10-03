"""On-disk cache of analysis runs — hosting mode only.

In hosting mode (`uv run host.py`, or `HILDE_HOSTING=1`) a completed
`/api/analysis` payload is written to disk keyed by (dataset, feature_cols,
config), so restarting the server does not recompute it. Dev runs are
unaffected: `is_hosting()` is false and nothing touches the disk.

Reusing a stored run means a re-request does not re-execute UMAP/HDBSCAN — the
frontend surfaces that with a banner, and the "Use cached results" toggle
bypasses this module entirely.

Cache-only mode (`HILDE_CACHE_ONLY=1`, implies hosting) is for a server too weak
to build anything: the stored runs are the only runs there are —
`backend/run_listing.py` lists them for the UI — and nothing in this module
writes to or deletes from the cache directory in that mode.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

HOSTING_ENV = "HILDE_HOSTING"
CACHE_ONLY_ENV = "HILDE_CACHE_ONLY"
MAINTENANCE_ENV = "HILDE_MAINTENANCE"
CACHE_DIR_ENV = "HILDE_CACHE_DIR"

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "hilde_runs"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def is_cache_only() -> bool:
    return _flag(CACHE_ONLY_ENV)


def is_maintenance() -> bool:
    """The UI shows a maintenance notice instead of the app and requests nothing."""
    return _flag(MAINTENANCE_ENV)


def is_hosting() -> bool:
    return _flag(HOSTING_ENV) or is_cache_only()


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
    except (OSError, ValueError, EOFError):
        # A truncated/corrupt entry must never break a request — just recompute.
        # gzip raises EOFError on truncation, which is not an OSError, hence the
        # explicit catch. Delete the file so the next build replaces it, instead of
        # every request for this config hitting the same bad entry forever.
        # Not in cache-only mode: no build will ever replace it, the directory may
        # be mounted read-only, and the entry may simply still be uploading.
        if not is_cache_only():
            path.unlink(missing_ok=True)
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

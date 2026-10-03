"""On-disk cache of analysis runs — hosting mode only.

In hosting mode (`uv run host.py`, or `HILDE_HOSTING=1`) a completed
`/api/analysis` payload is written to disk keyed by (dataset, feature_cols,
config), so restarting the server does not recompute it. Dev runs are
unaffected: `is_hosting()` is false and nothing touches the disk.

Reusing a stored run means a re-request does not re-execute UMAP/HDBSCAN — the
frontend surfaces that with a banner, and the "Use cached results" toggle
bypasses this module entirely.

Cache-only mode (`HILDE_CACHE_ONLY=1`, implies hosting) is for a server too weak
to build anything: the stored runs are the only runs there are. `list_metas()`
is what the UI offers in place of the free-form configuration, and nothing in
this module writes to or deletes from the cache directory in that mode.
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
CACHE_DIR_ENV = "HILDE_CACHE_DIR"

_DEFAULT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "hilde_runs"

# (file name, mtime) -> that entry's `meta`. Reading a meta means decompressing
# and parsing the whole payload, which a listing must not redo on every request.
_meta_memo: dict[tuple[str, int], dict[str, Any]] = {}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


def is_cache_only() -> bool:
    return _flag(CACHE_ONLY_ENV)


def is_hosting() -> bool:
    return _flag(HOSTING_ENV) or is_cache_only()


def cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV, "").strip()
    return Path(override) if override else _DEFAULT_DIR


def _path_for(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return cache_dir() / f"{digest}.json.gz"


def is_stored(key: str) -> bool:
    return _path_for(key).is_file()


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


def list_metas() -> list[dict[str, Any]]:
    """The `meta` block of every readable stored run. Unreadable entries are skipped."""
    metas = []
    for path in sorted(cache_dir().glob("*.json.gz")):
        try:
            memo_key = (path.name, path.stat().st_mtime_ns)
            if memo_key not in _meta_memo:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    _meta_memo[memo_key] = json.load(fh)["meta"]
            metas.append(_meta_memo[memo_key])
        except (OSError, ValueError, EOFError, KeyError, TypeError):
            continue
    return metas


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

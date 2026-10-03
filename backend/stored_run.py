"""Serve a stored run as the gzip bytes it already is — cache-only mode only.

Parsing an entry to re-encode it costs ~10x its gzipped size in memory (a 3.3 MB
entry is ~35 MB of Python objects, ~50 MB while parsing) and a few hundred ms of
CPU, only to produce the bytes the file already holds. So `/api/analysis` sends
the file itself with `Content-Encoding: gzip`, and inflates it chunk by chunk for
a client that does not accept gzip. Neither path builds the tree in memory.

The body is the entry as stored, `{"meta": …, "tree": …}`: it has no `status`
or `cached` field. `frontend/src/api.ts::runAnalysis` only polls while
`status === "running"` and only throws on `"error"`, so it returns this body as
the finished run; `cached` is read only by the hosting banner, which cache-only
mode does not show.

Routes that need the tree server-side (`/api/movement`) still parse it, into
the byte-bounded `backend/tree_cache.py`.
"""

from __future__ import annotations

import gzip
import threading
import zlib
from collections.abc import Iterator
from pathlib import Path

from fastapi.responses import FileResponse, Response, StreamingResponse

from backend import run_cache

_CHUNK = 1 << 16
_HEADERS = {"Vary": "Accept-Encoding"}

# Entries already inflated end to end once, by (path, size, mtime): a complete
# file is checked once per process, a replaced one again. Bounded — a stamp is
# ~200 bytes, and clearing only costs re-checking.
_CHECKED_MAX = 4096
_checked: set[tuple[Path, int, int]] = set()
_checked_lock = threading.Lock()


def accepts_gzip(accept_encoding: str) -> bool:
    """Whether an `Accept-Encoding` value admits gzip (`gzip;q=0` does not)."""
    for part in accept_encoding.lower().split(","):
        coding, _, params = part.strip().partition(";")
        if coding.strip() not in ("gzip", "x-gzip", "*"):
            continue
        q = params.strip()
        if q.startswith("q="):
            try:
                return float(q[2:]) > 0
            except ValueError:
                return False
        return True
    return False


def _is_complete(path: Path) -> bool:
    """Inflate the whole file without keeping it: a truncated or corrupt entry
    must be a miss (409), not a body the browser fails to decode halfway."""
    try:
        st = path.stat()
        stamp = (path, st.st_size, st.st_mtime_ns)
        with _checked_lock:
            if stamp in _checked:
                return True
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
    except (OSError, EOFError, zlib.error):
        return False
    with _checked_lock:
        if len(_checked) >= _CHECKED_MAX:
            _checked.clear()
        _checked.add(stamp)
    return True


def response(key: str, accept_encoding: str) -> Response | None:
    """The stored entry for `key` as an HTTP response, or None on a miss."""
    path = run_cache._path_for(key)
    if not path.is_file() or not _is_complete(path):
        return None
    if accepts_gzip(accept_encoding):
        return FileResponse(
            path,
            media_type="application/json",
            headers={**_HEADERS, "Content-Encoding": "gzip"},
        )
    return StreamingResponse(
        _inflate(path), media_type="application/json", headers=_HEADERS
    )


def _inflate(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            yield chunk

"""The stored runs, listed for the cache-only UI's run picker.

A parameter sweep leaves ~3,000 entries per dataset of up to ~100 MB of JSON
each, and the server this mode exists for cannot parse them just to learn what
they are. So an entry's `meta` is read off the first few kB of its file —
`backend/app.py::_build` writes it first — once: what was read is kept in an
index (`backend/run_index.py`), in memory and next to the cache.

The index is valid while the cache directory's mtime stays what it was. Adding,
replacing or removing an entry renames a file into or out of the directory,
which bumps it (`run_cache.store` and rsync both do). When it did change, every
entry is stat-ed and only new or changed ones are read.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend import run_cache, run_index
from backend.run_index import UNREACHABLE, UNREADABLE, Index, Record

_META_PREFIX = '{"meta": '
_CHUNK = 8192  # compressed bytes read first; doubled until the meta is complete
_RACY_S = 2.0  # mtimes are coarse: a just-changed directory may change again unseen
_RETRY_S = 30.0  # how soon an unreadable entry is looked at again

KeyOf = Callable[[str, list[str], dict[str, Any]], str]

_lock = threading.Lock()  # one scan at a time; the others wait for its result
_index: Index | None = None


def _dir_mtime(root: Path) -> int | None:
    try:
        return root.stat().st_mtime_ns
    except OSError:
        return None


def _read_meta(path: Path) -> dict[str, Any]:
    """The `meta` of an entry, decompressing only as much of the file as it takes."""
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw, chunk = b"", _CHUNK
    with open(path, "rb") as fh:
        try:  # the head is all we want: no read-ahead of the rest
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        while True:
            buf = fh.read(chunk)
            raw += inflate.decompress(buf) if buf else inflate.flush()
            text = raw.decode("utf-8", "ignore")
            if text.startswith(_META_PREFIX):
                try:
                    return json.JSONDecoder().raw_decode(text, len(_META_PREFIX))[0]
                except ValueError:
                    pass  # not all of it yet
            if not buf:
                return json.loads(text)["meta"]
            chunk *= 2


def _record(index: Index, name: str, mtime: int, key_of: KeyOf) -> Record:
    try:
        meta = _read_meta(index.root / name)
        dataset, cols, config = meta["dataset"], meta["feature_cols"], meta["config"]
        if run_cache._path_for(key_of(dataset, cols, config)).name != name:
            return (mtime, UNREACHABLE, ())
        return index.record(mtime, (dataset, tuple(cols), meta.get("n_total")), config)
    except (OSError, ValueError, EOFError, KeyError, TypeError, zlib.error):
        return (mtime, UNREADABLE, ())


def _scan(prev: Index | None, root: Path, key_of: KeyOf) -> Index:
    before = _dir_mtime(root)
    old = prev.files if prev is not None else {}
    index = (
        Index(root, keys=list(prev.keys), groups=list(prev.groups))
        if prev
        else Index(root)
    )
    try:
        entries = [e for e in os.scandir(root) if e.name.endswith(".json.gz")]
    except OSError:
        entries = []
    for entry in entries:
        try:
            mtime = entry.stat().st_mtime_ns
        except OSError:
            continue
        rec = old.get(entry.name)
        if rec is None or rec[0] != mtime or rec[1] == UNREADABLE:
            rec = _record(index, entry.name, mtime, key_of)
        index.files[entry.name] = rec
    racy = before is None or time.time_ns() - before < _RACY_S * 1e9
    index.stamp = None if racy or _dir_mtime(root) != before else before
    if index.unreadable():
        index.expires = time.monotonic() + _RETRY_S
    return index


def _current(key_of: KeyOf) -> Index:
    global _index
    root = run_cache.cache_dir()
    with _lock:
        index = _index if _index is not None and _index.root == root else None
        if index is None:
            try:  # before the mtime is taken: creating it bumps the directory's
                (root / run_index.INDEX_DIR).mkdir(exist_ok=True)
            except OSError:
                pass  # read-only or absent: the index lives in memory only
            index = run_index.load(root)
        if (
            index is not None
            and index.stamp is not None
            and index.stamp == _dir_mtime(root)
            and time.monotonic() < index.expires
        ):
            _index = index
            return index
        prev, _index = index, _scan(index, root, key_of)
        if prev is None or (prev.files, prev.stamp) != (_index.files, _index.stamp):
            run_index.save(_index)
        return _index


def summary(key_of: KeyOf, known: set[str]) -> list[dict[str, Any]]:
    """The datasets a request can reach a stored run of, with how many runs each.

    Reachable means: the dataset is in `known`, and the entry sits under the file
    name its own signature (`key_of`) hashes to — a copied-in file that does not
    is one no request will ever look up. Unreadable entries are skipped.
    """
    index = _current(key_of)
    counts: dict[str, int] = {}
    for _, g, _ in index.files.values():
        dataset = index.groups[g][0] if g >= 0 else None
        if dataset in known:
            counts[dataset] = counts.get(dataset, 0) + 1
    return [{"dataset": d, "n_runs": n} for d, n in sorted(counts.items())]


def encoded(key_of: KeyOf, dataset: str) -> bytes | None:
    """The stored runs of `dataset`, gzipped (`run_index.encode`), or None if none."""
    index = _current(key_of)
    with _lock:
        if dataset not in index.encoded:
            index.encoded[dataset] = run_index.encode(index, dataset)
        return index.encoded[dataset] or None

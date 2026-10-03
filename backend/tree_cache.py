"""Parsed runs kept in memory by request signature: a bounded LRU.

A tree build (UMAP+HDBSCAN+ZADU) is expensive, so a finished payload stays here
for the job poll and for the routes that read the tree server-side
(`/api/movement`, `/api/counterfactual/apply`). Bounded by count: a payload is
~8-10 MB on the smallest realistic dataset, and a hyperparameter sweep visits a
new key every build, so an unbounded dict grows into the container's limit.

Cache-only mode (`load`) also bounds it by bytes. There only `/api/movement`
parses an entry, and a parsed entry holds ~8-12x its gzipped size (a 3.3 MB
entry is ~35 MB, ~50 MB while parsing), so eight of the largest would be ~300 MB
on a 640 MB box. The cost is estimated from the file size; the newest entry is
kept even over the budget.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

MAX_ENTRIES = 8
MAX_BYTES = 48 << 20  # only entries put with a cost count, i.e. cache-only loads
PARSED_PER_GZ_BYTE = 12  # the high end of the measured 8.2-12.0

_entries: OrderedDict[str, tuple[dict[str, Any], int]] = OrderedDict()
_lock = threading.Lock()  # request threads share the LRU
_load_lock = threading.Lock()  # one cache-only parse at a time


def get(key: str) -> dict[str, Any] | None:
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return None
        _entries.move_to_end(key)
        return entry[0]


def put(key: str, payload: dict[str, Any], cost: int = 0) -> None:
    with _lock:
        _entries.pop(key, None)
        _evict(cost)
        _entries[key] = (payload, cost)


def clear() -> None:
    with _lock:
        _entries.clear()


def keys() -> list[str]:
    with _lock:
        return list(_entries)


def _evict(cost: int) -> bool:
    """Make room for one more entry of `cost` bytes, oldest out first. Caller locks."""
    evicted = False
    while _entries and (
        len(_entries) >= MAX_ENTRIES
        or sum(c for _, c in _entries.values()) + cost > MAX_BYTES
    ):
        _entries.popitem(last=False)
        evicted = True
    return evicted


def load(
    key: str, gz_bytes: int, parse: Callable[[str], dict[str, Any] | None]
) -> dict[str, Any] | None:
    """The payload under `key`, parsed by `parse` on a miss — cache-only mode.

    Room is made before parsing and one entry parses at a time, so the parse peak
    never stacks on a full cache or on a second parse.
    """
    with _load_lock:
        payload = get(key)
        if payload is not None:
            return payload
        cost = gz_bytes * PARSED_PER_GZ_BYTE
        with _lock:
            evicted = _evict(cost)
        if evicted:
            # An evicted tree can still be pinned by a reference cycle (a 409's
            # traceback holds the route frame that held the payload); only the
            # full collector frees it before the next parse.
            gc.collect()
        payload = parse(key)
        if payload is not None:
            put(key, payload, cost)
        return payload

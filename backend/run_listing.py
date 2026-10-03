"""The stored runs, listed for the cache-only UI's run picker.

A parameter sweep leaves thousands of entries of up to ~100 MB of JSON each, and
the server this mode exists for cannot parse them just to learn what they are.
So an entry's `meta` is read off the head of its file — `backend/app.py::_build`
writes it first — and the listing is memoized until a file in the cache
directory appears, disappears or changes.

Runs are grouped by (dataset, feature columns): the columns are the bulk of a
`meta` (4096 names on Olivetti) and the same for every run of a sweep.
"""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend import run_cache

_HEAD_CHARS = 1 << 17  # past the longest meta a registry dataset produces
_META_PREFIX = '{"meta": '

_Stamp = list[tuple[Path, int]]

# One listing at a time: a second request waits for the first one's memo instead
# of reading every entry again alongside it.
_lock = threading.Lock()
_memo: tuple[_Stamp, list[dict[str, Any]]] | None = None


def _read_meta(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        head = fh.read(_HEAD_CHARS)
        if head.startswith(_META_PREFIX):
            try:
                return json.JSONDecoder().raw_decode(head, len(_META_PREFIX))[0]
            except ValueError:
                pass  # a meta longer than the head: parse the whole entry after all
        return json.loads(head + fh.read())["meta"]


def _stamp() -> _Stamp:
    stamp = []
    for path in sorted(run_cache.cache_dir().glob("*.json.gz")):
        try:
            stamp.append((path, path.stat().st_mtime_ns))
        except OSError:
            continue
    return stamp


def list_runs(
    key_of: Callable[[str, list[str], dict[str, Any]], str], known: set[str]
) -> list[dict[str, Any]]:
    """The runs a request can reach, as `{dataset, feature_cols, n_total, configs}`.

    Reachable means: the dataset is in `known`, and the entry sits under the file
    name its own signature (`key_of`) hashes to — a copied-in file that does not
    is one no request will ever look up. Unreadable entries are skipped.
    """
    global _memo
    with _lock:
        stamp = _stamp()
        if _memo is not None and _memo[0] == stamp:
            return _memo[1]
        groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for path, _ in stamp:
            try:
                meta = _read_meta(path)
                dataset, cols, config = (
                    meta["dataset"],
                    meta["feature_cols"],
                    meta["config"],
                )
                if dataset not in known:
                    continue
                if run_cache._path_for(key_of(dataset, cols, config)) != path:
                    continue
                group = groups.setdefault(
                    (dataset, tuple(cols)),
                    {
                        "dataset": dataset,
                        "feature_cols": cols,
                        "n_total": meta.get("n_total"),
                        "configs": [],
                    },
                )
                group["configs"].append(config)
            except (OSError, ValueError, EOFError, KeyError, TypeError):
                continue
        runs = sorted(
            groups.values(), key=lambda g: (g["dataset"], len(g["feature_cols"]))
        )
        for group in runs:
            group["configs"].sort(key=lambda c: json.dumps(c, sort_keys=True))
        _memo = (stamp, runs)
        return runs

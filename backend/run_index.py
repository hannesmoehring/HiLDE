"""What `backend/run_listing.py` has read off the stored runs, and its two forms.

* On disk, `<cache dir>/.listing/index.json.gz`, so a restart reads one small
  file instead of the head of every entry. A subdirectory: writing it leaves the
  cache directory's own mtime, which says whether the index is still complete,
  alone.
* On the wire, one dataset at a time (`encode`): the knobs every run shares
  once, the others as value tables plus one row of indices per run, gzipped.

Bump `VERSION` whenever what a record means changes — including
`backend/app.py::_cache_key`, since a record says whether its file sits under
the name its own signature hashes to.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERSION = 1
INDEX_DIR = ".listing"
INDEX_FILE = "index.json.gz"

UNREACHABLE = -1  # a record's group: under a name its own signature does not hash to
UNREADABLE = -2  # a record's group: not (yet) an entry, e.g. an upload in progress
MISSING = object()  # a knob the run was requested without

Group = tuple[str, tuple[str, ...], Any]  # dataset, feature columns, n_total
Record = tuple[int, int, tuple[Any, ...]]  # file mtime, group, a value per knob


@dataclass
class Index:
    root: Path
    stamp: int | None = None  # the directory mtime it is complete for; None: check
    expires: float = float("inf")  # monotonic; sooner while an entry is unreadable
    keys: list[str] = field(default_factory=list)  # knob names, in `values` order
    groups: list[Group] = field(default_factory=list)
    files: dict[str, Record] = field(default_factory=dict)
    encoded: dict[str, bytes] = field(default_factory=dict)  # per dataset, lazily

    def __post_init__(self) -> None:
        self._ids = {g: i for i, g in enumerate(self.groups)}

    def record(self, mtime: int, group: Group, config: dict[str, Any]) -> Record:
        if group not in self._ids:
            self._ids[group] = len(self.groups)
            self.groups.append(group)
        self.keys.extend(k for k in config if k not in self.keys)
        values = tuple(config.get(k, MISSING) for k in self.keys)
        return (mtime, self._ids[group], values)

    def unreadable(self) -> bool:
        return any(g == UNREADABLE for _, g, _ in self.files.values())


def save(index: Index) -> None:
    """Write `index` next to the cache; a read-only mount keeps it in memory only."""
    files = {}
    for name, (mtime, g, values) in index.files.items():
        if g == UNREADABLE:
            continue  # looked at again on the next scan
        missing = [i for i, v in enumerate(values) if v is MISSING]
        vals = [None if v is MISSING else v for v in values]
        files[name] = [mtime, g, vals, missing] if missing else [mtime, g, vals]
    doc = {
        "version": VERSION,
        "stamp": None if index.unreadable() else index.stamp,
        "keys": index.keys,
        "groups": [list(g) for g in index.groups],
        "files": files,
    }
    path = index.root / INDEX_DIR / INDEX_FILE
    tmp = path.with_name(f"{INDEX_FILE}.{os.getpid()}.tmp")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(doc, fh, separators=(",", ":"))
        tmp.replace(path)
    except (OSError, TypeError, ValueError):
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def load(root: Path) -> Index | None:
    try:
        with gzip.open(root / INDEX_DIR / INDEX_FILE, "rt", encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("version") != VERSION:
            return None
        groups = [(d, tuple(c), n) for d, c, n in doc["groups"]]
        index = Index(root, stamp=doc["stamp"], keys=doc["keys"], groups=groups)
        for name, (mtime, g, vals, *missing) in doc["files"].items():
            values = list(vals)
            for i in missing[0] if missing else ():
                values[i] = MISSING
            index.files[name] = (mtime, g, tuple(values))
        return index
    except (OSError, ValueError, EOFError, KeyError, TypeError, zlib.error):
        return None


def _order(value: Any) -> tuple[str, Any]:
    """Sort key for one knob's values: numerically within a type, never across."""
    scalar = isinstance(value, (bool, int, float, str))
    return (type(value).__name__, value if scalar else json.dumps(value))


def encode(index: Index, dataset: str) -> bytes:
    """The runs of `dataset` as gzipped JSON (b"" if none):
    `{dataset, groups: [{feature_cols, n_total}], fixed: {knob: value},
    knobs: [name], values: [[value]], runs: [[group, value index per knob]]}`.
    A run's config is `fixed` plus `knobs[k]: values[k][i]` for each index
    `i >= 0` (-1: requested without it) — exactly the config it was stored
    under, so sending it back is a cache hit."""
    runs = [
        (g, values)
        for _, g, values in index.files.values()
        if g >= 0 and index.groups[g][0] == dataset
    ]
    if not runs:
        return b""
    order = sorted({g for g, _ in runs}, key=lambda g: len(index.groups[g][1]))
    renum = {g: i for i, g in enumerate(order)}
    fixed: dict[str, Any] = {}
    knobs, tables, columns = [], [], []
    for k, name in enumerate(index.keys):
        col = [values[k] if k < len(values) else MISSING for _, values in runs]
        # Keyed by the JSON text: 1 and 1.0 hash to different cache keys.
        distinct = {json.dumps(v): v for v in col if v is not MISSING}
        if len(distinct) == 1 and MISSING not in col:
            fixed[name] = col[0]
            continue
        if not distinct:
            continue
        texts = sorted(distinct, key=lambda t: _order(distinct[t]))
        pos = {t: i for i, t in enumerate(texts)}
        knobs.append(name)
        tables.append([distinct[t] for t in texts])
        columns.append([-1 if v is MISSING else pos[json.dumps(v)] for v in col])
    rows = sorted([renum[g], *(c[i] for c in columns)] for i, (g, _) in enumerate(runs))
    body = {
        "dataset": dataset,
        "groups": [
            {"feature_cols": list(index.groups[g][1]), "n_total": index.groups[g][2]}
            for g in order
        ],
        "fixed": fixed,
        "knobs": knobs,
        "values": tables,
        "runs": rows,
    }
    return gzip.compress(json.dumps(body, separators=(",", ":")).encode(), 6)

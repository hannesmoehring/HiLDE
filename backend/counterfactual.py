"""Counterfactual sessions — immutable edits applied to a COPY of a dataset.

A cluster movement is a preview; *Apply* turns the previewed feature deltas into
data. It never touches the dataset itself: an edit is recorded, and the analysis
is rebuilt on a counterfactual copy of the frame with the edit applied. Every
edit is immutable and content-addressed, so Undo and Reset are nothing more than
switching back to an earlier key.

TWO STORES, DELIBERATELY
========================

* **Edit history** — small, never evicted by the frame cache. `cf_id ->
  CounterfactualEdit`, expiring only by session lifetime: `COUNTERFACTUAL_TTL`
  since the edit (or any of its descendants) was last accessed, swept on insert.
  Undo across a long session and across tabs keeps working because the history
  is what the keys resolve through.
* **Materialized frames** — large, bounded: an LRU of `_FRAME_CACHE_MAX` copies
  keyed by `cf_id`. On a miss the frame is REBUILT by replaying the edit chain
  from the base loader, which is possible precisely because the history is never
  evicted. A copy costs a full frame in memory (MNIST: 70 000 × 784 float64 is
  ~440 MB), so the bound matters on the wide image datasets.

KEY CONVENTION
==============

`"{base}@cf:{cf_id}"`. `backend/datasets.py::load` resolves it; plain keys are
unchanged; counterfactual keys are NOT listed by `/api/datasets`. Every endpoint
that takes a `dataset` accepts either form through the same function, so an
analysis, a predicate, a movement and a further Apply all work on a
counterfactual run unchanged. The `analysis_id` of a run already digests the
dataset key, so a counterfactual run is a different run everywhere.

IDENTITY
========

`cf_id = sha256(base_dataset | parent_id | canonical-json(edit fields))[:12]`.
The base dataset is part of the identity, so the same edit on two datasets never
collides; the parent is part of it, so the same edit stacked at two different
depths is two snapshots; and because the content is the identity, the same edit
applied twice on the same parent is ONE snapshot — a duplicate click cannot
fork the history.

WHICH ROWS CHANGE
=================

An edit moves ALL members of the source cluster — the serialized child's
`row_indices`, never the ≤5000-point preview sample. Those are POSITIONS in the
dataframe (see `_child_positions` in backend/movement.py), and the whole
pipeline is positional, so the edit is applied with `iloc`. `row_ids` are
recorded alongside for the export and the UI; the two coincide only while the
loader's index is a RangeIndex, which is why both are kept.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

KEY_MARK = "@cf:"

# Seconds since an edit — or any descendant of it — was last touched before the
# history forgets it. Long on purpose: the history is tiny and an analyst may
# come back to a tab hours later expecting Undo to work.
COUNTERFACTUAL_TTL = 24 * 60 * 60

# Materialized frames kept in memory. Each is a full copy of the base frame.
_FRAME_CACHE_MAX = 8


class CounterfactualUnknown(KeyError):
    """A counterfactual id that was never registered, or has expired."""


@dataclass(frozen=True)
class CounterfactualEdit:
    """One immutable edit: the previewed movement written into data."""

    cf_id: str
    base_dataset: str
    parent: str | None  # cf_id this edit stacks on; None = directly on the base
    node_id: str
    source_child_index: int
    target: dict[str, Any]  # the movement's target, JSON-safe
    strength: float
    row_positions: tuple[int, ...]  # positions in the frame — what `iloc` uses
    row_ids: tuple[int, ...]  # the frame's `row_id` values at those positions
    deltas_raw: dict[str, float]  # feature -> applied raw delta; MUTABLE features only
    created_at: float

    @property
    def dataset_key(self) -> str:
        return dataset_key(self.base_dataset, self.cf_id)

    def summary(self) -> dict[str, Any]:
        """The `CounterfactualEdit` of frontend/src/types.ts."""
        return {
            "cf_id": self.cf_id,
            "parent": self.parent,
            "base_dataset": self.base_dataset,
            "dataset_key": self.dataset_key,
            "node_id": self.node_id,
            "source_child_index": self.source_child_index,
            "target": dict(self.target),
            "strength": self.strength,
            "n_rows": len(self.row_positions),
            "features_changed": [f for f, d in self.deltas_raw.items() if d != 0.0],
            "deltas_raw": dict(self.deltas_raw),
            "created_at": self.created_at,
        }


_lock = threading.Lock()
_edits: dict[str, CounterfactualEdit] = {}
_last_access: dict[str, float] = {}
_frames: OrderedDict[str, pd.DataFrame] = OrderedDict()


# ── Keys ────────────────────────────────────────────────────────────────────


def dataset_key(base_dataset: str, cf_id: str) -> str:
    return f"{base_dataset}{KEY_MARK}{cf_id}"


def split_key(key: str) -> tuple[str, str | None]:
    """`"Iris (Low)@cf:1a2b3c"` -> `("Iris (Low)", "1a2b3c")`; a plain key -> `(key, None)`."""
    base, mark, cf_id = key.partition(KEY_MARK)
    return (base, cf_id) if mark else (key, None)


def is_counterfactual_key(key: str) -> bool:
    return KEY_MARK in key


# ── Identity ────────────────────────────────────────────────────────────────


def edit_id(
    *,
    base_dataset: str,
    parent: str | None,
    node_id: str,
    source_child_index: int,
    target: dict[str, Any],
    strength: float,
    row_positions: tuple[int, ...],
    deltas_raw: dict[str, float],
) -> str:
    """Content-derived id. Deltas are rounded to 12 significant digits so two
    computations of the same movement that differ in the last bits (a BLAS
    thread-count change between server runs, say) still name the same edit."""
    fields = {
        "node_id": node_id,
        "source_child_index": source_child_index,
        "target": target,
        "strength": float(f"{strength:.12g}"),
        "row_positions": list(row_positions),
        "deltas_raw": {k: float(f"{v:.12g}") for k, v in sorted(deltas_raw.items())},
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    signature = f"{base_dataset}|{parent or ''}|{canonical}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]


# ── History ─────────────────────────────────────────────────────────────────


def register(
    *,
    base_dataset: str,
    parent: str | None,
    node_id: str,
    source_child_index: int,
    target: dict[str, Any],
    strength: float,
    row_positions: list[int],
    row_ids: list[int],
    deltas_raw: dict[str, float],
) -> CounterfactualEdit:
    """Record an edit on top of `parent` (or the base) and return it. Idempotent:
    the same edit on the same parent returns the existing entry."""
    positions = tuple(int(p) for p in row_positions)
    now = time.time()
    with _lock:
        _sweep_locked(now)
        if parent is not None and parent not in _edits:
            raise CounterfactualUnknown(parent)
        cf_id = edit_id(
            base_dataset=base_dataset,
            parent=parent,
            node_id=node_id,
            source_child_index=source_child_index,
            target=target,
            strength=strength,
            row_positions=positions,
            deltas_raw=deltas_raw,
        )
        edit = _edits.get(cf_id)
        if edit is None:
            edit = CounterfactualEdit(
                cf_id=cf_id,
                base_dataset=base_dataset,
                parent=parent,
                node_id=node_id,
                source_child_index=int(source_child_index),
                target=dict(target),
                strength=float(strength),
                row_positions=positions,
                row_ids=tuple(int(r) for r in row_ids),
                deltas_raw={str(k): float(v) for k, v in deltas_raw.items()},
                created_at=now,
            )
            _edits[cf_id] = edit
        _touch_locked(cf_id, now)
        return edit


def get(cf_id: str) -> CounterfactualEdit:
    with _lock:
        edit = _edits.get(cf_id)
        if edit is None:
            raise CounterfactualUnknown(cf_id)
        _touch_locked(cf_id, time.time())
        return edit


def chain(cf_id: str) -> list[CounterfactualEdit]:
    """Every edit from the base down to `cf_id`, in application order."""
    with _lock:
        edits = _chain_locked(cf_id)
        _touch_locked(cf_id, time.time())
        return edits


def _chain_locked(cf_id: str) -> list[CounterfactualEdit]:
    edits: list[CounterfactualEdit] = []
    cursor: str | None = cf_id
    while cursor is not None:
        edit = _edits.get(cursor)
        if edit is None:
            raise CounterfactualUnknown(cf_id)
        edits.append(edit)
        cursor = edit.parent
    edits.reverse()
    return edits


def _touch_locked(cf_id: str, now: float) -> None:
    """Refresh the access time of an edit and of every ancestor: a chain stays
    alive as long as its youngest member is in use."""
    cursor: str | None = cf_id
    while cursor is not None:
        _last_access[cursor] = now
        edit = _edits.get(cursor)
        cursor = None if edit is None else edit.parent


def _sweep_locked(now: float) -> None:
    """Forget edits untouched for `COUNTERFACTUAL_TTL`, and their frames."""
    stale = [k for k, t in _last_access.items() if now - t > COUNTERFACTUAL_TTL]
    for cf_id in stale:
        _edits.pop(cf_id, None)
        _last_access.pop(cf_id, None)
        _frames.pop(cf_id, None)


# ── Frames ──────────────────────────────────────────────────────────────────


def apply_edit(frame: pd.DataFrame, edit: CounterfactualEdit) -> None:
    """Write one edit into `frame` IN PLACE — only ever called on a copy.

    Positional (`iloc`), because the tree is positional. A mutable feature
    stored as an integer column is widened to float first: a counterfactual
    delta is a real number, and the copy is ours to retype.
    """
    positions = np.asarray(edit.row_positions, dtype=int)
    for feature, delta in edit.deltas_raw.items():
        if delta == 0.0:
            continue
        if not pd.api.types.is_float_dtype(frame[feature]):
            frame[feature] = frame[feature].astype("float64")
        col = frame.columns.get_loc(feature)
        frame.iloc[positions, col] = frame.iloc[positions, col].to_numpy() + delta


def replay(base: pd.DataFrame, edits: list[CounterfactualEdit]) -> pd.DataFrame:
    """A deep copy of `base` with every edit applied in order. `base` is untouched."""
    frame = base.copy(deep=True)
    for edit in edits:
        apply_edit(frame, edit)
    return frame


def resolve_frame(
    base_dataset: str, cf_id: str, load_base: Callable[[str], pd.DataFrame]
) -> pd.DataFrame:
    """The materialized frame for `base@cf:cf_id`: from the LRU, or replayed.

    `load_base` is the plain loader (memoized upstream); it is called outside
    the lock because a first load of a large dataset can take seconds.
    """
    with _lock:
        edits = _chain_locked(cf_id)
        if edits[-1].base_dataset != base_dataset:
            # The key names a base the edit was never made on — a mismatched key
            # is an unknown key, not a frame of somebody else's edit.
            raise CounterfactualUnknown(dataset_key(base_dataset, cf_id))
        _touch_locked(cf_id, time.time())
        cached = _frames.get(cf_id)
        if cached is not None:
            _frames.move_to_end(cf_id)
            return cached
    frame = replay(load_base(base_dataset), edits)
    with _lock:
        _frames[cf_id] = frame
        _frames.move_to_end(cf_id)
        while len(_frames) > _FRAME_CACHE_MAX:
            _frames.popitem(last=False)
    return frame


def changed_rows(
    base_dataset: str, cf_id: str, load_base: Callable[[str], pd.DataFrame]
) -> tuple[list[str], list[dict[str, Any]]]:
    """For the export: every row any edit in the chain moved, with the original
    and the counterfactual value of every feature any edit changed.

    Returns `(columns, rows)`; columns are `row_id`, then `feature` and
    `feature__cf` per changed feature, in first-changed order.
    """
    edits = chain(cf_id)
    frame = resolve_frame(base_dataset, cf_id, load_base)
    base = load_base(base_dataset)
    positions = sorted({p for e in edits for p in e.row_positions})
    features: list[str] = []
    for edit in edits:
        for feature, delta in edit.deltas_raw.items():
            if delta != 0.0 and feature not in features:
                features.append(feature)
    columns = ["row_id"] + [c for f in features for c in (f, f"{f}__cf")]
    has_row_id = "row_id" in base.columns
    rows: list[dict[str, Any]] = []
    for p in positions:
        record: dict[str, Any] = {
            "row_id": int(base["row_id"].iloc[p]) if has_row_id else int(p)
        }
        for f in features:
            record[f] = float(base[f].iloc[p])
            record[f"{f}__cf"] = float(frame[f].iloc[p])
        rows.append(record)
    return columns, rows


# ── Introspection (tests) ───────────────────────────────────────────────────


def clear() -> None:
    """Forget every edit and frame (tests)."""
    with _lock:
        _edits.clear()
        _last_access.clear()
        _frames.clear()


def known_ids() -> list[str]:
    with _lock:
        return list(_edits)


def frame_cache_size() -> int:
    with _lock:
        return len(_frames)

"""Apply a movement: recompute it on the server, then record it as an edit.

`POST /api/counterfactual/apply` carries the same fields as `/api/movement`,
with `strength` required and positive. The server never trusts client deltas:
it recomputes the movement at that strength — through the same validation,
the same artifact cache and (for a cold UMAP node) the same job path — and
takes the applied raw deltas from its OWN response. The edit is then stacked on
whatever `dataset` names: the base, or an existing counterfactual key.

Response, the `CounterfactualApplyResponse` of frontend/src/types.ts::

    {"status": "ok", "dataset_key": "Iris (Low)@cf:1a2b3c4d5e6f", "cf_id": …,
     "edits": [<CounterfactualEdit summary>, …],   # the whole chain, base first
     "n_rows_changed": 64}

Status codes follow `/api/movement` (400 / 409); a strength that is missing,
non-finite, zero or above one is a 400 — applying nothing is not an edit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from backend import counterfactual as cf
from backend.movement import (
    MovementError,
    MovementRequestData,
    ValidatedMovement,
    compute_validated_movement,
    validate_request,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


@dataclass(frozen=True)
class ApplyRequestData:
    """`MovementRequest` with a required strength."""

    analysis_id: str
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any]
    node_id: str
    source_child_index: int
    target: dict[str, Any]
    strength: float | None = None

    def as_movement(self) -> MovementRequestData:
        return MovementRequestData(**asdict(self))


def validate_apply(
    req: ApplyRequestData, payload: Mapping[str, Any] | None
) -> ValidatedMovement:
    """The movement validation, plus: the strength must be in (0, 1]."""
    s = req.strength
    if (
        s is None
        or isinstance(s, bool)
        or not isinstance(s, int | float)
        or not math.isfinite(s)
        or not 0.0 < float(s) <= 1.0
    ):
        raise MovementError(
            400,
            f"Apply needs a strength in (0, 1]; got {s!r}. A strength of zero "
            "changes nothing and is not an edit.",
        )
    return validate_request(req.as_movement(), payload)


def _edit_target(v: ValidatedMovement) -> dict[str, Any]:
    if v.target_kind == "cluster":
        return {"kind": "cluster", "child_index": v.target_child_index}
    x, y = v.target_xy or (0.0, 0.0)
    return {"kind": "point", "x": float(x), "y": float(y)}


def apply_validated(
    v: ValidatedMovement, df: pd.DataFrame, dataset_key: str
) -> dict[str, Any]:
    """Recompute the movement at `v.strength`, register the edit, answer.

    `df` is the frame `dataset_key` resolves to (base or counterfactual) — the
    movement is computed on it, so the source cluster's rows and the deltas are
    relative to exactly the data the analyst is looking at. Row POSITIONS come
    from the tree; `row_ids` are read off the frame for the export and the UI.
    """
    response = compute_validated_movement(v, df)
    base_dataset, parent = cf.split_key(dataset_key)
    positions = [int(p) for p in (v.source_node.get("row_indices") or [])]
    if "row_id" in df.columns:
        row_ids = [int(r) for r in df["row_id"].iloc[positions].tolist()]
    else:
        row_ids = list(positions)
    deltas = {
        str(c["feature"]): float(c["applied_delta_raw"] or 0.0)
        for c in response["feature_changes"]
        if c["mutable"]
    }
    edit = cf.register(
        base_dataset=base_dataset,
        parent=parent,
        node_id=v.node_id,
        source_child_index=v.source_child_index,
        target=_edit_target(v),
        strength=float(response["applied_strength"]),
        row_positions=positions,
        row_ids=row_ids,
        deltas_raw=deltas,
    )
    return {
        "status": "ok",
        "dataset_key": edit.dataset_key,
        "cf_id": edit.cf_id,
        "edits": [e.summary() for e in cf.chain(edit.cf_id)],
        "n_rows_changed": len(positions),
    }

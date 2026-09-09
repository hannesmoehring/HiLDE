"""Counterfactual sessions: edits, keys, frames, replay, and the rebuild on a key.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_counterfactual
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_counterfactual.py

The `__main__` block auto-discovers every module-level `test_*` function in
definition order. The Iris payload is shared with `test_movement` (memoized).
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend import app as backend_app
from backend import counterfactual as cf
from backend import datasets as ds
from backend import movement as movement_module
from backend.counterfactual_apply import (
    ApplyRequestData,
    apply_validated,
    validate_apply,
)
from backend.movement import MovementError, clear_artifact_cache, compute_movement
from backend.serialize import (
    SCHEMA_VERSION,
    analysis_id,
    effective_config,
    serialize_tree,
)
from backend.tests import test_movement as tm
from src.config_defaults import default_config
from src.datasets import DATASETS
from src.evaluation.evaluate import start_evaluation

DATASET = tm.DATASET  # "Iris (Low)"


def _expect_error(status: int, fn, *args: Any, **kwargs: Any) -> MovementError:
    try:
        fn(*args, **kwargs)
    except MovementError as exc:
        assert exc.status_code == status, f"expected {status}, got {exc}"
        return exc
    raise AssertionError(f"expected a {status} MovementError, none was raised")


def _apply(
    payload: dict[str, Any], df: pd.DataFrame, dataset_key: str, **overrides: Any
) -> dict[str, Any]:
    """Validate + apply, exactly as the route does, without the HTTP layer."""
    meta = payload["meta"]
    fields: dict[str, Any] = {
        "analysis_id": meta["analysis_id"],
        "dataset": dataset_key,
        "feature_cols": meta["feature_cols"],
        "config": meta["config"],
        "node_id": "root/1",
        "source_child_index": 0,
        "target": {"kind": "cluster", "child_index": 1},
        "strength": 0.5,
    }
    fields.update(overrides)
    validated = validate_apply(ApplyRequestData(**fields), payload)
    return apply_validated(validated, df, dataset_key)


def _payload_for(dataset_key: str, df: pd.DataFrame, **config_overrides: Any):
    """Build + serialize a run on `df` under `dataset_key`, as `_build` does."""
    requested = {"method": "PCA", "hierarchical_layers": 2, **config_overrides}
    config = default_config()
    config.update(requested)  # type: ignore[typeddict-item]
    config["dataset_choice"] = dataset_key
    feats = ds.default_feature_cols(df)
    tree = start_evaluation(df, feats, config)
    return {
        "meta": {
            "dataset": dataset_key,
            "feature_cols": feats,
            "config": requested,
            "n_total": len(df),
            "analysis_id": analysis_id(dataset_key, feats, requested),
            "effective_config": effective_config(config),
            "schema_version": SCHEMA_VERSION,
        },
        "tree": serialize_tree(tree),  # type: ignore[arg-type]
    }


def _edit_of(response: dict[str, Any]) -> cf.CounterfactualEdit:
    return cf.get(response["cf_id"])


def _movable_node(tree: dict[str, Any]) -> str:
    """The first node (breadth first) with two or more child clusters. A rebuild
    on counterfactual data may reshape the hierarchy, so a test must not assume
    `root/1` is still internal — that is the honest result, not a failure."""
    queue = [tree]
    while queue:
        node = queue.pop(0)
        children = node.get("children") or []
        if len(children) >= 2 and node.get("embedding_original") is not None:
            return str(node["id"])
        queue.extend(c for c in children if not c.get("is_leaf"))
    raise AssertionError("no node with two child clusters")


# ── The edit lands on exactly the source rows × the mutable features ────────


def test_apply_changes_exactly_the_source_rows_of_the_mutable_features():
    cf.clear()
    payload, df = tm._payload("PCA")
    base_id = id(ds.load(DATASET))
    before = df.copy(deep=True)

    response = _apply(payload, df, DATASET)
    assert response["status"] == "ok"
    assert response["dataset_key"] == f"{DATASET}@cf:{response['cf_id']}"
    assert len(response["edits"]) == 1
    edit = _edit_of(response)
    positions = np.asarray(edit.row_positions, dtype=int)
    assert response["n_rows_changed"] == positions.size
    assert (
        positions.size
        == tm.find_node(payload["tree"], "root/1")["children"][0]["n_points"]
    )

    # The base is untouched: same object as before, same values.
    assert id(ds.load(DATASET)) == base_id
    assert df.equals(before)

    frame = ds.load(response["dataset_key"])
    assert frame is not df
    feats = payload["meta"]["feature_cols"]
    diff = frame[feats].to_numpy(dtype=float) - before[feats].to_numpy(dtype=float)
    untouched = np.ones(len(df), dtype=bool)
    untouched[positions] = False
    assert np.all(diff[untouched] == 0.0)
    expected = np.array([edit.deltas_raw[f] for f in feats])
    assert np.all(np.abs(diff[positions] - expected) < 1e-12)
    assert np.any(expected != 0.0)
    # Labels and ids are never counterfactuals.
    for col in df.columns:
        if col not in feats:
            assert frame[col].equals(before[col]), col

    # JSON-safe, like every other response.
    assert json.loads(json.dumps(response, allow_nan=False))["status"] == "ok"


def test_immutable_features_stay_exactly_as_they_were():
    cf.clear()
    payload, df = tm._label_payload(tm._MIXED_FEATURES)
    # This run's root has only leaf children, so the movement is made on the root.
    response = _apply(
        payload,
        df,
        DATASET,
        node_id="root",
        target={"kind": "point", "x": 1.0, "y": -0.5},
    )
    edit = _edit_of(response)
    assert "target_setosa" not in edit.deltas_raw  # immutable: not even recorded
    frame = ds.load(response["dataset_key"])
    assert frame["target_setosa"].equals(df["target_setosa"])
    assert edit.summary()["features_changed"] == [
        f for f in tm._MIXED_FEATURES if not f.startswith("target_")
    ]


# ── Positions, not ids ──────────────────────────────────────────────────────

_SYNTHETIC = "CF test (Low)"


@functools.cache
def _synthetic_frame() -> pd.DataFrame:
    """Three blobs, a NON-RANGE index, and row ids that differ from positions."""
    rng = np.random.default_rng(7)
    centres = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 1.0], [0.0, 6.0, -1.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(30, 3)) for c in centres])
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame["target_a"] = np.repeat([True, False, False], 30)
    frame["row_id"] = 5000 + np.arange(len(frame))
    frame.index = 1000 + 2 * np.arange(len(frame))  # positions ≠ labels ≠ row ids
    return frame


def _with_synthetic_dataset(fn):
    """Register the synthetic loader for the duration of one test."""
    DATASETS[_SYNTHETIC] = _synthetic_frame
    try:
        return fn()
    finally:
        DATASETS.pop(_SYNTHETIC, None)


def test_non_range_index_frames_change_the_right_rows_and_export_row_ids():
    cf.clear()

    def run() -> None:
        df = ds.load(_SYNTHETIC)
        payload = _payload_for(
            _SYNTHETIC, df, hclust_min_cluster_size=10, hclust_min_samples=5
        )
        root = payload["tree"]
        assert len(root["children"] or []) >= 2, "the blobs must split at the root"
        # A source cluster larger than the preview cap: every member must change,
        # never just the sample.
        original_cap = movement_module.MAX_MOVEMENT_PREVIEW_POINTS
        movement_module.MAX_MOVEMENT_PREVIEW_POINTS = 5
        try:
            response = _apply(
                payload,
                df,
                _SYNTHETIC,
                node_id="root",
                target={"kind": "cluster", "child_index": 1},
                strength=1.0,
            )
        finally:
            movement_module.MAX_MOVEMENT_PREVIEW_POINTS = original_cap
        edit = _edit_of(response)
        source_rows = root["children"][0]["row_indices"]
        assert list(edit.row_positions) == list(source_rows)
        assert len(edit.row_positions) > 5
        # Positions are positions; row ids are what the frame says at them.
        assert list(edit.row_ids) == [5000 + p for p in edit.row_positions]
        assert all(p < len(df) for p in edit.row_positions)

        frame = ds.load(response["dataset_key"])
        positions = np.asarray(edit.row_positions)
        moved = frame["f0"].to_numpy() - df["f0"].to_numpy()
        assert np.all(np.abs(moved[positions] - edit.deltas_raw["f0"]) < 1e-12)
        mask = np.ones(len(df), dtype=bool)
        mask[positions] = False
        assert np.all(moved[mask] == 0.0)
        # The index labels survived untouched, so `loc` by label agrees with iloc.
        assert frame.index.equals(df.index)
        label = df.index[positions[0]]
        assert (
            abs(frame.loc[label, "f0"] - df.loc[label, "f0"] - edit.deltas_raw["f0"])
            < 1e-12
        )

        columns, rows = cf.changed_rows(_SYNTHETIC, edit.cf_id, ds.load)
        assert columns[0] == "row_id"
        assert [r["row_id"] for r in rows] == sorted(edit.row_ids)
        assert set(columns) >= {"f0", "f0__cf"}
        first = rows[0]
        assert abs(first["f0__cf"] - first["f0"] - edit.deltas_raw["f0"]) < 1e-12

    _with_synthetic_dataset(run)


# ── History vs frames ───────────────────────────────────────────────────────


def test_history_survives_frame_eviction_and_ids_are_content_derived():
    cf.clear()
    payload, df = tm._payload("PCA")
    keys: list[str] = []
    for i in range(1, 10):  # nine siblings on the base; the frame LRU holds eight
        response = _apply(payload, df, DATASET, strength=i / 10)
        keys.append(response["dataset_key"])
        ds.load(response["dataset_key"])  # materialize
    assert len(set(keys)) == 9
    assert cf.frame_cache_size() == cf._FRAME_CACHE_MAX == 8
    assert len(cf.known_ids()) == 9  # the history kept every edit

    # The evicted (oldest) key still resolves: replayed from the base.
    first = cf.get(cf.split_key(keys[0])[1])
    frame = ds.load(keys[0])
    positions = np.asarray(first.row_positions)
    feats = payload["meta"]["feature_cols"]
    expected = df[feats].to_numpy(dtype=float).copy()
    expected[positions] += np.array([first.deltas_raw[f] for f in feats])
    assert np.all(np.abs(frame[feats].to_numpy(dtype=float) - expected) < 1e-12)

    # Same edit, same parent -> the same id, and still nine entries.
    again = _apply(payload, df, DATASET, strength=0.1)
    assert again["dataset_key"] == keys[0]
    assert len(cf.known_ids()) == 9

    # The base dataset is part of the identity.
    common: dict[str, Any] = {
        "parent": None,
        "node_id": "root/1",
        "source_child_index": 0,
        "target": {"kind": "cluster", "child_index": 1},
        "strength": 0.5,
        "row_positions": (1, 2, 3),
        "deltas_raw": {"a": 0.25},
    }
    assert cf.edit_id(base_dataset="A", **common) != cf.edit_id(
        base_dataset="B", **common
    )
    # …and so is the parent: stacking the same edit deeper is a new snapshot.
    assert cf.edit_id(base_dataset="A", **common) != cf.edit_id(
        base_dataset="A", **{**common, "parent": "abc123"}
    )


def test_chain_of_two_edits_applies_in_order_and_the_parent_key_still_resolves():
    cf.clear()
    payload, df = tm._payload("PCA")
    first = _apply(payload, df, DATASET, strength=0.5)
    frame_a = ds.load(first["dataset_key"])
    edit_a = _edit_of(first)

    # The second edit is made ON the counterfactual run: its own payload, built
    # on the counterfactual frame under the counterfactual key.
    payload_a = _payload_for(first["dataset_key"], frame_a)
    second = _apply(
        payload_a,
        frame_a,
        first["dataset_key"],
        node_id=_movable_node(payload_a["tree"]),
        strength=0.5,
    )
    edit_b = _edit_of(second)
    assert edit_b.parent == edit_a.cf_id
    assert [e["cf_id"] for e in second["edits"]] == [edit_a.cf_id, edit_b.cf_id]

    feats = payload["meta"]["feature_cols"]
    expected = df[feats].to_numpy(dtype=float).copy()
    expected[np.asarray(edit_a.row_positions)] += [edit_a.deltas_raw[f] for f in feats]
    expected_a = expected.copy()
    expected[np.asarray(edit_b.row_positions)] += [edit_b.deltas_raw[f] for f in feats]
    frame_b = ds.load(second["dataset_key"])
    assert np.all(np.abs(frame_b[feats].to_numpy(dtype=float) - expected) < 1e-12)
    # Undo = the parent key, which still resolves to exactly the first edit.
    assert np.all(
        np.abs(ds.load(first["dataset_key"])[feats].to_numpy(dtype=float) - expected_a)
        < 1e-12
    )
    # And the frame cache holds both without either replacing the other.
    assert cf.frame_cache_size() >= 2


def test_expired_edits_are_swept_but_ancestors_of_live_edits_survive():
    cf.clear()
    payload, df = tm._payload("PCA")
    first = _apply(payload, df, DATASET, strength=0.3)
    parent = first["cf_id"]
    stale = _apply(payload, df, DATASET, strength=0.7)["cf_id"]
    # A child on top of `parent`, registered directly (no second build needed).
    edit = cf.get(parent)
    child = cf.register(
        base_dataset=DATASET,
        parent=parent,
        node_id=edit.node_id,
        source_child_index=edit.source_child_index,
        target=edit.target,
        strength=0.9,
        row_positions=list(edit.row_positions),
        row_ids=list(edit.row_ids),
        deltas_raw=edit.deltas_raw,
    )
    long_ago = 0.0
    with cf._lock:
        cf._last_access[parent] = long_ago
        cf._last_access[stale] = long_ago
    cf.chain(child.cf_id)  # touching the child refreshes its whole ancestry
    with cf._lock:
        cf._sweep_locked(cf.COUNTERFACTUAL_TTL + 10.0)  # "now", far past the stale one
    ids = set(cf.known_ids())
    assert parent in ids and child.cf_id in ids
    assert stale not in ids


# ── The rebuild on a counterfactual key ─────────────────────────────────────


def test_analysis_on_a_counterfactual_key_rebuilds_without_touching_the_run_cache():
    cf.clear()
    payload, df = tm._payload("PCA")
    # A large move: far enough that the hierarchy cannot come out the same.
    response = _apply(
        payload,
        df,
        DATASET,
        target={"kind": "point", "x": 60.0, "y": 60.0},
        strength=1.0,
    )
    key = response["dataset_key"]
    frame = ds.load(key)

    env_before = {k: os.environ.get(k) for k in ("HILDE_HOSTING", "HILDE_CACHE_DIR")}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["HILDE_HOSTING"] = "1"
        os.environ["HILDE_CACHE_DIR"] = tmp
        try:
            req = backend_app.AnalysisRequest(
                dataset=key,
                feature_cols=payload["meta"]["feature_cols"],
                config=payload["meta"]["config"],
            )
            cache_key = backend_app._cache_key(key, req.feature_cols, req.config)
            backend_app._build(req, frame, cache_key)
            assert list(Path(tmp).iterdir()) == [], (
                "hosting mode persisted a counterfactual"
            )
        finally:
            for k, v in env_before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    rebuilt = backend_app._cache_get(cache_key)
    assert rebuilt is not None
    assert rebuilt["meta"]["dataset"] == key
    assert rebuilt["meta"]["analysis_id"] != payload["meta"]["analysis_id"]
    assert rebuilt["tree"] != payload["tree"]

    # Movement on the counterfactual run: a new artifact under the new id.
    clear_artifact_cache()
    moved = compute_movement(
        tm._request(
            rebuilt, node_id="root", target={"kind": "point", "x": 0.0, "y": 0.0}
        ),
        rebuilt,
        frame,
    )
    assert moved["status"] == "ok"
    assert moved["analysis_id"] == rebuilt["meta"]["analysis_id"]


# ── Keys, listing, errors ───────────────────────────────────────────────────


def test_counterfactual_keys_are_not_listed_and_unknown_ids_are_404():
    cf.clear()
    assert not any(cf.is_counterfactual_key(k) for k in ds.dataset_keys())
    assert cf.split_key("Iris (Low)@cf:abc") == ("Iris (Low)", "abc")
    assert cf.split_key("Iris (Low)") == ("Iris (Low)", None)
    assert ds.base_key("Iris (Low)@cf:abc") == "Iris (Low)"

    try:
        ds.load("Iris (Low)@cf:deadbeef0000")
    except cf.CounterfactualUnknown:
        pass
    else:
        raise AssertionError("an unknown counterfactual id must not resolve")

    from fastapi import HTTPException

    for key, fragment in (
        ("Iris (Low)@cf:deadbeef0000", "expired or unknown"),
        ("No such dataset@cf:deadbeef0000", "Unknown dataset"),
        ("No such dataset", "Unknown dataset"),
    ):
        try:
            backend_app._load_dataset(key)
        except HTTPException as exc:
            assert exc.status_code == 404 and fragment in str(exc.detail), key
        else:
            raise AssertionError(f"{key} must be a 404")


def test_apply_rejects_a_strength_that_changes_nothing():
    payload, _ = tm._payload("PCA")
    for bad in (None, 0.0, -0.2, 1.5, float("nan")):
        exc = _expect_error(
            400,
            validate_apply,
            ApplyRequestData(**_fields(payload, strength=bad)),
            payload,
        )
        assert "strength" in exc.detail.lower()
    # Everything the movement endpoint refuses, Apply refuses the same way.
    _expect_error(409, validate_apply, ApplyRequestData(**_fields(payload)), None)
    _expect_error(
        400,
        validate_apply,
        ApplyRequestData(**_fields(payload, node_id="root/0")),
        payload,
    )


def _fields(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    meta = payload["meta"]
    fields: dict[str, Any] = {
        "analysis_id": meta["analysis_id"],
        "dataset": DATASET,
        "feature_cols": meta["feature_cols"],
        "config": meta["config"],
        "node_id": "root/1",
        "source_child_index": 0,
        "target": {"kind": "cluster", "child_index": 1},
        "strength": 0.5,
    }
    fields.update(overrides)
    return fields


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"OK — {_name}")
    print("OK — counterfactual sessions passed")

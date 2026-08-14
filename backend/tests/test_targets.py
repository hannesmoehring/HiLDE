"""Selection target-value stats over the real dataset registry.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
or with pytest:
    PYTHONPATH=. pytest backend/tests/test_targets.py
"""

from __future__ import annotations

import json

from backend import datasets as ds
from backend.targets import compute_targets

_KEYS = {
    "feature",
    "is_boolean",
    "sel_min",
    "sel_max",
    "sel_mean",
    "global_min",
    "global_max",
    "global_mean",
}


def _targets_of(df) -> list[str]:
    feats = set(ds.default_feature_cols(df))
    return [str(c) for c in df.columns if c != "row_id" and c not in feats]


def test_one_hot_targets_report_class_shares():
    df = ds.load("Iris (Low)")
    node = list(range(150))
    setosa = [i for i in node if bool(df["target_setosa"].iloc[i])]

    out = compute_targets(df, _targets_of(df), node, setosa)

    assert out["n_selected"] == len(setosa) == 50
    by_feature = {t["feature"]: t for t in out["targets"]}
    assert set(by_feature) == {"target_setosa", "target_versicolor", "target_virginica"}
    for t in out["targets"]:
        assert set(t.keys()) == _KEYS
        assert t["is_boolean"] is True
        # One-hot mean is a class share; the dataset base rate is the reference.
        assert t["global_mean"] == 1 / 3
    assert by_feature["target_setosa"]["sel_mean"] == 1.0
    assert by_feature["target_versicolor"]["sel_mean"] == 0.0

    # Starlette encodes with allow_nan=False.
    json.dumps(out, allow_nan=False)


def test_continuous_target_reports_a_range():
    df = ds.load("Swiss roll (Low)")
    out = compute_targets(df, _targets_of(df), list(range(1500)), list(range(200)))

    (t,) = out["targets"]
    assert t["feature"] == "target_manifold_position"
    assert t["is_boolean"] is False
    assert (
        t["global_min"]
        <= t["sel_min"]
        <= t["sel_mean"]
        <= t["sel_max"]
        <= t["global_max"]
    )


def test_edge_cases_stay_json_safe():
    df = ds.load("Iris (Low)")

    empty = compute_targets(df, ["target_setosa"], [0, 1, 2], [])
    assert empty["n_selected"] == 0
    assert empty["targets"][0]["sel_mean"] is None  # no NaN on the wire
    json.dumps(empty, allow_nan=False)

    # A column that is not in the frame is dropped, not an error.
    assert compute_targets(df, ["not_a_column"], [0, 1, 2], [0])["targets"] == []


if __name__ == "__main__":
    test_one_hot_targets_report_class_shares()
    test_continuous_target_reports_a_range()
    test_edge_cases_stay_json_safe()
    print("OK — target stats passed")

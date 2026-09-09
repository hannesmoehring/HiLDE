"""Per-step bookkeeping of the greedy predicate generator and its backend wrapper.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_predicate.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend import datasets as ds
from backend.predicate import compute_predicate
from src.analysis.predicate_generator import generate_predicate

_STEP_FIELDS = (
    "predicate_f1_before",
    "predicate_f1_after",
    "predicate_f1_gain",
    "predicate_n_matched",
    "predicate_precision",
    "predicate_recall",
)

_TOL = 1e-9


def _selected(rows: list[dict]) -> list[dict]:
    return sorted(
        (r for r in rows if r["in_predicate"]), key=lambda r: r["predicate_step"]
    )


def _tie_case() -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """A compact 35-point selection over five features, two of them identical.

    `a` and `a_dup` carry the same values, so the greedy tie-break is exercised on
    every step, and the RCM 1.0 and 0.9 runs select the same clauses in different
    orders — the cheap stand-in for the wine E2 regression.
    """
    rng = np.random.default_rng(11)
    base = rng.normal(size=(240, 4))
    background = np.column_stack(
        [base[:, 0], base[:, 0], base[:, 1], base[:, 2], base[:, 3]]
    )
    centre = np.array([0.8, 0.8, -0.6, 0.4, 0.0])
    sel = np.sort(np.argsort(np.linalg.norm(background - centre, axis=1))[:35]).tolist()
    sel_df = pd.DataFrame(background[sel], columns=["a", "a_dup", "b", "c", "d"])
    return sel_df, background, sel


def _tie_rows(threshold: float = 1.0, labelled: bool = True) -> list[dict]:
    sel_df, background, sel = _tie_case()
    return generate_predicate(
        "db",
        sel_df,
        background,
        threshold=threshold,
        selected_indices=sel if labelled else None,
    )


def _degenerate_frame() -> tuple[pd.DataFrame, list[str], list[int]]:
    """A selection whose per-feature range already spans the background.

    Every clause then matches every point, so no clause beats the empty
    conjunction and the greedy loop stops before its first step.
    """
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, size=(120, 3))
    values[0] = -1.0
    values[1] = 2.0
    cols = ["p", "q", "r"]
    return pd.DataFrame(values, columns=cols), cols, [0, 1, 2, 3, 4]


def _assert_internally_consistent(rows: list[dict]) -> None:
    """Steps contiguous from 0, each transition closed, gains telescoping."""
    chosen = _selected(rows)
    assert [r["predicate_step"] for r in chosen] == list(range(len(chosen)))

    baseline = rows[0]["predicate_baseline_f1"]
    for row in chosen:
        assert (
            abs(
                row["predicate_f1_before"]
                + row["predicate_f1_gain"]
                - row["predicate_f1_after"]
            )
            < _TOL
        )
    total = baseline + sum(r["predicate_f1_gain"] for r in chosen)
    assert abs(total - rows[0]["predicate_f1"]) < _TOL


def test_selected_clause_gains_are_consistent():
    rows = _tie_rows()
    chosen = _selected(rows)
    assert chosen, "the tie case must select at least one clause"

    for row in chosen:
        for field in _STEP_FIELDS:
            assert row[field] is not None
        assert row["predicate_f1_gain"] > 0
        assert (
            abs(
                row["predicate_f1_before"]
                + row["predicate_f1_gain"]
                - row["predicate_f1_after"]
            )
            < _TOL
        )
        assert row["predicate_n_matched"] >= 1


def test_gains_telescope_to_the_final_f1():
    rows = _tie_rows()
    baseline = rows[0]["predicate_baseline_f1"]
    gains = sum(r["predicate_f1_gain"] for r in _selected(rows))
    assert abs(baseline + gains - rows[0]["predicate_f1"]) < _TOL
    # The baseline is stamped identically on every row, like predicate_f1.
    assert {r["predicate_baseline_f1"] for r in rows} == {baseline}


def test_steps_are_contiguous_from_zero():
    chosen = _selected(_tie_rows())
    assert [r["predicate_step"] for r in chosen] == list(range(len(chosen)))


def test_first_clause_is_the_best_standalone_clause():
    rows = _tie_rows()
    first = _selected(rows)[0]
    # Step 0 scores every clause against the all-matching mask, so the clause it
    # accepts is the standalone maximum and its gain closes on the baseline.
    assert first["clause_f1"] == max(r["clause_f1"] for r in rows)
    assert (
        abs(
            first["predicate_f1_gain"]
            - (first["clause_f1"] - first["predicate_baseline_f1"])
        )
        < _TOL
    )


def test_unselected_clauses_have_no_step_fields():
    rows = _tie_rows()
    unselected = [r for r in rows if not r["in_predicate"]]
    assert unselected
    for row in unselected:
        assert row["predicate_step"] is None
        for field in _STEP_FIELDS:
            assert row[field] is None
        assert isinstance(row["predicate_baseline_f1"], float)


def test_no_label_path_leaves_step_fields_none():
    rows = _tie_rows(labelled=False)
    for row in rows:
        assert row["predicate_step"] is None
        for field in _STEP_FIELDS:
            assert field in row and row[field] is None
        assert row["predicate_baseline_f1"] == 0.0
        assert row["predicate_f1"] == 0.0


def test_identical_columns_break_ties_by_column_order():
    rows = _tie_rows()
    by_feature = {r["feature"]: r for r in rows}
    assert by_feature["a"]["clause_f1"] == by_feature["a_dup"]["clause_f1"]
    # The earlier column wins the tie; the later one then gains nothing.
    assert by_feature["a"]["predicate_step"] == 0
    assert by_feature["a_dup"]["in_predicate"] is False


def test_predicate_with_no_clause_reports_the_baseline_as_final():
    df, cols, sel = _degenerate_frame()
    out = compute_predicate(df, cols, True, list(range(len(df))), sel, "local")

    assert out["summary"]["n_features_used"] == 0
    assert out["summary"]["predicate_baseline_f1"] == out["summary"]["predicate_f1"]
    assert all(not r["in_predicate"] for r in out["full"])


def test_backend_response_is_json_safe():
    df = ds.load("Iris (Low)")
    feats = ds.default_feature_cols(df)
    node = list(range(len(df)))
    sel = [i for i in node if bool(df["target_setosa"].iloc[i])]

    out = compute_predicate(df, feats, True, node, sel, "local")

    # Starlette encodes with allow_nan=False — the new floats must survive it.
    assert len(json.dumps(out, allow_nan=False)) > 0
    summary = out["summary"]
    assert summary["predicate_baseline_f1"] <= summary["predicate_f1"]
    assert summary["trimmed_predicate_f1"] == out["trimmed"][0]["predicate_f1"]
    assert summary["trimmed_baseline_f1"] == out["trimmed"][0]["predicate_baseline_f1"]


def test_summary_reports_the_background_size_the_baseline_is_scored_on():
    # The baseline is 2|S|/(N+|S|); the summary carries N so the UI can label it
    # "no clauses (all N points)" with the real count, not a placeholder letter.
    df = ds.load("Iris (Low)")
    feats = ds.default_feature_cols(df)
    node = [i for i in range(len(df)) if i % 3 != 0]  # a 100-row sub-node
    sel = [i for i, row in enumerate(node) if bool(df["target_setosa"].iloc[row])]

    local = compute_predicate(df, feats, True, node, sel, "local")["summary"]
    assert local["n_background"] == len(node)
    assert local["n_selected"] == len(sel)
    expected = 2 * len(sel) / (len(node) + len(sel))
    assert abs(local["predicate_baseline_f1"] - expected) < _TOL

    whole = compute_predicate(df, feats, True, node, sel, "global")["summary"]
    assert whole["n_background"] == len(df)
    expected = 2 * len(sel) / (len(df) + len(sel))
    assert abs(whole["predicate_baseline_f1"] - expected) < _TOL


def test_rcm_runs_are_independently_consistent():
    df = ds.load("Iris (Low)")
    feats = ds.default_feature_cols(df)
    node = list(range(len(df)))
    sel = [i for i in node if bool(df["target_setosa"].iloc[i])]

    out = compute_predicate(df, feats, True, node, sel, "local")

    full_clauses = {r["feature"] for r in out["full"] if r["in_predicate"]}
    trimmed_clauses = {r["feature"] for r in out["trimmed"] if r["in_predicate"]}
    assert full_clauses != trimmed_clauses, "this fixture is meant to diverge"
    _assert_internally_consistent(out["full"])
    _assert_internally_consistent(out["trimmed"])


def test_greedy_clause_order_is_stable():
    # Regression stand-in for the wine E2 check: the tie-break must not reorder
    # the greedy construction. The 0.9 run legitimately differs from the 1.0 one.
    strict = [r["feature"] for r in _selected(_tie_rows(threshold=1.0))]
    relaxed = [r["feature"] for r in _selected(_tie_rows(threshold=0.9))]
    assert strict == ["a", "b", "c", "d"]
    assert relaxed == ["a", "d", "c", "b"]


if __name__ == "__main__":
    test_selected_clause_gains_are_consistent()
    test_gains_telescope_to_the_final_f1()
    test_steps_are_contiguous_from_zero()
    test_first_clause_is_the_best_standalone_clause()
    test_unselected_clauses_have_no_step_fields()
    test_no_label_path_leaves_step_fields_none()
    test_identical_columns_break_ties_by_column_order()
    test_predicate_with_no_clause_reports_the_baseline_as_final()
    test_backend_response_is_json_safe()
    test_summary_reports_the_background_size_the_baseline_is_scored_on()
    test_rcm_runs_are_independently_consistent()
    test_greedy_clause_order_is_stable()
    print("OK — predicate step bookkeeping passed")

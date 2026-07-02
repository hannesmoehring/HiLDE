"""Round-trip test for the tree serializer against a real analysis tree.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
or with pytest:
    PYTHONPATH=. pytest backend/tests/test_serialize.py
"""

from __future__ import annotations

import json

from backend import datasets as ds
from backend.serialize import serialize_tree
from src.config_defaults import default_config
from src.evaluation.evaluate import start_evaluation

_NODE_KEYS = {
    "id", "is_leaf", "depth", "n_points", "row_indices",
    "embedding_original", "embedding_original_variance", "rel_position",
    "kde", "rel_characteristics", "outlier_scores", "scores", "children",
}


def _build():
    config = default_config()
    config["hierarchical_layers"] = 2
    df = ds.load("Iris (Low)")
    feats = ds.default_feature_cols(df)
    tree = start_evaluation(df, feats, config)
    return serialize_tree(tree)


def _walk(node, seen):
    seen.append(node)
    for child in node["children"] or []:
        _walk(child, seen)


def test_serialized_tree_is_json_safe_and_well_formed():
    root = _build()

    # Starlette encodes with allow_nan=False — non-finite floats must be gone.
    blob = json.dumps(root, allow_nan=False)
    assert len(blob) > 0

    nodes = []
    _walk(root, nodes)
    assert len(nodes) > 1  # root + at least one child

    for n in nodes:
        assert set(n.keys()) == _NODE_KEYS, f"unexpected keys on {n['id']}: {set(n.keys()) ^ _NODE_KEYS}"
        assert isinstance(n["id"], str)
        assert isinstance(n["is_leaf"], bool)
        assert isinstance(n["depth"], int)
        assert n["n_points"] == len(n["row_indices"])
        # embedding is Nx2 or empty
        for xy in n["embedding_original"]:
            assert len(xy) == 2
        if n["kde"] is not None:
            assert n["kde"]["resolution"] == len(n["kde"]["grid"])
        for rc in n["rel_characteristics"]:
            assert set(rc.keys()) == {"feature", "z_mean", "z_std", "raw_mean"}
        if n["is_leaf"]:
            assert n["children"] is None
            assert n["outlier_scores"] is None
        else:
            assert isinstance(n["children"], list)

    # scaler is server-only and must not leak to the wire
    assert "scaler" not in root


if __name__ == "__main__":
    test_serialized_tree_is_json_safe_and_well_formed()
    print("OK — serialize round-trip passed")

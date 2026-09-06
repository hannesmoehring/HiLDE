"""Alignment, validation and refit-parity tests for cluster movement.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_movement.py

The `__main__` block auto-discovers every module-level `test_*` function in
definition order, so a new test only has to be defined — never registered.

This file grows with the feature: what is here now covers the run identity and
the serialized `effective_config` the movement endpoint is built on. The
alignment, validation, PCA and UMAP cases land with `backend/movement.py`.
"""

from __future__ import annotations

from typing import Any

from backend import datasets as ds
from backend.app import _cache_key
from backend.serialize import (
    SCHEMA_VERSION,
    analysis_id,
    effective_config,
    run_signature,
    serialize_tree,
)
from src.config_defaults import default_config
from src.evaluation.evaluate import start_evaluation

DATASET = "Iris (Low)"

_payload_cache: dict[str, tuple[dict[str, Any], Any]] = {}


def _payload(method: str) -> tuple[dict[str, Any], Any]:
    """Build a real analysis and serialize it exactly as `backend/app.py::_build`.

    Two layers so the tree has an internal node below the root (`root/1`), which
    the node-lookup and validation cases need. Memoized: the build is the
    expensive part of this module.
    """
    if method in _payload_cache:
        return _payload_cache[method]
    requested = {"method": method, "hierarchical_layers": 2}
    config = default_config()
    config.update(requested)  # type: ignore[typeddict-item]
    config["dataset_choice"] = DATASET  # the build banner names what it is fed
    df = ds.load(DATASET)
    feats = ds.default_feature_cols(df)
    tree = start_evaluation(df, feats, config)  # mutates `config` in place
    payload = {
        "meta": {
            "dataset": DATASET,
            "feature_cols": feats,
            "config": requested,
            "n_total": len(df),
            "analysis_id": analysis_id(DATASET, feats, requested),
            "effective_config": effective_config(config),
            "schema_version": SCHEMA_VERSION,
        },
        "tree": serialize_tree(tree),  # type: ignore[arg-type]
    }
    _payload_cache[method] = (payload, df)
    return payload, df


# ── Run identity ────────────────────────────────────────────────────────────


def test_analysis_id_matches_the_tree_cache_key_shape():
    # One identity for a run, not two: `analysis_id` hashes exactly the string
    # `backend/app.py::_cache_key` builds (plus the schema version). If either
    # side drifts, this fails rather than silently 409-ing every movement.
    payload, _ = _payload("PCA")
    meta = payload["meta"]
    assert run_signature(
        meta["dataset"], meta["feature_cols"], meta["config"]
    ) == _cache_key(meta["dataset"], meta["feature_cols"], meta["config"])
    assert (
        analysis_id(meta["dataset"], meta["feature_cols"], meta["config"])
        == meta["analysis_id"]
    )
    assert analysis_id(meta["dataset"], meta["feature_cols"], {}) != meta["analysis_id"]


def test_effective_config_records_the_refit_knobs_the_build_used():
    # `effective_config` is taken AFTER `start_evaluation` returned, i.e. from the
    # config `compute_analysis_tree` mutated in place — the only config a refit
    # may read. It carries the reducer knobs and nothing that is not JSON.
    payload, _ = _payload("PCA")
    meta = payload["meta"]
    effective = meta["effective_config"]
    assert meta["schema_version"] == SCHEMA_VERSION == 2
    assert effective["method"] == "PCA"
    assert effective["normalize"] is True
    assert isinstance(effective["umap_n_neighbors"], int)
    assert "selected_df" not in effective and "analysis_tree" not in effective
    # The requested config is what the id digests, so the id is reproducible
    # from a request alone; the effective one is what the refit reads.
    assert meta["config"] == {"method": "PCA", "hierarchical_layers": 2}


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"OK — {_name}")
    print("OK — movement foundations passed")

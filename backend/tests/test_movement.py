"""Alignment, validation and refit-parity tests for cluster movement.

Run standalone (no pytest required):
    PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
or with pytest (a dev dependency):
    PYTHONPATH=. .venv/bin/python -m pytest backend/tests/test_movement.py

The `__main__` block auto-discovers every module-level `test_*` function in
definition order, so a new test only has to be defined — never registered.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend import datasets as ds
from backend.app import _cache_key
from backend.movement import (
    ALIGNMENT_RMSE_MAX,
    MovementError,
    MovementRequestData,
    ProjectionArtifact,
    _refit_node_projection,
    align_embeddings,
    build_projection_artifact,
    find_node,
    require_alignment,
    validate_request,
)
from backend.serialize import (
    SCHEMA_VERSION,
    analysis_id,
    effective_config,
    run_signature,
    serialize_tree,
)
from src.analysis.analysis_routine import fit_node_projection
from src.config_defaults import default_config
from src.evaluation.evaluate import start_evaluation

DATASET = "Iris (Low)"

# Canary tolerances (binding, see the spec's alignment-canary amendment). A
# same-process refit must reproduce the serialized embedding at ~identity. These
# are not to be widened: a failure here means a refit is reading a different
# configuration than the build did.
PCA_CANARY_RMSE = 1e-6
UMAP_CANARY_RMSE = 1e-3

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


def _request(payload: dict[str, Any], **overrides: Any) -> MovementRequestData:
    meta = payload["meta"]
    fields: dict[str, Any] = {
        "analysis_id": meta["analysis_id"],
        "dataset": meta["dataset"],
        "feature_cols": meta["feature_cols"],
        "config": meta["config"],
        "node_id": "root",
        "source_child_index": 0,
        "target": {"kind": "point", "x": 0.5, "y": -0.25},
        "strength": None,
    }
    fields.update(overrides)
    return MovementRequestData(**fields)


def _expect_error(status: int, fn, *args: Any, **kwargs: Any) -> MovementError:
    try:
        fn(*args, **kwargs)
    except MovementError as exc:
        assert exc.status_code == status, (
            f"expected {status}, got {exc.status_code}: {exc.detail}"
        )
        return exc
    raise AssertionError(f"expected a {status} MovementError, none was raised")


def _cloud(n: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 2))


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _artifact(embedding: np.ndarray) -> ProjectionArtifact:
    """A minimal artifact — `require_alignment` only reads `embedding`."""
    n = embedding.shape[0]
    return ProjectionArtifact(
        analysis_id="test",
        node_id="root",
        method="PCA",
        parent_row_indices=np.arange(n),
        X_parent=np.zeros((n, 2)),
        embedding=embedding,
        reducer=None,
        scaler=None,
    )


def _as_serialized(embedding: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in embedding]


# ── Alignment: the similarity transform ─────────────────────────────────────


def test_alignment_identity_is_identity():
    fitted = _cloud(seed=0)
    alignment = align_embeddings(fitted, fitted)
    assert alignment.rmse < 1e-12
    assert alignment.max_residual < 1e-12
    assert alignment.n_points == fitted.shape[0]
    assert abs(alignment.scale - 1.0) < 1e-9
    assert np.allclose(alignment.rotation, np.eye(2), atol=1e-9)
    assert np.allclose(alignment.translation, 0.0, atol=1e-9)
    assert np.allclose(alignment.to_visible(fitted), fitted, atol=1e-9)
    assert np.allclose(alignment.to_fitted(fitted), fitted, atol=1e-9)


def test_alignment_recovers_rotation():
    fitted = _cloud(seed=1)
    rotation = _rotation(0.7)
    visible = fitted @ rotation.T
    alignment = align_embeddings(fitted, visible)
    assert alignment.rmse < 1e-12
    assert np.linalg.det(alignment.rotation) > 0
    assert np.allclose(alignment.rotation, rotation, atol=1e-9)
    assert np.allclose(alignment.to_visible(fitted), visible, atol=1e-9)
    assert np.allclose(alignment.to_fitted(visible), fitted, atol=1e-9)


def test_alignment_allows_reflection():
    fitted = _cloud(seed=2)
    visible = fitted @ np.diag([1.0, -1.0])
    alignment = align_embeddings(fitted, visible)
    assert alignment.rmse < 1e-12
    # Reflection is deliberately NOT forced out: a refit may mirror the plot
    # without changing what it shows, and forcing det=+1 would reject it.
    assert np.linalg.det(alignment.rotation) < 0
    assert np.allclose(alignment.to_visible(fitted), visible, atol=1e-9)
    assert np.allclose(alignment.to_fitted(visible), fitted, atol=1e-9)


def test_alignment_recovers_translation_and_scale():
    fitted = _cloud(seed=3)
    visible = 3.5 * (fitted @ _rotation(-1.2).T) + np.array([2.0, -7.0])
    alignment = align_embeddings(fitted, visible)
    assert alignment.rmse < 1e-12
    assert abs(alignment.scale - 3.5) < 1e-9
    assert np.allclose(alignment.translation, [2.0, -7.0], atol=1e-8)
    assert np.allclose(alignment.to_visible(fitted), visible, atol=1e-8)
    assert np.allclose(alignment.to_fitted(visible), fitted, atol=1e-8)


def test_alignment_ignores_non_finite_serialized_points():
    fitted = _cloud(seed=4)
    visible = _as_serialized(2.0 * fitted + np.array([1.0, 1.0]))
    visible[3] = [None, None]  # type: ignore[list-item]  # serialize.py nulls non-finite
    alignment = align_embeddings(fitted, visible)
    assert alignment.n_points == fitted.shape[0] - 1
    assert alignment.rmse < 1e-12
    assert abs(alignment.scale - 2.0) < 1e-9


def test_unalignable_pair_is_rejected():
    fitted = _cloud(seed=5)
    visible = _cloud(seed=6)  # independent cloud: no similarity transform fits
    alignment = align_embeddings(fitted, visible)
    assert alignment.rmse > ALIGNMENT_RMSE_MAX
    exc = _expect_error(
        409,
        require_alignment,
        _artifact(fitted),
        {"embedding_original": _as_serialized(visible)},
    )
    assert "rebuild" in exc.detail.lower()


def test_alignment_shape_mismatch_is_rejected():
    _expect_error(
        409,
        align_embeddings,
        _cloud(n=10, seed=7),
        _as_serialized(_cloud(n=11, seed=8)),
    )


def _visible_at_alignment_rmse(fitted: np.ndarray, target: float) -> np.ndarray:
    """A visible cloud whose best similarity alignment to `fitted` has normalized
    RMSE `target` (to 1e-4): bisect the amplitude of one fixed noise pattern."""
    noise = np.random.default_rng(99).normal(size=fitted.shape)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if align_embeddings(fitted, fitted + mid * noise).rmse < target:
            lo = mid
        else:
            hi = mid
    visible = fitted + ((lo + hi) / 2.0) * noise
    assert abs(align_embeddings(fitted, visible).rmse - target) < 1e-4
    return visible


def test_alignment_gate_is_enforced_at_its_boundary():
    """`ALIGNMENT_RMSE_MAX` is a product decision (D1). These cases verify that the
    documented bound is ENFORCED at its edge — 0.049 passes, 0.051 is a 409 — not
    that 0.05 is the right number. The bound is on the RMS over the node's
    existing points; the worst single point is reported, not bounded."""
    fitted = _cloud(n=60, seed=9)
    just_under = _visible_at_alignment_rmse(fitted, ALIGNMENT_RMSE_MAX - 0.001)
    alignment = require_alignment(
        _artifact(fitted), {"embedding_original": _as_serialized(just_under)}
    )
    assert alignment.rmse < ALIGNMENT_RMSE_MAX
    assert alignment.max_residual >= alignment.rmse
    assert alignment.max_residual > ALIGNMENT_RMSE_MAX  # a passing alignment can
    # still hold single points past the bound — that is exactly why it is reported

    just_over = _visible_at_alignment_rmse(fitted, ALIGNMENT_RMSE_MAX + 0.001)
    exc = _expect_error(
        409,
        require_alignment,
        _artifact(fitted),
        {"embedding_original": _as_serialized(just_over)},
    )
    assert "rebuild" in exc.detail.lower()


# ── Node lookup ─────────────────────────────────────────────────────────────


def test_find_node_resolves_serialized_path_ids():
    payload, _ = _payload("PCA")
    tree = payload["tree"]
    assert find_node(tree, "root") is tree
    assert find_node(tree, "root/1")["id"] == "root/1"
    assert find_node(tree, "root/1/0")["id"] == "root/1/0"
    assert find_node(tree, "root/99") is None
    assert find_node(tree, "root/0/0") is None  # root/0 is a leaf
    assert find_node(tree, "") is None
    assert find_node(tree, "1") is None
    assert find_node(tree, "root/x") is None
    assert find_node(None, "root") is None


# ── Request validation ──────────────────────────────────────────────────────


def test_valid_requests_are_accepted():
    payload, _ = _payload("PCA")
    point = validate_request(_request(payload), payload)
    assert point.node_id == "root"
    assert point.target_kind == "point"
    assert point.target_xy == (0.5, -0.25)
    assert point.method == "PCA"
    assert point.effective_config["normalize"] is True
    assert len(point.children) >= 2

    cluster = validate_request(
        _request(payload, target={"kind": "cluster", "child_index": 1}), payload
    )
    assert cluster.target_child_index == 1
    assert cluster.target_node is cluster.children[1]
    assert cluster.source_node is cluster.children[0]


def test_analysis_identity_mismatches_are_409():
    payload, _ = _payload("PCA")
    _expect_error(409, validate_request, _request(payload), None)
    _expect_error(409, validate_request, _request(payload, analysis_id="nope"), payload)
    # Same id echoed back, but the request describes a different run.
    _expect_error(
        409, validate_request, _request(payload, dataset="Wine (Low)"), payload
    )
    _expect_error(
        409, validate_request, _request(payload, config={"method": "UMAP"}), payload
    )


def test_payload_predating_effective_config_is_409():
    payload, _ = _payload("PCA")
    legacy = {
        "meta": {k: v for k, v in payload["meta"].items() if k != "effective_config"},
        "tree": payload["tree"],
    }
    exc = _expect_error(409, validate_request, _request(payload), legacy)
    assert "rebuild" in exc.detail.lower()

    old_schema = {
        "meta": {**payload["meta"], "schema_version": SCHEMA_VERSION - 1},
        "tree": payload["tree"],
    }
    _expect_error(409, validate_request, _request(payload), old_schema)

    # A payload written before the feature landed has NONE of the three meta
    # keys. The schema check runs before the identity comparison, so the client
    # is told to rebuild — not that it asked about a "different run", which is
    # what the missing `analysis_id` would otherwise look like.
    stripped = {
        "meta": {
            k: v
            for k, v in payload["meta"].items()
            if k not in ("analysis_id", "effective_config", "schema_version")
        },
        "tree": payload["tree"],
    }
    exc = _expect_error(409, validate_request, _request(payload), stripped)
    assert "before movement support" in exc.detail
    assert "different analysis run" not in exc.detail


def test_bad_requests_are_400():
    payload, _ = _payload("PCA")
    _expect_error(400, validate_request, _request(payload, node_id="root/42"), payload)
    _expect_error(400, validate_request, _request(payload, node_id="root/0"), payload)
    _expect_error(
        400, validate_request, _request(payload, source_child_index=99), payload
    )
    _expect_error(
        400, validate_request, _request(payload, source_child_index=-1), payload
    )
    _expect_error(
        400,
        validate_request,
        _request(payload, target={"kind": "cluster", "child_index": 99}),
        payload,
    )
    _expect_error(
        400,
        validate_request,
        _request(
            payload, source_child_index=1, target={"kind": "cluster", "child_index": 1}
        ),
        payload,
    )
    _expect_error(
        400,
        validate_request,
        _request(payload, target={"kind": "point", "x": float("nan"), "y": 0.0}),
        payload,
    )
    _expect_error(
        400,
        validate_request,
        _request(payload, target={"kind": "point", "x": 0.0, "y": float("inf")}),
        payload,
    )
    _expect_error(
        400, validate_request, _request(payload, target={"kind": "elsewhere"}), payload
    )
    _expect_error(400, validate_request, _request(payload, strength=1.5), payload)
    _expect_error(400, validate_request, _request(payload, strength=-0.1), payload)
    for ok in (0.0, 0.5, 1.0):
        assert validate_request(_request(payload, strength=ok), payload).strength == ok


def test_unsupported_reducers_are_400_and_name_tsne_and_mds():
    payload, _ = _payload("PCA")
    for method in ("t-SNE", "MDS"):
        unsupported = {
            "meta": {
                **payload["meta"],
                "effective_config": {
                    **payload["meta"]["effective_config"],
                    "method": method,
                },
            },
            "tree": payload["tree"],
        }
        exc = _expect_error(400, validate_request, _request(payload), unsupported)
        # Four reducers exist; the rejection must name both unsupported ones so
        # the disabled UI state can explain itself.
        assert "t-SNE" in exc.detail and "MDS" in exc.detail
        assert "PCA" in exc.detail and "UMAP" in exc.detail


def test_parent_without_projection_is_400():
    payload, _ = _payload("PCA")
    tree = payload["tree"]
    unprojected = {**tree, "embedding_original": None}
    _expect_error(
        400,
        validate_request,
        _request(payload),
        {"meta": payload["meta"], "tree": unprojected},
    )


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


# ── The canary: a same-process refit must reproduce the visible embedding ───


def _canary(method: str, tolerance: float) -> dict[str, float]:
    """Refit every projected node and align it to the serialized embedding.

    What this proves is refit PARITY: the same rows, the same root scaler, the
    same `fit_node_projection` path (same `init="pca"`, same seed) reproduce the
    serialized embedding at ~identity. It is NOT the test for the
    `effective_config` amendment — on Iris the in-place clamp changes nothing,
    so a refit from the raw request config would pass here too. That guarantee
    is `test_refit_reads_effective_config_not_the_request_config` below.
    """
    payload, df = _payload(method)
    meta = payload["meta"]
    measured: dict[str, float] = {}

    nodes: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node["children"] or []:
            walk(child)

    walk(payload["tree"])
    projected = [n for n in nodes if n["embedding_original"] is not None]
    assert len(projected) >= 3, "expected several projected nodes to check"

    for node in projected:
        artifact = build_projection_artifact(
            analysis_id=meta["analysis_id"],
            node_id=node["id"],
            node=node,
            method=method,
            effective_config=meta["effective_config"],
            feature_cols=meta["feature_cols"],
            df=df,
        )
        assert artifact.X_parent.shape[0] == node["n_points"]
        assert list(artifact.parent_row_indices) == node["row_indices"]
        alignment = require_alignment(artifact, node)  # would raise above tolerance
        measured[node["id"]] = alignment.rmse
        assert alignment.rmse < tolerance, (
            f"{method} refit of {node['id']} does not reproduce the serialized "
            f"embedding: normalized RMSE {alignment.rmse:.3e} >= {tolerance:g}"
        )
    return measured


def test_pca_refit_reproduces_the_serialized_embedding():
    measured = _canary("PCA", PCA_CANARY_RMSE)
    print(f"   PCA canary RMSE: max={max(measured.values()):.3e}  {measured}")


def test_umap_refit_reproduces_the_serialized_embedding():
    measured = _canary("UMAP", UMAP_CANARY_RMSE)
    print(f"   UMAP canary RMSE: max={max(measured.values()):.3e}  {measured}")


def test_wide_pca_refit_matches_the_analysis_fit():
    # More columns than rows (Olivetti's shape, scaled down): the refit takes the
    # thin-SVD path instead of covariance_eigh and must give the same projection.
    X = np.random.default_rng(0).standard_normal((40, 300))
    config = default_config()
    config["method"] = "PCA"
    shared = fit_node_projection(X, config)
    refit = _refit_node_projection(X, config)
    assert shared is not None and refit is not None
    np.testing.assert_allclose(refit.embedding, shared.embedding, atol=1e-9)


def test_refit_does_not_mutate_the_source_dataframe():
    payload, df = _payload("PCA")
    meta = payload["meta"]
    before = df[meta["feature_cols"]].to_numpy(copy=True)
    build_projection_artifact(
        analysis_id=meta["analysis_id"],
        node_id="root/1",
        node=find_node(payload["tree"], "root/1"),
        method="PCA",
        effective_config=meta["effective_config"],
        feature_cols=meta["feature_cols"],
        df=df,
    )
    assert np.array_equal(before, df[meta["feature_cols"]].to_numpy())


# ═══════════════════════════════════════════════════════════════════════════
# APPEND BELOW THIS LINE — PCA movement tests (Phase 2), then UMAP (Phase 4).
# Nothing above needs editing: any module-level `test_*` function defined here
# is picked up by pytest and by the standalone runner alike. `_payload(method)`
# returns a memoized (payload, df) pair; `_request(payload, **overrides)` builds
# a request against it; `_expect_error(status, fn, …)` asserts a MovementError.
# ═══════════════════════════════════════════════════════════════════════════

# The header import block above is frozen; the Phase 2 additions import here.
from backend.movement import (
    MAX_MOVEMENT_PREVIEW_POINTS,
    _child_positions,
    _preview_indices,
    artifact_for,
    compute_movement,
    embedding_array,
    mutable_flags,
)

# Iris' `default_feature_cols` drops every `target_*` column, so the
# immutable-feature paths are unreachable from `_payload`. They need runs whose
# feature space contains a label because the analyst hand-picked one:
#   MIXED       — two mutable features left, so the click stays reachable;
#   ONE_MUTABLE — a single mutable feature, so the mutable component matrix has
#                 rank 1 and only the closest reachable point exists.
_MIXED_FEATURES = ["petal length (cm)", "petal width (cm)", "target_setosa"]
_ONE_MUTABLE_FEATURES = ["petal length (cm)", "target_setosa", "target_versicolor"]

_label_payload_cache: dict[tuple[str, ...], tuple[dict[str, Any], Any]] = {}


def _label_payload(feature_cols: list[str]) -> tuple[dict[str, Any], Any]:
    """`_payload`, but with a hand-picked feature list. Memoized per list."""
    key = tuple(feature_cols)
    if key in _label_payload_cache:
        return _label_payload_cache[key]
    requested = {"method": "PCA", "hierarchical_layers": 2}
    config = default_config()
    config.update(requested)  # type: ignore[typeddict-item]
    config["dataset_choice"] = DATASET
    df = ds.load(DATASET)
    tree = start_evaluation(df, list(feature_cols), config)  # mutates `config`
    payload = {
        "meta": {
            "dataset": DATASET,
            "feature_cols": list(feature_cols),
            "config": requested,
            "n_total": len(df),
            "analysis_id": analysis_id(DATASET, list(feature_cols), requested),
            "effective_config": effective_config(config),
            "schema_version": SCHEMA_VERSION,
        },
        "tree": serialize_tree(tree),  # type: ignore[arg-type]
    }
    _label_payload_cache[key] = (payload, df)
    return payload, df


def _movement(payload: dict[str, Any], df: Any, **overrides: Any):
    """Answer one movement and hand back the pieces the assertions need.

    `artifact_for` is the cached refit the response itself used, so the checks
    below run against the very reducer that produced the numbers.
    """
    request = _request(payload, **overrides)
    response = compute_movement(request, payload, df)
    validated = validate_request(request, payload)
    return response, validated, artifact_for(validated, df)


def _delta_input(response: dict[str, Any], artifact: ProjectionArtifact) -> np.ndarray:
    """Recover Δ in the reducer's INPUT space from the reported raw deltas."""
    raw = np.array(
        [c["recommended_delta_raw"] for c in response["feature_changes"]], dtype=float
    )
    if artifact.scaler is None:
        return raw
    return raw / np.asarray(artifact.scaler.scale_, dtype=float)


def _preview_xy(response: dict[str, Any]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in response["preview_points"]], dtype=float)


def _visible_source_points(validated: Any, artifact: ProjectionArtifact) -> np.ndarray:
    """The source cluster as the FRONTEND holds it: the parent's serialized rows."""
    parent = embedding_array(validated.node["embedding_original"])
    return parent[_child_positions(artifact, validated.source_node)]


def _null_space_vector(matrix: np.ndarray) -> np.ndarray:
    """One unit vector the matrix annihilates (`matrix @ v == 0`)."""
    _, singular, vt = np.linalg.svd(matrix)
    rank = int((singular > 1e-12).sum())
    return vt[rank]


# ── PCA movement: shape, sharing, and the frontend's own ghost formula ──────


def test_pca_free_point_movement_is_finite_and_shared():
    payload, df = _payload("PCA")
    response, validated, artifact = _movement(
        payload, df, target={"kind": "point", "x": 1.5, "y": 0.4}
    )
    assert response["status"] == "ok"
    assert response["method"] == "PCA"
    assert response["node_id"] == "root"
    assert response["source_child_index"] == 0
    assert response["target"]["kind"] == "point"
    assert response["target"]["child_index"] is None
    assert response["target"]["estimator"] == "pca_pinv"
    assert response["target_size"] is None
    # PCA reaches the reachable target exactly and interpolates linearly, so
    # there is nothing to search for and no trajectory to plot.
    assert response["recommended_strength"] == 1.0
    assert response["applied_strength"] == 1.0
    assert response["strength_trajectory"] == []
    assert response["metrics"]["support_ratio"] is None  # a UMAP concept
    assert response["metrics"]["alignment_rmse"] < PCA_CANARY_RMSE
    assert response["metrics"]["alignment_max_residual"] < PCA_CANARY_RMSE
    assert (
        response["metrics"]["alignment_max_residual"]
        >= response["metrics"]["alignment_rmse"]
    )
    assert response["metrics"]["preview_sampled"] is False
    assert response["source_size"] == validated.source_node["n_points"]
    assert len(response["preview_points"]) == response["source_size"]
    assert {c["feature"] for c in response["feature_changes"]} == set(
        validated.feature_cols
    )

    displacement = np.asarray(response["visible_displacement"], dtype=float)
    assert np.all(np.isfinite(displacement))
    assert np.linalg.norm(displacement) > 0.0

    # One shared translation: every preview point sits exactly where the client
    # would put its ghost, `Y_source + α · visible_displacement`.
    before = _visible_source_points(validated, artifact)
    after = _preview_xy(response)
    assert np.allclose(after - before, displacement, atol=1e-9)
    assert [p["row_id"] for p in response["preview_points"]] == list(
        validated.source_node["row_indices"]
    )


def test_pca_movement_preserves_pairwise_distances():
    # Cheap smoke only: a shared translation cannot change the cluster's shape,
    # so this is true by construction. The substantive checks are the row-space
    # and centroid-reaches-target cases below.
    payload, df = _payload("PCA")
    response, validated, artifact = _movement(
        payload, df, target={"kind": "point", "x": -1.0, "y": 2.0}
    )
    before = _visible_source_points(validated, artifact)
    after = _preview_xy(response)
    gram_before = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=-1)
    gram_after = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=-1)
    assert np.allclose(gram_before, gram_after, atol=1e-9)


# ── The math: minimum-norm solution, and it reaches the target ──────────────


def test_pca_delta_is_the_minimum_norm_row_space_solution():
    payload, df = _payload("PCA")
    response, validated, artifact = _movement(
        payload, df, target={"kind": "point", "x": -2.0, "y": 1.1}
    )
    delta = _delta_input(response, artifact)
    mutable = mutable_flags(validated.feature_cols)
    p_mutable = np.asarray(artifact.reducer.components_, dtype=float)[:, mutable]
    d = delta[mutable]

    # Δ = P_mutableᵀ w: the displacement lies in the ROW SPACE of the mutable
    # component matrix. That is what makes it the pseudo-inverse answer rather
    # than any other member of the (d−2)-dimensional solution set.
    w, *_ = np.linalg.lstsq(p_mutable.T, d, rcond=None)
    assert np.allclose(p_mutable.T @ w, d, atol=1e-10)

    # …and it is the smallest such answer: adding anything from the null space
    # keeps the projection identical while strictly growing the norm.
    detour = _null_space_vector(p_mutable)
    assert np.allclose(p_mutable @ detour, 0.0, atol=1e-10)
    assert np.linalg.norm(d + detour) > np.linalg.norm(d)


def test_pca_free_point_reaches_the_requested_target():
    payload, df = _payload("PCA")
    click = np.array([1.5, 0.4])
    response, validated, artifact = _movement(
        payload,
        df,
        target={"kind": "point", "x": float(click[0]), "y": float(click[1])},
    )
    delta = _delta_input(response, artifact)
    positions = _child_positions(artifact, validated.source_node)

    # The spec's own formulation: push the moved rows through the FROZEN reducer
    # and look at where their centroid lands on screen.
    moved = artifact.reducer.transform(artifact.X_parent[positions] + delta)
    reached = require_alignment(artifact, validated.node).to_visible(moved.mean(axis=0))
    residual = float(np.linalg.norm(reached[0] - click))
    print(f"   free-point centroid residual: {residual:.3e} (visible units)")
    assert residual < 1e-9
    assert response["metrics"]["projected_distance_after"] < 1e-9
    assert response["metrics"]["projected_distance_before"] > 1.0
    assert response["metrics"]["projected_distance_reduction"] > 1.0
    # A point target's feature-space destination is DEFINED as μ_s + Δ, so an
    # "after" distance would be (1−α)·‖Δ‖ by construction — content-free. The
    # backend sends null; only "before" (= ‖Δ‖, the size of the change) remains.
    assert response["metrics"]["feature_centroid_distance_before"] > 0.0
    assert response["metrics"]["feature_centroid_distance_after"] is None
    # A free target is under-determined: the panel has to say in which norm the
    # recommendation is the smallest one.
    assert validated.effective_config["normalize"] is True
    assert any("smallest standardized change" in w for w in response["warnings"])


def test_pca_cluster_movement_reaches_the_target_centroid():
    payload, df = _payload("PCA")
    response, validated, artifact = _movement(
        payload, df, target={"kind": "cluster", "child_index": 1}
    )
    assert response["target"]["kind"] == "cluster"
    assert response["target"]["child_index"] == 1
    assert response["target"]["estimator"] == "observed_centroid"
    assert response["target_size"] == validated.children[1]["n_points"]
    # Δ = μ_t − μ_s is unique: no min-norm caveat, and nothing else to warn about
    # on this run (every Iris feature is mutable, the move stays in range).
    assert response["warnings"] == []

    delta = _delta_input(response, artifact)
    source = _child_positions(artifact, validated.source_node)
    destination = _child_positions(artifact, validated.children[1])
    assert np.allclose(
        artifact.X_parent[source].mean(axis=0) + delta,
        artifact.X_parent[destination].mean(axis=0),
        atol=1e-9,
    )

    alignment = require_alignment(artifact, validated.node)
    moved = artifact.reducer.transform(artifact.X_parent[source] + delta)
    reached = alignment.to_visible(moved.mean(axis=0))[0]
    wanted = alignment.to_visible(artifact.embedding[destination].mean(axis=0))[0]
    residual = float(np.linalg.norm(reached - wanted))
    print(f"   cluster centroid residual:    {residual:.3e} (visible units)")
    assert residual < 1e-9
    assert np.allclose(
        [response["target"]["x"], response["target"]["y"]], wanted, atol=1e-9
    )
    assert response["metrics"]["projected_distance_after"] < 1e-9
    assert response["metrics"]["feature_centroid_distance_after"] < 1e-9
    assert response["metrics"]["feature_centroid_distance_before"] > 1.0


# ── Strength ────────────────────────────────────────────────────────────────


def test_pca_strength_scales_the_displacement_exactly():
    payload, df = _payload("PCA")
    target = {"kind": "point", "x": 1.5, "y": 0.4}
    full, validated, artifact = _movement(payload, df, target=target, strength=1.0)
    half, *_ = _movement(payload, df, target=target, strength=0.5)
    zero, *_ = _movement(payload, df, target=target, strength=0.0)
    assert (full["applied_strength"], half["applied_strength"]) == (1.0, 0.5)
    assert zero["applied_strength"] == 0.0
    # `strength: null` asks for the recommendation, which for PCA is full strength.
    recommended, *_ = _movement(payload, df, target=target, strength=None)
    assert recommended["applied_strength"] == 1.0
    assert recommended["preview_points"] == full["preview_points"]

    origin = _visible_source_points(validated, artifact)
    displacement = np.asarray(full["visible_displacement"], dtype=float)
    assert np.allclose(_preview_xy(zero), origin, atol=1e-9)
    assert np.allclose(_preview_xy(half), origin + 0.5 * displacement, atol=1e-9)
    # The returned vector is the FULL-strength one whatever α was applied — the
    # slider multiplies it client-side.
    assert np.allclose(half["visible_displacement"], displacement, atol=1e-12)
    assert np.allclose(zero["visible_displacement"], displacement, atol=1e-12)

    for at_full, at_half, at_zero in zip(
        full["feature_changes"],
        half["feature_changes"],
        zero["feature_changes"],
        strict=True,
    ):
        assert at_half["recommended_delta_raw"] == at_full["recommended_delta_raw"]
        assert at_half["applied_delta_raw"] == 0.5 * at_full["recommended_delta_raw"]
        assert at_zero["applied_delta_raw"] == 0.0
        assert at_zero["standardized_magnitude"] == at_full["standardized_magnitude"]

    assert abs(zero["metrics"]["projected_distance_reduction"]) < 1e-9
    assert (
        abs(
            half["metrics"]["projected_distance_after"]
            - 0.5 * half["metrics"]["projected_distance_before"]
        )
        < 1e-9
    )


# ── Raw units ───────────────────────────────────────────────────────────────


def test_pca_raw_deltas_reverse_the_root_standardization():
    payload, df = _payload("PCA")
    response, validated, artifact = _movement(
        payload, df, target={"kind": "cluster", "child_index": 1}
    )
    scaler = artifact.scaler
    assert scaler is not None, "this run normalizes, so a root scaler must exist"

    raw = df[validated.feature_cols].to_numpy(dtype=float)
    source_rows = np.asarray(validated.source_node["row_indices"], dtype=int)
    target_rows = np.asarray(validated.children[1]["row_indices"], dtype=int)
    # Δ derived independently of the response, straight from the input space.
    expected = artifact.X_parent[
        _child_positions(artifact, validated.children[1])
    ].mean(axis=0) - artifact.X_parent[
        _child_positions(artifact, validated.source_node)
    ].mean(axis=0)

    for j, change in enumerate(response["feature_changes"]):
        assert change["feature"] == validated.feature_cols[j]
        assert change["mutable"] is True
        # Reported means and values are the dataset's own units …
        assert abs(change["source_mean_raw"] - raw[source_rows, j].mean()) < 1e-9
        assert abs(change["target_value_raw"] - raw[target_rows, j].mean()) < 1e-9
        # … the raw delta is the standardized one times the ROOT scale …
        assert (
            abs(change["recommended_delta_raw"] - expected[j] * scaler.scale_[j]) < 1e-9
        )
        # … and the standardized magnitude stays in standardized units.
        assert abs(change["standardized_magnitude"] - abs(expected[j])) < 1e-9
        assert (
            abs(
                change["source_mean_raw"]
                + change["recommended_delta_raw"]
                - change["target_value_raw"]
            )
            < 1e-9
        )


def test_out_of_range_fraction_counts_points_leaving_the_dataset_range():
    payload, df = _payload("PCA")
    # α = 0 moves nothing, and every observed point is inside the observed range.
    unmoved, *_ = _movement(
        payload, df, target={"kind": "point", "x": 0.0, "y": 0.0}, strength=0.0
    )
    assert unmoved["metrics"]["out_of_range_fraction"] == 0.0
    assert not any("outside the range" in w for w in unmoved["warnings"])

    far, *_ = _movement(payload, df, target={"kind": "point", "x": 500.0, "y": 500.0})
    assert far["metrics"]["out_of_range_fraction"] == 1.0
    assert any("outside the range" in w for w in far["warnings"])


# ── Immutable features ──────────────────────────────────────────────────────


def test_pca_holds_label_features_at_exactly_zero():
    payload, df = _label_payload(_MIXED_FEATURES)
    response, validated, artifact = _movement(
        payload, df, target={"kind": "point", "x": 1.0, "y": -0.5}
    )
    assert validated.feature_cols == _MIXED_FEATURES
    assert [c["feature"] for c in response["feature_changes"]] == _MIXED_FEATURES
    changes = {c["feature"]: c for c in response["feature_changes"]}
    label = changes["target_setosa"]
    assert label["mutable"] is False
    # Exactly zero, not "small": an immutable feature is never solved for.
    assert label["recommended_delta_raw"] == 0.0
    assert label["applied_delta_raw"] == 0.0
    assert label["standardized_magnitude"] == 0.0
    assert changes["petal length (cm)"]["mutable"] is True
    assert changes["petal width (cm)"]["mutable"] is True
    assert (
        _delta_input(response, artifact)[_MIXED_FEATURES.index("target_setosa")] == 0.0
    )
    assert any("held fixed" in w for w in response["warnings"])
    # Two mutable features still span the plane, so the click stays reachable.
    assert response["metrics"]["projected_distance_after"] < 1e-9


def test_pca_rank_deficient_components_return_the_closest_reachable_point():
    payload, df = _label_payload(_ONE_MUTABLE_FEATURES)
    click = np.array([1.0, 3.0])
    response, validated, artifact = _movement(
        payload,
        df,
        target={"kind": "point", "x": float(click[0]), "y": float(click[1])},
    )
    mutable = mutable_flags(validated.feature_cols)
    assert int(mutable.sum()) == 1
    p_mutable = np.asarray(artifact.reducer.components_, dtype=float)[:, mutable]
    assert int(np.linalg.matrix_rank(p_mutable)) < 2

    assert any("closest reachable point" in w for w in response["warnings"])
    delta = _delta_input(response, artifact)
    assert np.count_nonzero(delta[~mutable]) == 0
    metrics = response["metrics"]
    # It still moves, and it moves as far as it can: the remaining gap is the
    # part of the request that no mutable feature can express.
    assert metrics["projected_distance_after"] > 1e-6
    assert metrics["projected_distance_after"] < metrics["projected_distance_before"]

    alignment = require_alignment(artifact, validated.node)
    source = _child_positions(artifact, validated.source_node)
    reached = artifact.embedding[source].mean(axis=0) + p_mutable @ delta[mutable]
    residual = alignment.to_fitted(click)[0] - reached
    # Least squares: what is left over is orthogonal to the reachable direction.
    assert abs(float(residual @ p_mutable[:, 0])) < 1e-9


# ── Sampling, immutability of the inputs, wire safety ───────────────────────


def test_preview_sampling_is_deterministic_and_bounded():
    small, sampled = _preview_indices(1000)
    assert sampled is False
    assert np.array_equal(small, np.arange(1000))

    n = MAX_MOVEMENT_PREVIEW_POINTS + 1
    picks, sampled = _preview_indices(n)
    assert sampled is True
    assert picks.size <= MAX_MOVEMENT_PREVIEW_POINTS
    assert picks[0] == 0 and picks[-1] == n - 1  # the boundary rows survive
    assert np.array_equal(picks, np.unique(picks))
    # No RNG anywhere: the same request draws the same ghosts every time.
    assert np.array_equal(picks, _preview_indices(n)[0])


def test_pca_movement_does_not_mutate_the_source_dataframe():
    payload, df = _payload("PCA")
    before = df.copy(deep=True)
    _movement(payload, df, target={"kind": "point", "x": 2.0, "y": -1.0})
    _movement(payload, df, target={"kind": "cluster", "child_index": 1}, strength=0.3)
    assert df.equals(before)


def test_pca_movement_response_is_json_safe():
    import json

    payload, df = _payload("PCA")
    for target in (
        {"kind": "point", "x": 0.0, "y": 0.0},
        {"kind": "cluster", "child_index": 1},
    ):
        response, *_ = _movement(payload, df, target=target)
        # Starlette encodes with allow_nan=False; a NaN would be a 500, not a plot.
        assert json.loads(json.dumps(response, allow_nan=False))["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════
# UMAP movement (Phase 4) and the background-job path.
#
# No exact-destination assertions live here: UMAP's inverse is approximate by
# construction (a transform→inverse→transform round trip on Iris lands a couple
# of embedding units away), so a test demanding exactness would fail on healthy
# code. What is asserted instead is finiteness, ordering, determinism, and that
# the estimators and warnings say which kind of answer the analyst is looking at.
# ═══════════════════════════════════════════════════════════════════════════

import threading
import time

from backend import movement as movement_module
from backend import movement_jobs
from backend.movement import (
    MAX_STRENGTH_SEARCH_POINTS,
    STRENGTH_GRID,
    SUPPORT_RATIO_WARN,
    _strength_search_indices,
    artifact_is_cached,
    clear_artifact_cache,
    compute_validated_movement,
    needs_background,
)


class _InverseFails:
    """A frozen reducer whose `inverse_transform` is unusable.

    Wraps the real one so `transform` — and therefore every projected position
    in the response — stays genuine; only the inverse is broken. Two modes,
    because the fallback triggers on both: an exception, and a finite-looking
    call that returns NaN.
    """

    def __init__(self, inner: Any, n_features: int, mode: str) -> None:
        self._inner = inner
        self._n_features = n_features
        self._mode = mode

    def transform(self, X: Any) -> Any:
        return self._inner.transform(X)

    def inverse_transform(self, Y: Any) -> Any:
        if self._mode == "raise":
            raise RuntimeError("forced inverse failure")
        rows = np.asarray(Y, dtype=float).reshape(-1, 2).shape[0]
        return np.full((rows, self._n_features), np.nan)


def _umap_pieces():
    """The UMAP payload plus the visible centroid of child 1 (a click inside data)."""
    payload, df = _payload("UMAP")
    validated = validate_request(_request(payload), payload)
    artifact = artifact_for(validated, df)
    visible = embedding_array(validated.node["embedding_original"])
    destination = visible[_child_positions(artifact, validated.children[1])].mean(
        axis=0
    )
    return payload, df, artifact, destination


def _point(xy: np.ndarray) -> dict[str, Any]:
    return {"kind": "point", "x": float(xy[0]), "y": float(xy[1])}


def _trajectory(response: dict[str, Any]) -> np.ndarray:
    return np.array(
        [p["projected_distance"] for p in response["strength_trajectory"]], dtype=float
    )


# ── UMAP: the inverse target ────────────────────────────────────────────────


def test_umap_free_point_movement_inverts_and_previews_finitely():
    payload, df, _artifact_unused, destination = _umap_pieces()
    response, validated, _ = _movement(payload, df, target=_point(destination))

    assert response["status"] == "ok"
    assert response["method"] == "UMAP"
    assert response["target"]["kind"] == "point"
    assert response["target"]["child_index"] is None
    assert response["target"]["estimator"] == "umap_inverse"
    assert response["target_size"] is None
    # The marker sits on the click: the inverse estimates the destination in
    # FEATURE space, it does not move the position the analyst pointed at.
    assert np.allclose(
        [response["target"]["x"], response["target"]["y"]], destination, atol=1e-9
    )
    # The inverse is approximate; the panel has to say so rather than implying a
    # guaranteed landing.
    assert any("approximate" in w for w in response["warnings"])

    assert response["source_size"] == validated.source_node["n_points"]
    assert len(response["preview_points"]) == response["source_size"]
    assert np.all(np.isfinite(_preview_xy(response)))
    assert np.all(np.isfinite(np.asarray(response["visible_displacement"], float)))
    for change in response["feature_changes"]:
        assert change["recommended_delta_raw"] is not None
        assert change["standardized_magnitude"] is not None

    # The whole α grid comes back, not only its argmin: a neighbour-driven
    # transform can plateau and jump, and the panel plots that.
    strengths = [p["strength"] for p in response["strength_trajectory"]]
    assert np.allclose(strengths, STRENGTH_GRID)
    trajectory = _trajectory(response)
    assert np.all(np.isfinite(trajectory))
    print(f"   UMAP strength trajectory: {[round(d, 4) for d in trajectory.tolist()]}")
    monotone = bool(np.all(np.diff(trajectory) <= 1e-12))
    print(f"   monotone non-increasing:  {monotone}")

    metrics = response["metrics"]
    assert metrics["alignment_rmse"] < UMAP_CANARY_RMSE
    assert metrics["alignment_max_residual"] >= metrics["alignment_rmse"]
    assert metrics["support_ratio"] is not None  # a free target always reports it
    assert metrics["preview_sampled"] is False
    assert metrics["projected_distance_after"] < metrics["projected_distance_before"]
    assert metrics["feature_centroid_distance_after"] is None  # point target (D3)


def test_umap_cluster_target_uses_the_observed_centroid():
    payload, df, _artifact_unused, destination = _umap_pieces()
    response, validated, artifact = _movement(
        payload, df, target={"kind": "cluster", "child_index": 1}
    )
    # NOT an inverse of the displayed centroid: the destination cluster's own
    # feature-space mean is observed and exact.
    assert response["target"]["estimator"] == "observed_centroid"
    assert response["target"]["child_index"] == 1
    assert response["target_size"] == validated.children[1]["n_points"]
    assert response["metrics"]["support_ratio"] is None  # nothing was inverted

    delta = _delta_input(response, artifact)
    source = _child_positions(artifact, validated.source_node)
    target_rows = _child_positions(artifact, validated.children[1])
    assert np.allclose(
        artifact.X_parent[source].mean(axis=0) + delta,
        artifact.X_parent[target_rows].mean(axis=0),
        atol=1e-9,
    )
    # Δ = μ_t − μ_s exactly, so the feature-space gap closes at full strength …
    assert response["applied_strength"] == 1.0
    assert response["metrics"]["feature_centroid_distance_after"] < 1e-9
    # … and the marker sits on the destination cluster's projected centroid.
    assert np.allclose(
        [response["target"]["x"], response["target"]["y"]], destination, atol=1e-9
    )
    # The min-norm caveat belongs to pinv solutions only; this one is unique.
    assert not any("smallest standardized change" in w for w in response["warnings"])


def test_umap_inverse_failure_falls_back_to_neighbour_interpolation():
    payload, df, artifact, destination = _umap_pieces()
    real = artifact.reducer
    n_features = artifact.X_parent.shape[1]
    for mode in ("raise", "nan"):
        artifact.reducer = _InverseFails(real, n_features, mode)
        try:
            response, _validated, _ = _movement(payload, df, target=_point(destination))
        finally:
            artifact.reducer = real
        assert response["target"]["estimator"] == "neighbor_interpolation", mode
        assert any(
            "estimated from the nearest observed points" in w
            for w in response["warnings"]
        ), mode
        # The fallback is a real destination, not a shrug: finite deltas, finite
        # ghosts, and it still moves toward the click.
        assert np.all(np.isfinite(_delta_input(response, artifact))), mode
        assert np.all(np.isfinite(_preview_xy(response))), mode
        assert (
            response["metrics"]["projected_distance_after"]
            < response["metrics"]["projected_distance_before"]
        ), mode

    # The healthy path is unchanged once the real reducer is back.
    healthy, _validated, _ = _movement(payload, df, target=_point(destination))
    assert healthy["target"]["estimator"] == "umap_inverse"


# ── UMAP: the strength grid ─────────────────────────────────────────────────


def test_umap_recommended_strength_is_never_worse_than_standing_still():
    payload, df, _artifact_unused, destination = _umap_pieces()
    for target in (_point(destination), {"kind": "cluster", "child_index": 1}):
        response, _validated, _ = _movement(payload, df, target=target)
        trajectory = _trajectory(response)
        recommended = response["recommended_strength"]
        if recommended is None:
            # No recommendation means α = 0 already won, and the panel says so.
            assert float(np.argmin(trajectory)) == 0.0
            assert any("does not reliably approach" in w for w in response["warnings"])
            assert response["applied_strength"] == 0.0
            continue
        index = int(np.argmin(np.abs(STRENGTH_GRID - recommended)))
        assert trajectory[index] == trajectory.min()
        assert trajectory[index] <= trajectory[0]
        # Ties resolve to the smaller movement: nothing earlier on the grid is
        # as good as the strength that was picked.
        assert np.all(trajectory[:index] > trajectory[index])
        assert response["applied_strength"] == recommended


def test_umap_strength_search_is_deterministic():
    payload, df, _artifact_unused, destination = _umap_pieces()
    target = _point(destination)
    first, _validated, _ = _movement(payload, df, target=target)
    second, _validated, _ = _movement(payload, df, target=target)
    # Same request, same answer — no unseeded RNG anywhere in the search.
    assert first["recommended_strength"] == second["recommended_strength"]
    assert first["strength_trajectory"] == second["strength_trajectory"]
    assert first["preview_points"] == second["preview_points"]
    assert first["feature_changes"] == second["feature_changes"]
    assert first["metrics"] == second["metrics"]


def test_umap_strength_search_sample_is_deterministic_and_bounded():
    assert np.array_equal(_strength_search_indices(250), np.arange(250))
    n = MAX_STRENGTH_SEARCH_POINTS * 3 + 7
    picks = _strength_search_indices(n)
    assert picks.size <= MAX_STRENGTH_SEARCH_POINTS
    assert picks[0] == 0 and picks[-1] == n - 1  # the boundary rows survive
    assert np.array_equal(picks, np.unique(picks))
    assert np.array_equal(picks, _strength_search_indices(n))
    # Tighter than the preview cap: the grid costs eleven transforms, not one.
    assert MAX_STRENGTH_SEARCH_POINTS < MAX_MOVEMENT_PREVIEW_POINTS


def test_umap_unsupported_target_warns_instead_of_being_rejected():
    payload, df, _artifact_unused, _destination = _umap_pieces()
    # A click in empty space far outside the embedding: the inverse is an
    # extrapolation. It is answered with a warning, never silently refused.
    response, _validated, _ = _movement(
        payload, df, target={"kind": "point", "x": 500.0, "y": 500.0}
    )
    assert response["status"] == "ok"
    assert response["metrics"]["support_ratio"] > SUPPORT_RATIO_WARN
    assert any("far from observed data" in w for w in response["warnings"])
    # The inverse of an unsupported position is still a point in feature space,
    # so the movement is computed and previewed; what changes is that the panel
    # is told the destination is an extrapolation.
    assert np.all(np.isfinite(_preview_xy(response)))
    assert np.all(np.isfinite(_trajectory(response)))


def test_umap_preview_sampling_is_flagged_and_deterministic():
    payload, df, _artifact_unused, destination = _umap_pieces()
    original = movement_module.MAX_MOVEMENT_PREVIEW_POINTS
    movement_module.MAX_MOVEMENT_PREVIEW_POINTS = 12  # Iris clusters are small
    try:
        first, validated, _ = _movement(payload, df, target=_point(destination))
        second, _validated, _ = _movement(payload, df, target=_point(destination))
    finally:
        movement_module.MAX_MOVEMENT_PREVIEW_POINTS = original
    assert first["metrics"]["preview_sampled"] is True
    assert 0 < len(first["preview_points"]) <= 12
    assert first["preview_points"] == second["preview_points"]
    # Centroids and feature changes come from the COMPLETE cluster, never the
    # sample: `source_size` still reports every source row.
    assert first["source_size"] == validated.source_node["n_points"]
    assert first["source_size"] > len(first["preview_points"])

    full, _validated, _ = _movement(payload, df, target=_point(destination))
    assert full["metrics"]["preview_sampled"] is False
    assert full["feature_changes"] == first["feature_changes"]


def test_umap_movement_does_not_mutate_the_source_dataframe():
    payload, df, _artifact_unused, destination = _umap_pieces()
    before = df.copy(deep=True)
    _movement(payload, df, target=_point(destination))
    _movement(payload, df, target={"kind": "cluster", "child_index": 1}, strength=0.3)
    assert df.equals(before)


def test_umap_movement_response_is_json_safe():
    import json

    payload, df, _artifact_unused, destination = _umap_pieces()
    for target in (_point(destination), {"kind": "cluster", "child_index": 1}):
        response, *_ = _movement(payload, df, target=target)
        # Starlette encodes with allow_nan=False; a NaN would be a 500, not a plot.
        assert json.loads(json.dumps(response, allow_nan=False))["status"] == "ok"


# ── Numba's threading layer: every UMAP call is serialized ──────────────────


class _OverlapDetector:
    """A reducer proxy that records any two calls overlapping in time.

    Numba's workqueue layer aborts the whole process on concurrent parallel
    regions, which no assertion can catch — so this asserts the guarantee one
    level up: no two UMAP calls are ever in flight at once.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._busy = 0
        self._guard = threading.Lock()
        self.overlaps = 0
        self.calls = 0

    def _enter(self) -> None:
        with self._guard:
            self.calls += 1
            if self._busy:
                self.overlaps += 1
            self._busy += 1

    def _leave(self) -> None:
        with self._guard:
            self._busy -= 1

    def transform(self, X: Any) -> Any:
        self._enter()
        try:
            time.sleep(0.005)  # widen the window so a race would show
            return self._inner.transform(X)
        finally:
            self._leave()

    def inverse_transform(self, Y: Any) -> Any:
        self._enter()
        try:
            return self._inner.inverse_transform(Y)
        finally:
            self._leave()


def test_concurrent_umap_movements_never_overlap_in_the_reducer():
    """Request threads (warm UMAP movements) and job threads (refits) share one
    Numba threading layer that is not thread-safe. `NUMBA_PARALLEL_LOCK` in
    src/analysis/dim_reducer.py serializes every UMAP fit, transform and
    inverse in the process; this runs four movements at once and checks that
    the reducer never saw two calls in flight."""
    payload, df, artifact, destination = _umap_pieces()
    real = artifact.reducer
    detector = _OverlapDetector(real)
    artifact.reducer = detector
    results: list[Any] = []
    errors: list[BaseException] = []

    def worker(alpha: float) -> None:
        try:
            results.append(
                _movement(payload, df, target=_point(destination), strength=alpha)[0]
            )
        except BaseException as exc:  # noqa: BLE001 - collected for the assertions
            errors.append(exc)

    try:
        threads = [
            threading.Thread(target=worker, args=(a,)) for a in (0.2, 0.5, 0.8, 1.0)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120.0)
    finally:
        artifact.reducer = real
    assert errors == [], errors
    assert len(results) == 4 and all(r["status"] == "ok" for r in results)
    assert detector.calls >= 4 * (len(STRENGTH_GRID) + 1)
    assert detector.overlaps == 0, f"{detector.overlaps} overlapping UMAP calls"

    # Fits, too: two cold refits on different nodes at once complete.
    meta = payload["meta"]
    fits: list[Any] = []

    def refit(node_id: str) -> None:
        fits.append(
            build_projection_artifact(
                analysis_id=meta["analysis_id"],
                node_id=node_id,
                node=find_node(payload["tree"], node_id),
                method="UMAP",
                effective_config=meta["effective_config"],
                feature_cols=meta["feature_cols"],
                df=df,
            )
        )

    pair = [threading.Thread(target=refit, args=(n,)) for n in ("root/0", "root/1")]
    for t in pair:
        t.start()
    for t in pair:
        t.join(timeout=120.0)
    assert len(fits) == 2


# ── The background-job path ─────────────────────────────────────────────────


def _poll(job_id: str, timeout: float = 120.0):
    """Poll a movement job to a terminal state, as the frontend does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = movement_jobs.get(job_id)
        assert job is not None, "the job vanished mid-poll"
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("movement job never reached a terminal state")


def test_only_an_uncached_umap_refit_needs_a_background_job():
    umap_payload, umap_df = _payload("UMAP")
    pca_payload, pca_df = _payload("PCA")
    umap = validate_request(_request(umap_payload), umap_payload)
    pca = validate_request(_request(pca_payload), pca_payload)

    clear_artifact_cache()
    # PCA is a pinv solve against an artifact: never a polling round trip, even
    # cold — that is the interactive path behind the strength slider.
    assert needs_background(pca) is False
    assert needs_background(umap) is True  # the reducer still has to be refitted

    artifact_for(umap, umap_df)
    assert artifact_is_cached(umap.analysis_id, umap.node_id) is True
    assert needs_background(umap) is False  # a warm artifact answers inline
    artifact_for(pca, pca_df)  # leave both caches warm for the tests below


def test_movement_job_runs_polls_and_returns_the_response():
    payload, df, _artifact_unused, destination = _umap_pieces()
    request = _request(payload, target=_point(destination))
    validated = validate_request(request, payload)

    movement_jobs.clear()
    job = movement_jobs.submit(
        movement_jobs.job_key(request),
        lambda: compute_validated_movement(validated, df),
    )
    # Before it finishes the envelope is the running variant …
    running = movement_jobs.envelope(job)
    assert running["status"] in {"running", "ok"}
    finished = _poll(job.id)
    assert finished.status == "done", finished.detail

    # … and the terminal envelope is the MovementResponse ITSELF — no "done"
    # wrapper, because a movement response already carries its own status.
    response = movement_jobs.envelope(finished)
    assert response["status"] == "ok"
    assert "job_id" not in response
    assert response["node_id"] == validated.node_id
    assert response["method"] == "UMAP"
    import json

    assert json.loads(json.dumps(response, allow_nan=False))["status"] == "ok"

    # Polling again still finds it: the result is held in memory (never
    # persisted) until the bounded store prunes it.
    assert movement_jobs.envelope(movement_jobs.get(job.id)) == response
    movement_jobs.clear()


def test_movement_job_reattaches_instead_of_starting_a_second_run():
    movement_jobs.clear()
    release = threading.Event()

    def slow() -> dict[str, Any]:
        release.wait(timeout=30.0)
        return {"status": "ok", "node_id": "root"}

    key = "movement-key"
    first = movement_jobs.submit(key, slow)
    second = movement_jobs.submit(key, slow)
    # A debounced slider or a retried POST must re-attach, not start a second
    # UMAP fit of the same node.
    assert second.id == first.id
    assert movement_jobs.envelope(second) == {"status": "running", "job_id": first.id}
    release.set()
    assert _poll(first.id).status == "done"
    assert movement_jobs.envelope(movement_jobs.get(first.id))["node_id"] == "root"
    movement_jobs.clear()


def test_movement_job_reports_a_failure_through_the_poll():
    movement_jobs.clear()

    def broken() -> dict[str, Any]:
        raise MovementError(409, "the refit could not be reconciled")

    job = movement_jobs.submit("movement-failure", broken)
    assert _poll(job.id).status == "error"
    envelope = movement_jobs.envelope(movement_jobs.get(job.id))
    assert envelope["status"] == "error"
    assert envelope["job_id"] == job.id
    assert "could not be reconciled" in envelope["detail"]
    movement_jobs.clear()


# ── The effective-config plumbing (H2) ──────────────────────────────────────


def _config_handed_to_the_fit(payload: dict[str, Any], df: Any) -> dict[str, Any]:
    """Run one cold movement against `payload` and return the config that reached
    `fit_node_projection` — the only place a refit's knobs are consumed."""
    seen: list[dict[str, Any]] = []
    real_fit = movement_module.fit_node_projection

    def spy(X: Any, config: Any) -> Any:
        seen.append(dict(config))
        return real_fit(X, config)

    clear_artifact_cache()
    movement_module.fit_node_projection = spy
    try:
        compute_movement(_request(payload), payload, df)
    finally:
        movement_module.fit_node_projection = real_fit
        clear_artifact_cache()
    assert len(seen) == 1, "expected exactly one refit"
    return seen[0]


def test_refit_reads_effective_config_not_the_request_config():
    """The regression test for the effective-config amendment.

    A data-driven canary cannot prove this: on every bundled dataset the clamp
    changes nothing, and umap-learn truncates `n_neighbors` to n−1 internally
    anyway, so raw and effective configs produce identical fits. So the payload
    here is built with `meta.config` and `meta.effective_config` deliberately
    DISAGREEING on a refit-relevant knob, and the assertion is on the plumbing:
    the value that reaches the fit is the effective one. PCA is used because
    the knob does not change a PCA fit, which keeps the alignment gate out of
    the picture — the point is what was handed to the fit, not what it drew.
    """
    payload, df = _payload("PCA")
    meta = payload["meta"]
    requested = {**meta["config"], "umap_n_neighbors": 15}
    divergent = {
        "meta": {
            **meta,
            "config": requested,
            "analysis_id": analysis_id(DATASET, meta["feature_cols"], requested),
            "effective_config": {**meta["effective_config"], "umap_n_neighbors": 7},
        },
        "tree": payload["tree"],
    }
    assert _config_handed_to_the_fit(divergent, df)["umap_n_neighbors"] == 7

    # Negative control: the bug this test exists to catch — a refit built from
    # the REQUEST config — must make the assertion above fail. If this block
    # ever passes with 7, the test has gone vacuous.
    real_refit = movement_module.refit_config

    def from_the_request(_effective: Any) -> Any:
        config = default_config()
        config.update(requested)  # type: ignore[typeddict-item]
        return config

    movement_module.refit_config = from_the_request
    try:
        assert _config_handed_to_the_fit(divergent, df)["umap_n_neighbors"] == 15
    finally:
        movement_module.refit_config = real_refit


# ── The artifact cache under concurrency (H1) ───────────────────────────────


def _race_two_cold_requests(fit: Any) -> tuple[list[Any], list[BaseException]]:
    """Two threads ask for the same cold node; the second one starts only once
    the first is INSIDE `fit`, so it must find that fit in flight."""
    payload, df = _payload("PCA")
    validated = validate_request(_request(payload, node_id="root/1"), payload)
    clear_artifact_cache()
    started = threading.Event()
    release = threading.Event()
    real_fit = movement_module.fit_node_projection

    def gated(X: Any, config: Any) -> Any:
        started.set()
        assert release.wait(timeout=30.0), "the gated fit was never released"
        return fit(real_fit, X, config)

    results: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(artifact_for(validated, df))
        except BaseException as exc:  # noqa: BLE001 - collected for the assertions
            errors.append(exc)

    movement_module.fit_node_projection = gated
    try:
        first = threading.Thread(target=worker)
        first.start()
        assert started.wait(timeout=30.0), "the first request never reached the fit"
        second = threading.Thread(target=worker)
        second.start()
        second.join(timeout=0.5)
        assert second.is_alive(), "the second request did not wait for the first"
        release.set()
        first.join(timeout=60.0)
        second.join(timeout=60.0)
        assert not first.is_alive() and not second.is_alive()
    finally:
        release.set()
        movement_module.fit_node_projection = real_fit
    return results, errors


def test_concurrent_cold_requests_share_one_refit():
    calls: list[int] = []

    def counted(real_fit: Any, X: Any, config: Any) -> Any:
        calls.append(1)
        return real_fit(X, config)

    results, errors = _race_two_cold_requests(counted)
    assert errors == []
    # The LRU is shared between request and job threads: one fit, one artifact,
    # both callers get the very same object, and nothing is left in flight.
    assert len(calls) == 1, f"fit_node_projection ran {len(calls)} times"
    assert len(results) == 2 and results[0] is results[1]
    assert not movement_module._artifact_in_flight
    assert artifact_is_cached(results[0].analysis_id, results[0].node_id)


def test_failed_refit_reaches_every_waiter_and_leaves_nothing_cached():
    def broken(_real_fit: Any, _X: Any, _config: Any) -> Any:
        raise RuntimeError("forced refit failure")

    results, errors = _race_two_cold_requests(broken)
    assert results == []
    assert len(errors) == 2 and all(isinstance(e, RuntimeError) for e in errors)
    assert not movement_module._artifact_in_flight
    payload, df = _payload("PCA")
    validated = validate_request(_request(payload, node_id="root/1"), payload)
    assert not artifact_is_cached(validated.analysis_id, validated.node_id)
    # The failure is not sticky: the next request refits for real.
    assert artifact_for(validated, df).node_id == "root/1"


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"OK — {_name}")
    print("OK — movement alignment + canary passed")

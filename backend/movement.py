"""Cluster movement — counterfactual preview of a shared feature-space translation.

An analyst picks a source child cluster and a destination (a free coordinate in
the visible projection, or a sibling cluster). This module answers with ONE
feature-space displacement `Δ` that every source point receives:

    X_source' = X_source + strength · Δ

so the cluster translates without morphing — pairwise distances inside it are
preserved exactly. Nothing is written back: neither the source dataframe nor the
analysis tree is mutated, and the result is not a new analysis run.

HTTP CONTRACT
=============

`POST /api/movement`  (mirrors `/api/analysis`: a job id when work is needed, the
result inline when it is not)

Request::

    {
      "analysis_id": "8a42e8d4…",          # AnalysisMeta.analysis_id of the run on screen
      "dataset": "Iris (Low)",
      "feature_cols": ["sepal length (cm)", "..."],
      "config": {"method": "UMAP"},         # the client's knobs — see below
      "node_id": "root/1",                  # the PARENT node (TreeNode.id)
      "source_child_index": 0,
      "target": {"kind": "point", "x": 2.41, "y": -1.82},
      # or:     {"kind": "cluster", "child_index": 2}
      "strength": null                      # null = use the recommended strength
    }

`config` is echoed only so the run signature can be checked. Refits read the
serialized `meta.effective_config` — the post-build, MUTATED config — and NEVER
this field: `compute_analysis_tree` clamps `hclust_umap_n_components` and
`umap_n_neighbors` against the root size before any per-node reducer is fit, so
every visible embedding was produced with the clamped values. Refitting from the
raw request config builds a different reducer whenever the clamp fired, and the
alignment gate then rejects the movement for an invisible reason.

Response — the `MovementResponse` of `frontend/src/types.ts`::

    {
      "status": "ok", "analysis_id": …, "node_id": …, "method": "PCA",
      "source_child_index": 0,
      "target": {"kind": …, "child_index": null, "x": …, "y": …,
                 "estimator": "pca_pinv" | "umap_inverse"
                            | "neighbor_interpolation" | "observed_centroid"},
      "recommended_strength": …, "applied_strength": …,
      "source_size": …, "target_size": …,
      "feature_changes": [{"feature", "mutable", "source_mean_raw",
                           "target_value_raw", "recommended_delta_raw",
                           "applied_delta_raw", "standardized_magnitude"}],
      "preview_points": [{"row_id", "x", "y"}],
      "visible_displacement": [dx, dy],
      "strength_trajectory": [{"strength", "projected_distance"}],
      "metrics": {"projected_distance_before", "projected_distance_after",
                  "projected_distance_reduction",
                  "feature_centroid_distance_before",
                  "feature_centroid_distance_after", "out_of_range_fraction",
                  "support_ratio", "alignment_rmse", "alignment_max_residual",
                  "preview_sampled"},
      "warnings": [str, …]
    }

COORDINATE CONVENTION (binding): every 2D quantity that leaves this module —
target x/y, preview points, `visible_displacement`, projected distances — is in
VISIBLE coordinates, the units of the parent's serialized `embedding_original`.
The refitted reducer's own coordinates never cross the wire; they are mapped
through the stored alignment transform first.

STATUS CODES
------------
400  the request cannot describe a movement: unknown/leaf/childless node,
     invalid source or destination child, source == destination, non-finite
     coordinates, strength outside [0, 1], unsupported reducer (t-SNE and MDS
     are not supported), parent with fewer than 1 child for a point target or 2
     for a cluster target, parent whose `embedding_original` is null.
409  the request is well-formed but the server cannot honour it against the run
     the client holds: `analysis_id` does not match, the analysis payload is
     gone from the cache, the payload predates `effective_config` (rebuild
     required), or the refit fails to align with the stored embedding.

`GET /api/movement/jobs/{job_id}` — `{"status": "running", "job_id"}`,
`{"status": "error", "job_id", "detail"}`, or the finished `MovementResponse`
itself. Note this differs from `/api/analysis/jobs/{job_id}`: there is no
`"done"` wrapper, because a movement response already carries a `status` of its
own (`"ok"`) that a wrapper would collide with. `MovementJob` in
`frontend/src/types.ts` is the same three-way union.

Only a UMAP request that has to refit a reducer becomes a job. PCA is always
inline — it is one pinv solve — and so is a UMAP request whose projection
artifact is already cached. Validation runs synchronously either way, so a 400
or 409 is an HTTP error rather than a job failure discovered one poll later.

Movement results are never persisted. Only reusable projection artifacts are
cached, in memory, keyed by (analysis_id, node_id) with a bounded LRU that is
shared between request and job threads under a lock; concurrent cold requests
for one node wait for a single refit rather than each running their own.

`metrics.feature_centroid_distance_after` is `null` for POINT targets: their
feature-space destination is *defined* as `μ_source + Δ`, so the value would be
`(1−α)·‖Δ‖` by construction and read as if the click had been reached whether or
not it was. Cluster targets have an observed destination and report it.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from backend.serialize import SCHEMA_VERSION, _finite
from backend.serialize import analysis_id as run_analysis_id
from src.analysis.analysis_routine import fit_node_projection
from src.analysis.dim_reducer import NUMBA_PARALLEL_LOCK, ReductionResult
from src.config_defaults import default_config

if TYPE_CHECKING:
    import pandas as pd

    from src.types import Config

# The reducers a movement can be expressed in. PCA is linear (exact inversion of
# a shared displacement); UMAP carries an approximate `inverse_transform`. The
# other two reducers in `src/analysis/dim_reducer.py` have neither: t-SNE has no
# out-of-sample transform at all, and MDS only reconstructs from a distance
# matrix. Both are named in the rejection message so the UI can say which.
SUPPORTED_METHODS = ("PCA", "UMAP")
UNSUPPORTED_METHODS = ("t-SNE", "MDS")

# Rejection threshold for the refit→visible alignment. A PRODUCT decision, not a
# derived tolerance. What it bounds, exactly: the root-mean-square residual of
# the similarity alignment over the node's EXISTING points, normalized by the
# visible cloud's RMS radius — how far, on average, the refit places the points
# the analyst is already looking at from where they are drawn, as a fraction of
# the plot's own scale. It does NOT bound individual residuals (single points can
# exceed 5 %; the response reports the worst one as `alignment_max_residual`),
# and it says nothing about the error of NEW counterfactual points, which the
# alignment never saw. A same-process refit lands at ~1e-16 for PCA and well
# under 1e-3 for UMAP, so this is a tripwire, not a tuning knob: a request that
# trips it is answered with 409 + "rebuild", never with a widened bound and a
# preview drawn in the wrong place. Never widen.
ALIGNMENT_RMSE_MAX = 0.05

# Bounded LRU of fitted reducers, same reasoning as `_TREE_CACHE_MAX` in
# backend/app.py: an artifact pins the parent's scaled matrix, its embedding and
# a fitted UMAP model (tens of MB on a wide dataset), and every node the analyst
# visits is a new key, so an unbounded dict grows monotonically into the
# container's memory limit.
_ARTIFACT_CACHE_MAX = 4


class MovementError(Exception):
    """A movement request the server will not answer.

    `status_code` is the HTTP status the route must return — 400 when the request
    cannot describe a movement at all, 409 when it is well formed but cannot be
    honoured against the run the client holds. The route translates this into an
    `HTTPException`; this module stays framework-free.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ── Node lookup ─────────────────────────────────────────────────────────────


def find_node(tree: Mapping[str, Any] | None, node_id: str) -> dict[str, Any] | None:
    """Resolve a serialized node from the path id `serialize_node` emits.

    `"root"`, `"root/1"`, `"root/2/0"` — each segment after `root` is the index
    of a child in `children`. `None` for anything that does not resolve.
    """
    if tree is None or not node_id:
        return None
    parts = node_id.split("/")
    if parts[0] != "root":
        return None
    node: Any = tree
    for part in parts[1:]:
        children = node.get("children") or []
        if not part.isdigit():
            return None
        index = int(part)
        if not 0 <= index < len(children):
            return None
        node = children[index]
    return node


# ── Request + validation ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MovementRequestData:
    """The `MovementRequest` of `frontend/src/types.ts`, framework-free.

    Field names match the TS contract exactly, so a route holding a pydantic
    model builds one with `MovementRequestData(**req.model_dump())`.
    """

    analysis_id: str
    dataset: str
    feature_cols: list[str]
    config: dict[str, Any]
    node_id: str
    source_child_index: int
    target: dict[str, Any]
    strength: float | None = None


@dataclass(frozen=True)
class ValidatedMovement:
    """A request that named a real, movable configuration. Everything downstream
    (artifact construction, the displacement solve, the response builder) reads
    this rather than the raw request — in particular `effective_config`, never
    `MovementRequestData.config`."""

    analysis_id: str
    dataset: str
    feature_cols: list[str]
    node_id: str
    node: dict[str, Any]  # the serialized PARENT node
    method: str  # "PCA" | "UMAP", as the run was BUILT
    effective_config: dict[str, Any]
    source_child_index: int
    target_kind: str  # "point" | "cluster"
    target_child_index: int | None
    target_xy: tuple[float, float] | None  # visible coordinates
    strength: float | None  # None = use the recommended strength

    @property
    def children(self) -> list[dict[str, Any]]:
        return self.node["children"] or []

    @property
    def source_node(self) -> dict[str, Any]:
        return self.children[self.source_child_index]

    @property
    def target_node(self) -> dict[str, Any] | None:
        if self.target_child_index is None:
            return None
        return self.children[self.target_child_index]


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_compatible_run(
    req: MovementRequestData, payload: Mapping[str, Any] | None
) -> tuple[dict[str, Any], Any]:
    """The 409 family: the request is well formed, but not about the run we hold.

    Returns the payload's `(meta, tree)`.
    """
    if payload is None:
        raise MovementError(
            409,
            "The analysis this movement refers to is no longer on the server. "
            "Rebuild it and try the movement again.",
        )
    meta = dict(payload.get("meta") or {})
    # A cached payload can predate the movement fields entirely: run_cache.py keys
    # on (dataset, feature_cols, config), not on the code, so an old entry is
    # still a cache hit. Refusing here is the whole point of the schema version —
    # the alternative is silently refitting from the raw config. Checked BEFORE
    # the identity comparison: a legacy payload carries no `analysis_id` at all,
    # so the id check would fail too, with the wrong diagnosis ("different run"
    # instead of "rebuild required").
    if int(meta.get("schema_version") or 0) < SCHEMA_VERSION or not meta.get(
        "effective_config"
    ):
        raise MovementError(
            409,
            "This analysis was built before movement support was added, so the "
            "configuration its projections were fitted with was not recorded. "
            "Rebuild the analysis to enable movement.",
        )
    expected = run_analysis_id(req.dataset, req.feature_cols, req.config)
    if req.analysis_id != expected or meta.get("analysis_id") != req.analysis_id:
        raise MovementError(
            409,
            "This movement was requested against a different analysis run than "
            "the one the server holds. Rebuild the analysis and try again.",
        )
    return meta, payload.get("tree")


def validate_request(
    req: MovementRequestData, payload: Mapping[str, Any] | None
) -> ValidatedMovement:
    """Check a movement request against the serialized run, or raise `MovementError`.

    `payload` is the cached `/api/analysis` payload (`{"meta": …, "tree": …}`);
    `None` means the cache no longer holds it.
    """
    meta, tree = _require_compatible_run(req, payload)
    effective = dict(meta["effective_config"])

    node = find_node(tree, req.node_id)
    if node is None:
        raise MovementError(400, f"Unknown node: {req.node_id!r}")
    if node.get("is_leaf"):
        raise MovementError(
            400,
            f"Node {req.node_id!r} is a leaf: it has no child clusters to move.",
        )
    children = node.get("children") or []
    if not children:
        raise MovementError(400, f"Node {req.node_id!r} has no child clusters.")
    if node.get("embedding_original") is None:
        raise MovementError(
            400,
            f"Node {req.node_id!r} has no projection, so there is nothing to move "
            "within.",
        )

    requested_method = str(effective.get("method") or "")
    # Report the canonical spelling ("PCA" / "UMAP") downstream — the response
    # contract types it as those two literals — while the refit keeps reading
    # `effective_config` verbatim.
    method = next((m for m in SUPPORTED_METHODS if m == requested_method.upper()), "")
    if not method:
        raise MovementError(
            400,
            f"Movement is only available for {' and '.join(SUPPORTED_METHODS)} "
            f"projections. This run uses {requested_method or 'an unknown reducer'}; "
            f"{UNSUPPORTED_METHODS[0]} and {UNSUPPORTED_METHODS[1]} are not "
            "supported (neither offers a usable inverse of the projection).",
        )

    if req.strength is not None and (
        not _is_finite_number(req.strength) or not 0.0 <= float(req.strength) <= 1.0
    ):
        raise MovementError(
            400,
            f"Strength must be null or a number in [0, 1]; got {req.strength!r}.",
        )

    if not 0 <= req.source_child_index < len(children):
        raise MovementError(
            400,
            f"Source child index {req.source_child_index} is out of range for "
            f"{req.node_id!r} ({len(children)} children).",
        )

    target = dict(req.target or {})
    kind = target.get("kind")
    target_child_index: int | None = None
    target_xy: tuple[float, float] | None = None
    if kind == "point":
        x, y = target.get("x"), target.get("y")
        if not _is_finite_number(x) or not _is_finite_number(y):
            raise MovementError(
                400, f"Target coordinates must be finite numbers; got {x!r}, {y!r}."
            )
        target_xy = (float(x), float(y))  # type: ignore[arg-type]
    elif kind == "cluster":
        index = target.get("child_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise MovementError(
                400, f"Cluster target needs an integer child_index; got {index!r}."
            )
        if not 0 <= index < len(children):
            raise MovementError(
                400,
                f"Destination child index {index} is out of range for "
                f"{req.node_id!r} ({len(children)} children).",
            )
        if index == req.source_child_index:
            raise MovementError(400, "Source and destination clusters must differ.")
        target_child_index = index
    else:
        raise MovementError(
            400, f"Unknown target kind {kind!r}; expected 'point' or 'cluster'."
        )

    # A point target needs one cluster to move; a cluster target needs a second
    # one to move it towards.
    required_children = 2 if kind == "cluster" else 1
    if len(children) < required_children:
        raise MovementError(
            400,
            f"Node {req.node_id!r} has {len(children)} child cluster(s); a "
            f"{kind} target needs at least {required_children}.",
        )

    return ValidatedMovement(
        analysis_id=req.analysis_id,
        dataset=req.dataset,
        feature_cols=list(req.feature_cols),
        node_id=req.node_id,
        node=node,
        method=method,
        effective_config=effective,
        source_child_index=req.source_child_index,
        target_kind=str(kind),
        target_child_index=target_child_index,
        target_xy=target_xy,
        strength=None if req.strength is None else float(req.strength),
    )


# ── Parent reconstruction ───────────────────────────────────────────────────


def refit_config(effective: Mapping[str, Any]) -> Config:
    """A complete `Config` carrying the run's *effective* knobs.

    `default_config()` supplies the keys a reducer fit never reads; the recorded
    effective config overrides every knob that shaped the visible embeddings.
    Never build this from a request's `config` field — see the module docstring.
    """
    config = default_config()
    config.update(effective)  # type: ignore[typeddict-item]
    return config


def root_scaled_matrix(
    df: pd.DataFrame, feature_cols: list[str], config: Config
) -> tuple[np.ndarray, StandardScaler | None]:
    """The root feature matrix, scaled exactly as `compute_analysis_tree` did.

    One `StandardScaler` fit on the WHOLE frame when `config["normalize"]`, no
    scaling otherwise. Per-node matrices are slices of this — the analysis fits
    each node's reducer on `X_orig` sliced from the root-scaled matrix, not on a
    per-node rescaling. `to_numpy()` copies, so the source frame is untouched.
    """
    values = df[feature_cols].to_numpy()
    if config["normalize"]:
        scaler = StandardScaler()
        return scaler.fit_transform(values), scaler
    return values, None


@dataclass
class ProjectionArtifact:
    """A node's reducer, brought back. The analysis serializes coordinates and
    discards the fitted reducers (in local *and* hosted mode), so a refit is the
    only path to one — not an optimization that could be skipped."""

    analysis_id: str
    node_id: str
    method: str
    parent_row_indices: np.ndarray  # indices into the source dataframe
    X_parent: np.ndarray  # root-scaled rows of this node
    embedding: np.ndarray  # the REFIT embedding, in fitted coordinates
    reducer: object
    scaler: StandardScaler | None  # the root scaler, or None when normalize=False


# The LRU is shared between request threads and `movement_jobs` worker threads,
# so every read and write of it happens under this lock. The lock guards the
# dictionary operations ONLY — never a fit and never a wait: a refit can take
# minutes, and holding the lock across it would queue every movement on the
# server behind one cold UMAP node.
_artifact_lock = threading.Lock()
_artifact_cache: OrderedDict[tuple[str, str], ProjectionArtifact] = OrderedDict()
# Cold refits in flight. A second request for the same node waits on the first
# fit's Future instead of running its own — two different first UMAP requests on
# one node (a point target, then a cluster target) used to refit the reducer
# twice and let the second overwrite the first.
_artifact_in_flight: dict[tuple[str, str], Future[ProjectionArtifact]] = {}


def _refit_node_projection(X: np.ndarray, config: Config) -> ReductionResult | None:
    """`fit_node_projection`, except for a wide PCA node (more columns than rows).

    The analysis' PCA uses `covariance_eigh`, which forms the p x p covariance:
    on Olivetti's 400 x 4096 root that peaks at ~650 MiB and takes 6-24 s, more
    than a 640 MiB container holds. LAPACK's thin SVD of the same matrix needs
    ~66 MiB and 0.3 s. Both are RNG-free and sign-fixed by `svd_flip`, so the
    components agree to rounding and `require_alignment` still checks the refit
    against the projection on screen. Tall nodes keep the shared path unchanged.
    """
    n, p = X.shape
    if str(config["method"]).lower() != "pca" or p <= n:
        return fit_node_projection(X, config)
    if n < 2:  # fit_node_projection's guard (_MIN_EMBED_DIMS)
        return None
    try:
        pca = PCA(n_components=2, svd_solver="full")
        embedding = pca.fit_transform(X)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return ReductionResult(embedding, pca, pca.explained_variance_ratio_)


def build_projection_artifact(
    *,
    analysis_id: str,
    node_id: str,
    node: Mapping[str, Any],
    method: str,
    effective_config: Mapping[str, Any],
    feature_cols: list[str],
    df: pd.DataFrame,
) -> ProjectionArtifact:
    """Refit one node's projection. Uncached — `artifact_for` is the cached entry."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise MovementError(
            409,
            f"The dataset no longer has the feature columns this analysis used: "
            f"{missing[:5]}. Rebuild the analysis.",
        )
    config = refit_config(effective_config)
    X_root, scaler = root_scaled_matrix(df, feature_cols, config)
    row_indices = np.asarray(node["row_indices"], dtype=int)
    if row_indices.size == 0 or int(row_indices.max()) >= len(df):
        raise MovementError(
            409,
            "The analysis refers to rows this dataset no longer has. Rebuild it "
            "and try the movement again.",
        )
    X_parent = X_root[row_indices]
    result = _refit_node_projection(X_parent, config)
    if result is None:
        raise MovementError(
            409,
            "The projection for this node could not be refitted. Rebuild the "
            "analysis and try the movement again.",
        )
    return ProjectionArtifact(
        analysis_id=analysis_id,
        node_id=node_id,
        method=method,
        parent_row_indices=row_indices,
        X_parent=X_parent,
        embedding=np.asarray(result.embedding, dtype=float),
        reducer=result.reducer,
        scaler=scaler,
    )


def artifact_for(v: ValidatedMovement, df: pd.DataFrame) -> ProjectionArtifact:
    """The parent's projection artifact, from the bounded LRU or freshly refitted.

    Under the lock: look the key up in the LRU, then in the in-flight table; if
    it is in neither, register a new Future and own it. The lock is released
    before anything slow — the owner fits outside it, a waiter blocks on the
    Future outside it. The owner then stores the artifact and pops the Future
    under the lock again, and resolves the Future after releasing it. A fit that
    raises pops its Future and propagates the exception to every waiter.
    """
    key = (v.analysis_id, v.node_id)
    with _artifact_lock:
        cached = _artifact_cache.get(key)
        if cached is not None:
            _artifact_cache.move_to_end(key)
            return cached
        pending = _artifact_in_flight.get(key)
        owner = pending is None
        if pending is None:
            pending = _artifact_in_flight[key] = Future()
    if not owner:
        return pending.result()  # re-raises the owner's failure, if that is the outcome
    try:
        artifact = build_projection_artifact(
            analysis_id=v.analysis_id,
            node_id=v.node_id,
            node=v.node,
            method=v.method,
            effective_config=v.effective_config,
            feature_cols=v.feature_cols,
            df=df,
        )
    except BaseException as exc:
        with _artifact_lock:
            _artifact_in_flight.pop(key, None)
        pending.set_exception(exc)
        raise
    with _artifact_lock:
        _artifact_cache[key] = artifact
        _artifact_cache.move_to_end(key)
        while len(_artifact_cache) > _ARTIFACT_CACHE_MAX:
            _artifact_cache.popitem(last=False)
        _artifact_in_flight.pop(key, None)
    pending.set_result(artifact)
    return artifact


def clear_artifact_cache() -> None:
    """Drop every cached artifact (tests; a rebuild invalidates them by key anyway).

    Fits in flight are left alone: their waiters hold the Future, and the
    artifact is stored when the fit returns.
    """
    with _artifact_lock:
        _artifact_cache.clear()


# ── Alignment ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Alignment:
    """The similarity transform from FITTED coordinates to VISIBLE ones.

    `visible ≈ scale · rotation · fitted + translation`, with `rotation`
    orthogonal — reflections allowed, because a refit may mirror the projection
    without changing what it means. Both directions are kept: a clicked target
    arrives in visible coordinates and must reach the reducer in fitted ones,
    and every preview coordinate must come back out in visible ones.
    """

    scale: float
    rotation: np.ndarray  # 2x2 orthogonal; det ±1
    translation: np.ndarray  # length-2
    rmse: float  # normalized: residual RMSE / RMS radius of the visible cloud
    # The single worst point, in the same normalized units. `ALIGNMENT_RMSE_MAX`
    # bounds the RMS only, so this can sit well above it on a passing alignment.
    max_residual: float
    n_points: int  # points that carried finite coordinates in both embeddings

    def to_visible(self, fitted: np.ndarray) -> np.ndarray:
        a = np.asarray(fitted, dtype=float).reshape(-1, 2)
        return a @ (self.scale * self.rotation).T + self.translation

    def to_fitted(self, visible: np.ndarray) -> np.ndarray:
        if self.scale <= 0.0:
            raise MovementError(
                409,
                "The refitted projection collapsed to a single point and cannot "
                "be aligned with the one on screen. Rebuild the analysis.",
            )
        b = np.asarray(visible, dtype=float).reshape(-1, 2)
        return (b - self.translation) @ self.rotation / self.scale


def embedding_array(embedding: Any) -> np.ndarray:
    """Serialized `embedding_original` (`[[x|null, y|null], …]`) -> Nx2 float array.

    `serialize.py` maps non-finite coordinates to `null`; they come back as NaN
    and are excluded from the alignment fit rather than silently read as 0.
    """
    if embedding is None:
        return np.empty((0, 2), dtype=float)
    rows = [
        [
            math.nan if xy[0] is None else float(xy[0]),
            math.nan if xy[1] is None else float(xy[1]),
        ]
        for xy in embedding
    ]
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def align_embeddings(fitted: np.ndarray, visible: Any) -> Alignment:
    """Fit the similarity transform taking `fitted` onto `visible`.

    Orthogonal Procrustes with a uniform scale (the closed-form Umeyama
    solution): centre both clouds, take `R = U Vᵀ` of `SVD(visibleᵀ fitted)` —
    the determinant is NOT forced to +1, so a mirrored refit aligns — then
    `s = Σσ / Σ‖fittedᶜ‖²` and `t` from the centroids.
    """
    A = np.asarray(fitted, dtype=float).reshape(-1, 2)
    B = embedding_array(visible) if not isinstance(visible, np.ndarray) else visible
    B = np.asarray(B, dtype=float).reshape(-1, 2)
    if A.shape[0] != B.shape[0]:
        raise MovementError(
            409,
            f"The refitted projection has {A.shape[0]} points but the stored one "
            f"has {B.shape[0]}. Rebuild the analysis.",
        )
    usable = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    if int(usable.sum()) < 2:
        raise MovementError(
            409,
            "The stored projection for this node has fewer than two usable "
            "points, so the refit cannot be aligned with it. Rebuild the analysis.",
        )
    a, b = A[usable], B[usable]
    a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - a_mean, b - b_mean

    u, sigma, vt = np.linalg.svd(bc.T @ ac)
    rotation = u @ vt
    denom = float((ac**2).sum())
    scale = float(sigma.sum() / denom) if denom > 0.0 else 0.0
    translation = b_mean - scale * (rotation @ a_mean)

    residual = b - (a @ (scale * rotation).T + translation)
    per_point = np.sqrt((residual**2).sum(axis=1))
    rmse_raw = float(np.sqrt((per_point**2).mean()))
    max_raw = float(per_point.max())
    radius = float(np.sqrt((bc**2).sum(axis=1).mean()))
    if radius > 0.0:
        rmse, max_residual = rmse_raw / radius, max_raw / radius
    else:  # the visible cloud is a single point — anything else is unalignable
        rmse = max_residual = 0.0 if rmse_raw == 0.0 else math.inf
    return Alignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        rmse=rmse,
        max_residual=max_residual,
        n_points=int(usable.sum()),
    )


def require_alignment(
    artifact: ProjectionArtifact, node: Mapping[str, Any]
) -> Alignment:
    """Align a refit to the node's serialized embedding, or refuse the movement.

    A large error means the refit is not the projection on screen, so every
    preview coordinate would be drawn in the wrong place. That is a 409 asking
    for a rebuild — never a threshold to widen.
    """
    alignment = align_embeddings(artifact.embedding, node.get("embedding_original"))
    if (
        not math.isfinite(alignment.rmse)
        or alignment.rmse > ALIGNMENT_RMSE_MAX
        or alignment.scale <= 0.0
    ):
        raise MovementError(
            409,
            "The refitted projection does not reproduce the one on screen "
            f"(normalized alignment error {alignment.rmse:.4g}, limit "
            f"{ALIGNMENT_RMSE_MAX:g}). Rebuild the analysis and try the movement "
            "again.",
        )
    return alignment


# ── Mutable features ────────────────────────────────────────────────────────

# Columns a movement never edits. `backend/datasets.py::default_feature_cols`
# keeps them out of the feature space already, so they reach this module only
# when the analyst added them by hand — and a counterfactual that rewrites the
# ground truth (or the row identifier) is not one worth recommending. A
# configurable checklist is explicitly a later iteration; this is the default
# set, and for now the only one.
IMMUTABLE_COLUMNS = ("row_id",)
IMMUTABLE_PREFIX = "target_"

# The preview carries one record per moved point. Above this many, it is
# subsampled deterministically and flagged `preview_sampled`; every metric and
# every centroid still comes from the COMPLETE cluster.
MAX_MOVEMENT_PREVIEW_POINTS = 5000

# Above this share of moved points leaving the dataset's observed range, the
# counterfactual is extrapolating and the panel has to say so.
OUT_OF_RANGE_WARN_FRACTION = 0.05


def mutable_flags(feature_cols: list[str]) -> np.ndarray:
    """Boolean mask over `feature_cols`: True = this movement may change it."""
    return np.array(
        [
            c not in IMMUTABLE_COLUMNS and not str(c).startswith(IMMUTABLE_PREFIX)
            for c in feature_cols
        ],
        dtype=bool,
    )


# ── The displacement ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Displacement:
    """One shared feature-space translation, and how it was arrived at."""

    delta_input: np.ndarray  # length d, reducer INPUT space; immutable entries 0
    target_input: np.ndarray  # the feature-space destination it aims at
    target_fitted: np.ndarray  # length 2, the destination in FITTED coordinates
    estimator: str  # "pca_pinv" | "observed_centroid"
    warnings: list[str]


def _min_norm_note(normalize: bool) -> str:
    """The under-determination sentence for a free-point PCA target.

    `pinv` returns ONE answer out of a (d−2)-dimensional solution space — the
    smallest — and "smallest" is measured in the reducer's INPUT space, which is
    root-standardized units when the run normalized and raw units when it did
    not. That distinction changes which features the recommendation prefers, so
    it is stated rather than left implicit.
    """
    if normalize:
        return (
            "Many different feature changes reach this position; this is the "
            "smallest standardized change that reaches this point (smallest "
            "measured in root-standardized units — one unit is one standard "
            "deviation over the whole dataset)."
        )
    return (
        "Many different feature changes reach this position; this is the "
        "smallest raw change that reaches this point (smallest measured in the "
        "dataset's own units, because this analysis was built without "
        "normalization)."
    )


def free_point_displacement(
    *,
    components: np.ndarray,
    mutable: np.ndarray,
    source_mean_input: np.ndarray,
    source_mean_fitted: np.ndarray,
    target_fitted: np.ndarray,
    normalize: bool,
) -> Displacement:
    """The minimum-norm shared translation carrying the source centroid to a click.

    PCA is affine, so a displacement `Δ` moves every projected point by exactly
    `P Δ` (`P = components_`; the `mean_` centering cancels in the difference).
    Solving `P_mutable δ = y* − ȳ_source` with the pseudo-inverse gives the
    smallest such δ; immutable features keep a hard zero, they are not solved
    for and then discarded.

    Rank < 2 — fewer than two mutable features, or two whose loadings are
    collinear — leaves the pseudo-inverse returning the least-squares answer,
    which is exactly the closest reachable point. That is reported, never passed
    off as an exact hit.

    No `inverse_transform` and no per-point solve: one displacement for the whole
    cluster is what preserves each point's residual outside the two retained
    components.
    """
    d_fitted = np.asarray(target_fitted, dtype=float) - source_mean_fitted
    delta = np.zeros(components.shape[1], dtype=float)
    notes: list[str] = []
    p_mutable = components[:, mutable]
    if p_mutable.shape[1] == 0:
        notes.append(
            "No feature in this analysis may be changed — every selected column "
            "is a row identifier or a label — so no movement can be recommended."
        )
    else:
        delta[mutable] = np.linalg.pinv(p_mutable) @ d_fitted
        if int(np.linalg.matrix_rank(p_mutable)) < 2:
            notes.append(
                "The features that may change move this projection along a single "
                "direction, so the selected position cannot be reached exactly. "
                "The recommendation goes to the closest reachable point."
            )
        notes.append(_min_norm_note(normalize))
    return Displacement(
        delta_input=delta,
        target_input=source_mean_input + delta,
        target_fitted=np.asarray(target_fitted, dtype=float),
        estimator="pca_pinv",
        warnings=notes,
    )


def cluster_displacement(
    *,
    mutable: np.ndarray,
    source_mean_input: np.ndarray,
    target_mean_input: np.ndarray,
    target_fitted: np.ndarray,
) -> Displacement:
    """The shared translation taking the source centroid onto another cluster's.

    `Δ = μ_target − μ_source`, both means in the reducer's INPUT space, immutable
    entries zeroed afterwards. This solution is unique — no pseudo-inverse, so no
    min-norm caveat — and because PCA is linear the projected source centroid
    lands on the projected target centroid exactly whenever every feature is
    mutable.
    """
    delta = np.asarray(target_mean_input, dtype=float) - source_mean_input
    delta[~mutable] = 0.0
    return Displacement(
        delta_input=delta,
        target_input=np.asarray(target_mean_input, dtype=float),
        target_fitted=np.asarray(target_fitted, dtype=float),
        estimator="observed_centroid",
        warnings=[],
    )


# ── Response construction ───────────────────────────────────────────────────


def _pca_components(artifact: ProjectionArtifact) -> np.ndarray:
    """The refitted PCA's 2xd component matrix, in the reducer's input space."""
    components = getattr(artifact.reducer, "components_", None)
    matrix = None if components is None else np.asarray(components, dtype=float)
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] != 2:
        raise MovementError(
            409,
            "The refitted projection for this node is not the 2D PCA the movement "
            "needs. Rebuild the analysis and try the movement again.",
        )
    return matrix


def _require_finite(values: np.ndarray, what: str) -> np.ndarray:
    """Guard the aggregates the response types as plain numbers.

    Starlette encodes with `allow_nan=False`, and `visible_displacement` /
    `target.x` / `target.y` are not nullable in the TS contract — a NaN centroid
    has to become a 409, not a 500 at encode time.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise MovementError(
            409,
            f"The projection produced a non-finite {what}, so no movement can be "
            "derived from it. Rebuild the analysis and try the movement again.",
        )
    return array


def _child_positions(
    artifact: ProjectionArtifact, child: Mapping[str, Any]
) -> np.ndarray:
    """Row positions of a child cluster inside its parent's matrices.

    Serialized `row_indices` are global (positions in the dataframe) at every
    depth, so a child's rows have to be located in the parent's own ordering
    before they can index `X_parent` or `embedding`.
    """
    lookup = {int(g): i for i, g in enumerate(artifact.parent_row_indices)}
    try:
        return np.array(
            [lookup[int(g)] for g in (child["row_indices"] or [])], dtype=int
        )
    except KeyError as exc:
        raise MovementError(
            409,
            "A child cluster refers to rows its parent does not contain. Rebuild "
            "the analysis and try the movement again.",
        ) from exc


def _raw_scaling(
    scaler: StandardScaler | None, n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(scale, mean)` undoing the root standardization: `raw = input·scale + mean`.

    Identity when the run was built with `normalize=False` — then the reducer's
    input space already IS the dataset's own units.
    """
    if scaler is None:
        return np.ones(n_features), np.zeros(n_features)
    return np.asarray(scaler.scale_, dtype=float), np.asarray(scaler.mean_, dtype=float)


def _preview_indices(n: int) -> tuple[np.ndarray, bool]:
    """Local indices of the points to preview, and whether that is a subsample.

    Deterministic by construction (an evenly spaced stride, endpoints included) —
    no RNG, so the same request draws the same ghosts every time.
    """
    if n <= MAX_MOVEMENT_PREVIEW_POINTS:
        return np.arange(n), False
    picks = np.linspace(0, n - 1, MAX_MOVEMENT_PREVIEW_POINTS).round().astype(int)
    return np.unique(picks), True


def _out_of_range_fraction(
    df: pd.DataFrame,
    feature_cols: list[str],
    mutable: np.ndarray,
    source_raw: np.ndarray,
    applied_delta_raw: np.ndarray,
) -> float | None:
    """Share of MOVED points landing outside the dataset's observed raw range.

    Binding definition: a point counts when at least one MUTABLE feature leaves
    the FULL dataset's `[min, max]`, measured in raw units. Immutable features
    are excluded because nothing pushed them anywhere. `None` when there is
    nothing to measure (no rows, or no mutable feature).
    """
    if source_raw.shape[0] == 0 or not mutable.any():
        return None
    low = df[feature_cols].min().to_numpy(dtype=float)[mutable]
    high = df[feature_cols].max().to_numpy(dtype=float)[mutable]
    moved = source_raw[:, mutable] + applied_delta_raw[mutable]
    outside = (moved < low) | (moved > high)
    return float(outside.any(axis=1).mean())


def compute_pca_movement(v: ValidatedMovement, df: pd.DataFrame) -> dict[str, Any]:
    """The `MovementResponse` for a validated PCA request, computed inline.

    PCA needs no background job: the only real work is the reducer refit, which
    `artifact_for` caches per (analysis_id, node_id). Every 2D quantity that
    leaves here is in VISIBLE coordinates — the click is converted into fitted
    coordinates to be solved, and results are mapped back through the alignment.

    Raises `MovementError` (409) when the refit cannot be reconciled with the
    projection on screen.
    """
    if v.method != "PCA":
        raise MovementError(
            400, f"compute_pca_movement received a {v.method} run, not a PCA one."
        )

    artifact = artifact_for(v, df)
    alignment = require_alignment(artifact, v.node)
    components = _pca_components(artifact)
    mutable = mutable_flags(v.feature_cols)
    normalize = bool(v.effective_config.get("normalize", True))

    source_positions = _child_positions(artifact, v.source_node)
    X_source = artifact.X_parent[source_positions]
    Y_source = artifact.embedding[source_positions]  # fitted coordinates
    source_mean_input = _require_finite(X_source.mean(axis=0), "source centroid")
    source_mean_fitted = _require_finite(Y_source.mean(axis=0), "source centroid")

    target_size: int | None = None
    if v.target_kind == "cluster":
        target_positions = _child_positions(artifact, v.target_node or {})
        displacement = cluster_displacement(
            mutable=mutable,
            source_mean_input=source_mean_input,
            target_mean_input=_require_finite(
                artifact.X_parent[target_positions].mean(axis=0),
                "destination centroid",
            ),
            target_fitted=_require_finite(
                artifact.embedding[target_positions].mean(axis=0),
                "destination centroid",
            ),
        )
        target_size = int(target_positions.size)
    else:
        displacement = free_point_displacement(
            components=components,
            mutable=mutable,
            source_mean_input=source_mean_input,
            source_mean_fitted=source_mean_fitted,
            target_fitted=alignment.to_fitted(np.asarray(v.target_xy, dtype=float))[0],
            normalize=normalize,
        )

    delta = _require_finite(displacement.delta_input, "displacement")
    # PCA reaches the reachable target exactly and interpolates linearly on the
    # way, so there is no strength to search for: full strength IS the
    # recommendation. (The α grid exists for UMAP, whose transform is not linear.)
    recommended_strength = 1.0
    applied = recommended_strength if v.strength is None else float(v.strength)

    d_fitted = components @ delta  # full-strength projected displacement, fitted
    source_centroid_vis = alignment.to_visible(source_mean_fitted)[0]
    moved_centroid_vis = alignment.to_visible(source_mean_fitted + applied * d_fitted)[
        0
    ]
    target_vis = _require_finite(
        alignment.to_visible(displacement.target_fitted)[0], "destination coordinate"
    )
    visible_displacement = (
        alignment.to_visible(source_mean_fitted + d_fitted)[0] - source_centroid_vis
    )

    picks, preview_sampled = _preview_indices(int(source_positions.size))
    moved_vis = alignment.to_visible(Y_source[picks] + applied * d_fitted)
    preview_row_ids = artifact.parent_row_indices[source_positions[picks]]
    preview_points = [
        {"row_id": int(row_id), "x": float(xy[0]), "y": float(xy[1])}
        for row_id, xy in zip(preview_row_ids, moved_vis, strict=True)
        if np.isfinite(xy).all()
    ]

    scale, offset = _raw_scaling(artifact.scaler, len(v.feature_cols))
    recommended_raw = delta * scale
    source_mean_raw = source_mean_input * scale + offset
    target_raw = displacement.target_input * scale + offset
    feature_changes = [
        {
            "feature": str(column),
            "mutable": bool(mutable[j]),
            "source_mean_raw": _finite(source_mean_raw[j]),
            "target_value_raw": _finite(target_raw[j]),
            "recommended_delta_raw": _finite(recommended_raw[j]),
            "applied_delta_raw": _finite(recommended_raw[j] * applied),
            # |Δ| in the reducer's input space — root-standardized units when the
            # run normalized, raw units when it did not.
            "standardized_magnitude": _finite(abs(delta[j])),
        }
        for j, column in enumerate(v.feature_cols)
    ]

    out_of_range = _out_of_range_fraction(
        df,
        v.feature_cols,
        mutable,
        X_source * scale + offset,
        applied * recommended_raw,
    )

    notes = list(displacement.warnings)
    held = [
        c
        for c, is_mutable in zip(v.feature_cols, mutable, strict=True)
        if not is_mutable
    ]
    if held:
        shown = ", ".join(held[:4]) + ("…" if len(held) > 4 else "")
        notes.append(
            f"{len(held)} selected column(s) are row identifiers or labels "
            f"({shown}) and are held fixed, so the requested destination may not "
            "be reached exactly."
        )
    if out_of_range is not None and out_of_range > OUT_OF_RANGE_WARN_FRACTION:
        notes.append(
            f"{out_of_range:.0%} of the moved points fall outside the range this "
            "dataset covers in at least one changed feature — the counterfactual "
            "is extrapolating beyond the observed data."
        )

    projected_before = float(np.linalg.norm(target_vis - source_centroid_vis))
    projected_after = float(np.linalg.norm(target_vis - moved_centroid_vis))
    feature_before = float(
        np.linalg.norm(displacement.target_input - source_mean_input)
    )
    # Cluster targets only: their destination is an OBSERVED centroid. A point
    # target's feature-space destination is defined as μ_source + Δ, so the
    # remaining distance would be (1−α)·‖Δ‖ by construction — exactly 0 at full
    # strength whether or not the click was actually reached — and reporting it
    # next to the genuine projected distances would pass it off as measured.
    feature_after = (
        float(
            np.linalg.norm(
                displacement.target_input - (source_mean_input + applied * delta)
            )
        )
        if v.target_kind == "cluster"
        else None
    )

    return {
        "status": "ok",
        "analysis_id": v.analysis_id,
        "node_id": v.node_id,
        "method": v.method,
        "source_child_index": v.source_child_index,
        "target": {
            "kind": v.target_kind,
            "child_index": v.target_child_index,
            "x": float(target_vis[0]),
            "y": float(target_vis[1]),
            "estimator": displacement.estimator,
        },
        "recommended_strength": recommended_strength,
        "applied_strength": applied,
        "source_size": int(source_positions.size),
        "target_size": target_size,
        "feature_changes": feature_changes,
        "preview_points": preview_points,
        "visible_displacement": [
            float(visible_displacement[0]),
            float(visible_displacement[1]),
        ],
        # UMAP's α grid; PCA interpolates exactly, so there is no trajectory to plot.
        "strength_trajectory": [],
        "metrics": {
            "projected_distance_before": _finite(projected_before),
            "projected_distance_after": _finite(projected_after),
            "projected_distance_reduction": _finite(projected_before - projected_after),
            "feature_centroid_distance_before": _finite(feature_before),
            "feature_centroid_distance_after": None
            if feature_after is None
            else _finite(feature_after),
            "out_of_range_fraction": None
            if out_of_range is None
            else _finite(out_of_range),
            # A UMAP concept: how much observed data supports an inverted click.
            "support_ratio": None,
            "alignment_rmse": _finite(alignment.rmse),
            "alignment_max_residual": _finite(alignment.max_residual),
            "preview_sampled": bool(preview_sampled),
        },
        "warnings": notes,
    }


# ── UMAP movement ───────────────────────────────────────────────────────────

# The α grid the recommended strength is chosen on. UMAP's out-of-sample
# transform is neighbour-driven, so the projected distance is not linear in α
# and cannot be solved for the way PCA's is — it has to be sampled.
STRENGTH_GRID = np.linspace(0.0, 1.0, 11)

# The α grid runs one `reducer.transform` per strength, so it is evaluated on a
# deterministic subsample of the source cluster while the final metrics use the
# (larger) preview set. The recommended α is therefore chosen on slightly less
# data than the numbers next to it are measured on, and the two can disagree —
# deterministically, and only ever by which grid point wins, never by an
# unrepeatable draw.
MAX_STRENGTH_SEARCH_POINTS = 1000

# Neighbours consulted when `inverse_transform` fails and the destination has to
# be estimated from observed points instead.
NEIGHBOR_FALLBACK_K = 10

# Above this support ratio the clicked position sits far outside the observed
# cloud (its distance to the nearest embedded point is several typical
# neighbour gaps), so its inverse is an extrapolation and the panel says so.
SUPPORT_RATIO_WARN = 3.0


def _strength_search_indices(n: int) -> np.ndarray:
    """Local indices of the points the α grid is evaluated on.

    Same evenly spaced stride as `_preview_indices`, a tighter cap: the grid
    costs eleven `reducer.transform` calls, the preview one. Deterministic — no
    RNG, so the recommended strength for a given request never moves.
    """
    if n <= MAX_STRENGTH_SEARCH_POINTS:
        return np.arange(n)
    picks = np.linspace(0, n - 1, MAX_STRENGTH_SEARCH_POINTS).round().astype(int)
    return np.unique(picks)


def _umap_transform(reducer: Any, X: np.ndarray) -> np.ndarray:
    """Forward moved points through the FROZEN reducer.

    Never a refit: a new fit may rotate, reflect or reshape the projection, and
    the preview would then be drawn against an embedding the analyst is not
    looking at.

    Under `NUMBA_PARALLEL_LOCK`: `transform` runs Numba parallel regions, and
    the workqueue threading layer aborts the process if two threads enter one
    concurrently — a slider re-request racing a job-thread refit was enough.
    """
    try:
        with NUMBA_PARALLEL_LOCK:
            moved = np.asarray(
                reducer.transform(np.asarray(X, dtype=float)), dtype=float
            ).reshape(-1, 2)
    except Exception as exc:
        raise MovementError(
            409,
            "The fitted projection for this node could not place the moved points "
            f"({type(exc).__name__}). Rebuild the analysis and try the movement "
            "again.",
        ) from exc
    return moved


def _umap_inverse(
    reducer: Any, target_fitted: np.ndarray, n_features: int
) -> np.ndarray | None:
    """`inverse_transform` of one FITTED coordinate, or `None` when unusable.

    UMAP's inverse is approximate by construction; `None` here means it was not
    merely inaccurate but unusable — it raised, or returned non-finite values —
    which is what the neighbour fallback exists for.
    """
    try:
        with NUMBA_PARALLEL_LOCK:  # same reason as `_umap_transform`
            x = np.asarray(
                reducer.inverse_transform(np.asarray([target_fitted], dtype=float)),
                dtype=float,
            ).reshape(-1)
    except Exception:  # noqa: BLE001 - any inverse failure means "use the fallback"
        return None
    if x.size != n_features or not np.all(np.isfinite(x)):
        return None
    return x


def _neighbor_target(
    embedding_fitted: np.ndarray,
    X_parent: np.ndarray,
    target_fitted: np.ndarray,
    k: int = NEIGHBOR_FALLBACK_K,
) -> np.ndarray:
    """Inverse-distance-weighted mean of the k displayed points nearest the click.

    The fallback destination when `inverse_transform` cannot be used. Distances
    are taken in fitted coordinates, which rank the same as visible ones (the
    alignment is a similarity transform), and the weights come back in the
    reducer's own input space through `X_parent`.
    """
    Y = np.asarray(embedding_fitted, dtype=float).reshape(-1, 2)
    usable = np.flatnonzero(np.isfinite(Y).all(axis=1))
    if usable.size == 0:
        raise MovementError(
            409,
            "This projection has no usable points to estimate a destination from. "
            "Rebuild the analysis and try the movement again.",
        )
    click = np.asarray(target_fitted, dtype=float).reshape(2)
    distances = np.linalg.norm(Y[usable] - click, axis=1)
    # Stable sort, not argpartition: ties must resolve the same way every call.
    take = usable[np.argsort(distances, kind="stable")[: min(k, usable.size)]]
    d = np.linalg.norm(Y[take] - click, axis=1)
    if float(d.min()) <= 0.0:  # the click landed on a point; that point IS the answer
        return np.asarray(X_parent[take[d <= 0.0]].mean(axis=0), dtype=float)
    weights = 1.0 / d
    return np.asarray(
        (X_parent[take] * weights[:, None]).sum(axis=0) / weights.sum(), dtype=float
    )


def _support_ratio(
    embedding_fitted: np.ndarray, target_fitted: np.ndarray
) -> float | None:
    """How far a clicked position sits outside the observed cloud.

    `dist(click, nearest embedded point) / median nearest-neighbour distance in
    the parent embedding` — a ratio, so it reads the same in fitted and visible
    coordinates. ~1 means the click sits where points normally sit; large means
    the inverse is extrapolating. `None` when the parent is too small or
    degenerate to define a typical gap.
    """
    Y = np.asarray(embedding_fitted, dtype=float).reshape(-1, 2)
    Y = Y[np.isfinite(Y).all(axis=1)]
    if Y.shape[0] < 2:
        return None
    click = np.asarray(target_fitted, dtype=float).reshape(2)
    nearest = float(np.min(np.linalg.norm(Y - click, axis=1)))
    neighbours = NearestNeighbors(n_neighbors=2).fit(Y)
    gaps, _ = neighbours.kneighbors(Y)
    typical = float(np.median(gaps[:, 1]))
    if not math.isfinite(typical) or typical <= 0.0 or not math.isfinite(nearest):
        return None
    return nearest / typical


def umap_point_displacement(
    *,
    reducer: Any,
    embedding_fitted: np.ndarray,
    X_parent: np.ndarray,
    mutable: np.ndarray,
    source_mean_input: np.ndarray,
    target_fitted: np.ndarray,
) -> Displacement:
    """The shared translation carrying the source centroid toward a clicked point.

    `x_target = reducer.inverse_transform([y*_fitted])[0]`, then
    `Δ = x_target − μ_source` with immutable entries zeroed. The click arrives in
    visible coordinates and is converted through the alignment BEFORE inversion —
    the reducer only ever sees its own coordinates.

    Unlike PCA there is no exactness claim to make: UMAP's inverse is
    approximate, so applying Δ is not guaranteed to land the moved centroid on
    the click. That is stated rather than implied, and the achieved position is
    measured by forwarding the moved points through the frozen reducer.
    """
    notes: list[str] = []
    x_target = _umap_inverse(reducer, target_fitted, int(source_mean_input.size))
    if x_target is not None:
        estimator = "umap_inverse"
        notes.append(
            "UMAP's inverse projection is approximate, so this recommendation is "
            "not guaranteed to land the cluster's centre exactly on the position "
            "you clicked. The preview shows where the frozen projection actually "
            "puts the moved points."
        )
    else:
        x_target = _neighbor_target(embedding_fitted, X_parent, target_fitted)
        estimator = "neighbor_interpolation"
        notes.append(
            "UMAP could not invert the position you clicked, so the destination "
            "was estimated from the nearest observed points. The recommended "
            "feature values are an estimate, not an inverse."
        )
    delta = np.asarray(x_target, dtype=float) - source_mean_input
    delta[~mutable] = 0.0
    return Displacement(
        delta_input=delta,
        target_input=source_mean_input + delta,
        target_fitted=np.asarray(target_fitted, dtype=float),
        estimator=estimator,
        warnings=notes,
    )


def compute_umap_movement(v: ValidatedMovement, df: pd.DataFrame) -> dict[str, Any]:
    """The `MovementResponse` for a validated UMAP request.

    Three things differ from the PCA path. The destination for a free click comes
    from `inverse_transform` (or the neighbour fallback), not from a pinv solve.
    The recommended strength is searched on an α grid instead of being 1 by
    construction, and the whole grid is returned: a neighbour-driven transform
    can plateau and jump, so the argmin alone would hide how trustworthy the
    intermediate ghosts are. And every projected position is obtained by
    forwarding points through the FROZEN reducer — a refit could rotate, reflect
    or reshape the projection the analyst is looking at.

    Note that `reducer.transform` of the source rows does not reproduce the
    coordinates the fit gave them (out-of-sample placement is its own
    computation), so the α = 0 entry of the trajectory can differ slightly from
    `projected_distance_before`, which is measured from the centroid actually on
    screen. Both are honest; they answer different questions.
    """
    if v.method != "UMAP":
        raise MovementError(
            400, f"compute_umap_movement received a {v.method} run, not a UMAP one."
        )

    artifact = artifact_for(v, df)
    alignment = require_alignment(artifact, v.node)
    mutable = mutable_flags(v.feature_cols)

    source_positions = _child_positions(artifact, v.source_node)
    X_source = artifact.X_parent[source_positions]
    Y_source = artifact.embedding[source_positions]  # fitted coordinates
    source_mean_input = _require_finite(X_source.mean(axis=0), "source centroid")
    source_mean_fitted = _require_finite(Y_source.mean(axis=0), "source centroid")

    support_ratio: float | None = None
    target_size: int | None = None
    if v.target_kind == "cluster":
        # Deliberately NOT an inverse of the displayed centroid: the destination
        # cluster's own feature-space mean is observed, exact and unique, where
        # an inverse of its projected centroid would be an estimate of something
        # already known.
        target_positions = _child_positions(artifact, v.target_node or {})
        displacement = cluster_displacement(
            mutable=mutable,
            source_mean_input=source_mean_input,
            target_mean_input=_require_finite(
                artifact.X_parent[target_positions].mean(axis=0),
                "destination centroid",
            ),
            target_fitted=_require_finite(
                artifact.embedding[target_positions].mean(axis=0),
                "destination centroid",
            ),
        )
        target_size = int(target_positions.size)
    else:
        target_fitted = alignment.to_fitted(np.asarray(v.target_xy, dtype=float))[0]
        displacement = umap_point_displacement(
            reducer=artifact.reducer,
            embedding_fitted=artifact.embedding,
            X_parent=artifact.X_parent,
            mutable=mutable,
            source_mean_input=source_mean_input,
            target_fitted=target_fitted,
        )
        support_ratio = _support_ratio(artifact.embedding, target_fitted)

    delta = _require_finite(displacement.delta_input, "displacement")
    target_vis = _require_finite(
        alignment.to_visible(displacement.target_fitted)[0], "destination coordinate"
    )
    source_centroid_vis = alignment.to_visible(source_mean_fitted)[0]

    # The α grid: translate a deterministic sample, forward it through the frozen
    # reducer, and measure the moved centroid's distance to the destination — all
    # in visible coordinates, like every other 2D quantity on the wire.
    search = _strength_search_indices(int(source_positions.size))
    X_search = X_source[search]
    grid_centroids = [
        alignment.to_visible(
            _require_finite(
                _umap_transform(artifact.reducer, X_search + alpha * delta).mean(
                    axis=0
                ),
                "moved centroid",
            )
        )[0]
        for alpha in STRENGTH_GRID
    ]
    grid_distances = [
        float(np.linalg.norm(target_vis - centroid)) for centroid in grid_centroids
    ]
    # argmin takes the FIRST minimum, so a tie resolves to the smaller movement.
    best = int(np.argmin(grid_distances))
    # best == 0 means no positive strength beat standing still — including ties,
    # which are not improvements. Then there is no recommendation to give.
    recommended_strength = None if best == 0 else float(STRENGTH_GRID[best])
    if v.strength is not None:
        applied = float(v.strength)
    else:
        applied = 0.0 if recommended_strength is None else recommended_strength

    picks, preview_sampled = _preview_indices(int(source_positions.size))
    moved_fitted = _umap_transform(artifact.reducer, X_source[picks] + applied * delta)
    moved_vis = alignment.to_visible(moved_fitted)
    moved_centroid_vis = alignment.to_visible(
        _require_finite(moved_fitted.mean(axis=0), "moved centroid")
    )[0]
    preview_row_ids = artifact.parent_row_indices[source_positions[picks]]
    preview_points = [
        {"row_id": int(row_id), "x": float(xy[0]), "y": float(xy[1])}
        for row_id, xy in zip(preview_row_ids, moved_vis, strict=True)
        if np.isfinite(xy).all()
    ]
    # Indicative only for UMAP: the transform is not linear, so the client may
    # not interpolate ghosts along this vector the way it does for PCA.
    visible_displacement = _require_finite(
        grid_centroids[-1] - grid_centroids[0], "projected displacement"
    )

    scale, offset = _raw_scaling(artifact.scaler, len(v.feature_cols))
    recommended_raw = delta * scale
    source_mean_raw = source_mean_input * scale + offset
    target_raw = displacement.target_input * scale + offset
    feature_changes = [
        {
            "feature": str(column),
            "mutable": bool(mutable[j]),
            "source_mean_raw": _finite(source_mean_raw[j]),
            "target_value_raw": _finite(target_raw[j]),
            "recommended_delta_raw": _finite(recommended_raw[j]),
            "applied_delta_raw": _finite(recommended_raw[j] * applied),
            "standardized_magnitude": _finite(abs(delta[j])),
        }
        for j, column in enumerate(v.feature_cols)
    ]

    out_of_range = _out_of_range_fraction(
        df,
        v.feature_cols,
        mutable,
        X_source * scale + offset,
        applied * recommended_raw,
    )

    notes = list(displacement.warnings)
    if recommended_strength is None:
        notes.append(
            "This UMAP movement does not reliably approach the selected destination."
        )
    if support_ratio is not None and support_ratio > SUPPORT_RATIO_WARN:
        notes.append(
            "The selected position is far from observed data. Its inverse is uncertain."
        )
    held = [
        c
        for c, is_mutable in zip(v.feature_cols, mutable, strict=True)
        if not is_mutable
    ]
    if held:
        shown = ", ".join(held[:4]) + ("…" if len(held) > 4 else "")
        notes.append(
            f"{len(held)} selected column(s) are row identifiers or labels "
            f"({shown}) and are held fixed, so the requested destination may not "
            "be reached exactly."
        )
    if out_of_range is not None and out_of_range > OUT_OF_RANGE_WARN_FRACTION:
        notes.append(
            f"{out_of_range:.0%} of the moved points fall outside the range this "
            "dataset covers in at least one changed feature — the counterfactual "
            "is extrapolating beyond the observed data."
        )

    projected_before = float(np.linalg.norm(target_vis - source_centroid_vis))
    projected_after = float(np.linalg.norm(target_vis - moved_centroid_vis))
    feature_before = float(
        np.linalg.norm(displacement.target_input - source_mean_input)
    )
    # Cluster targets only: their destination is an OBSERVED centroid. A point
    # target's feature-space destination is defined as μ_source + Δ, so the
    # remaining distance would be (1−α)·‖Δ‖ by construction — exactly 0 at full
    # strength whether or not the click was actually reached — and reporting it
    # next to the genuine projected distances would pass it off as measured.
    feature_after = (
        float(
            np.linalg.norm(
                displacement.target_input - (source_mean_input + applied * delta)
            )
        )
        if v.target_kind == "cluster"
        else None
    )

    return {
        "status": "ok",
        "analysis_id": v.analysis_id,
        "node_id": v.node_id,
        "method": v.method,
        "source_child_index": v.source_child_index,
        "target": {
            "kind": v.target_kind,
            "child_index": v.target_child_index,
            "x": float(target_vis[0]),
            "y": float(target_vis[1]),
            "estimator": displacement.estimator,
        },
        "recommended_strength": recommended_strength,
        "applied_strength": applied,
        "source_size": int(source_positions.size),
        "target_size": target_size,
        "feature_changes": feature_changes,
        "preview_points": preview_points,
        "visible_displacement": [
            float(visible_displacement[0]),
            float(visible_displacement[1]),
        ],
        "strength_trajectory": [
            {"strength": float(alpha), "projected_distance": _finite(distance)}
            for alpha, distance in zip(STRENGTH_GRID, grid_distances, strict=True)
        ],
        "metrics": {
            "projected_distance_before": _finite(projected_before),
            "projected_distance_after": _finite(projected_after),
            "projected_distance_reduction": _finite(projected_before - projected_after),
            "feature_centroid_distance_before": _finite(feature_before),
            "feature_centroid_distance_after": None
            if feature_after is None
            else _finite(feature_after),
            "out_of_range_fraction": None
            if out_of_range is None
            else _finite(out_of_range),
            "support_ratio": None if support_ratio is None else _finite(support_ratio),
            "alignment_rmse": _finite(alignment.rmse),
            "alignment_max_residual": _finite(alignment.max_residual),
            "preview_sampled": bool(preview_sampled),
        },
        "warnings": notes,
    }


# ── Dispatch ────────────────────────────────────────────────────────────────


def artifact_is_cached(analysis_id: str, node_id: str) -> bool:
    """Whether this node's reducer is already in the LRU (no refit needed)."""
    with _artifact_lock:
        return (analysis_id, node_id) in _artifact_cache


def needs_background(v: ValidatedMovement) -> bool:
    """Whether answering this request needs a background job.

    PCA never does: its whole computation is one pinv solve against an artifact
    the LRU almost always holds, and routing it through a poll would turn an
    instant slider into a round trip. A UMAP request does when the reducer still
    has to be refitted — the analysis discards fitted reducers, so the first
    movement on a node pays for a full UMAP fit, which can outlast the response
    timeout a proxy enforces (see backend/jobs.py).
    """
    return v.method != "PCA" and not artifact_is_cached(v.analysis_id, v.node_id)


def compute_validated_movement(
    v: ValidatedMovement, df: pd.DataFrame
) -> dict[str, Any]:
    """Answer an already-validated request — inline, or on a job thread.

    Split out of `compute_movement` so the route can validate synchronously (a
    bad request must be an HTTP 400/409, not a job that fails a second later)
    and only then decide whether the computation itself needs a job.
    """
    if v.method == "PCA":
        return compute_pca_movement(v, df)
    return compute_umap_movement(v, df)


def compute_movement(
    req: MovementRequestData, payload: Mapping[str, Any] | None, df: pd.DataFrame
) -> dict[str, Any]:
    """Entry point for `POST /api/movement`: validate the request, then answer it.

    `payload` is the cached `/api/analysis` payload for the run the client holds
    (`None` when the cache no longer has it); `df` is the loaded dataset. Raises
    `MovementError` — `status_code` 400 or 409 — for everything the server will
    not answer; the route turns that into an `HTTPException`.

    This computes the whole answer here and now. The route splits the same two
    steps apart — `validate_request` then `compute_validated_movement` — so that
    a UMAP request whose reducer still has to be refitted can run on a job
    thread while its validation errors stay synchronous HTTP responses. t-SNE
    and MDS never reach either path: `validate_request` refuses them by name.
    """
    return compute_validated_movement(validate_request(req, payload), df)

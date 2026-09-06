# Spec: Cluster-to-Point and Cluster-to-Cluster Movement

Status: source plan (Hannes) + binding amendments from the code-verified review of 28.08.
Blocks marked **AMENDMENT (binding)** override the surrounding original text where they
conflict.

## Goal

Allow an analyst to select a source cluster and request a shared feature-space
adjustment that moves it toward: (1) an arbitrary point clicked in the current 2D
projection; or (2) another cluster in the same projection. If the click resolves to
another cluster, the system automatically switches from cluster-to-point to
cluster-to-cluster movement. The first implementation supports projections built with
PCA and UMAP. The result is a counterfactual preview and feature-change recommendation.
It does not immediately modify the source dataset or rebuild the hierarchy.

## Core interpretation

The operation is a **shared translation** of the source cluster:

```text
X_source_moved = X_source + strength × Δ
```

`Δ` is one feature-space displacement vector; every source point receives the same
displacement; `strength` lies in [0, 1] (0 = unchanged, 1 = complete recommendation).
One shared displacement preserves all pairwise distances inside the source cluster in
the original feature space — the cluster moves without morphing its internal shape.

## Target semantics

**Cluster-to-point**: triggered when the user clicks an empty location in the
projection. The clicked 2D coordinate is converted into an estimated feature-space
destination using the fitted PCA or UMAP projection.

**Cluster-to-cluster**: triggered when the user clicks a point belonging to another
cluster, another cluster's centroid label, another cluster's legend entry, or
sufficiently close to a point belonging to another cluster. The target becomes the
entire destination cluster; the target representative is the destination cluster's
centroid in the feature space used by the analysis.

**Invalid targets** (do not start a movement): clicking the selected source cluster or
its centroid/legend entry; clicking outside the plotting region; selecting a target from
another hierarchical layer; using a projection method other than PCA or UMAP.

> **AMENDMENT (binding):** four methods exist in `src/analysis/dim_reducer.py` — the
> unsupported-method gate and the disabled-state UI copy must cover t-SNE AND MDS by
> name.

## Scope decisions

The first version will: move one child cluster within its parent node; target a free
projection coordinate or a sibling cluster; apply one shared feature delta to every
source point; operate against the projection currently visible on screen; show a
frozen-projection preview; report changes in raw feature units and standardized
magnitudes; expose movement strength 0–100%; preserve the original dataset and analysis
tree.

It will not: morph the source distribution to match the target distribution;
independently optimize every source point; guarantee that HDBSCAN merges the clusters
after rebuilding; support t-SNE or MDS; write counterfactual values back into the
dataset; interpret the movement causally; treat the preview as a new completed analysis
run.

## User interaction

1. **Select the source cluster** (existing behaviour unchanged). Then show a `Move C2…`
   action in the selected cluster's layer-side panel or the cluster projection toolbar.
2. **Enter movement mode**: keep the source selection; dim unrelated navigation
   controls; crosshair cursor; instruction "Click an empty position, or click another
   cluster to use it as the destination." Normal drill-down clicks are suspended for
   this projection while movement mode is active.
3. **Resolve the target**, precedence: centroid label / legend entry → cluster; point of
   another cluster → cluster; nearest other-cluster point within a 10 px screen-space
   snap radius (evaluated after zoom/pan transforms) → cluster; otherwise → free 2D
   point.
4. **Calculate the recommendation** with a loading state ("Calculating PCA/UMAP
   movement…"); UMAP may require fitting or restoring a reducer and can take noticeably
   longer.
5. **Preview**: original source cluster; moved cluster as translucent ghost points;
   original and proposed centroids; arrow between them; the point target or highlighted
   target cluster; requested and achieved destination distance.
6. **Inspect feature changes** in a movement panel (target, strength, projected distance
   reduction, original-space centroid distance reduction, changed-feature count; per
   feature: source mean, target mean, recommended change, applied change, standardized
   magnitude). Sort by descending absolute standardized change.
7. **Adjust strength** with a 0–100% slider updating applied raw deltas, ghost
   positions, projected-distance metrics, and plausibility warnings. Debounce UMAP
   preview requests; PCA previews update client-side after the backend returns the
   complete displacement.
8. **Cancel** via a visible button, Escape, navigating to another node, or a new
   analysis build.

> **DECISION (Hannes, 06.09) — navigation.** "Navigating to another node" means leaving
> the movement's parent projection. Drilling DEEPER keeps the parent projection on screen
> in the layer stack, so the preview persists while its node is on the path; it is
> cancelled when the node leaves the path (a sibling, a step back up, another dataset, a
> new build). Implemented as `nodeIsOnPath` in `frontend/src/movement.ts`.

## Mathematical definition

With `Xₛ` the source points in the reducer's feature space, `μₛ` their mean, `Yₛ` the
displayed source points, `ȳₛ` their mean, and α the strength: `Xₛ' = Xₛ + αΔ`.

### Immutable features

The displacement applies only to mutable features. Default mutable = all selected
feature columns except `row_id` and `target_*`. If a `target_*` column was manually
included as an analysis feature, keep it immutable by default and report that the
requested destination may not be reached exactly. A later iteration may add a feature
checklist for the mutable set.

## PCA implementation

### Free point target

With `y*` the clicked 2D coordinate, `d_y = y* − ȳₛ`, and `P` the PCA components for the
mutable features, find the minimum-norm mutable displacement with
`delta_mutable = np.linalg.pinv(P_mutable) @ projected_displacement`, writing zeros for
immutable features. If the mutable component matrix has rank < 2, calculate the closest
reachable point and report that the target cannot be reached exactly. At full strength
`mean(PCA.transform(Xₛ + Δ))` must match the reachable target within numerical
tolerance. Do not call `inverse_transform` per point — one shared displacement preserves
each point's residual outside the first two components.

> **AMENDMENT (binding) — under-determination must be stated to the user.** The pinv
> solution is one canonical answer out of a (d−2)-dimensional solution space, and its
> minimality is measured in the reducer's INPUT space: root-standardized units when
> `config["normalize"]` is on, raw units when off. The movement panel must carry one
> sentence ("smallest standardized change that reaches this point" / "smallest raw
> change" respectively) and the docs must state the norm-space caveat. Cluster targets
> (Δ = μₜ − μₛ) are unique and need no such label.

### Cluster target

`Δ = μₜ − μₛ` with `μₜ` the destination-cluster mean in PCA input space; immutable
entries zero. Because PCA is linear, the projected source centroid reaches the
corresponding reachable target centroid exactly when all relevant features are mutable.

### PCA strength

PCA is linear, so preview coordinates interpolate exactly. Original formula:
`Yₛ' = Yₛ + α(PΔ)`; frontend may update the ghost preview locally while the slider
moves.

> **AMENDMENT (binding):** that formula holds in FITTED-reducer coordinates; the visible
> embedding differs by the alignment similarity transform (see Projection artifact
> management). The backend returns the full-strength projected displacement ALREADY
> mapped into visible coordinates (`d_vis`), and the client-side update is
> `ghost = Yₛ + α·d_vis`. More generally: every 2D quantity in the response (preview
> points, target coordinate, displacements, distances) is in visible coordinates; the
> frontend never sees fitted coordinates.

## UMAP implementation

### Free point target

`x_target = reducer.inverse_transform([y*])[0]`, then `Δ = x_target − μₛ`, immutable
entries zero. UMAP inversion is approximate — applying Δ does not guarantee the moved
centroid lands on the click. Validate by forwarding moved points through the frozen
reducer: `Y_source_moved = reducer.transform(X_source + α·Δ)`.

### UMAP inverse fallback

If `inverse_transform` fails or returns non-finite values: take the k nearest displayed
parent points to the click, retrieve them in the reducer's feature space, use their
inverse-distance-weighted mean as `x_target`. Return
`target_estimator = "umap_inverse" | "neighbor_interpolation"` so the UI can explain the
estimate.

### Cluster target

Do NOT inverse-transform the displayed centroid; use the destination cluster's actual
feature-space centroid: `Δ = μₜ − μₛ`. Use the frozen reducer only for the ghost
preview.

### Recommended UMAP strength

Evaluate `strengths = np.linspace(0, 1, 11)`: translate a deterministic source sample,
project through the frozen reducer, take the moved projected centroid's distance to the
target. Choose the smallest distance; ties → smaller movement. Return
`recommended_strength`. If no positive strength improves on zero, return no
recommendation and show "This UMAP movement does not reliably approach the selected
destination."

> **AMENDMENT (binding) — return the trajectory, not only the argmin.** UMAP's
> out-of-sample transform is neighbor-driven: distance vs α can plateau and jump, and
> intermediate-α ghosts are the least trustworthy. The grid is computed anyway — return
> `strength_trajectory: [{strength, projected_distance}]`, plot it as a mini-chart in
> the panel, and show a caveat line when the trajectory is non-monotone. Document that
> the recommended α is selected on the ≤1000-point deterministic sample while final
> metrics use the full preview set (small mismatches possible, deterministic).

### UMAP support warning

For a free target in an empty region compute
`support_ratio = dist(target, nearest embedded points) / typical local neighbor distance
in the parent embedding`; return the ratio and warnings ("The selected position is far
from observed data. Its inverse is uncertain."). Do not silently reject unless inversion
fails completely.

## Frozen projection requirement

The preview must use the reducer fitted for the visible parent node. Never recompute the
embedding after applying the movement (a new fit may rotate/reflect/reshape). Label the
result "Counterfactual preview in the current frozen projection". A later
"Apply and rebuild" is a separate operation.

## Analysis identity

Add a stable `analysis_id` to the analysis metadata (backend/app.py +
frontend/src/types.ts), derived from the existing cache key
(dataset + feature columns + built configuration + cache schema version). The movement
endpoint must use the configuration stored in the completed analysis, not the current
unsaved values in the configuration rail. Add a cache schema version so older serialized
runs are not mistaken for compatible results.

> **AMENDMENT (binding) — effective config, the critical one.** `compute_analysis_tree`
> mutates its config IN PLACE before any per-node reducer is fit: it clamps
> `hclust_umap_n_components` and `umap_n_neighbors` against the root size. Every visible
> per-node embedding was fit with the MUTATED values; a movement refit from the raw
> request config rebuilds a different reducer whenever the clamp fired, and the
> alignment gate then rejects movements for an invisible reason. Required: at build
> time, serialize the post-build (mutated) config into the payload meta as
> `effective_config`; movement refits read ONLY `effective_config`. Cached payloads
> without it → HTTP 409 (rebuild required). This is what the schema-version bump is for.

## Projection artifact management

The analysis serializes coordinates and discards fitted reducers (verified: in local AND
hosted mode — the refit path is the only path, not an optimization). Implement a bounded
projection-artifact cache in `backend/movement.py`:

```python
@dataclass
class ProjectionArtifact:
    analysis_id: str
    node_id: str
    method: str
    parent_row_indices: np.ndarray
    X_parent: np.ndarray
    embedding: np.ndarray
    reducer: object
    scaler: StandardScaler | None
```

keyed by analysis_id + node_id, bounded LRU.

**Artifact creation** on a movement request: resolve the analysis payload; resolve the
parent node from `node_id`; load the dataset; recreate the root scaler exactly as the
analysis did; select the parent node's rows; fit PCA or UMAP using the built
configuration; cache.

> **AMENDMENT (binding):** build reducers exclusively through the existing
> `src/analysis/dim_reducer.fit_dimensionality_reducer` /
> `analysis_routine._embed_original` path — never construct PCA/UMAP directly in
> movement code. This inherits parity by construction, including the `init="pca"` UMAP
> determinism fix that exists only there, and the per-node-on-root-scaled-rows fitting
> semantics (verified: `_embed_original(ctx.X_orig)` with `X_orig` sliced per node;
> HDBSCAN's −1 label is skipped when building children, so child indices exclude noise).

**Alignment with the stored projection**: align the refitted embedding to the serialized
visible embedding with a similarity transform (centering, optional uniform scale,
rotation/reflection, translation); store both directions; convert clicked targets into
fitted coordinates before inversion and preview coordinates back into visible
coordinates. Compute normalized alignment RMSE; above tolerance, reject the movement and
ask the user to rebuild rather than showing a misaligned preview.

> **AMENDMENT (binding) — alignment canary.** Add a test: a same-process refit of a
> node's embedding must align to the serialized one at ~identity (normalized
> RMSE ≈ 0; tolerance 1e-6 for PCA, 1e-3 for UMAP). This is the regression canary for
> the effective-config amendment. A large RMSE in production is a red flag to surface
> (409 + message), never a tolerance to widen.

## Backend API

Create `backend/movement.py` and `backend/movement_jobs.py`; add routes in
`backend/app.py`.

**Request** (point target shown; cluster target uses
`{"kind": "cluster", "child_index": 2}`; `strength: null` requests the recommended
strength):

```json
{
  "analysis_id": "8a42e8d4…", "dataset": "Iris (Low)",
  "feature_cols": ["sepal length (cm)", "sepal width (cm)"],
  "config": {"method": "UMAP"}, "node_id": "root/1",
  "source_child_index": 0,
  "target": {"kind": "point", "x": 2.41, "y": -1.82},
  "strength": null
}
```

**Validation**: analysis id matches the run signature; node exists, is internal, has
children; source child exists; destination child exists for cluster targets; source ≠
destination; finite coordinates; method is PCA or UMAP; strength null or in [0, 1];
source/target rows belong to the parent; participating feature columns numeric;
serialized parent embedding available. Invalid → HTTP 400; analysis payload or
compatible reducer state gone → HTTP 409 (rebuild).

> **AMENDMENT (binding):** add — parent has ≥1 child for point targets and ≥2 for
> cluster targets; `embedding_original` non-null for the parent.

**Response** (abridged; full field set as in the original plan):
`status, analysis_id, node_id, method, source_child_index, target {kind, child_index?,
x, y, estimator}, recommended_strength, applied_strength, source_size, target_size,
feature_changes[{feature, mutable, source_mean_raw, target_value_raw,
recommended_delta_raw, applied_delta_raw, standardized_magnitude}],
preview_points[{row_id, x, y}], metrics{projected_distance_before/after/reduction,
feature_centroid_distance_before/after, out_of_range_fraction, support_ratio,
alignment_rmse, preview_sampled}, warnings[]` — plus (amended) `strength_trajectory`.
All 2D fields in visible coordinates (see PCA strength amendment).

> **AMENDMENT (binding) — `out_of_range_fraction` was undefined.** Definition: the
> fraction of moved source points having at least one MUTABLE feature outside the full
> dataset's observed raw [min, max]. Raw units; whole dataset as reference. Warn above
> 0.05.

**Background jobs**: first UMAP request may refit; mirror the analysis-job pattern
(`POST /api/movement`, `GET /api/movement/jobs/{job_id}` via `backend/jobs.py`). Cached
artifacts may return inline; a cache miss runs as a background job. Do not persist
movement results; cache only reusable projection artifacts in memory.

## Preview sampling

`MAX_MOVEMENT_PREVIEW_POINTS = 5000`, `MAX_STRENGTH_SEARCH_POINTS = 1000`. Above the
limits: deterministic sample (include boundary points if practical), return
`preview_sampled: true`, keep feature centroids computed from the complete cluster,
explain sampling in the UI. PCA previews all points unless response size forces
sampling.

## Frontend types and API client

Extend `frontend/src/types.ts` with `MovementTarget`
(`{kind:"point"; x; y} | {kind:"cluster"; child_index}`), `MovementRequest`, and
response types (status, feature changes, preview points, metrics incl.
`strength_trajectory`, warnings, job polling). Implement `runMovement()` in
`frontend/src/api.ts` following the existing `runAnalysis` polling behaviour, including
`POLL_RETRIES` retry of temporary polling failures.

## Scatter-chart changes

`frontend/src/charts/ClusterScatter.tsx` + `frontend/src/charts/props.ts`: add
`movementMode?`, `movementSource?`, `movementTarget?`, `movementPreview?`,
`onMovementTarget?`.

**Point events**: preserve row id, child index, screen position, and projection position
in the interaction callback. Normal mode: clustered point click → existing drill-down;
noise click → no movement action. Movement mode: other-cluster point → cluster target;
source-cluster point → ignore with a hint; noise point → free point at its projected
coordinate; background → free point.

**Noise interaction**: noise is rendered as one non-interactive path; in movement mode
add transparent hit targets so noise points can serve as point destinations without
changing the visible × rendering.

**Coordinate inversion**: for a background click, remove the current D3 zoom transform,
invert the equal-aspect plot transform, and return `embedding_original` units. Suppress
background target creation after a drag or zoom gesture.

## Movement state

Original: store movement state in `Navigation` in `frontend/src/App.tsx`.

> **AMENDMENT (binding):** there is no `Navigation` structure — `App.tsx` holds flat
> `useState` (`treePath`, `analysis`, …). Implement the state machine exactly as typed
> in the original plan (idle / choosing-target / loading / ready / error) but as its own
> hook `useMovement` in a new `frontend/src/movement.ts`, cleared by effects watching
> `treePath`, `analysis`, and the dataset key.

Clear on: dataset change, new analysis, tree change, navigation away from the parent,
source change, Escape/Cancel. Ignore stale responses by checking analysis id, node id,
source child, target, and request generation before committing to state.

## Movement panel

New `frontend/src/components/ClusterMovementPanel.tsx`: start/cancel movement mode;
source and resolved destination; loading and failure states; strength slider; feature
changes; projection and feature-space distance metrics; support, range, alignment, and
sampling warnings; the frozen-projection note; the min-norm sentence (amended); the
strength-trajectory mini-chart (amended); JSON/CSV export of the recommendation
(feature, source mean, target value, recommended raw delta, applied raw delta,
standardized magnitude). No `Apply` button in the first version.

## Styling

`frontend/src/styles.css`: movement toolbar, crosshair, source emphasis, destination
marker, arrow, translucent ghosts, delta table/bars, warning styles, loading, sampled
badge. Categorical colors stay reserved for clusters; ghosts use outline/dashed/neutral
accent.

## Backend tests

`backend/tests/test_movement.py`, standalone + pytest.

**PCA**: free target → finite displacement; moved centroid reaches reachable target at
full strength; cluster movement aligns mutable centroid with target centroid; identical
displacement for every point; pairwise distances unchanged; α=0 identity; α=0.5 half
displacement; immutable columns zero; rank-deficient components → closest reachable +
warning; raw deltas reverse root standardization.

> **AMENDMENT (binding):** the pairwise-distance case is tautological (translation
> preserves distances by construction) — keep as cheap smoke, but the substantive
> assertions are: Δ lies in the row space of the mutable component matrix
> (`Δ = P_mutableᵀ w` — what makes it the pinv solution), immutable entries exactly
> zero, centroid-reaches-target, raw-delta unscaling.

**UMAP** (small deterministic synthetic or Iris): finite inverse movement; cluster
targets use observed centroid, not inverse UMAP; finite previews; recommended grid
strength never worse than zero-strength; deterministic strength search; forced
inverse-failure exercises the neighbor fallback; unsupported targets warn; deterministic
sampling; original dataframe unchanged. No unrealistically exact destination
assertions.

**Alignment**: identity → identity; rotated, reflected, translated+scaled embeddings
align; excessive error rejects. Plus the same-process refit canary (amendment above).

**Validation**: unsupported reducer 400; missing node 400; leaf 400; invalid
source/target child 400; source == target rejected; non-finite coordinates rejected;
strength outside [0,1] rejected; mismatched analysis identity 409; responses JSON-safe.

## Frontend verification

Extract target resolution and coordinate conversion into pure functions. Manual
checklist: drill-down unchanged; movement mode does not change the tree path; cluster /
centroid-label / legend clicks resolve to cluster targets; empty space and noise points
resolve to point targets; source clicks start nothing; zoomed/panned clicks resolve
correctly; drag creates no target; Escape cancels; navigation clears; stale responses
dropped; PCA slider smooth; UMAP slider debounced; sampled previews badged; unsupported
methods (t-SNE AND MDS) show an explanatory disabled state.

## Documentation

README.md, backend/README.md, frontend/README.md, src/README.md: movement as
counterfactual preview; target resolution; shared-translation semantics; PCA exactness
AND the min-norm/norm-space caveat (amended); UMAP approximation, uncertainty, and the
non-monotone trajectory caveat (amended); frozen-projection behaviour; mutable-feature
defaults; preview sampling; `out_of_range_fraction` definition (amended); why movement
does not guarantee a future HDBSCAN merge; supported and unsupported reducers.

## Implementation sequence

Phase 1 shared backend model (ids + effective_config + schema version, node lookup,
validation, root scaling + parent reconstruction, artifact cache, alignment, response
types) → Phase 2 PCA movement (+ tests) → Phase 3 frontend interaction (state machine,
action, target resolution, inversion, snapping, ghosts/marker/arrow, panel,
cancellation + stale guards) → Phase 4 UMAP (inverse targets, fallback, observed
centroids, strength search + trajectory, support warnings, sampling, jobs, tests) →
Phase 5 documentation and verification (docs, full suite, production build, manual PCA +
UMAP on Iris, wide image dataset with sampling, local and hosted cached-analysis modes).

> **AMENDMENT (binding) — vertical-slice checkpoint.** Phases 1–3 form a complete,
> shippable feature. STOP after Phase 3, report, and get Hannes's go before Phase 4.

## Verification commands

```bash
PYTHONPATH=. .venv/bin/python -m backend.tests.test_movement
PYTHONPATH=. .venv/bin/python -m backend.tests.test_serialize
PYTHONPATH=. .venv/bin/python -m backend.tests.test_targets
PYTHONPATH=. .venv/bin/python -m backend.tests.test_predicate   # if present
cd frontend && npm run build
uv run ruff format --check .
```

## Acceptance criteria

As in the original plan (movement mode without tree-path change; point and cluster
targets auto-resolved; PCA reaches reachable targets / aligns mutable centroids; UMAP
inverse only for free points, observed centroids for clusters, achieved distance +
uncertainty reported; identical displacement per point; distances preserved; raw +
standardized reporting; frozen projection; unsupported reducers gated; no mutation of
dataframe or tree; deterministic, badged sampling; bounded artifacts; tests + build
green; documented limitations) — PLUS: `effective_config` serialized and used by all
refits; same-process alignment canary green; all response 2D quantities in visible
coordinates; `strength_trajectory` returned and displayed; `out_of_range_fraction`
computed per the amended definition; min-norm sentence present in the panel.

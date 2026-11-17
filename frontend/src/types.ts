// Data contract shared with the Python backend.
// Mirrors backend/serialize.py output and the FastAPI request/response shapes.

export interface NodeScores {
  n_points: number;
  k: number | null;
  trustworthiness: number | null;
  continuity: number | null;
  mrre_false: number | null;
  mrre_missing: number | null;
  stress: number | null;
  cadi: number | null;
}

export interface Characteristic {
  feature: string;
  z_mean: number | null;
  z_std: number | null;
  raw_mean: number | null;
  is_feature?: boolean; // false = column present in the dataset but not selected as a feature
}

// One node of the analysis tree. Internal nodes have children; leaves do not.
export interface TreeNode {
  id: string; // stable path id, e.g. "root", "root/2/0"
  is_leaf: boolean;
  depth: number;
  n_points: number;
  row_indices: number[]; // indices into the source dataframe
  embedding_original: [number | null, number | null][] | null; // Nx2 projection; null = not projectable
  embedding_original_variance: (number | null)[] | null; // PCA only
  rel_position: [number | null, number | null] | null; // MDS centroid in sibling layout
  rel_characteristics: Characteristic[];
  outlier_scores: (number | null)[] | null; // internal nodes only (GLOSH)
  scores: NodeScores | null;
  children: TreeNode[] | null; // internal nodes only
}

export interface AnalysisMeta {
  dataset: string;
  feature_cols: string[];
  config: Record<string, unknown>; // the knobs the client sent, verbatim
  n_total: number;
  // ── Movement prerequisites (added with the cluster-movement feature) ──────
  // `analysis_id` is a stable digest of (dataset, feature_cols, config, schema
  // version); the movement endpoint refuses a request whose id does not match the
  // run it holds. `effective_config` is the config AFTER `compute_analysis_tree`
  // mutated it in place (it clamps `hclust_umap_n_components` and
  // `umap_n_neighbors` against the root size), which is what every visible
  // per-node embedding was actually fit with — refitting from `config` would
  // build a different reducer whenever the clamp fired. Older cached payloads
  // predate all three and answer 409 on /api/movement.
  analysis_id?: string;
  effective_config?: Record<string, unknown>;
  schema_version?: number;
}

export interface AnalysisResponse {
  meta: AnalysisMeta;
  tree: TreeNode;
  cached: boolean; // true = a stored run was reused instead of recomputing
}

// A build that misses the cache runs as a background job: POST starts it, GET
// polls it. Keeping a long run off a single request is what avoids the 100s
// response timeout a proxy (Cloudflare) enforces.
export type AnalysisJob =
  | { status: "running"; job_id: string }
  | { status: "error"; job_id: string; detail: string }
  | ({ status: "done"; job_id: string } & AnalysisResponse);

// Server mode. The persistent run cache (and its banner/toggle) exist only when hosting.
export interface ModeInfo {
  hosting: boolean;
  cache_dir: string | null;
  cache_only: boolean; // the server refuses to build; only stored runs can be shown
  maintenance: boolean; // show a notice instead of the app, and request nothing
}

// One stored run, as the request that reproduces its cache key (cache-only mode).
export interface CachedRun {
  dataset: string;
  feature_cols: string[];
  config: Partial<AnalysisConfig>; // the knobs the run was requested with, verbatim
  n_total: number | null;
}

// /api/cached-runs: the datasets a cache-only server has stored runs of.
export interface CachedDataset {
  dataset: string;
  n_runs: number;
}

// How /api/cached-runs/{dataset} ships its runs (see backend/run_listing.py):
// the knobs every run shares once, the others as value tables plus, per run, its
// feature-column group and one index per knob (-1: requested without it).
export interface CachedRunListing {
  dataset: string;
  groups: { feature_cols: string[]; n_total: number | null }[];
  fixed: Partial<AnalysisConfig>;
  knobs: (keyof AnalysisConfig)[];
  values: (string | number | boolean | null)[][];
  runs: number[][];
}

export interface DatasetInfo {
  key: string;
  label: string;
}

export interface DatasetColumns {
  key: string;
  n_rows: number;
  columns: string[];
  default_feature_cols: string[];
  image: ImageSpec | null; // non-null = every row is an image of these dimensions
}

// Datasets whose rows are images (Digits, Olivetti faces, MNIST, Fashion-MNIST).
export interface ImageSpec {
  width: number;
  height: number;
}

export interface ImagePixels extends ImageSpec {
  pixels: number[]; // 0..255 greyscale, row-major
}

// ── Predicate (selection-time) ──────────────────────────────────────────────
export interface PredicateRow {
  feature: string;
  sel_min: number;
  sel_max: number;
  sel_range: number;
  global_min: number;
  global_max: number;
  clause_f1: number; // this clause ALONE against the selection ("Standalone F1")
  clause_precision: number;
  clause_recall: number;
  in_predicate: boolean;
  predicate_step: number | null; // zero-based greedy selection order
  predicate_f1: number; // F1 of the FINAL conjunction; identical on every row
  // ── Marginal contribution of this clause at the step it was added ─────────
  // Populated for selected clauses only; `null` on unselected clauses and on the
  // whole no-labels path. The gain is a property of the CONSTRUCTION ORDER, not
  // of the feature: it is conditional on the clauses already chosen and on the
  // fixed interval the RCM trim produced, so it is not feature importance.
  predicate_f1_before: number | null; // conjunction F1 before this clause
  predicate_f1_after: number | null; // conjunction F1 after this clause
  predicate_f1_gain: number | null; // after - before
  predicate_n_matched: number | null; // |M_k| — background points still matched
  predicate_precision: number | null; // precision of the conjunction after this step
  predicate_recall: number | null; // recall of the conjunction after this step
  // F1 of the EMPTY conjunction (matches all N background points). Identical on
  // every row, present even when nothing was selected. 2|S|/(N+|S|) — a function
  // of selection and background size only, so it is never an informative score.
  predicate_baseline_f1: number;
}

// Characteristics of a lasso selection rather than of a whole node: same records as
// TreeNode.rel_characteristics, but z-scored within the node being explored.
export interface CharacteristicsResponse {
  characteristics: Characteristic[];
}

export interface PredicateSummary {
  predicate_f1: number; // RCM 1.0 final conjunction F1
  predicate_baseline_f1: number; // RCM 1.0 empty-conjunction F1 (see PredicateRow)
  // The RCM 0.9 run is a SEPARATE greedy construction: its clause membership and
  // its order can both differ from the 1.0 run, so it carries its own trajectory
  // rather than reusing the numbers above. `null` when the trimmed run produced
  // no rows.
  trimmed_predicate_f1: number | null;
  trimmed_baseline_f1: number | null;
  n_features_used: number;
  n_features_total: number;
  n_selected: number;
  // Size of the background the baseline was scored on (the node in local scope,
  // the whole dataset in global scope) — the N in "no clauses (all N points)".
  n_background: number;
}

export interface PredicateResponse {
  full: PredicateRow[]; // RCM 1.0
  trimmed: PredicateRow[]; // RCM 0.9
  summary: PredicateSummary | null;
}

export type PredicateScope = "local" | "global";

// ── Target columns (the `target_*` labels, kept out of the feature space) ────
// Reported for a selection alongside the predicate, never as part of it.
export interface TargetStat {
  feature: string;
  is_boolean: boolean; // one-hot label column — its mean is a class share, not a magnitude
  sel_min: number | null;
  sel_max: number | null;
  sel_mean: number | null;
  global_min: number | null;
  global_max: number | null;
  global_mean: number | null;
}

export interface TargetsResponse {
  n_selected: number;
  targets: TargetStat[];
}

export interface RowsResponse {
  columns: string[];
  rows: Record<string, unknown>[];
}

// ── Config knobs the frontend exposes (overlaid on backend defaults) ─────────
export type DRMethod = "PCA" | "t-SNE" | "UMAP" | "MDS";

export interface AnalysisConfig {
  // hierarchical clustering
  hclust_normalize: boolean;
  hierarchical_layers: number;
  hclust_umap_n_components: number;
  hclust_min_samples: number;
  hclust_min_cluster_size: number;
  // exploration / per-node embedding
  normalize: boolean;
  method: DRMethod;
  pca_components: number;
  tsne_perplexity: number;
  tsne_learning_rate: number;
  tsne_random_state: number;
  umap_n_neighbors: number;
  umap_min_dist: number;
  umap_random_state: number;
  mds_metric: boolean; // true = metric MDS, false = non-metric
  mds_n_init: number;
  mds_max_iter: number;
  mds_random_state: number;
}

// ── Cluster movement (counterfactual preview) ───────────────────────────────
// A shared translation of one child cluster: X_source' = X_source + strength·Δ.
// Every source point receives the SAME feature-space displacement, so the
// cluster moves without morphing — pairwise distances inside it are preserved.
// Nothing is written back: the dataframe and the analysis tree are untouched.
//
// COORDINATE CONVENTION (binding): every 2D quantity crossing this contract —
// target x/y, preview points, displacements, distances — is in VISIBLE
// coordinates, i.e. the units of the parent node's serialized
// `embedding_original`. The backend refits the reducer to answer a movement and
// aligns that refit to the stored embedding with a similarity transform; the
// frontend never sees fitted-reducer coordinates.

export type MovementTarget =
  | { kind: "point"; x: number; y: number }
  | { kind: "cluster"; child_index: number };

export interface MovementRequest {
  analysis_id: string; // must match AnalysisMeta.analysis_id of the run on screen
  dataset: string;
  feature_cols: string[];
  config: Record<string, unknown>; // the client's knobs; the server refits from
  // `effective_config` regardless — sent only so the run signature can be checked
  node_id: string; // parent node, e.g. "root/1" (TreeNode.id)
  source_child_index: number;
  target: MovementTarget;
  strength: number | null; // null = "use the recommended strength"
}

// One row of the recommendation table. Raw units are the dataset's own units:
// the backend undoes the root standardization before reporting.
export interface MovementFeatureChange {
  feature: string;
  mutable: boolean; // false = held fixed (row_id / target_* columns), delta is 0
  source_mean_raw: number | null;
  target_value_raw: number | null;
  recommended_delta_raw: number | null; // the full-strength Δ for this feature
  applied_delta_raw: number | null; // recommended × applied_strength
  standardized_magnitude: number | null; // |Δ| in root-standardized units
}

export interface MovementPreviewPoint {
  row_id: number;
  x: number; // visible coordinates
  y: number;
}

export interface MovementMetrics {
  projected_distance_before: number | null;
  projected_distance_after: number | null;
  projected_distance_reduction: number | null;
  feature_centroid_distance_before: number | null;
  // Cluster targets only — `null` for point targets, whose feature-space
  // destination is DEFINED as μ_source + Δ, so the value would be (1−α)·‖Δ‖ by
  // construction and read as if the click had been reached whether or not it was.
  feature_centroid_distance_after: number | null;
  // Fraction of MOVED source points with at least one mutable feature outside
  // the FULL dataset's observed raw [min, max]. Raw units, whole dataset as the
  // reference. Above 0.05 the panel warns: the counterfactual is leaving the
  // region the data actually covers.
  out_of_range_fraction: number | null;
  // Free UMAP targets only: distance from the click to the nearest embedded
  // point, over the typical local neighbour distance. Large = the inverse is
  // an extrapolation.
  support_ratio: number | null;
  // Normalized RMSE of the refit→stored embedding alignment over the node's
  // EXISTING points (residual over the visible cloud's RMS radius). Near zero
  // means the refit reproduced the projection on screen; above the server's
  // bound (0.05) the movement is a 409, never tolerated away. The bound is on
  // the RMS only: `alignment_max_residual` is the single worst point, which can
  // sit above it on a passing alignment, and neither says anything about the
  // error of NEW counterfactual points.
  alignment_rmse: number | null;
  alignment_max_residual: number | null;
  preview_sampled: boolean; // true = preview_points is a deterministic subsample
}

// UMAP's out-of-sample transform is neighbour-driven, so projected distance vs
// strength can plateau and jump. The whole grid is returned, not just the
// argmin, and the panel plots it; a non-monotone trajectory earns a caveat.
export interface MovementStrengthPoint {
  strength: number;
  projected_distance: number;
}

export interface MovementResponse {
  status: "ok";
  analysis_id: string;
  node_id: string;
  method: string; // "PCA" | "UMAP" — the method the run was BUILT with
  source_child_index: number;
  target: {
    kind: "point" | "cluster";
    child_index: number | null;
    x: number; // visible coordinates
    y: number;
    // How the feature-space destination was obtained. "pca_pinv" = minimum-norm
    // solution in the reducer's input space; "umap_inverse" = UMAP
    // inverse_transform; "neighbor_interpolation" = the fallback when the
    // inverse fails; "observed_centroid" = a cluster target's actual mean.
    estimator: "pca_pinv" | "umap_inverse" | "neighbor_interpolation" | "observed_centroid";
  };
  recommended_strength: number | null; // null = no strength approached the target
  applied_strength: number;
  source_size: number;
  target_size: number | null; // cluster targets only
  feature_changes: MovementFeatureChange[];
  preview_points: MovementPreviewPoint[]; // moved source points, visible coordinates
  // Full-strength projected displacement in VISIBLE coordinates. The PCA slider
  // updates ghosts client-side as `ghost = Y_source + α · visible_displacement`;
  // for UMAP it is indicative only (the transform is not linear).
  visible_displacement: [number, number];
  strength_trajectory: MovementStrengthPoint[]; // UMAP: the whole α grid; PCA: []
  metrics: MovementMetrics;
  warnings: string[];
}

// Movement runs as a background job when a UMAP refit is needed; a cached
// artifact (and every PCA request) answers inline. Unlike AnalysisJob the
// finished variant is the response itself rather than a `done` wrapper: the
// analysis payload has no `status` of its own to collide with, and a movement
// response does (`"ok"`). Discriminating on that one field keeps the inline and
// the polled result literally the same object.
export type MovementJob =
  | { status: "running"; job_id: string }
  | { status: "error"; job_id: string; detail: string }
  | MovementResponse;

// Client-side state machine, owned by `useMovement` in frontend/src/movement.ts.
// (There is no `Navigation` struct in App.tsx — it holds flat useState.)
export type MovementState =
  | { phase: "idle" }
  | { phase: "choosing"; nodeId: string; sourceChildIndex: number }
  | {
      phase: "loading";
      nodeId: string;
      sourceChildIndex: number;
      target: MovementTarget;
    }
  | {
      phase: "ready";
      nodeId: string;
      sourceChildIndex: number;
      target: MovementTarget;
      response: MovementResponse;
      strength: number; // slider position; may lead the response for PCA
    }
  | {
      phase: "error";
      nodeId: string;
      sourceChildIndex: number;
      target: MovementTarget | null;
      message: string;
    };

// ── Counterfactual sessions (Apply) ─────────────────────────────────────────
// Apply writes the previewed deltas — at the applied strength, mutable features
// only, one rigid translation per moved row — into a server-side counterfactual
// COPY of the dataset and rebuilds the whole analysis on it. Nothing about the
// original dataframe or any existing payload changes. Edits are immutable and
// content-addressed, stacked as a chain; Undo and Reset are just a switch back
// to the parent key or the base. See backend/counterfactual.py.

export interface CounterfactualEdit {
  cf_id: string;
  parent: string | null; // the cf_id this edit stacks on; null = directly on the base
  base_dataset: string;
  dataset_key: string; // "{base}@cf:{cf_id}" — what every request sends as `dataset`
  node_id: string;
  source_child_index: number;
  target: MovementTarget;
  strength: number;
  n_rows: number; // every member of the source cluster, never the preview sample
  features_changed: string[];
  deltas_raw: Record<string, number>; // feature -> applied raw delta (mutable only)
  created_at: number;
}

/** `MovementRequest` with a required, positive strength. `dataset` is the base
 *  or an existing counterfactual key to stack on. */
export interface CounterfactualApplyRequest extends Omit<MovementRequest, "strength"> {
  strength: number;
}

export interface CounterfactualApplyResponse {
  status: "ok";
  dataset_key: string;
  cf_id: string;
  edits: CounterfactualEdit[]; // the whole chain, base first
  n_rows_changed: number;
}

// Same three-way union as MovementJob: an Apply on a cold UMAP node is a job.
export type CounterfactualJob =
  | { status: "running"; job_id: string }
  | { status: "error"; job_id: string; detail: string }
  | CounterfactualApplyResponse;

export interface CounterfactualChain {
  cf_id: string;
  dataset_key: string;
  base_dataset: string;
  edits: CounterfactualEdit[];
}

// Client-side state of the session, owned by `useCounterfactual` in
// frontend/src/counterfactual.ts. `applying` = the apply request is out;
// `rebuilding` = the analysis is being rebuilt on the new key while the current
// run stays on screen; `error` keeps the stack and the displayed run unchanged.
export type CounterfactualPhase = "idle" | "applying" | "rebuilding" | "error";

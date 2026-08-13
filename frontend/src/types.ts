// Data contract shared with the Python backend (see PLAN.md "Data Contract").
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
  config: Record<string, unknown>;
  n_total: number;
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
  clause_f1: number;
  clause_precision: number;
  clause_recall: number;
  in_predicate: boolean;
  predicate_step: number | null;
  predicate_f1: number;
}

// Characteristics of a lasso selection rather than of a whole node: same records as
// TreeNode.rel_characteristics, but z-scored within the node being explored.
export interface CharacteristicsResponse {
  characteristics: Characteristic[];
}

export interface PredicateSummary {
  predicate_f1: number;
  n_features_used: number;
  n_features_total: number;
  n_selected: number;
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

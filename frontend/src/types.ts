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

export interface Kde {
  grid: (number | null)[][]; // resolution x resolution density
  resolution: number; // 60
  extent: [number, number]; // normalized grid bounds, [-0.5, 0.5]
}

export interface Characteristic {
  feature: string;
  z_mean: number | null;
  z_std: number | null;
  raw_mean: number | null;
}

// One node of the analysis tree. Internal nodes have children; leaves do not.
export interface TreeNode {
  id: string; // stable path id, e.g. "root", "root/2/0"
  is_leaf: boolean;
  depth: number;
  n_points: number;
  row_indices: number[]; // indices into the source dataframe
  embedding_original: [number | null, number | null][]; // Nx2 projection
  embedding_original_variance: (number | null)[] | null; // PCA only
  rel_position: [number | null, number | null] | null; // MDS centroid in sibling layout
  kde: Kde | null;
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

export interface RowsResponse {
  columns: string[];
  rows: Record<string, unknown>[];
}

// ── Config knobs the frontend exposes (overlaid on backend defaults) ─────────
export type DRMethod = "PCA" | "t-SNE" | "UMAP";

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
}

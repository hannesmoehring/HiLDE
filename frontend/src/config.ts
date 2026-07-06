import type { AnalysisConfig } from "./types";

// Mirrors the defaults in src/ui/state.py::init_state (the ones the config panel exposes).
export const DEFAULT_CONFIG: AnalysisConfig = {
  hclust_normalize: true,
  hierarchical_layers: 1,
  hclust_umap_n_components: 2,
  hclust_min_samples: 5,
  hclust_min_cluster_size: 25,
  normalize: true,
  method: "UMAP",
  pca_components: 4,
  tsne_perplexity: 30,
  tsne_learning_rate: 200,
  tsne_random_state: 42,
  umap_n_neighbors: 15,
  umap_min_dist: 0.1,
  umap_random_state: 42,
  mds_metric: true,
  mds_n_init: 2,
  mds_max_iter: 100,
  mds_random_state: 42,
};

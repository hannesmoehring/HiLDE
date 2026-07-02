// Prop contracts for the six ported charts. Phase-4 subagents implement each
// chart against the interface here; the integrator (App/exploration wiring) codes
// against these same types, so components drop in without merge conflicts.
//
// Parity references point at the Streamlit source each chart replaces.
import type { Characteristic, NodeScores, PredicateRow, TreeNode } from "../types";

// A — KDE cluster topography. Replaces src/ui/visualization.py::cluster_gauss_kde.
// Small-multiples of per-child 2D KDE density contours, positioned by each child's
// MDS rel_position and size-scaled by n_points, with clickable centroids C0..Cn.
export interface KdeTopographyProps {
  node: TreeNode; // internal node whose children are drawn
  onSelectCluster: (childIndex: number) => void;
  selectedChild?: number | null;
  title?: string;
}

// B — Cluster characteristics bar. Replaces cluster_characteristics_fig.
// Grouped z-score bars per feature, error bars = z_std, dotted zero line,
// bar color by sign of z_mean (positive vs negative).
export interface CharacteristicsBarProps {
  data: Characteristic[];
  title?: string;
}

// C — Projection scatter. Replaces make_scatter_fig.
// 2D scatter of a node's embedding with lasso + box selection. Optional coloring
// by cluster label or by interactive-filter membership.
export interface ProjectionScatterProps {
  points: [number | null, number | null][]; // embedding_original of the node
  rowIds: number[]; // parallel to points; the node's row_indices
  method: string; // "PCA" | "t-SNE" | "UMAP" (axis labels)
  clusterLabels?: string[] | null; // color-by-cluster mode
  interactiveGroup?: string[] | null; // color-by-filter mode ("Matches filters"/"Other")
  onSelect: (localIndices: number[]) => void; // indices into points[]
  selected?: number[]; // controlled highlight
}

// D — PCA explained-variance bar. Replaces make_pca_variance_fig.
// Horizontal stacked bar, one segment per principal component.
export interface PcaVarianceBarProps {
  explainedVariance: number[]; // ratios, sum <= 1
}

// E — Predicate feature-range bands. Replaces make_feature_range_fig.
// Per feature: faint global track, translucent full (RCM 1.0) band, solid core
// (RCM 0.9) band; predicate-clause features highlighted.
export interface PredicateBandsProps {
  full: PredicateRow[]; // RCM 1.0
  trimmed: PredicateRow[]; // RCM 0.9
}

// F — DR-quality score tiles. Replaces src/ui/components/scores.py::render_node_scores.
// Tiles for Trustworthiness / Continuity / Stress / CADI + an MRRE caption.
export interface ScoreTilesProps {
  scores: NodeScores | null;
  title?: string;
}

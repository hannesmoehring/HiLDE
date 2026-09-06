// Prop contracts for the six ported charts. Phase-4 subagents implement each
// chart against the interface here; the integrator (App/exploration wiring) codes
// against these same types, so components drop in without merge conflicts.
//
// Parity references point at the Streamlit source each chart replaces.
import type { ReactNode } from "react";
import type {
  Characteristic,
  MovementResponse,
  MovementTarget,
  NodeScores,
  PredicateRow,
  TargetStat,
  TreeNode,
} from "../types";

// A — Cluster projection scatter. Replaces the KDE topography (which ported
// src/ui/visualization.py::cluster_gauss_kde): the parent node's 2D embedding
// with points colored by child cluster, HDBSCAN noise as grey ×, and a clickable
// legend. Clicking a point / legend chip / centroid label drills into that child.
export interface ClusterScatterProps {
  node: TreeNode; // internal node whose children are drawn
  onSelectCluster: (childIndex: number) => void;
  selectedChild?: number | null;
  title?: string;
  highlightRow?: number | null; // row id to ring (a point picked in the GLOSH outlier table)
  // ── Cluster movement ──────────────────────────────────────────────────────
  // Off by default: with `movementMode` false the chart behaves exactly as
  // before, and a click still drills down. While it is on, drill-down clicks are
  // suspended for this projection and a click resolves to a movement target
  // instead, by this precedence: centroid label / legend entry -> that cluster;
  // a point of another cluster -> that cluster; the nearest other-cluster point
  // within a 10 px screen-space snap radius (measured AFTER the zoom transform)
  // -> that cluster; otherwise -> a free point at the inverted coordinate.
  // Clicks on the source cluster resolve to nothing. Coordinates handed back are
  // in `embedding_original` units: undo the d3 zoom transform, then the
  // equal-aspect plot transform.
  movementMode?: boolean;
  movementSource?: number | null; // child index being moved
  movementTarget?: MovementTarget | null; // resolved destination, for the marker
  movementPreview?: MovementResponse | null; // ghost points, arrow, target marker
  onMovementTarget?: (target: MovementTarget) => void;
}

// B — Cluster characteristics bar. Replaces cluster_characteristics_fig.
// Grouped z-score bars per feature, error bars = z_std, dotted zero line,
// bar color by sign of z_mean (positive vs negative).
export interface CharacteristicsBarProps {
  data: Characteristic[];
  title?: string;
  nonFeatureOnly?: boolean; // hide feature columns; show only non-feature (target) columns
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
  toolbarExtra?: ReactNode; // right-bound slot in the Lasso/Box row
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

// E2 — Target-value bands. Same band geometry as E, for the `target_*` label
// columns that are deliberately excluded from the predicate. Rendered in the
// target hue so it never reads as a predicate clause.
export interface TargetBandsProps {
  targets: TargetStat[];
  nSelected: number;
}

// F — DR-quality score tiles. Replaces src/ui/components/scores.py::render_node_scores.
// Tiles for Trustworthiness / Continuity / Stress / CADI + an MRRE caption.
export interface ScoreTilesProps {
  scores: NodeScores | null;
  title?: string;
}

// Cache-only mode: choosing among the stored runs of one dataset. A sweep stores
// hundreds per dataset, so the picker is one selector per knob the runs actually
// differ in, not a list of runs.
import { DEFAULT_CONFIG } from "./config";
import type { AnalysisConfig, CachedRun } from "./types";

/** Something the stored runs of a dataset differ in, and the values it takes. */
export interface RunFacet {
  label: string;
  values: string[];
  valueOf: (run: CachedRun) => string;
}

// The names ConfigPanel.tsx gives the knobs; anything else shows under its key.
const KNOB_LABELS: Partial<Record<keyof AnalysisConfig, string>> = {
  normalize: "Normalize",
  hierarchical_layers: "Hierarchical layers",
  hclust_umap_n_components: "UMAP pre-reduction components",
  hclust_min_samples: "HDBSCAN min samples",
  hclust_min_cluster_size: "HDBSCAN min cluster size",
  method: "Method",
  tsne_perplexity: "t-SNE perplexity",
  tsne_learning_rate: "t-SNE learning rate",
  umap_n_neighbors: "UMAP n_neighbors",
  umap_min_dist: "UMAP min_dist",
  mds_metric: "Metric MDS",
};

export function runFacets(runs: CachedRun[]): RunFacet[] {
  const knobs = (Object.keys(DEFAULT_CONFIG) as (keyof AnalysisConfig)[])
    // One checkbox drives this and `normalize`; offering both would be the same choice twice.
    .filter((k) => k !== "hclust_normalize")
    .map((k) => ({
      label: KNOB_LABELS[k] ?? k,
      valueOf: (run: CachedRun) => String(run.config[k] ?? DEFAULT_CONFIG[k]),
    }));
  const features = {
    label: "Feature columns",
    valueOf: (run: CachedRun) => String(run.feature_cols.length),
  };
  return [features, ...knobs]
    .map((f) => ({
      ...f,
      values: Array.from(new Set(runs.map(f.valueOf))).sort(
        (a, b) => Number(a) - Number(b) || a.localeCompare(b),
      ),
    }))
    .filter((f) => f.values.length > 1);
}

/** The run a dataset opens on: the stored one nearest the app's own defaults. */
export function defaultRun(runs: CachedRun[]): CachedRun | null {
  const keys = Object.keys(DEFAULT_CONFIG) as (keyof AnalysisConfig)[];
  let best: CachedRun | null = null;
  let bestShared = -1;
  for (const run of runs) {
    const shared = keys.filter((k) => run.config[k] === DEFAULT_CONFIG[k]).length;
    if (shared > bestShared) {
      best = run;
      bestShared = shared;
    }
  }
  return best;
}

/** The stored run that takes `value` on `facet` and otherwise differs least from
 *  `current` — the stored combinations need not be a full grid. */
export function nearestRun(
  runs: CachedRun[],
  facets: RunFacet[],
  current: CachedRun,
  facet: RunFacet,
  value: string,
): CachedRun | null {
  let best: CachedRun | null = null;
  let bestShared = -1;
  for (const run of runs) {
    if (facet.valueOf(run) !== value) continue;
    const shared = facets.filter((f) => f.valueOf(run) === f.valueOf(current)).length;
    if (shared > bestShared) {
      best = run;
      bestShared = shared;
    }
  }
  return best;
}

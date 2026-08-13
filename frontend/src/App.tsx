import { useEffect, useMemo, useState, type ReactElement } from "react";
import { datasetColumns, getMode, listDatasets, runAnalysis } from "./api";
import { ClusterScatter } from "./charts/ClusterScatter";
import { ConfigPanel } from "./components/ConfigPanel";
import { ExplorationPanel } from "./components/ExplorationPanel";
import { LayerSide } from "./components/LayerSide";
import { OutlierPanel } from "./components/OutlierPanel";
import { DEFAULT_CONFIG } from "./config";
import { getNodeAtPath } from "./treeNav";
import type {
  AnalysisConfig,
  AnalysisResponse,
  DatasetColumns,
  DatasetInfo,
  ImageSpec,
  ModeInfo,
} from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetKey, setDatasetKey] = useState<string>("");
  const [columns, setColumns] = useState<DatasetColumns | null>(null);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [charNonFeatureOnly, setCharNonFeatureOnly] = useState(false);
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);

  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [treePath, setTreePath] = useState<number[]>([]);
  // Explore the node at the end of `treePath` whole, instead of waiting for a drill
  // into one of its clusters. Every navigation goes through `navigate` so the flag
  // can never outlive the path it was set for.
  const [exploreWhole, setExploreWhole] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function navigate(path: number[], whole = false) {
    setTreePath(path);
    setExploreWhole(whole);
  }

  // The config rail collapses to a 38px strip so the analysis canvas can widen.
  const [configOpen, setConfigOpen] = useState(true);

  // Hosting mode only: reuse stored runs, and say so when we do.
  const [mode, setMode] = useState<ModeInfo | null>(null);
  const [useCache, setUseCache] = useState(true);

  useEffect(() => {
    getMode()
      .then(setMode)
      .catch(() => setMode({ hosting: false, cache_dir: null }));
  }, []);

  useEffect(() => {
    listDatasets()
      .then((d) => {
        setDatasets(d);
        if (d.length) setDatasetKey(d[0].key);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // On dataset change, load its columns and reset the feature selection to the default.
  useEffect(() => {
    if (!datasetKey) return;
    setColumns(null);
    setAnalysis(null);
    navigate([]);
    datasetColumns(datasetKey)
      .then((c) => {
        setColumns(c);
        setFeatureCols(c.default_feature_cols);
        // Default the UMAP pre-reduction to the dataset's full feature dimensionality,
        // so clustering runs on the full space (the backend skips pre-reduction when
        // n_components >= n_features).
        setConfig((cfg) => ({
          ...cfg,
          hclust_umap_n_components: Math.max(2, c.default_feature_cols.length),
        }));
      })
      .catch((e) => setError(String(e)));
  }, [datasetKey]);

  const maxDims = useMemo(() => Math.max(2, featureCols.length), [featureCols]);

  // Label columns held out of the feature space (see backend default_feature_cols).
  // A `target_*` column the user checked in as a feature is not one: it is in the
  // predicate now, and reporting it twice would say otherwise.
  const targetCols = useMemo(
    () => (columns?.columns ?? []).filter((c) => c.startsWith("target_") && !featureCols.includes(c)),
    [columns, featureCols],
  );

  function patchConfig(patch: Partial<AnalysisConfig>) {
    setConfig((c) => ({ ...c, ...patch }));
  }

  function toggleFeature(col: string) {
    setFeatureCols((f) => (f.includes(col) ? f.filter((x) => x !== col) : [...f, col]));
  }

  async function build() {
    if (!datasetKey || featureCols.length === 0) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    navigate([]);
    try {
      const res = await runAnalysis(datasetKey, featureCols, config, useCache);
      setAnalysis(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  // Picking a new dataset clears `analysis`, but only in an effect — so one render
  // pairs the new dataset key with the previous dataset's tree. The panels below key
  // their requests off both, and row indices from a 6497-row tree sent against a
  // 150-row dataset are a 500, not an empty result. Never hand them a mismatched pair.
  const shownAnalysis = analysis && analysis.meta.dataset === datasetKey ? analysis : null;

  const runMeta = [datasetKey || "no dataset", `${featureCols.length} features`, config.method]
    .filter(Boolean)
    .join(", ");
  const cfgSummary = [
    config.method,
    `${config.hierarchical_layers} layers`,
    `mcs ${config.hclust_min_cluster_size}`,
  ].join(", ");

  return (
    <div className="app">
      <header className="topbar">
        <span className="topbar__brand">HiLDE</span>
        <span className="topbar__sub">
          <b>Hi</b>erarchical <b>L</b>ocal <b>D</b>ecomposition &amp; <b>E</b>xplanation
        </span>
        <span className="topbar__meta">{runMeta}</span>
      </header>

      <div className={configOpen ? "shell" : "shell shell--collapsed"}>
        <aside className="rail">
          {configOpen ? (
            <>
              <div className="rail__head">
                <span className="kicker">Configuration</span>
                <span className="rail__summary">{cfgSummary}</span>
                <button onClick={() => setConfigOpen(false)}>Hide</button>
              </div>

              <div className="cfg">
                <section className="cfg__block">
                  <h3>Dataset</h3>
                  <label className="field">
                    <span>Source</span>
                    {/* Locked during a build: swapping datasets mid-run would apply the
                        in-flight tree under the new dataset's key. */}
                    <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value)} disabled={loading}>
                      {datasets.map((d) => (
                        <option key={d.key} value={d.key}>
                          {d.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  {columns && (
                    <div className="cfg__meta">
                      <span>{columns.columns.length} columns</span>
                      <span>{featureCols.length} selected</span>
                    </div>
                  )}
                </section>

                {columns && (
                  <section className="cfg__block">
                    <div className="cfg__head">
                      <h3>Feature columns</h3>
                      <span className="cfg__count">
                        {featureCols.length}/{columns.columns.length - 1}
                      </span>
                      <div className="cfg__actions">
                        <button onClick={() => setFeatureCols(columns.default_feature_cols)}>Reset</button>
                        <button onClick={() => setFeatureCols([])}>None</button>
                      </div>
                    </div>
                    <div className="feature-picker__list">
                      {columns.columns
                        .filter((c) => c !== "row_id")
                        .map((c) => (
                          <label key={c} className={featureCols.includes(c) ? undefined : "is-off"}>
                            <input
                              type="checkbox"
                              checked={featureCols.includes(c)}
                              onChange={() => toggleFeature(c)}
                            />
                            {c}
                          </label>
                        ))}
                    </div>
                    <label className="field--check" style={{ marginTop: "0.6rem", marginBottom: 0 }}>
                      <input
                        type="checkbox"
                        checked={charNonFeatureOnly}
                        onChange={(e) => setCharNonFeatureOnly(e.target.checked)}
                      />
                      <span>Characteristics: non-feature columns only</span>
                    </label>
                  </section>
                )}

                <ConfigPanel config={config} maxDims={maxDims} onChange={patchConfig} />
              </div>

              <div className="cfg__build">
                <button className="primary" onClick={build} disabled={loading || featureCols.length === 0}>
                  {loading ? "Building…" : "Build & Apply"}
                </button>
                {mode?.hosting && (
                  <label className="field--check" style={{ marginBottom: 0 }} title={mode.cache_dir ?? undefined}>
                    <input
                      type="checkbox"
                      checked={useCache}
                      onChange={(e) => setUseCache(e.target.checked)}
                    />
                    <span>Use cached results</span>
                  </label>
                )}
              </div>
            </>
          ) : (
            <div className="rail__strip">
              <button onClick={() => setConfigOpen(true)} title="Show configuration">
                ›
              </button>
              <span className="rail__vert">Configuration</span>
              <span className="rail__vert rail__vert--faint">{cfgSummary}</span>
            </div>
          )}
        </aside>

        <main className="canvas">
          {/* Failures report on the canvas, not in the rail: the rail collapses to a
              38px strip and would otherwise swallow the only sign anything went wrong. */}
          {error && (
            <div className="banner banner--error" role="alert">
              <strong>Error</strong>
              <span>{error}</span>
              {!configOpen && <button onClick={() => setConfigOpen(true)}>Show configuration</button>}
            </div>
          )}

          {mode?.hosting && shownAnalysis?.cached && (
            <div className="banner">
              <strong>Cached</strong>
              <span>
                This dataset and configuration were computed before, so the stored run was reused —
                nothing was recomputed. Uncheck <em>Use cached results</em> and rebuild to force a fresh run.
              </span>
            </div>
          )}

          {loading && <div className="empty">Reducing, clustering and scoring …</div>}

          {shownAnalysis && (
            <Navigation
              analysis={shownAnalysis}
              treePath={treePath}
              exploreWhole={exploreWhole}
              navigate={navigate}
              dataset={datasetKey}
              featureCols={featureCols}
              targetCols={targetCols}
              config={config}
              charNonFeatureOnly={charNonFeatureOnly}
              imageSpec={columns?.image ?? null}
            />
          )}

          {!shownAnalysis && !loading && (
            <div className="empty">Pick features and press Build &amp; Apply to compute a run.</div>
          )}
        </main>
      </div>
    </div>
  );
}

function Navigation(props: {
  analysis: AnalysisResponse;
  treePath: number[];
  exploreWhole: boolean;
  navigate: (path: number[], whole?: boolean) => void;
  dataset: string;
  featureCols: string[];
  targetCols: string[];
  config: AnalysisConfig;
  charNonFeatureOnly: boolean;
  imageSpec: ImageSpec | null;
}) {
  const { analysis, treePath, exploreWhole, navigate, dataset, featureCols, targetCols, config, charNonFeatureOnly, imageSpec } =
    props;
  const root = analysis.tree;
  const nLayers = config.hierarchical_layers;

  // Point picked in a layer's GLOSH outlier table; ringed in that layer's scatter.
  // One at a time across layers, cleared whenever we drill in or out.
  const [outlierPick, setOutlierPick] = useState<{ layer: number; rowId: number } | null>(null);
  const pathKey = treePath.join(",");
  useEffect(() => {
    setOutlierPick(null);
  }, [pathKey, root]);

  const layerViews: ReactElement[] = [];
  let explorationPath: number[] | null = null;
  let waiting = false;

  for (let L = 1; L <= nLayers; L++) {
    const parentPath = treePath.slice(0, L - 1);
    const node = getNodeAtPath(root, parentPath);
    if (node.is_leaf || (node.children && node.children.length === 0)) {
      explorationPath = parentPath;
      break;
    }
    const selectedChild = treePath.length >= L ? treePath[L - 1] : null;
    const child = selectedChild != null ? node.children![selectedChild] : null;
    // Whole-layer exploration lands on the layer whose own node the path ends at —
    // the deepest one rendered, i.e. the one that would otherwise be waiting for a
    // cluster click. `node` here *is* that node, so it is what gets explored.
    const exploringHere = exploreWhole && treePath.length === L - 1;
    layerViews.push(
      <section className="panel layer" key={`layer-${L}`}>
        <div className="panel__head">
          <span className="kicker">Layer {L}</span>
          <span className="panel__title">
            {L === 1 ? "Cluster projection — root" : `Sub-projection — layer ${L}`}
          </span>
          <span className="panel__meta">
            {node.row_indices.length} points, {node.children?.length ?? 0} clusters
          </span>
        </div>
        <div className="layer__cols">
          <div>
            <ClusterScatter
              node={node}
              selectedChild={selectedChild}
              onSelectCluster={(i) => navigate([...parentPath, i])}
              highlightRow={outlierPick?.layer === L ? outlierPick.rowId : null}
            />
            {/* The projection above is what this acts on, so the action sits under it
                rather than in the side column, which is about the selected child. */}
            <div className="layer__explore">
              <p className="hint">
                {exploringHere
                  ? "Exploring every point in this layer — its clusters are not split up."
                  : "Or take the layer whole, without drilling into one cluster."}
              </p>
              <button
                className={exploringHere ? "primary" : undefined}
                aria-pressed={exploringHere}
                onClick={() => navigate(parentPath, !exploringHere)}
                title={
                  exploringHere
                    ? "Go back to picking a cluster to drill into"
                    : `Open the exploration panel on all ${node.row_indices.length} points of this layer`
                }
              >
                {exploringHere ? "Exploring entire layer" : "Explore entire layer"}
              </button>
            </div>
          </div>
          <div className="layer__side">
            {child ? (
              <LayerSide
                parent={node}
                child={child}
                childIndex={selectedChild!}
                dataset={dataset}
                featureCols={featureCols}
                config={config}
                charNonFeatureOnly={charNonFeatureOnly}
              />
            ) : (
              <p className="hint">
                {exploringHere
                  ? "The whole layer is being explored below. Select a cluster to drill into one instead."
                  : "Select a cluster to see its DR quality, characteristics and predicate."}
              </p>
            )}
          </div>
        </div>
        <OutlierPanel
          node={node}
          dataset={dataset}
          selectedRow={outlierPick?.layer === L ? outlierPick.rowId : null}
          onSelectRow={(rowId) => setOutlierPick(rowId == null ? null : { layer: L, rowId })}
        />
      </section>,
    );
    if (exploringHere) {
      explorationPath = parentPath;
      break;
    }
    if (treePath.length < L) {
      waiting = true;
      break;
    }
  }

  if (!waiting && explorationPath === null) explorationPath = treePath.slice(0, nLayers);

  const explorationNode = explorationPath !== null ? getNodeAtPath(root, explorationPath) : null;
  const pathLabel = explorationPath && explorationPath.length ? explorationPath.map((c) => `C${c}`).join(" → ") : "root";
  // The explored node is whatever the deepest layer holds as its selected child, so
  // that layer already reports its scores. A non-empty path says so either way: under
  // "explore entire layer" the node is the layer's own, but a layer's own node is the
  // layer above's selected child, and that side column reports it. Only an empty path
  // has nothing above it — a leaf root, or the whole of layer 1 — and there the
  // exploration panel is the only place the scores can appear.
  const scoresShownByLayer = explorationPath !== null && explorationPath.length > 0;

  return (
    <>
      {layerViews}
      {waiting && <div className="empty">Click a cluster in the projection above to drill in.</div>}
      {explorationNode && (
        <ExplorationPanel
          dataset={dataset}
          featureCols={featureCols}
          targetCols={targetCols}
          config={config}
          node={explorationNode}
          pathLabel={pathLabel}
          imageSpec={imageSpec}
          showScores={!scoresShownByLayer}
          charNonFeatureOnly={charNonFeatureOnly}
        />
      )}
    </>
  );
}

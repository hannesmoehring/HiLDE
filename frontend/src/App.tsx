import { useEffect, useMemo, useState, type ReactElement } from "react";
import { datasetColumns, getMode, listDatasets, runAnalysis } from "./api";
import { CharacteristicsBar } from "./charts/CharacteristicsBar";
import { ClusterScatter } from "./charts/ClusterScatter";
import { ScoreTiles } from "./charts/ScoreTiles";
import { ConfigPanel } from "./components/ConfigPanel";
import { ExplorationPanel } from "./components/ExplorationPanel";
import { OutlierPanel } from "./components/OutlierPanel";
import { DEFAULT_CONFIG } from "./config";
import { getNodeAtPath } from "./treeNav";
import type { AnalysisConfig, AnalysisResponse, DatasetColumns, DatasetInfo, ModeInfo } from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetKey, setDatasetKey] = useState<string>("");
  const [columns, setColumns] = useState<DatasetColumns | null>(null);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [charNonFeatureOnly, setCharNonFeatureOnly] = useState(false);
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);

  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [treePath, setTreePath] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setTreePath([]);
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
    setTreePath([]);
    try {
      const res = await runAnalysis(datasetKey, featureCols, config, useCache);
      setAnalysis(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

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

          {mode?.hosting && analysis?.cached && (
            <div className="banner">
              <strong>Cached</strong>
              <span>
                This dataset and configuration were computed before, so the stored run was reused —
                nothing was recomputed. Uncheck <em>Use cached results</em> and rebuild to force a fresh run.
              </span>
            </div>
          )}

          {loading && <div className="empty">Reducing, clustering and scoring …</div>}

          {analysis && (
            <Navigation
              analysis={analysis}
              treePath={treePath}
              setTreePath={setTreePath}
              dataset={datasetKey}
              featureCols={featureCols}
              config={config}
              charNonFeatureOnly={charNonFeatureOnly}
            />
          )}

          {!analysis && !loading && (
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
  setTreePath: (p: number[]) => void;
  dataset: string;
  featureCols: string[];
  config: AnalysisConfig;
  charNonFeatureOnly: boolean;
}) {
  const { analysis, treePath, setTreePath, dataset, featureCols, config, charNonFeatureOnly } = props;
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
              onSelectCluster={(i) => setTreePath([...parentPath, i])}
              highlightRow={outlierPick?.layer === L ? outlierPick.rowId : null}
            />
          </div>
          <div className="layer__side">
            {child ? (
              <>
                <ScoreTiles scores={child.scores} title={`C${selectedChild} — DR quality`} />
                <CharacteristicsBar
                  data={child.rel_characteristics}
                  title={`C${selectedChild} characteristics`}
                  nonFeatureOnly={charNonFeatureOnly}
                />
              </>
            ) : (
              <p className="hint">Select a cluster to see its DR quality and characteristics.</p>
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
    if (treePath.length < L) {
      waiting = true;
      break;
    }
  }

  if (!waiting && explorationPath === null) explorationPath = treePath.slice(0, nLayers);

  const explorationNode = explorationPath !== null ? getNodeAtPath(root, explorationPath) : null;
  const pathLabel = explorationPath && explorationPath.length ? explorationPath.map((c) => `C${c}`).join(" → ") : "root";

  return (
    <>
      {layerViews}
      {waiting && <div className="empty">Click a cluster in the projection above to drill in.</div>}
      {explorationNode && (
        <ExplorationPanel
          dataset={dataset}
          featureCols={featureCols}
          config={config}
          node={explorationNode}
          pathLabel={pathLabel}
          nonFeatureOnly={charNonFeatureOnly}
        />
      )}
    </>
  );
}

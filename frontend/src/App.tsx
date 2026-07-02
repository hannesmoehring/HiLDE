import { useEffect, useMemo, useState, type ReactElement } from "react";
import { datasetColumns, listDatasets, runAnalysis } from "./api";
import { CharacteristicsBar } from "./charts/CharacteristicsBar";
import { KdeTopography } from "./charts/KdeTopography";
import { ScoreTiles } from "./charts/ScoreTiles";
import { ConfigPanel } from "./components/ConfigPanel";
import { ExplorationPanel } from "./components/ExplorationPanel";
import { DEFAULT_CONFIG } from "./config";
import { getNodeAtPath } from "./treeNav";
import type { AnalysisConfig, AnalysisResponse, DatasetColumns, DatasetInfo } from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetKey, setDatasetKey] = useState<string>("");
  const [columns, setColumns] = useState<DatasetColumns | null>(null);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);

  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [treePath, setTreePath] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      const res = await runAnalysis(datasetKey, featureCols, config);
      setAnalysis(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>SHD — Dimensionality Reduction Explorer</h1>
      </header>

      <section className="panel">
        <h2>General configuration</h2>
        <div className="general-config">
          <label className="field">
            <span>Dataset</span>
            <select value={datasetKey} onChange={(e) => setDatasetKey(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.key} value={d.key}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>

          {columns && (
            <div className="feature-picker">
              <div className="feature-picker__head">
                <span>
                  Feature columns ({featureCols.length}/{columns.columns.length - 1})
                </span>
                <button onClick={() => setFeatureCols(columns.default_feature_cols)}>Reset</button>
                <button onClick={() => setFeatureCols([])}>None</button>
              </div>
              <div className="feature-picker__list">
                {columns.columns
                  .filter((c) => c !== "row_id")
                  .map((c) => (
                    <label key={c}>
                      <input
                        type="checkbox"
                        checked={featureCols.includes(c)}
                        onChange={() => toggleFeature(c)}
                      />
                      {c}
                    </label>
                  ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Analysis configuration</h2>
        <ConfigPanel config={config} maxDims={maxDims} onChange={patchConfig} />
        <div className="build-row">
          <button className="primary" onClick={build} disabled={loading || featureCols.length === 0}>
            {loading ? "Building…" : "Build & Apply"}
          </button>
          {error && <span className="error">{error}</span>}
        </div>
      </section>

      {analysis && <Navigation analysis={analysis} treePath={treePath} setTreePath={setTreePath} dataset={datasetKey} featureCols={featureCols} config={config} />}
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
}) {
  const { analysis, treePath, setTreePath, dataset, featureCols, config } = props;
  const root = analysis.tree;
  const nLayers = config.hierarchical_layers;

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
        <h2>{L === 1 ? "Cluster topography (root)" : `Sub-topography — layer ${L}`}</h2>
        <div className="layer__cols">
          <KdeTopography
            node={node}
            selectedChild={selectedChild}
            onSelectCluster={(i) => setTreePath([...parentPath, i])}
          />
          {child && (
            <div className="layer__side">
              <ScoreTiles scores={child.scores} title={`C${selectedChild} — DR quality`} />
              <CharacteristicsBar data={child.rel_characteristics} title={`C${selectedChild} characteristics`} />
            </div>
          )}
        </div>
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
      {waiting && <p className="hint panel">Click a cluster in the topography above to drill in.</p>}
      {explorationNode && (
        <ExplorationPanel
          dataset={dataset}
          featureCols={featureCols}
          config={config}
          node={explorationNode}
          pathLabel={pathLabel}
        />
      )}
    </>
  );
}

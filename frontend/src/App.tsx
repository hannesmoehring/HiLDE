import { useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import {
  counterfactualRowsUrl,
  datasetColumns,
  getMode,
  listCachedRuns,
  listDatasets,
  runAnalysis,
} from "./api";
import { defaultRun, nearestRun, runFacets } from "./cachedRuns";
import { ClusterScatter } from "./charts/ClusterScatter";
import {
  ClusterMovementPanel,
  MovementStartRow,
} from "./components/ClusterMovementPanel";
import { ConfigPanel } from "./components/ConfigPanel";
import { ExplorationPanel } from "./components/ExplorationPanel";
import { LayerSide } from "./components/LayerSide";
import { OutlierPanel } from "./components/OutlierPanel";
import { DEFAULT_CONFIG } from "./config";
import { useCounterfactual, type CounterfactualController } from "./counterfactual";
import { movementPropsFor, useMovement } from "./movement";
import { getNodeAtPath } from "./treeNav";
import type {
  AnalysisConfig,
  AnalysisResponse,
  CachedRun,
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
  const analysisRef = useRef(analysis);
  analysisRef.current = analysis;
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

  // Counterfactual session: the stack of applied movements and the ONE dataset
  // key every request goes out with. A rebuild on a counterfactual key uses the
  // run's own feature columns and built config — never the rail's unsaved knobs
  // — and commits the new run together with the stack, then goes to root:
  // node ids like `root/1` are positional and may name entirely different rows
  // after a rebuild, so "the same path" would preserve nothing.
  const cf = useCounterfactual({
    baseDataset: datasetKey,
    rebuild: (key) => {
      const meta = analysisRef.current?.meta;
      if (!meta) return Promise.reject(new Error("there is no run on screen to rebuild"));
      return runAnalysis(key, meta.feature_cols, meta.config as Partial<AnalysisConfig>, true);
    },
    onSwitched: (res) => {
      setAnalysis(res);
      navigate([]);
    },
  });
  const cfBusy = cf.phase === "applying" || cf.phase === "rebuilding";
  const cfHead = cf.stack.length ? cf.stack[cf.stack.length - 1] : null;

  // The config rail collapses to a 38px strip so the analysis canvas can widen.
  const [configOpen, setConfigOpen] = useState(true);

  // Hosting mode only: reuse stored runs, and say so when we do.
  const [mode, setMode] = useState<ModeInfo | null>(null);
  const [useCache, setUseCache] = useState(true);

  // Cache-only server: nothing can be built, so the stored runs are the only
  // inputs there are. `null` until the listing is in.
  const cacheOnly = mode?.cache_only ?? false;
  const [cachedRuns, setCachedRuns] = useState<CachedRun[] | null>(null);
  const [runPick, setRunPick] = useState<CachedRun | null>(null);
  const runsHere = useMemo(
    () => (cachedRuns ?? []).filter((r) => r.dataset === datasetKey),
    [cachedRuns, datasetKey],
  );
  const facets = useMemo(() => runFacets(runsHere), [runsHere]);
  const firstRun = useMemo(() => defaultRun(runsHere), [runsHere]);
  // A dataset switch leaves `runPick` on the old dataset; fall to this one's default.
  const selectedRun = runPick && runsHere.includes(runPick) ? runPick : firstRun;

  useEffect(() => {
    getMode()
      .then(setMode)
      .catch(() =>
        setMode({ hosting: false, cache_dir: null, cache_only: false, maintenance: false }),
      );
  }, []);

  // Waits for the mode: a cache-only server offers the datasets that have a stored
  // run, not the registry.
  useEffect(() => {
    if (!mode || mode.maintenance) return; // under maintenance nothing is requested
    if (mode.cache_only) {
      listCachedRuns()
        .then((runs) => {
          const keys = Array.from(new Set(runs.map((r) => r.dataset)));
          setCachedRuns(runs);
          setDatasets(keys.map((k) => ({ key: k, label: k })));
          if (keys.length) setDatasetKey(keys[0]);
        })
        .catch((e) => setError(String(e)));
      return;
    }
    listDatasets()
      .then((d) => {
        setDatasets(d);
        if (d.length) setDatasetKey(d[0].key);
      })
      .catch((e) => setError(String(e)));
  }, [mode]);

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

  // Cache-only: the selected stored run *is* the configuration. Once its dataset's
  // columns are in (the effect above has just put that dataset's defaults on the
  // rail), replace them with the run's own features and knobs and show the run.
  // The request repeats the run's stored signature verbatim, so it is a cache hit.
  useEffect(() => {
    if (!selectedRun || columns?.key !== selectedRun.dataset) return;
    let stale = false;
    setFeatureCols(selectedRun.feature_cols);
    setConfig({ ...DEFAULT_CONFIG, ...selectedRun.config });
    setLoading(true);
    setError(null);
    setAnalysis(null);
    navigate([]);
    runAnalysis(selectedRun.dataset, selectedRun.feature_cols, selectedRun.config)
      .then((res) => {
        if (!stale) setAnalysis(res);
      })
      .catch((e) => {
        if (!stale) setError(String(e));
      })
      .finally(() => {
        if (!stale) setLoading(false);
      });
    return () => {
      stale = true;
      setLoading(false);
    };
  }, [selectedRun, columns]);

  const maxDims = useMemo(() => Math.max(2, featureCols.length), [featureCols]);

  // Label columns held out of the feature space (see backend default_feature_cols).
  // A `target_*` column the user checked in as a feature is not one: it is in the
  // predicate now, and reporting it twice would say otherwise.
  const targetCols = useMemo(
    () =>
      (columns?.columns ?? []).filter(
        (c) => c.startsWith("target_") && !featureCols.includes(c),
      ),
    [columns, featureCols],
  );

  function patchConfig(patch: Partial<AnalysisConfig>) {
    setConfig((c) => ({ ...c, ...patch }));
  }

  function toggleFeature(col: string) {
    setFeatureCols((f) =>
      f.includes(col) ? f.filter((x) => x !== col) : [...f, col],
    );
  }

  async function build() {
    if (!datasetKey || featureCols.length === 0) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    navigate([]);
    try {
      // On a counterfactual session the rail rebuilds the counterfactual data:
      // the edits are the data now, the rail only changes how it is analysed.
      const res = await runAnalysis(cf.activeDatasetKey, featureCols, config, useCache);
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
  const shownAnalysis =
    analysis && analysis.meta.dataset === cf.activeDatasetKey ? analysis : null;

  const runMeta = [
    datasetKey || "no dataset",
    `${featureCols.length} features`,
    config.method,
  ]
    .filter(Boolean)
    .join(", ");
  const cfgSummary = [
    config.method,
    `${config.hierarchical_layers} layers`,
    `mcs ${config.hclust_min_cluster_size}`,
  ].join(", ");

  if (mode?.maintenance) {
    return (
      <div className="app">
        <header className="topbar">
          <span className="topbar__brand">HiLDE</span>
        </header>
        <main className="canvas">
          <div className="banner" role="alert">
            <strong>Maintenance</strong>
            <span>
              HiLDE is under maintenance and does not work right now. Please check back
              later.
            </span>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="topbar__brand">HiLDE</span>
        <span className="topbar__sub">
          <b>Hi</b>erarchical <b>L</b>ocal <b>D</b>ecomposition &amp; <b>E</b>
          xplanation
        </span>
        <a
          className="topbar__video"
          href="https://youtu.be/BHsBJ5zTIYA"
          target="_blank"
          rel="noreferrer"
        >
          ▶ Intro video
        </a>
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
                    <span>
                      Source
                      {cf.stack.length > 0 && (
                        <span
                          className="cf-badge"
                          title={`Counterfactual data: ${cf.stack.length} edit(s) applied on top of ${datasetKey}. Changing the dataset resets the session.`}
                        >
                          cf
                        </span>
                      )}
                    </span>
                    {/* Locked during a build: swapping datasets mid-run would apply the
                        in-flight tree under the new dataset's key. Also locked while
                        a counterfactual apply/rebuild is out, for the same reason. */}
                    <select
                      value={datasetKey}
                      onChange={(e) => setDatasetKey(e.target.value)}
                      disabled={loading || cfBusy}
                    >
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

                {cacheOnly && (
                  <section className="cfg__block">
                    <h3>Cached run</h3>
                    <p className="hint cfg__note">
                      {runsHere.length} stored for this dataset
                      {facets.length > 0 && ". Choose among them by the settings they differ in"}
                      . The full settings of the run on screen are below, read-only.
                    </p>
                    {/* A plain wrapper: the rail gives a block's first direct field
                        the stacked layout, and these are all label-and-value rows. */}
                    <div>
                      {selectedRun &&
                        facets.map((f) => (
                          <label className="field" key={f.label}>
                            <span>{f.label}</span>
                            <select
                              value={f.valueOf(selectedRun)}
                              onChange={(e) =>
                                setRunPick(
                                  nearestRun(runsHere, facets, selectedRun, f, e.target.value),
                                )
                              }
                            >
                              {f.values.map((v) => (
                                <option key={v} value={v}>
                                  {v === "true" ? "yes" : v === "false" ? "no" : v}
                                </option>
                              ))}
                            </select>
                          </label>
                        ))}
                    </div>
                  </section>
                )}

                {columns && (
                  <section className="cfg__block">
                    <div className="cfg__head">
                      <h3>Feature columns</h3>
                      <span className="cfg__count">
                        {featureCols.length}/{columns.columns.length - 1}
                      </span>
                      <div className="cfg__actions">
                        <button
                          onClick={() =>
                            setFeatureCols(columns.default_feature_cols)
                          }
                          disabled={cacheOnly}
                        >
                          Reset
                        </button>
                        <button onClick={() => setFeatureCols([])} disabled={cacheOnly}>
                          None
                        </button>
                      </div>
                    </div>
                    <div className="feature-picker__list">
                      {columns.columns
                        .filter((c) => c !== "row_id")
                        .map((c) => (
                          <label
                            key={c}
                            className={
                              featureCols.includes(c) ? undefined : "is-off"
                            }
                          >
                            <input
                              type="checkbox"
                              checked={featureCols.includes(c)}
                              onChange={() => toggleFeature(c)}
                              disabled={cacheOnly}
                            />
                            {c}
                          </label>
                        ))}
                    </div>
                    <label
                      className="field--check"
                      style={{ marginTop: "0.6rem", marginBottom: 0 }}
                    >
                      <input
                        type="checkbox"
                        checked={charNonFeatureOnly}
                        onChange={(e) =>
                          setCharNonFeatureOnly(e.target.checked)
                        }
                      />
                      <span>Characteristics: non-feature columns only</span>
                    </label>
                  </section>
                )}

                {/* Cache-only: the knobs describe the stored run and cannot be changed. */}
                <fieldset className="cfg__lock" disabled={cacheOnly}>
                  <ConfigPanel
                    config={config}
                    maxDims={maxDims}
                    onChange={patchConfig}
                  />
                </fieldset>
              </div>

              <div className="cfg__build">
                {/* `cfBusy` gates this the way it gates the dataset selector and
                    Undo/Reset. Build is the one control that can move App state
                    from under an apply that is already out: it nulls `analysis`,
                    which the pending rebuild reads, and it commits a run built on
                    the key it captured. The counterfactual hook's stale check
                    cannot see either — Build changes no base, generation or head. */}
                <button
                  className="primary"
                  onClick={build}
                  disabled={loading || featureCols.length === 0 || cfBusy || cacheOnly}
                  title={
                    cacheOnly
                      ? "Disabled on this server — only cached runs are available"
                      : cfBusy
                        ? "Waiting for the counterfactual rebuild"
                        : undefined
                  }
                >
                  {loading ? "Building…" : "Build & Apply"}
                </button>
                {mode?.hosting && !cacheOnly && (
                  <label
                    className="field--check"
                    style={{ marginBottom: 0 }}
                    title={mode.cache_dir ?? undefined}
                  >
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
              <button
                onClick={() => setConfigOpen(true)}
                title="Show configuration"
              >
                ›
              </button>
              <span className="rail__vert">Configuration</span>
              <span className="rail__vert rail__vert--faint">{cfgSummary}</span>
            </div>
          )}
        </aside>

        <main className="canvas">
          {cacheOnly && (
            <div className="banner" role="note">
              <strong>Cached only</strong>
              <span>
                Only cached inputs are available at the moment: this server does not
                have the computing capacity to build new runs. Pick a dataset and one
                of its stored runs under <em>Configuration</em>.
                {cachedRuns?.length === 0 && " No stored runs were found on this server."}
              </span>
            </div>
          )}

          {/* Failures report on the canvas, not in the rail: the rail collapses to a
              38px strip and would otherwise swallow the only sign anything went wrong. */}
          {error && (
            <div className="banner banner--error" role="alert">
              <strong>Error</strong>
              <span>{error}</span>
              {!configOpen && (
                <button onClick={() => setConfigOpen(true)}>
                  Show configuration
                </button>
              )}
            </div>
          )}

          {cf.phase === "applying" && (
            <div className="banner" role="status">
              <strong>Applying</strong>
              <span>Writing the movement into a counterfactual copy of the dataset…</span>
            </div>
          )}
          {cf.phase === "rebuilding" && (
            <div className="banner" role="status">
              <strong>Rebuilding</strong>
              <span>
                Rebuilding the analysis on the counterfactual data… the run below stays on
                screen until the new one is ready.
              </span>
            </div>
          )}
          {cf.phase === "error" && cf.error && (
            <div className="banner banner--error" role="alert">
              <strong>Counterfactual</strong>
              <span>{cf.error} — the run on screen is unchanged.</span>
            </div>
          )}
          {cfHead && (
            <div className="banner banner--cf">
              <strong>Counterfactual data</strong>
              <span className="banner__chips">
                {cf.stack.length} edit{cf.stack.length === 1 ? "" : "s"} applied ·
                {cfHead.edits.map((e) => (
                  <span
                    key={e.cf_id}
                    className="banner__chip"
                    title={`${e.node_id}: C${e.source_child_index} → ${
                      e.target.kind === "cluster"
                        ? `C${e.target.child_index}`
                        : `point (${e.target.x.toFixed(2)}, ${e.target.y.toFixed(2)})`
                    } at ${Math.round(e.strength * 100)}% — ${e.n_rows} rows, ${e.features_changed.length} feature(s): ${e.features_changed.join(", ")}`}
                  >
                    C{e.source_child_index} →{" "}
                    {e.target.kind === "cluster" ? `C${e.target.child_index}` : "point"} @{" "}
                    {Math.round(e.strength * 100)}%
                  </span>
                ))}
              </span>
              <span className="banner__note">
                Hierarchy rebuilt on counterfactual data — cluster numbering does not
                correspond to the previous run.
              </span>
              <button onClick={cf.undo} disabled={cfBusy} title="Back to the previous data">
                Undo
              </button>
              <button onClick={cf.reset} disabled={cfBusy} title="Back to the original data">
                Reset
              </button>
              <a
                className="button-link"
                href={counterfactualRowsUrl(cfHead.cfId)}
                download={`counterfactual_${cfHead.cfId}.csv`}
                title="The changed rows: row_id, original and counterfactual values"
              >
                Export rows
              </a>
            </div>
          )}

          {mode?.hosting && !cacheOnly && shownAnalysis?.cached && (
            <div className="banner">
              <strong>Cached</strong>
              <span>
                This dataset and configuration were computed before, so the
                stored run was reused — nothing was recomputed. Uncheck{" "}
                <em>Use cached results</em> and rebuild to force a fresh run.
              </span>
            </div>
          )}

          {loading && (
            <div className="empty">
              {cacheOnly ? "Loading the stored run …" : "Reducing, clustering and scoring …"}
            </div>
          )}

          {shownAnalysis && (
            // Keyed on the active dataset key: a switch remounts the whole layer
            // stack, so the movement machine starts idle against the new run in
            // the very render that shows it.
            <Navigation
              key={cf.activeDatasetKey}
              analysis={shownAnalysis}
              treePath={treePath}
              exploreWhole={exploreWhole}
              navigate={navigate}
              dataset={cf.activeDatasetKey}
              counterfactual={cf}
              featureCols={featureCols}
              targetCols={targetCols}
              config={config}
              charNonFeatureOnly={charNonFeatureOnly}
              imageSpec={columns?.image ?? null}
              cacheOnly={cacheOnly}
            />
          )}

          {!shownAnalysis && !loading && (
            <div className="empty">
              {cacheOnly
                ? "Pick a stored run under Configuration."
                : "Pick features and press Build & Apply to compute a run."}
            </div>
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
  counterfactual: CounterfactualController;
  featureCols: string[];
  targetCols: string[];
  config: AnalysisConfig;
  charNonFeatureOnly: boolean;
  imageSpec: ImageSpec | null;
  cacheOnly: boolean;
}) {
  const {
    analysis,
    treePath,
    exploreWhole,
    navigate,
    dataset,
    counterfactual,
    featureCols,
    targetCols,
    config,
    charNonFeatureOnly,
    imageSpec,
    cacheOnly,
  } = props;
  const root = analysis.tree;
  const nLayers = config.hierarchical_layers;
  // What the shown embedding was computed with. The rail's Method knob can be changed
  // without rebuilding, so it describes the *next* run, not the coordinates on screen.
  const builtMethod =
    typeof analysis.meta.config.method === "string"
      ? analysis.meta.config.method
      : config.method;

  // Cluster movement — a counterfactual preview, not an edit. The hook owns the
  // whole state machine and clears itself on a dataset / analysis / path change,
  // so nothing here has to remember to tear it down.
  const movement = useMovement({ analysis, datasetKey: dataset, treePath });

  // Apply is enabled only when the preview on screen is exactly what the server
  // would write (`movement.settled`) and no counterfactual apply/rebuild is out.
  // A failed apply leaves the button usable again.
  const cfBusy =
    counterfactual.phase === "applying" || counterfactual.phase === "rebuilding";
  // A cache-only server never applies: the rebuild that follows is a new run.
  const applyDisabledReason: string | null = cacheOnly
    ? "not available on this server — it needs a rebuild, and only cached runs are served"
    : cfBusy
      ? "a counterfactual apply or rebuild is in progress"
      : !analysis.meta.analysis_id
        ? "this run predates movement support — rebuild it first"
        : movement.state.phase !== "ready"
          ? "pick a destination and wait for the preview"
          : !movement.settled
            ? "waiting for the preview at this strength"
            : null;
  const onApply = () => {
    const s = movement.state;
    if (s.phase !== "ready" || applyDisabledReason) return;
    counterfactual.apply({
      analysis_id: analysis.meta.analysis_id ?? "",
      dataset,
      feature_cols: analysis.meta.feature_cols,
      config: analysis.meta.config,
      node_id: s.nodeId,
      source_child_index: s.sourceChildIndex,
      target: s.target,
      strength: s.strength,
    });
  };
  // Point picked in a layer's GLOSH outlier table; ringed in that layer's scatter.
  // One at a time across layers, cleared whenever we drill in or out.
  const [outlierPick, setOutlierPick] = useState<{
    layer: number;
    rowId: number;
  } | null>(null);
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
    // A running movement belongs to exactly one node. Its reading — controls,
    // tiles, feature table — takes a full-width row of THAT layer card, under
    // both columns; every other layer only ever shows the start button.
    const movingHere =
      movement.state.phase !== "idle" && movement.state.nodeId === node.id;
    layerViews.push(
      <section className="panel layer" key={`layer-${L}`}>
        <div className="panel__head">
          <span className="kicker">Layer {L}</span>
          <span className="panel__title">
            {L === 1
              ? "Cluster projection — root"
              : `Sub-projection — layer ${L}`}
          </span>
          <span className="panel__meta">
            {node.row_indices.length} points, {node.children?.length ?? 0}{" "}
            clusters
          </span>
        </div>
        <div className="layer__cols">
          <div>
            <ClusterScatter
              node={node}
              selectedChild={selectedChild}
              onSelectCluster={(i) => navigate([...parentPath, i])}
              highlightRow={outlierPick?.layer === L ? outlierPick.rowId : null}
              {...movementPropsFor(movement, node.id)}
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
                {exploringHere
                  ? "Exploring entire layer"
                  : "Explore entire layer"}
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
            <MovementStartRow
              nodeId={node.id}
              selectedChildIndex={selectedChild}
              builtMethod={builtMethod}
              cacheOnly={cacheOnly}
              active={movingHere}
              onStart={movement.start}
            />
          </div>
        </div>
        {movingHere && (
          <div className="layer__movement">
            <ClusterMovementPanel
              state={movement.state}
              preview={movement.preview}
              builtMethod={builtMethod}
              onStrength={movement.setStrength}
              onCancel={movement.cancel}
              onApply={onApply}
              applyDisabledReason={applyDisabledReason}
              applying={cfBusy}
            />
          </div>
        )}
        <OutlierPanel
          node={node}
          dataset={dataset}
          selectedRow={outlierPick?.layer === L ? outlierPick.rowId : null}
          onSelectRow={(rowId) =>
            setOutlierPick(rowId == null ? null : { layer: L, rowId })
          }
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

  if (!waiting && explorationPath === null)
    explorationPath = treePath.slice(0, nLayers);

  const explorationNode =
    explorationPath !== null ? getNodeAtPath(root, explorationPath) : null;
  const pathLabel =
    explorationPath && explorationPath.length
      ? explorationPath.map((c) => `C${c}`).join(" → ")
      : "root";
  // The explored node is whatever the deepest layer holds as its selected child, so
  // that layer already reports its scores. A non-empty path says so either way: under
  // "explore entire layer" the node is the layer's own, but a layer's own node is the
  // layer above's selected child, and that side column reports it. Only an empty path
  // has nothing above it — a leaf root, or the whole of layer 1 — and there the
  // exploration panel is the only place the scores can appear.
  const scoresShownByLayer =
    explorationPath !== null && explorationPath.length > 0;

  return (
    <>
      {layerViews}
      {waiting && (
        <div className="empty">
          Click a cluster in the projection above to drill in.
        </div>
      )}
      {explorationNode && (
        <ExplorationPanel
          dataset={dataset}
          featureCols={featureCols}
          targetCols={targetCols}
          config={config}
          builtMethod={builtMethod}
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

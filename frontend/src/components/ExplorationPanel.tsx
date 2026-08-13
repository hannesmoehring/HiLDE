// Leaf exploration: scatter + selection -> predicate bands + selected-points table,
// plus an interactive feature-range filter mode.
// Parity with src/ui/components/exploration.py::render_cluster_exploration.
import { useEffect, useMemo, useState } from "react";
import { fetchRows, fetchSelectionCharacteristics, fetchTargets, runPredicate } from "../api";
import { CharacteristicsBar } from "../charts/CharacteristicsBar";
import { PcaVarianceBar } from "../charts/PcaVarianceBar";
import { PredicateBands } from "../charts/PredicateBands";
import { ProjectionScatter } from "../charts/ProjectionScatter";
import { ScoreTiles } from "../charts/ScoreTiles";
import { TargetBands } from "../charts/TargetBands";
import type {
  AnalysisConfig,
  Characteristic,
  ImageSpec,
  PredicateResponse,
  PredicateScope,
  RowsResponse,
  TargetsResponse,
  TreeNode,
} from "../types";
import { PointImage } from "./PointImage";

interface Props {
  dataset: string;
  featureCols: string[];
  targetCols: string[]; // `target_*` label columns — reported, never predicated on
  config: AnalysisConfig;
  node: TreeNode;
  pathLabel: string;
  imageSpec: ImageSpec | null; // non-null = table rows can be opened as images
  // The layer above already reports this node's scores in its side column, so they
  // are shown here only when there is no layer above — i.e. the root is a leaf.
  showScores: boolean;
  charNonFeatureOnly: boolean;
}

// The predicate describes the current selection; the characteristics describe the
// whole node, so they answer different questions about different point sets.
type View = "predicate" | "characteristics";

// Per-feature standardized (z-score) matrix for the node's rows, computed in the
// browser — mirrors the fresh StandardScaler in Streamlit's compute_data_layer.
interface ZData {
  cols: string[];
  Z: number[][];
  bounds: [number, number][]; // [min,max] of each standardized column
}

function standardize(cols: string[], rows: Record<string, unknown>[]): ZData {
  const n = rows.length || 1;
  const raw = rows.map((r) => cols.map((c) => Number(r[c]) || 0));
  const mean = cols.map((_, j) => raw.reduce((s, row) => s + row[j], 0) / n);
  const std = cols.map((_, j) => {
    const m = mean[j];
    const v = raw.reduce((s, row) => s + (row[j] - m) ** 2, 0) / n;
    return Math.sqrt(v) || 1;
  });
  const Z = raw.map((row) => row.map((v, j) => (v - mean[j]) / std[j]));
  const bounds = cols.map((_, j) => {
    const col = Z.map((r) => r[j]);
    return [Math.min(...col), Math.max(...col)] as [number, number];
  });
  return { cols, Z, bounds };
}

// Target columns sit at the right of the selected-points table, fenced off from
// the feature columns so nobody reads a label as something the predicate used.
function cellClass(col: string, targets: Set<string>, first: string | undefined): string | undefined {
  if (!targets.has(col)) return undefined;
  return col === first ? "is-target is-target-first" : "is-target";
}

function toCsv(rows: RowsResponse): string {
  const header = rows.columns.join(",");
  const body = rows.rows
    .map((r) => rows.columns.map((c) => JSON.stringify(r[c] ?? "")).join(","))
    .join("\n");
  return `${header}\n${body}`;
}

export function ExplorationPanel({
  dataset,
  featureCols,
  targetCols,
  config,
  node,
  pathLabel,
  imageSpec,
  showScores,
  charNonFeatureOnly,
}: Props) {
  const [selected, setSelected] = useState<number[]>([]);
  const [view, setView] = useState<View>("predicate");
  const [scope, setScope] = useState<PredicateScope>("global");
  const [predicate, setPredicate] = useState<PredicateResponse | null>(null);
  const [charSel, setCharSel] = useState<Characteristic[] | null>(null);
  const [charFailed, setCharFailed] = useState(false);
  const [targets, setTargets] = useState<TargetsResponse | null>(null);
  const [rows, setRows] = useState<RowsResponse | null>(null);
  const [imageRow, setImageRow] = useState<number | null>(null); // dataframe row id, not a table position

  // Interactive feature-range filter mode.
  const [interactive, setInteractive] = useState(false);
  const [zData, setZData] = useState<ZData | null>(null);
  const [filterFeatures, setFilterFeatures] = useState<string[]>([]);
  const [ranges, setRanges] = useState<Record<string, [number, number]>>({});

  // Reset everything when the explored node changes.
  useEffect(() => {
    setSelected([]);
    setPredicate(null);
    setCharSel(null);
    setTargets(null);
    setRows(null);
    setZData(null);
    setFilterFeatures([]);
    setRanges({});
  }, [node.id]);

  // A new selection rebuilds the table, so the row the image was opened from is gone.
  useEffect(() => {
    setImageRow(null);
  }, [selected]);

  // Fetch + standardize the node's feature values when interactive mode is on.
  useEffect(() => {
    if (!interactive) {
      setZData(null);
      return;
    }
    let cancelled = false;
    fetchRows(dataset, node.row_indices, featureCols)
      .then((r) => !cancelled && setZData(standardize(featureCols, r.rows)))
      .catch(() => !cancelled && setZData(null));
    return () => {
      cancelled = true;
    };
  }, [interactive, node.id, dataset, featureCols]);

  // "Matches filters" / "Other" per point (interactive mode only).
  const interactiveGroup = useMemo(() => {
    if (!interactive || !zData) return null;
    return zData.Z.map((row) =>
      filterFeatures.every((f) => {
        const j = zData.cols.indexOf(f);
        if (j < 0) return true;
        const [lo, hi] = ranges[f] ?? zData.bounds[j];
        return row[j] >= lo && row[j] <= hi;
      })
        ? "Matches filters"
        : "Other",
    );
  }, [interactive, zData, filterFeatures, ranges]);

  // In interactive mode the filter *is* the selection (drives the table).
  useEffect(() => {
    if (interactive && interactiveGroup) {
      setSelected(
        interactiveGroup.flatMap((g, i) =>
          g === "Matches filters" ? [i] : [],
        ),
      );
    }
  }, [interactive, interactiveGroup]);

  // The table shows the features the predicate speaks about, then the labels it
  // deliberately ignores — same order the header/CSV are marked up in.
  const tableCols = useMemo(() => [...featureCols, ...targetCols], [featureCols, targetCols]);
  const targetSet = useMemo(() => new Set(targetCols), [targetCols]);
  const firstTarget = targetCols[0];

  // Selection -> predicate (skipped in interactive mode) + target values + rows table.
  useEffect(() => {
    if (selected.length === 0) {
      setPredicate(null);
      setTargets(null);
      setRows(null);
      return;
    }
    let cancelled = false;
    if (!interactive) {
      runPredicate({
        dataset,
        feature_cols: featureCols,
        config,
        row_indices: node.row_indices,
        selected_local_indices: selected,
        scope,
      })
        .then((p) => !cancelled && setPredicate(p))
        .catch(() => !cancelled && setPredicate(null));
    } else {
      setPredicate(null);
    }
    if (targetCols.length > 0) {
      fetchTargets({
        dataset,
        target_cols: targetCols,
        row_indices: node.row_indices,
        selected_local_indices: selected,
      })
        .then((t) => !cancelled && setTargets(t))
        .catch(() => !cancelled && setTargets(null));
    }
    fetchRows(
      dataset,
      selected.map((i) => node.row_indices[i]),
      tableCols,
    )
      .then((r) => !cancelled && setRows(r))
      .catch(() => !cancelled && setRows(null));
    return () => {
      cancelled = true;
    };
  }, [selected, scope, interactive, node.id, dataset, featureCols, targetCols, tableCols, config]);

  // Selection -> characteristics, on its own so the cost is only paid while the tab
  // is open. Unlike the predicate this holds in interactive mode too: the filtered
  // points are a selection like any other. Clearing first matters even when the tab
  // is hidden, or reopening it flashes the previous selection's numbers.
  useEffect(() => {
    setCharSel(null);
    setCharFailed(false);
    if (view !== "characteristics" || selected.length === 0) return;
    let cancelled = false;
    fetchSelectionCharacteristics({
      dataset,
      feature_cols: featureCols,
      row_indices: node.row_indices,
      selected_local_indices: selected,
    })
      .then((c) => !cancelled && setCharSel(c.characteristics))
      .catch(() => !cancelled && setCharFailed(true));
    return () => {
      cancelled = true;
    };
  }, [view, selected, node.id, dataset, featureCols]);

  const variance = (node.embedding_original_variance ?? []).filter(
    (v): v is number => v !== null,
  );
  const showVariance = config.method === "PCA" && variance.length > 0;

  function exportCsv() {
    if (!rows) return;
    const blob = new Blob([toCsv(rows)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "selected_points.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="exploration panel">
      <h2>
        <span className="kicker">Exploration</span>
        <span className="panel__title">{pathLabel}</span>
      </h2>
      {(showScores || showVariance) && (
        <div className="exploration__summary">
          {showScores && <ScoreTiles scores={node.scores} title="DR quality — this cluster" />}
          {showVariance && <PcaVarianceBar explainedVariance={variance} />}
        </div>
      )}

      <div className="exploration__cols">
        <div className="exploration__analysis">
          <label className="field--check" style={{ marginBottom: "0.35rem" }}>
            <input
              type="checkbox"
              checked={interactive}
              onChange={(e) => setInteractive(e.target.checked)}
            />
            <span>Interactive feature ranges</span>
          </label>

          <div className="tabs" role="tablist" aria-label="Selection explanation">
            <button
              role="tab"
              aria-selected={view === "predicate"}
              className={view === "predicate" ? "is-active" : undefined}
              onClick={() => setView("predicate")}
            >
              Predicate
            </button>
            <button
              role="tab"
              aria-selected={view === "characteristics"}
              className={view === "characteristics" ? "is-active" : undefined}
              onClick={() => setView("characteristics")}
            >
              Characteristics
            </button>
          </div>

          {/* One bounded viewport for both tabs. A wide dataset yields a band per
              feature — 784 of them on MNIST — which would otherwise run the page
              on for thousands of pixels below the plot. */}
          <div className="exploration__view" role="tabpanel">
            {view === "characteristics" ? (
              selected.length === 0 ? (
                <p className="hint">
                  Use lasso or box selection in the plot to capture points.
                </p>
              ) : charFailed ? (
                <p className="hint">Could not compute characteristics for this selection.</p>
              ) : charSel === null ? (
                <p className="hint">Computing characteristics…</p>
              ) : (
                <CharacteristicsBar
                  data={charSel}
                  title={`Selection characteristics — vs. ${pathLabel}`}
                  nonFeatureOnly={charNonFeatureOnly}
                />
              )
            ) : interactive ? (
              <InteractiveFilters
                zData={zData}
                features={filterFeatures}
                ranges={ranges}
                onFeatures={setFilterFeatures}
                onRange={(f, r) => setRanges((prev) => ({ ...prev, [f]: r }))}
                matched={selected.length}
              />
            ) : selected.length === 0 ? (
              <p className="hint">
                Use lasso or box selection in the plot to capture points.
              </p>
            ) : (
              <>
                <div className="scope-toggle">
                  <label>
                    <input
                      type="radio"
                      checked={scope === "global"}
                      onChange={() => setScope("global")}
                    />
                    Whole dataset (global)
                  </label>
                  <label>
                    <input
                      type="radio"
                      checked={scope === "local"}
                      onChange={() => setScope("local")}
                    />
                    This cluster (local)
                  </label>
                </div>
                {predicate?.summary && (
                  <div className="predicate-summary">
                    <span>
                      Predicate F1: {predicate.summary.predicate_f1.toFixed(2)}
                    </span>
                    <span>
                      Features used: {predicate.summary.n_features_used} /{" "}
                      {predicate.summary.n_features_total}
                    </span>
                    <span>Selected: {predicate.summary.n_selected}</span>
                  </div>
                )}
                {predicate && (
                  <PredicateBands
                    full={predicate.full}
                    trimmed={predicate.trimmed}
                  />
                )}
              </>
            )}
          </div>

          {targets && targets.targets.length > 0 && selected.length > 0 && (
            <div className="target-values">
              <h4>Target values — selection</h4>
              <TargetBands targets={targets.targets} nSelected={targets.n_selected} />
            </div>
          )}
        </div>

        <div className="exploration__plot">
          <ProjectionScatter
            points={node.embedding_original}
            rowIds={node.row_indices}
            method={config.method}
            interactiveGroup={interactive ? interactiveGroup : null}
            onSelect={interactive ? () => {} : setSelected}
            selected={interactive ? [] : selected}
          />
        </div>
      </div>

      <div className="exploration__table">
        <h3>Selected points: {selected.length}</h3>
        {rows && rows.rows.length > 0 && (
          <>
            <button onClick={exportCsv}>Export selected points to CSV</button>
            {imageSpec && (
              <p className="hint">Click a row to see the image behind that point.</p>
            )}
            <div className="exploration__rows">
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {rows.columns.map((c) => (
                        <th key={c} className={cellClass(c, targetSet, firstTarget)}>
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {/* Row i of the table is selected[i] of the node, i.e. dataframe row
                        node.row_indices[selected[i]] — the id /api/rows was asked for. */}
                    {rows.rows.slice(0, 50).map((r, i) => {
                      const rowId = node.row_indices[selected[i]];
                      const active = imageSpec != null && imageRow === rowId;
                      return (
                        <tr
                          key={i}
                          className={imageSpec ? (active ? "is-pickable is-active" : "is-pickable") : undefined}
                          onClick={imageSpec ? () => setImageRow(active ? null : rowId) : undefined}
                        >
                          {rows.columns.map((c) => (
                            <td key={c} className={cellClass(c, targetSet, firstTarget)}>
                              {String(r[c])}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {imageSpec && imageRow != null && (
                <PointImage
                  dataset={dataset}
                  rowId={imageRow}
                  onClose={() => setImageRow(null)}
                />
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function InteractiveFilters(props: {
  zData: ZData | null;
  features: string[];
  ranges: Record<string, [number, number]>;
  onFeatures: (f: string[]) => void;
  onRange: (f: string, r: [number, number]) => void;
  matched: number;
}) {
  const { zData, features, ranges, onFeatures, onRange, matched } = props;
  if (!zData) return <p className="hint">Loading feature values…</p>;

  return (
    <div className="interactive-filters">
      <p className="hint">
        Standardized (z-score) ranges. Matching points are highlighted in the
        plot — {matched} match.
      </p>
      <label className="field">
        <span>Features to filter</span>
        <select
          multiple
          value={features}
          size={Math.min(6, zData.cols.length)}
          onChange={(e) =>
            onFeatures(Array.from(e.target.selectedOptions, (o) => o.value))
          }
        >
          {zData.cols.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
      {features.map((f) => {
        const j = zData.cols.indexOf(f);
        const [bMin, bMax] = zData.bounds[j] ?? [-3, 3];
        const [lo, hi] = ranges[f] ?? [bMin, bMax];
        return (
          <div key={f} className="range-row">
            <span className="range-row__label">{f}</span>
            <input
              type="range"
              min={bMin}
              max={bMax}
              step={(bMax - bMin) / 100 || 0.01}
              value={lo}
              onChange={(e) =>
                onRange(f, [Math.min(Number(e.target.value), hi), hi])
              }
            />
            <input
              type="range"
              min={bMin}
              max={bMax}
              step={(bMax - bMin) / 100 || 0.01}
              value={hi}
              onChange={(e) =>
                onRange(f, [lo, Math.max(Number(e.target.value), lo)])
              }
            />
            <span className="range-row__vals">
              [{lo.toFixed(2)}, {hi.toFixed(2)}]
            </span>
          </div>
        );
      })}
    </div>
  );
}

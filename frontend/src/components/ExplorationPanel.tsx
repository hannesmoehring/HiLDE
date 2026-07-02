// Leaf exploration: scatter + selection -> predicate bands + selected-points table,
// plus an interactive feature-range filter mode.
// Parity with src/ui/components/exploration.py::render_cluster_exploration.
import { useEffect, useMemo, useState } from "react";
import { fetchRows, runPredicate } from "../api";
import { CharacteristicsBar } from "../charts/CharacteristicsBar";
import { PcaVarianceBar } from "../charts/PcaVarianceBar";
import { PredicateBands } from "../charts/PredicateBands";
import { ProjectionScatter } from "../charts/ProjectionScatter";
import { ScoreTiles } from "../charts/ScoreTiles";
import type {
  AnalysisConfig,
  PredicateResponse,
  PredicateScope,
  RowsResponse,
  TreeNode,
} from "../types";

interface Props {
  dataset: string;
  featureCols: string[];
  config: AnalysisConfig;
  node: TreeNode;
  pathLabel: string;
}

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

function toCsv(rows: RowsResponse): string {
  const header = rows.columns.join(",");
  const body = rows.rows
    .map((r) => rows.columns.map((c) => JSON.stringify(r[c] ?? "")).join(","))
    .join("\n");
  return `${header}\n${body}`;
}

export function ExplorationPanel({ dataset, featureCols, config, node, pathLabel }: Props) {
  const [selected, setSelected] = useState<number[]>([]);
  const [scope, setScope] = useState<PredicateScope>("local");
  const [predicate, setPredicate] = useState<PredicateResponse | null>(null);
  const [rows, setRows] = useState<RowsResponse | null>(null);

  // Interactive feature-range filter mode.
  const [interactive, setInteractive] = useState(false);
  const [zData, setZData] = useState<ZData | null>(null);
  const [filterFeatures, setFilterFeatures] = useState<string[]>([]);
  const [ranges, setRanges] = useState<Record<string, [number, number]>>({});

  // Reset everything when the explored node changes.
  useEffect(() => {
    setSelected([]);
    setPredicate(null);
    setRows(null);
    setZData(null);
    setFilterFeatures([]);
    setRanges({});
  }, [node.id]);

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
      setSelected(interactiveGroup.flatMap((g, i) => (g === "Matches filters" ? [i] : [])));
    }
  }, [interactive, interactiveGroup]);

  // Selection -> predicate (skipped in interactive mode) + selected-rows table.
  useEffect(() => {
    if (selected.length === 0) {
      setPredicate(null);
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
    fetchRows(dataset, selected.map((i) => node.row_indices[i]), featureCols)
      .then((r) => !cancelled && setRows(r))
      .catch(() => !cancelled && setRows(null));
    return () => {
      cancelled = true;
    };
  }, [selected, scope, interactive, node.id, dataset, featureCols, config]);

  const variance = (node.embedding_original_variance ?? []).filter(
    (v): v is number => v !== null,
  );

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
      <h2>Exploration — {pathLabel}</h2>
      <ScoreTiles scores={node.scores} title="DR quality — this cluster" />
      {config.method === "PCA" && variance.length > 0 && <PcaVarianceBar explainedVariance={variance} />}

      <div className="exploration__cols">
        <div className="exploration__analysis">
          <label className="field--check" style={{ marginBottom: "0.75rem" }}>
            <input
              type="checkbox"
              checked={interactive}
              onChange={(e) => setInteractive(e.target.checked)}
            />
            <span>Interactive feature ranges</span>
          </label>

          {interactive ? (
            <InteractiveFilters
              zData={zData}
              features={filterFeatures}
              ranges={ranges}
              onFeatures={setFilterFeatures}
              onRange={(f, r) => setRanges((prev) => ({ ...prev, [f]: r }))}
              matched={selected.length}
            />
          ) : selected.length === 0 ? (
            <p className="hint">Use lasso or box selection in the plot to capture points.</p>
          ) : (
            <>
              <div className="scope-toggle">
                <label>
                  <input type="radio" checked={scope === "local"} onChange={() => setScope("local")} />
                  This cluster (local)
                </label>
                <label>
                  <input type="radio" checked={scope === "global"} onChange={() => setScope("global")} />
                  Whole dataset (global)
                </label>
              </div>
              {predicate?.summary && (
                <div className="predicate-summary">
                  <span>Predicate F1: {predicate.summary.predicate_f1.toFixed(2)}</span>
                  <span>
                    Features used: {predicate.summary.n_features_used} /{" "}
                    {predicate.summary.n_features_total}
                  </span>
                  <span>Selected: {predicate.summary.n_selected}</span>
                </div>
              )}
              {predicate && <PredicateBands full={predicate.full} trimmed={predicate.trimmed} />}
            </>
          )}
          <CharacteristicsBar data={node.rel_characteristics} title="Cluster characteristics" />
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
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>{rows.columns.map((c) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {rows.rows.slice(0, 50).map((r, i) => (
                    <tr key={i}>
                      {rows.columns.map((c) => (
                        <td key={c}>{String(r[c])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
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
        Standardized (z-score) ranges. Matching points are highlighted in the plot — {matched} match.
      </p>
      <label className="field">
        <span>Features to filter</span>
        <select
          multiple
          value={features}
          size={Math.min(6, zData.cols.length)}
          onChange={(e) => onFeatures(Array.from(e.target.selectedOptions, (o) => o.value))}
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
              onChange={(e) => onRange(f, [Math.min(Number(e.target.value), hi), hi])}
            />
            <input
              type="range"
              min={bMin}
              max={bMax}
              step={(bMax - bMin) / 100 || 0.01}
              value={hi}
              onChange={(e) => onRange(f, [lo, Math.max(Number(e.target.value), lo)])}
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

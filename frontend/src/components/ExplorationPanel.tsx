// Leaf exploration: scatter + selection -> predicate bands + selected-points table,
// plus an interactive column-range filter (the "Ranges" tab).
// Parity with src/ui/components/exploration.py::render_cluster_exploration.
import { useEffect, useMemo, useState } from "react";
import { fetchRows, fetchSelectionCharacteristics, fetchTargets, runPredicate } from "../api";
import { CharacteristicsBar } from "../charts/CharacteristicsBar";
import { PcaVarianceBar } from "../charts/PcaVarianceBar";
import { PredicateBands } from "../charts/PredicateBands";
import { ProjectionScatter } from "../charts/ProjectionScatter";
import { ScoreTiles } from "../charts/ScoreTiles";
import { TargetBands } from "../charts/TargetBands";
import { useDebounced } from "../hooks/useDebounced";
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
import { collectRangeData, RangeFilters } from "./RangeFilters";
import type { RangeData } from "./RangeFilters";

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
// whole node; the ranges *make* a selection rather than explain one. Three questions
// about three point sets, so they take turns in one viewport rather than stacking.
type View = "predicate" | "characteristics" | "ranges";

// How long a slider drag settles before the node is re-scanned and the selected rows
// refetched. Long enough to swallow a drag, short enough to feel like a live filter.
const RANGE_SETTLE_MS = 120;

// Shown instead of a predicate or a characteristics chart when the selection covers
// every row of the explored node. Both endpoints compare the selection against that
// node, so the answer would be a comparison of the node with itself: an F1 of 1.00
// over 0 clauses, and z-scores that are exactly 0 with a standard deviation of exactly
// 1. Saying so is more use than drawing it.
const WHOLE_NODE_HINT =
  "The selection is the entire node, so the comparison would be self-referential — " +
  "every bar would read 0 by construction. Narrow it with a range or a lasso.";

// Target columns sit at the right of the selected-points table, fenced off from
// the feature columns so nobody reads a label as something the predicate used.
function cellClass(col: string, targets: Set<string>, first: string | undefined): string | undefined {
  if (!targets.has(col)) return undefined;
  return col === first ? "is-target is-target-first" : "is-target";
}

/** `ranges` minus every column not in `keep`. Returns the same object when nothing is
 *  dropped, so the state update bails out instead of re-rendering. */
function dropRanges(
  prev: Record<string, [number, number]>,
  keep: Set<string>,
): Record<string, [number, number]> {
  const gone = Object.keys(prev).filter((c) => !keep.has(c));
  if (gone.length === 0) return prev;
  const next = { ...prev };
  for (const c of gone) delete next[c];
  return next;
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

  // Column-range filter state. The tab *is* the mode — there is no separate flag that
  // could drift out of step with which panel is on screen.
  const [rangeData, setRangeData] = useState<RangeData | null>(null);
  const [filterCols, setFilterCols] = useState<string[]>([]);
  const [ranges, setRanges] = useState<Record<string, [number, number]>>({});
  const settledRanges = useDebounced(ranges, RANGE_SETTLE_MS);
  const interactive = view === "ranges";

  // Reset everything when the explored node changes.
  useEffect(() => {
    setSelected([]);
    setPredicate(null);
    setCharSel(null);
    setTargets(null);
    setRows(null);
    setRangeData(null);
    setFilterCols([]);
    setRanges({});
  }, [node.id]);

  // A new selection rebuilds the table, so the row the image was opened from is gone.
  useEffect(() => {
    setImageRow(null);
  }, [selected]);

  // The table shows the features the predicate speaks about, then the labels it
  // deliberately ignores — same order the header/CSV are marked up in. The ranges tab
  // offers exactly this set too: it filters, so a label is a fair thing to slice on.
  const tableCols = useMemo(() => [...featureCols, ...targetCols], [featureCols, targetCols]);
  const targetSet = useMemo(() => new Set(targetCols), [targetCols]);
  const firstTarget = targetCols[0];

  // A filtered column can leave the table — unticked in the feature picker, or all of
  // them at once via "None". Drop the pick and its window with it, so the tab badge and
  // the range rows keep describing the same set of columns.
  useEffect(() => {
    const keep = new Set(tableCols);
    setFilterCols((prev) =>
      prev.every((c) => keep.has(c)) ? prev : prev.filter((c) => keep.has(c)),
    );
    setRanges((prev) => dropRanges(prev, keep));
  }, [tableCols]);

  // Fetch the node's raw feature + target values while the ranges tab is open.
  useEffect(() => {
    if (!interactive) {
      setRangeData(null);
      return;
    }
    let cancelled = false;
    fetchRows(dataset, node.row_indices, tableCols)
      .then((r) => !cancelled && setRangeData(collectRangeData(tableCols, targetCols, r.rows)))
      .catch(() => !cancelled && setRangeData(null));
    return () => {
      cancelled = true;
    };
  }, [interactive, node.id, dataset, tableCols, targetCols]);

  // The clauses the conjunction actually runs over: a picked column that is present in
  // the fetched values with a finite window. `filtering` is derived from *these* rather
  // than from the picked columns, because the two can disagree — unticking a column in
  // the feature picker (or the one-click "None") takes it out of `tableCols`, refetches
  // `rangeData` without it and drops its clause. A `filtering` flag read off
  // `filterCols` would then let an empty conjunction — `[].every(...)` is true for every
  // row — hand the whole node back as if it were a filter result.
  const clauses = useMemo(() => {
    if (!interactive || !rangeData) return [];
    return filterCols
      .map((c) => {
        const j = rangeData.cols.indexOf(c);
        const [lo, hi] = settledRanges[c] ?? rangeData.bounds[j] ?? [NaN, NaN];
        return { j, lo, hi };
      })
      .filter((c) => c.j >= 0 && Number.isFinite(c.lo) && Number.isFinite(c.hi));
  }, [interactive, rangeData, filterCols, settledRanges]);
  const filtering = clauses.length > 0;

  // "Matches filters" / "Other" per point (ranges tab only). The column positions are
  // resolved once rather than per point: on a wide dataset this runs over cols x rows.
  const interactiveGroup = useMemo(() => {
    if (!filtering || !rangeData) return null;
    return rangeData.values.map((row) =>
      clauses.every(({ j, lo, hi }) => row[j] >= lo && row[j] <= hi)
        ? "Matches filters"
        : "Other",
    );
  }, [filtering, rangeData, clauses]);

  // While the ranges tab is filtering, the filter *is* the selection (drives the
  // table, the target bands, and — once you switch tabs — the predicate).
  useEffect(() => {
    if (interactiveGroup) {
      setSelected(
        interactiveGroup.flatMap((g, i) =>
          g === "Matches filters" ? [i] : [],
        ),
      );
    }
  }, [interactiveGroup]);

  // A selection indexes into the node it was made in. On the render that swaps the
  // node, the reset effect above has not been applied yet, so `selected` still holds
  // the previous node's positions — mapping those through the new node's row_indices
  // yields rows nobody asked for. Sit the round out; the reset lands next render.
  const staleSelection = selected.some((i) => i >= node.row_indices.length);

  // A selection that *is* the node answers nothing: the predicate separates it from
  // itself, and the characteristics endpoint z-scores it against itself, which is where
  // the exactly-zero bars under "Selection characteristics — vs. {node}" came from. A
  // strict subset is the intended workflow (narrow in Ranges, inspect in Characteristics)
  // and is carried across tabs untouched; only the whole node is refused.
  const wholeNodeSelection = selected.length > 0 && selected.length === node.row_indices.length;

  // …and refusing it would strand the user on the Ranges tab with a selection they
  // cannot use, so leaving that tab drops it. A whole-node lasso made on another tab is
  // the user's own doing and stays put, hint and all.
  useEffect(() => {
    if (interactive) return;
    setSelected((prev) =>
      prev.length > 0 && prev.length === node.row_indices.length ? [] : prev,
    );
  }, [interactive, node.row_indices.length]);

  // Selection -> predicate (skipped in interactive mode) + target values + rows table.
  useEffect(() => {
    if (selected.length === 0 || staleSelection || wholeNodeSelection) {
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
  }, [selected, staleSelection, wholeNodeSelection, scope, interactive, node.id, dataset, featureCols, targetCols, tableCols, config]);

  // Selection -> characteristics, on its own so the cost is only paid while the tab
  // is open. Unlike the predicate this holds in interactive mode too: the filtered
  // points are a selection like any other. Clearing first matters even when the tab
  // is hidden, or reopening it flashes the previous selection's numbers.
  useEffect(() => {
    setCharSel(null);
    setCharFailed(false);
    if (view !== "characteristics" || selected.length === 0 || staleSelection || wholeNodeSelection)
      return;
    let cancelled = false;
    fetchSelectionCharacteristics({
      dataset,
      feature_cols: featureCols,
      config,
      row_indices: node.row_indices,
      selected_local_indices: selected,
    })
      .then((c) => !cancelled && setCharSel(c.characteristics))
      .catch(() => !cancelled && setCharFailed(true));
    return () => {
      cancelled = true;
    };
    // `config.normalize` rather than `config`: it is the only field the endpoint
    // reads, and the whole object is minted fresh on every config keystroke.
  }, [view, selected, staleSelection, wholeNodeSelection, node.id, dataset, featureCols, config.normalize]);

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
      {showScores && (
        <div className="exploration__summary">
          <ScoreTiles scores={node.scores} title="DR quality — this cluster" />
        </div>
      )}

      <div className="exploration__cols">
        <div className="exploration__analysis">
          <div className="tabs" role="tablist" aria-label="Selection view">
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
            <button
              role="tab"
              aria-selected={view === "ranges"}
              className={view === "ranges" ? "is-active" : undefined}
              onClick={() => setView("ranges")}
            >
              Ranges
              {filterCols.length > 0 && (
                <span className="tabs__badge">{filterCols.length}</span>
              )}
            </button>
          </div>

          {/* One bounded viewport for all three tabs. A wide dataset yields a band
              (and a pickable column) per feature — 784 of them on MNIST — which would
              otherwise run the page on for thousands of pixels below the plot. */}
          <div className="exploration__view" role="tabpanel">
            {view === "ranges" ? (
              <RangeFilters
                data={rangeData}
                active={filterCols}
                ranges={ranges}
                matched={filtering ? selected.length : null}
                narrowing={clauses.length}
                onActive={setFilterCols}
                onRange={(c, r) => setRanges((prev) => ({ ...prev, [c]: r }))}
                onClear={() => {
                  setFilterCols([]);
                  setRanges({});
                  // The filter *was* the selection; leaving it ringed after clearing
                  // would leave a selection with nothing on screen explaining it, and
                  // Clear disables itself so it could not be pressed again.
                  setSelected([]);
                }}
              />
            ) : view === "characteristics" ? (
              selected.length === 0 ? (
                <p className="hint">
                  Use lasso or box selection in the plot to capture points.
                </p>
              ) : wholeNodeSelection ? (
                <p className="hint">{WHOLE_NODE_HINT}</p>
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
            ) : selected.length === 0 ? (
              <p className="hint">
                Use lasso or box selection in the plot to capture points.
              </p>
            ) : wholeNodeSelection ? (
              <p className="hint">{WHOLE_NODE_HINT}</p>
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
                {/* An empty conjunction matches everything, so its F1 is reported
                    as 1.00 over 0 clauses — "Predicate F1: 1.00 · Features used:
                    0 / 2" reads as a perfect explanation of nothing. LayerSide
                    already says so instead of drawing it. */}
                {predicate?.summary && predicate.summary.n_features_used === 0 ? (
                  <p className="hint">
                    No feature range separates this selection from the rest of the{" "}
                    {scope === "global" ? "dataset" : "cluster"}.
                  </p>
                ) : (
                  <>
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
            points={node.embedding_original ?? []}
            rowIds={node.row_indices}
            method={config.method}
            interactiveGroup={interactiveGroup}
            onSelect={filtering ? () => {} : setSelected}
            selected={filtering ? [] : selected}
            toolbarExtra={
              showVariance ? <PcaVarianceBar explainedVariance={variance} /> : undefined
            }
          />
        </div>
      </div>

      <div className="exploration__table">
        <h3>Selected points: {selected.length}</h3>
        {/* The table is empty in this state because nothing was requested for it; say
            why here too, since the Ranges tab has no room for the hint above. */}
        {wholeNodeSelection && <p className="hint">{WHOLE_NODE_HINT}</p>}
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

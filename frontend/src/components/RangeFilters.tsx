// The "Ranges" exploration tab: pick columns, slide a [min, max] window over each,
// and the points inside *every* window become the selection.
//
// Both the feature columns and the held-out `target_*` label columns are offered.
// That is safe here in a way it would not be for the predicate: a range filter is a
// question the user asks, not an explanation the tool induces, so slicing on a label
// explains nothing away — it just asks "where do the points with this label sit?".
// Targets stay marked in the target hue throughout so the two never blur together.
//
// Windows are in raw column units. The old z-score axis picked out exactly the same
// points (standardizing is monotone, and the bounds were the column's own min/max
// either way) but printed "[-0.43, 2.31]" where a label wants "[0, 1]".
import { format } from "d3";
import { useMemo, useState } from "react";

const fmt = format(".4~g");
const BINS = 28;

/** Raw per-column values for one node's rows, plus each column's span within it. */
export interface RangeData {
  cols: string[];
  targets: Set<string>;
  values: number[][]; // rows x cols; anything non-numeric is NaN
  bounds: [number, number][]; // [min, max] over the finite values; [NaN, NaN] if none
}

/** `/api/rows` sends NaN as null, and `Number(null)` is 0 — a missing cell must not
 *  read as a real zero sitting in the middle of somebody's range. */
function numeric(cell: unknown): number {
  if (cell === null || cell === undefined || cell === "") return NaN;
  const v = Number(cell);
  return Number.isFinite(v) ? v : NaN;
}

export function collectRangeData(
  cols: string[],
  targetCols: string[],
  rows: Record<string, unknown>[],
): RangeData {
  const values = rows.map((r) => cols.map((c) => numeric(r[c])));
  const bounds = cols.map((_, j) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of values) {
      const v = row[j];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return (lo <= hi ? [lo, hi] : [NaN, NaN]) as [number, number];
  });
  return { cols, targets: new Set(targetCols), values, bounds };
}

/** Bin counts normalized to the tallest bin, so a row draws at a fixed height. */
function histogram(values: number[], min: number, max: number): number[] {
  const bins = new Array<number>(BINS).fill(0);
  const span = max - min;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const b = span > 0 ? Math.min(BINS - 1, Math.floor(((v - min) / span) * BINS)) : 0;
    bins[b] += 1;
  }
  const peak = Math.max(...bins, 1);
  return bins.map((c) => c / peak);
}

interface Props {
  data: RangeData | null;
  active: string[]; // chosen columns, in pick order
  ranges: Record<string, [number, number]>;
  matched: number | null; // null = nothing is being filtered, whatever is picked
  narrowing: number; // picked columns that actually contribute a clause
  onActive: (cols: string[]) => void;
  onRange: (col: string, range: [number, number]) => void;
  onClear: () => void;
}

export function RangeFilters({ data, active, ranges, matched, narrowing, onActive, onRange, onClear }: Props) {
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    const hit = (c: string) => q === "" || c.toLowerCase().includes(q);
    const pick = (want: boolean) =>
      data.cols.filter((c) => data.targets.has(c) === want && hit(c));
    return [
      { key: "Features", cols: pick(false) },
      { key: "Targets", cols: pick(true) },
    ].filter((g) => g.cols.length > 0);
  }, [data, query]);

  if (!data) return <p className="hint">Loading column values…</p>;

  return (
    <div className="range-filters">
      <p className="hint">
        Points inside every range are highlighted in the plot, and become the selection
        driving the table and the target bands below.
      </p>

      <div className="range-filters__head">
        {/* The count speaks about the clauses that are actually applied, never about
            the picked columns: a pick whose column has left the table, or whose window
            still spans the whole column, filters nothing and must not be counted as if
            it did. */}
        {matched === null ? (
          <span className="range-filters__count">
            {active.length === 0
              ? "No ranges yet"
              : "No range narrows this cluster yet — every window still spans its whole column"}
          </span>
        ) : (
          <span className="range-filters__count">
            <b>{matched}</b> of {data.values.length} points match {narrowing}{" "}
            {narrowing === 1 ? "range" : "ranges"}
          </span>
        )}
        <button onClick={onClear} disabled={active.length === 0}>
          Clear
        </button>
      </div>

      <input
        type="search"
        className="range-filters__search"
        placeholder="Find a column…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Find a column to filter on"
      />
      <div className="feature-picker__list range-filters__list">
        {groups.map((g) => (
          <div key={g.key} className="range-filters__group">
            <div className="range-filters__group-head">{g.key}</div>
            {g.cols.map((c) => {
              const on = active.includes(c);
              // [NaN, NaN] = the column holds nothing numeric in this cluster.
              const usable = Number.isFinite(data.bounds[data.cols.indexOf(c)][0]);
              return (
                <label
                  key={c}
                  className={on ? undefined : "is-off"}
                  title={usable ? c : `${c} — no numeric values in this cluster`}
                >
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={!usable}
                    onChange={() =>
                      onActive(on ? active.filter((x) => x !== c) : [...active, c])
                    }
                  />
                  <span className={data.targets.has(c) ? "is-target" : undefined}>{c}</span>
                </label>
              );
            })}
          </div>
        ))}
        {groups.length === 0 && <p className="hint">No column matches “{query}”.</p>}
      </div>

      {active.map((col) => {
        const j = data.cols.indexOf(col);
        if (j < 0) return null;
        return (
          <RangeRow
            key={col}
            col={col}
            isTarget={data.targets.has(col)}
            column={data.values.map((r) => r[j])}
            bounds={data.bounds[j]}
            range={ranges[col]}
            onRange={onRange}
            onRemove={() => onActive(active.filter((x) => x !== col))}
          />
        );
      })}
    </div>
  );
}

function RangeRow(props: {
  col: string;
  isTarget: boolean;
  column: number[];
  bounds: [number, number];
  range: [number, number] | undefined;
  onRange: (col: string, range: [number, number]) => void;
  onRemove: () => void;
}) {
  const { col, isTarget, column, bounds, range, onRange, onRemove } = props;
  const [min, max] = bounds;
  const [lo, hi] = range ?? bounds;
  const bins = useMemo(() => histogram(column, min, max), [column, min, max]);

  const span = max - min;
  const at = (v: number) => (span > 0 ? ((v - min) / span) * 100 : 0);
  const loPct = at(lo);
  const hiPct = at(hi);
  const step = span > 0 ? span / 200 : 1;

  return (
    <div className={isTarget ? "range-row is-target" : "range-row"}>
      <div className="range-row__head">
        <span className="range-row__name">{col}</span>
        <span className="range-row__vals">
          {fmt(lo)} – {fmt(hi)}
        </span>
        <button className="range-row__drop" onClick={onRemove} title={`Remove ${col}`}>
          ×
        </button>
      </div>

      <div className="range-track">
        <div className="range-hist" aria-hidden="true">
          {bins.map((h, i) => {
            const centre = ((i + 0.5) / BINS) * 100;
            return (
              <span
                key={i}
                className={centre >= loPct && centre <= hiPct ? "is-in" : undefined}
                style={{ height: `${Math.max(h * 100, 2)}%` }}
              />
            );
          })}
        </div>
        <div className="range-track__bar">
          <span
            className="range-track__fill"
            style={{ left: `${loPct}%`, right: `${100 - hiPct}%` }}
          />
        </div>
        {/* Two inputs stacked on one track: the lower thumb never crosses the upper. */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={lo}
          aria-label={`${col} minimum`}
          onChange={(e) => onRange(col, [Math.min(Number(e.target.value), hi), hi])}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={hi}
          aria-label={`${col} maximum`}
          onChange={(e) => onRange(col, [lo, Math.max(Number(e.target.value), lo)])}
        />
      </div>

      <div className="range-row__ends">
        <span>{fmt(min)}</span>
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

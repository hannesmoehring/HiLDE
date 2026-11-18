// "Overview": the whole cluster hierarchy as one treemap at the top of the
// canvas, with a colour switch, its legend and a details column. It only reads
// the tree and the path, and only ever changes the path through `onNavigate`.
import { useMemo, useState } from "react";
import { TreemapOverview } from "../charts/TreemapOverview";
import { buildOverview, flatten, pathKeys, pathLabelOf, type OverviewTile } from "../overview";
import { KIND_FILL, NO_VALUE_FILL, RAMP, metricsFor, scaleFor } from "../overviewMetrics";
import type { TreeNode } from "../types";
import { OverviewDetails } from "./OverviewDetails";

interface Props {
  root: TreeNode;
  treePath: number[];
  onNavigate: (path: number[]) => void;
}

const OPEN_KEY = "hilde.overview.open";

// Collapsed or not is a per-viewer convenience; storage may be unavailable.
function readOpen(): boolean {
  try {
    return window.localStorage.getItem(OPEN_KEY) !== "0";
  } catch {
    return true;
  }
}

export function OverviewPanel({ root, treePath, onNavigate }: Props) {
  const [open, setOpen] = useState(readOpen);
  const [metricKey, setMetricKey] = useState("kind");
  const [inspected, setInspected] = useState<OverviewTile | null>(null);

  const tree = useMemo(() => buildOverview(root), [root]);
  const tiles = useMemo(() => flatten(tree), [tree]);
  const metrics = useMemo(() => metricsFor(tree), [tree]);
  const metric = metrics.find((m) => m.key === metricKey) ?? metrics[0];
  const scale = useMemo(() => scaleFor(metric, tree), [metric, tree]);
  const onPath = pathKeys(root, treePath);
  const hasCharacteristics = tiles.some((t) => (t.node?.rel_characteristics.length ?? 0) > 0);

  const current = tiles.find((t) => t.key === (onPath[onPath.length - 1] ?? tree.key)) ?? tree;
  // A rebuild swaps the tree under a remembered tile; never describe a stale one.
  const shown = inspected && tiles.includes(inspected) ? inspected : current;
  const groups = [...new Set(metrics.map((m) => m.group))];

  function toggle() {
    const next = !open;
    setOpen(next);
    try {
      window.localStorage.setItem(OPEN_KEY, next ? "1" : "0");
    } catch {
      // Not remembered, which is fine.
    }
  }

  const nClusters = tiles.filter((t) => t.kind === "internal" || t.kind === "leaf").length;
  const crumbs: number[][] = [[], ...treePath.map((_, i) => treePath.slice(0, i + 1))].slice(0, onPath.length + 1);

  return (
    <section className="panel overview">
      <div className="panel__head">
        <span className="kicker">Overview</span>
        <span className="panel__title">Cluster hierarchy</span>
        <span className="panel__meta">
          {nClusters} clusters, area = points
          <button
            className="overview__toggle"
            aria-expanded={open}
            aria-controls="overview-body"
            onClick={toggle}
          >
            {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {open && (
        <div id="overview-body" className="overview__body">
          <div className="overview__controls">
            <nav className="overview__crumbs" aria-label="Current path">
              {crumbs.map((p, i) => (
                <span key={i}>
                  {i > 0 && <span aria-hidden="true"> → </span>}
                  <button
                    aria-current={i === crumbs.length - 1 ? "location" : undefined}
                    onClick={() => onNavigate(p)}
                    title={i === 0 ? "Back to the root" : `Open ${pathLabelOf(p)}`}
                  >
                    {i === 0 ? "root" : `C${p[p.length - 1]}`}
                  </button>
                </span>
              ))}
            </nav>
            <label className="overview__select">
              <span>Colour tiles by</span>
              <select value={metric.key} onChange={(e) => setMetricKey(e.target.value)}>
                {groups.map((g) => (
                  <optgroup key={g} label={g}>
                    {metrics
                      .filter((m) => m.group === g)
                      .map((m) => (
                        <option key={m.key} value={m.key}>
                          {m.label}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </label>
          </div>

          <Legend metricLabel={metric.label} format={metric.format} domain={metric.value ? scale.domain : null} categorical={!metric.value} />

          <div className="overview__cols">
            <TreemapOverview
              root={tree}
              fill={scale.fill}
              pathKeys={onPath}
              onActivate={(t) => onNavigate(t.path)}
              onInspect={setInspected}
            />
            <OverviewDetails
              tile={shown}
              total={root.n_points}
              metric={metric}
              source={shown === inspected ? "hover" : "path"}
              hasCharacteristics={hasCharacteristics}
            />
          </div>
          <p className="hint overview__note">
            Each tile’s second line names the feature whose mean moves most relative to its parent; under
            0.3 SD it says “close to parent”. Hatched tiles are the points HDBSCAN left unassigned when it
            split that node, so every node’s area equals its size. Click a tile (or Enter on it) to open it
            below; an unassigned tile opens the layer that shows those points as noise.
          </p>
        </div>
      )}
    </section>
  );
}

function Legend(props: {
  metricLabel: string;
  format: (v: number) => string;
  domain: [number, number] | null;
  categorical: boolean;
}) {
  const { metricLabel, format, domain, categorical } = props;
  const hatch = <span className="overview__swatch overview__swatch--hatch" style={{ background: KIND_FILL.unassigned }} />;
  if (categorical)
    return (
      <div className="overview__legend">
        <span><span className="overview__swatch" style={{ background: KIND_FILL.internal }} />internal node (frames its sub-clusters)</span>
        <span><span className="overview__swatch" style={{ background: KIND_FILL.leaf }} />leaf</span>
        <span>{hatch}unassigned (HDBSCAN noise)</span>
      </div>
    );
  return (
    <div className="overview__legend">
      {domain ? (
        <span>
          {metricLabel}: {format(domain[0])}
          <span className="overview__ramp" aria-hidden="true">
            {RAMP.map((c) => (
              <i key={c} style={{ background: c }} />
            ))}
          </span>
          {format(domain[1])}
        </span>
      ) : (
        <span>{metricLabel}: no tile has a value</span>
      )}
      <span><span className="overview__swatch" style={{ background: NO_VALUE_FILL }} />no value</span>
      <span>{hatch}hatched = unassigned, same scale</span>
    </div>
  );
}

// "Overview": the whole cluster hierarchy as one treemap at the top of the
// canvas, with a colour switch, its legend and a details column. The map zooms
// on its own (useMapFocus) without touching the layer stack; it only ever changes
// the path through `onNavigate`, when a tile is explored or a path crumb clicked.
import { useMemo, useState } from "react";
import { TreemapOverview } from "../charts/TreemapOverview";
import { useMapFocus } from "../hooks/useMapFocus";
import {
  buildOverview,
  flatten,
  pathKeys,
  pathLabelOf,
  type OverviewTile,
} from "../overview";
import {
  CONSTANT_FILL,
  KIND_FILL,
  NO_VALUE_FILL,
  RAMP,
  metricsFor,
  scaleFor,
  type MetricScale,
} from "../overviewMetrics";
import { maxClusterDepth, targetOf } from "../overviewTarget";
import type { TreeNode } from "../types";
import { ExploreAction } from "./ExploreAction";
import { MapCrumbs } from "./MapCrumbs";
import { MapStatus } from "./MapStatus";
import { OverviewDetails } from "./OverviewDetails";

interface Props {
  root: TreeNode;
  treePath: number[];
  // How deep the layer stack below renders. The rail can lower it without a
  // rebuild, so the tree may hold nodes the stack cannot open; the overview must
  // never point at a node the stack is not showing.
  nLayers: number;
  normalized: boolean; // the built run's config.normalize
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

export function OverviewPanel({
  root,
  treePath: rawPath,
  nLayers,
  normalized,
  onNavigate: navigateRaw,
}: Props) {
  const [open, setOpen] = useState(readOpen);
  const [metricKey, setMetricKey] = useState("kind");
  const [inspected, setInspected] = useState<OverviewTile | null>(null);

  const depth = Math.max(0, nLayers);
  // A deeper tile opens its deepest ancestor the stack can show.
  const treePath = rawPath.slice(0, depth);
  const onNavigate = (path: number[]) => navigateRaw(path.slice(0, depth));

  const tree = useMemo(
    () => buildOverview(root, normalized),
    [root, normalized],
  );
  const tiles = useMemo(() => flatten(tree), [tree]);
  const metrics = useMemo(() => metricsFor(tree), [tree]);
  const metric = metrics.find((m) => m.key === metricKey) ?? metrics[0];
  const scale = useMemo(() => scaleFor(metric, tree), [metric, tree]);
  const onPath = pathKeys(root, treePath);
  // The stack cuts the tree off: say so wherever the map promises a target.
  const cut = maxClusterDepth(tiles) > depth;
  const hasCharacteristics = tiles.some(
    (t) => (t.node?.rel_characteristics.length ?? 0) > 0,
  );

  const currentKey = onPath[onPath.length - 1] ?? tree.key;
  const map = useMapFocus(tree, currentKey);
  // A rebuild swaps the tree under a remembered tile; never describe a stale one.
  const hovered = inspected && tiles.includes(inspected) ? inspected : null;
  const shown = hovered ?? map.selected ?? map.focus;
  const source = hovered ? "hover" : map.selected ? "selected" : "focus";
  // The button acts on what is pinned, never on a passing hover or focus preview:
  // pressing it blurs the map, which drops that preview before the click lands.
  const exploreTile = map.selected ?? map.focus;
  const exploreTarget = targetOf(exploreTile, depth);

  // Exactly what activating a tile did before the map could zoom.
  function explore(t: OverviewTile) {
    const path = targetOf(t, depth).path;
    const landing = pathKeys(root, path);
    map.explored(t, landing[landing.length - 1] ?? tree.key);
    onNavigate(path);
  }
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

  const nClusters = tiles.filter(
    (t) => t.kind === "internal" || t.kind === "leaf",
  ).length;
  const crumbs: number[][] = [
    [],
    ...treePath.map((_, i) => treePath.slice(0, i + 1)),
  ].slice(0, onPath.length + 1);

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
              <span className="overview__crumbs-label">Layers</span>
              {crumbs.map((p, i) => (
                <span key={i}>
                  {i > 0 && <span aria-hidden="true"> → </span>}
                  <button
                    aria-current={
                      i === crumbs.length - 1 ? "location" : undefined
                    }
                    onClick={() => onNavigate(p)}
                    title={
                      i === 0 ? "Back to the root" : `Open ${pathLabelOf(p)}`
                    }
                  >
                    {i === 0 ? "root" : `C${p[p.length - 1]}`}
                  </button>
                </span>
              ))}
            </nav>
            <label className="overview__select">
              <span>Colour tiles by</span>
              <select
                value={metric.key}
                onChange={(e) => setMetricKey(e.target.value)}
              >
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

          <Legend
            metricLabel={metric.label}
            holeNote={metric.holeNote}
            format={metric.format}
            scale={scale}
            categorical={!metric.value}
          />

          <div className="overview__cols">
            <div className="overview__map">
              <MapCrumbs chain={map.chain} onZoom={map.zoomTo} />
              <MapStatus focus={map.focus} selected={map.selected} />
              <TreemapOverview
                tree={tree}
                focus={map.focus}
                fill={scale.fill}
                pathKeys={onPath}
                selectedKey={map.selected?.key ?? null}
                targetOf={(t) => targetOf(t, depth)}
                onZoom={map.zoomTo}
                onZoomOut={map.zoomOut}
                onSelect={map.select}
                onExplore={explore}
                onInspect={setInspected}
              />
            </div>
            <OverviewDetails
              tile={shown}
              total={root.n_points}
              metric={metric}
              source={source}
              isCurrent={shown.key === currentKey}
              action={
                <ExploreAction
                  tile={exploreTile}
                  target={exploreTarget}
                  isOpen={exploreTarget.path.join(",") === treePath.join(",")}
                  onExplore={() => explore(exploreTile)}
                />
              }
              hasCharacteristics={hasCharacteristics}
              normalized={normalized}
            />
          </div>
          <p className="hint overview__note">
            Each tile’s second line names the feature whose mean moves most
            relative to its parent; under 0.3 SD it says “close to parent”.
            {cut &&
              ` The layer stack is limited to ${depth} layer${depth === 1 ? "" : "s"}: the map still zooms all the way down, but exploring a deeper tile opens its deepest ancestor the stack shows, and an unassigned tile at that depth or below opens its parent in the exploration panel without marking the noise.`}
          </p>
        </div>
      )}
    </section>
  );
}

function Legend(props: {
  metricLabel: string;
  holeNote?: string;
  format: (v: number) => string;
  scale: MetricScale;
  categorical: boolean;
}) {
  const { metricLabel, holeNote, format, scale, categorical } = props;
  const { domain } = scale;
  const hatch = (
    <span
      className="overview__swatch overview__swatch--hatch"
      style={{ background: KIND_FILL.unassigned }}
    />
  );
  if (categorical)
    return (
      <div className="overview__legend">
        <span>
          <span
            className="overview__swatch"
            style={{ background: KIND_FILL.internal }}
          />
          internal node (frames its sub-clusters)
        </span>
        <span>
          <span
            className="overview__swatch"
            style={{ background: KIND_FILL.leaf }}
          />
          leaf
        </span>
        <span>{hatch}unassigned (HDBSCAN noise)</span>
      </div>
    );
  // Only claim the hatched tiles sit on the ramp where they actually have a value.
  const hatchNote =
    scale.unassigned !== "absent" && holeNote
      ? holeNote
      : {
          all: "hatched = unassigned, same scale",
          some: "hatched = unassigned, same scale where its mean is known",
          none: "hatched = unassigned, no value under this metric",
          absent: null,
        }[scale.unassigned];
  return (
    <div className="overview__legend">
      {!domain ? (
        <span>{metricLabel}: no tile has a value</span>
      ) : scale.constant ? (
        <span>
          {metricLabel}:
          <span
            className="overview__swatch overview__swatch--after"
            style={{ background: CONSTANT_FILL }}
          />
          every tile = {format(domain[0])}
        </span>
      ) : (
        <span>
          {metricLabel}: {format(domain[0])}
          <span className="overview__ramp" aria-hidden="true">
            {RAMP.map((c) => (
              <i key={c} style={{ background: c }} />
            ))}
          </span>
          {format(domain[1])}
        </span>
      )}
      <span>
        <span
          className="overview__swatch"
          style={{ background: NO_VALUE_FILL }}
        />
        no value
      </span>
      {hatchNote && (
        <span>
          {hatch}
          {hatchNote}
        </span>
      )}
    </div>
  );
}

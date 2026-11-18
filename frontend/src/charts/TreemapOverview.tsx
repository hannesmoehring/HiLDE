// Treemap of the whole cluster hierarchy, every depth at once. Area = points.
// The map never zooms: it is the overview, and the layer stack below is the
// zoom. It only outlines where the stack currently is.
//
// Props are declared here rather than in charts/props.ts: that file is the
// contract for the six charts ported from Streamlit, and this one is new.
import { useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import * as d3 from "d3";
import { useResize } from "../hooks/useResize";
import { fmtCount, fmtPct, kindLabel, shiftName, type OverviewTile } from "../overview";
import { textOn } from "../overviewMetrics";
import type { TileTarget } from "../overviewTarget";
import { theme } from "./theme";

export interface TreemapOverviewProps {
  root: OverviewTile;
  fill: (t: OverviewTile) => string;
  pathKeys: string[]; // nodes on the current treePath, root excluded
  // What activating a tile opens; the panel owns the layer-stack depth.
  targetOf: (t: OverviewTile) => TileTarget;
  onActivate: (t: OverviewTile) => void;
  onInspect: (t: OverviewTile | null) => void; // hover or keyboard focus
}

const HEADER = 17; // label band of a frame (root / internal node)
const PAD = 2;
const LINE = 13;
const CHAR_W = 6.1; // IBM Plex Sans at 11px, a little generous
const fits = (s: string, w: number) => s.length * CHAR_W + 8 <= w;

type Cell = d3.HierarchyRectangularNode<OverviewTile>;

const isFrame = (t: OverviewTile) => t.children.length > 0;

// Below this in either direction a tile has no area to see, click or hover, so
// it is not drawn and not a keyboard stop; its nearest drawn ancestor stands in.
const MIN_EXTENT = 1;
const drawable = (c: Cell) => c.x1 - c.x0 >= MIN_EXTENT && c.y1 - c.y0 >= MIN_EXTENT;

function heightFor(width: number): number {
  return Math.round(Math.max(280, Math.min(460, width * 0.48)));
}

// Lines for a tile, dropped from the end until they fit.
function tileLines(c: Cell): string[] {
  const t = c.data;
  const w = c.x1 - c.x0;
  const h = c.y1 - c.y0;
  const lines = [t.label];
  const name = t.kind === "unassigned" ? null : shiftName(t);
  if (name) lines.push(name);
  lines.push(`n = ${fmtCount(t.size)}`);
  const room = Math.max(0, Math.floor((h - 4) / LINE));
  if (!room || !fits(t.label, w)) return [];
  return lines.slice(0, room).filter((s) => fits(s, w));
}

// A frame's one-line header, segments dropped from the end until it fits.
function headerText(c: Cell): string {
  const t = c.data;
  const parts = [t.label];
  const name = shiftName(t);
  if (name) parts.push(name);
  parts.push(`n = ${fmtCount(t.size)}`);
  for (let k = parts.length; k > 0; k--) {
    const s = parts.slice(0, k).join(" · ");
    if (fits(s, c.x1 - c.x0)) return s;
  }
  return "";
}

function ariaLabel(t: OverviewTile, target: TileTarget, hidden: number): string {
  const share = t.owner ? `, ${fmtPct(t.size / t.owner.n_points)} of its parent` : "";
  const name = shiftName(t);
  const small = hidden ? `, ${hidden} tile${hidden === 1 ? "" : "s"} inside too small to show` : "";
  return `${t.pathLabel}, ${kindLabel(t.kind)}, ${fmtCount(t.size)} points${share}${
    name && t.kind !== "unassigned" ? `, ${name}` : ""
  }${small}. Activate: ${target.what}.`;
}

export function TreemapOverview({ root, fill, pathKeys, targetOf, onActivate, onInspect }: TreemapOverviewProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const width = size.width;
  const height = heightFor(width);
  const hatchId = `ov-hatch-${useId().replace(/:/g, "")}`;
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const tileRefs = useRef(new Map<string, SVGGElement>());

  const cells = useMemo(() => {
    if (width <= 0) return [] as Cell[];
    const h = d3
      .hierarchy(root, (t) => t.children)
      .sum((t) => (t.children.length ? 0 : t.size))
      // Largest first, as squarify wants, but unassigned always last so it sits
      // in the same corner of every frame.
      .sort((a, b) => {
        const ua = a.data.kind === "unassigned" ? 1 : 0;
        const ub = b.data.kind === "unassigned" ? 1 : 0;
        return ua - ub || (b.value ?? 0) - (a.value ?? 0);
      });
    const layout = d3
      .treemap<OverviewTile>()
      .tile(d3.treemapSquarify)
      .size([width, height])
      .paddingInner(PAD)
      .paddingOuter(PAD)
      // Only a frame big enough to hold a label gets a band; on a deep tree of
      // many small clusters a fixed band would eat the children it frames.
      .paddingTop((n) => (n.x1 - n.x0 >= 40 && n.y1 - n.y0 >= 3 * HEADER ? HEADER : PAD));
    return layout(h).descendants();
  }, [root, width, height]);

  // Pre-order, so the arrow keys walk a parent before its children. A tile too
  // small to draw is skipped along with its subtree: its children are no bigger.
  const order = useMemo(() => {
    const out: Cell[] = [];
    const walk = (c: Cell) => {
      if (!drawable(c)) return;
      out.push(c);
      c.children?.forEach(walk);
    };
    if (cells[0]) walk(cells[0]);
    return out;
  }, [cells]);
  // Every laid-out cell, so a path outline can still find a collapsed node.
  const allByKey = useMemo(() => new Map(cells.map((c) => [c.data.key, c])), [cells]);
  const byKey = useMemo(() => new Map(order.map((c) => [c.data.key, c])), [order]);
  // Descendants not drawn, per drawn frame: told in its accessible name.
  const hiddenIn = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of order)
      if (c.children) m.set(c.data.key, c.descendants().filter((d) => !byKey.has(d.data.key)).length);
    return m;
  }, [order, byKey]);

  // The selected node may be too small to draw; its nearest drawn ancestor then
  // carries the current marker and the tab stop.
  const drawnAncestor = (key: string): string => {
    let c = allByKey.get(key);
    while (c && !byKey.has(c.data.key)) c = c.parent ?? undefined;
    return c?.data.key ?? root.key;
  };
  const selectedKey = drawnAncestor(pathKeys[pathKeys.length - 1] ?? root.key);
  // One tab stop for the whole map (roving tabindex): a deep tree has hundreds of
  // tiles, and tabbing through all of them to reach the layers would be a trap.
  const tabKey = focusKey && byKey.has(focusKey) ? focusKey : byKey.has(selectedKey) ? selectedKey : root.key;

  function focusAt(i: number) {
    const c = order[Math.max(0, Math.min(order.length - 1, i))];
    if (c) tileRefs.current.get(c.data.key)?.focus();
  }

  function onKey(e: KeyboardEvent, c: Cell) {
    const i = order.indexOf(c);
    const moves: Record<string, () => void> = {
      ArrowRight: () => focusAt(i + 1),
      ArrowDown: () => focusAt(i + 1),
      ArrowLeft: () => focusAt(i - 1),
      ArrowUp: () => focusAt(i - 1),
      Home: () => focusAt(0),
      End: () => focusAt(order.length - 1),
      Enter: () => onActivate(c.data),
      " ": () => onActivate(c.data),
    };
    const move = moves[e.key];
    if (move) {
      e.preventDefault();
      move();
    }
  }

  const inspect = (key: string | null) => {
    onInspect(key ? (byKey.get(key)?.data ?? null) : null);
  };

  const hover = hoverKey ? byKey.get(hoverKey) : undefined;
  const focus = focusKey ? byKey.get(focusKey) : undefined;

  return (
    <div ref={ref} className="treemap" style={{ height }}>
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="group"
          aria-label="Treemap of the cluster hierarchy. Arrow keys move between tiles, Enter opens one."
          onMouseLeave={() => {
            setHoverKey(null);
            inspect(focusKey);
          }}
        >
          {/* The hatch follows the label ink: ink lines vanish on the dark ramp
              steps, and the hatch is all that tells a label-less unassigned tile
              from a leaf. */}
          <defs>
            <pattern id={hatchId} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={6} stroke={theme.textPrimary} strokeOpacity={0.16} strokeWidth={1.5} />
            </pattern>
            <pattern id={`${hatchId}-light`} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={6} stroke="#ffffff" strokeOpacity={0.4} strokeWidth={1.5} />
            </pattern>
          </defs>

          {order.map((c) => {
            const t = c.data;
            const w = Math.max(0, c.x1 - c.x0);
            const h = Math.max(0, c.y1 - c.y0);
            const bg = fill(t);
            const ink = textOn(bg);
            const frame = isFrame(t);
            const header = frame && c.y1 - c.y0 >= 3 * HEADER && w >= 40 ? headerText(c) : "";
            const lines = frame ? [] : tileLines(c);
            return (
              <g
                key={t.key}
                ref={(el) => {
                  if (el) tileRefs.current.set(t.key, el);
                  else tileRefs.current.delete(t.key);
                }}
                className="treemap__tile"
                role="button"
                tabIndex={t.key === tabKey ? 0 : -1}
                aria-label={ariaLabel(t, targetOf(t), hiddenIn.get(t.key) ?? 0)}
                aria-current={t.key === selectedKey && t.kind !== "unassigned" ? "true" : undefined}
                onClick={() => onActivate(t)}
                onKeyDown={(e) => onKey(e, c)}
                onMouseEnter={() => {
                  setHoverKey(t.key);
                  inspect(t.key);
                }}
                onFocus={() => {
                  setFocusKey(t.key);
                  inspect(t.key);
                }}
                onBlur={() => {
                  setFocusKey((k) => (k === t.key ? null : k));
                  if (!hoverKey) inspect(null);
                }}
              >
                <rect x={c.x0} y={c.y0} width={w} height={h} fill={bg} stroke={theme.borderStrong} strokeWidth={0.75} />
                {t.kind === "unassigned" && (
                  <rect x={c.x0} y={c.y0} width={w} height={h} fill={`url(#${ink === theme.textPrimary ? hatchId : `${hatchId}-light`})`} pointerEvents="none" />
                )}
                {header && (
                  <text x={c.x0 + 5} y={c.y0 + 12} fontSize={11} fill={ink} pointerEvents="none">
                    <tspan fontWeight={600}>{header.split(" · ")[0]}</tspan>
                    {header.includes(" · ") && <tspan>{header.slice(header.indexOf(" · "))}</tspan>}
                  </text>
                )}
                {lines.map((s, i) => (
                  <text
                    key={i}
                    x={c.x0 + 4}
                    y={c.y0 + 12 + i * LINE}
                    fontSize={11}
                    fontWeight={i === 0 ? 600 : 400}
                    fill={ink}
                    // A halo in the tile's own fill keeps the label clear of the hatch.
                    stroke={t.kind === "unassigned" ? bg : "none"}
                    strokeWidth={3}
                    paintOrder="stroke"
                    pointerEvents="none"
                  >
                    {s}
                  </text>
                ))}
              </g>
            );
          })}

          {/* Where the layer stack is: ancestors thin, the selected node heavy.
              White under the ink, like Ring, so it holds on the dark ramp steps. */}
          <g pointerEvents="none">
            {[...new Set(pathKeys.map(drawnAncestor))].map((k) => {
              const c = byKey.get(k);
              if (!c || k === root.key) return null;
              const sel = k === selectedKey;
              const box = {
                x: c.x0,
                y: c.y0,
                width: Math.max(0, c.x1 - c.x0),
                height: Math.max(0, c.y1 - c.y0),
                fill: "none",
              };
              return (
                <g key={k}>
                  <rect {...box} stroke="#ffffff" strokeWidth={sel ? 5 : 3} />
                  <rect {...box} stroke={theme.textPrimary} strokeWidth={sel ? 2.5 : 1.25} />
                </g>
              );
            })}
            {/* Hover ring dashed, keyboard focus solid; both inset so they never
                hide the path outline, or drawn around a tile too small to inset in. */}
            {hover && hover !== focus && <Ring c={hover} dashed />}
            {focus && <Ring c={focus} />}
          </g>
        </svg>
      )}
    </div>
  );
}

function Ring({ c, dashed = false }: { c: Cell; dashed?: boolean }) {
  const w = c.x1 - c.x0;
  const h = c.y1 - c.y0;
  const inset = w > 8 && h > 8 ? 2 : -2;
  const box = { x: c.x0 + inset, y: c.y0 + inset, width: w - 2 * inset, height: h - 2 * inset, fill: "none" };
  return (
    <>
      <rect {...box} stroke="#ffffff" strokeWidth={4} />
      <rect {...box} stroke={theme.textPrimary} strokeWidth={2} strokeDasharray={dashed ? "4 2" : undefined} />
    </>
  );
}

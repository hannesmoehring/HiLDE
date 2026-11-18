// Geometry and words for TreemapOverview: the squarified layout of the focused
// subtree, which tiles are big enough to draw, and what each tile says.
import * as d3 from "d3";
import { fmtCount, fmtPct, kindLabel, shiftName, type OverviewTile } from "../overview";
import { isFrame } from "../overviewFocus";
import type { TileTarget } from "../overviewTarget";

export type Cell = d3.HierarchyRectangularNode<OverviewTile>;

export const HEADER = 17; // label band of a frame (root / internal node)
const PAD = 2;
export const LINE = 13;
const CHAR_W = 6.1; // IBM Plex Sans at 11px, a little generous
const fits = (s: string, w: number) => s.length * CHAR_W + 8 <= w;

// Below this in either direction a tile has no area to see, click or hover, so
// it is not drawn and not a keyboard stop; its nearest drawn ancestor stands in.
const MIN_EXTENT = 1;
export const drawable = (c: Cell) => c.x1 - c.x0 >= MIN_EXTENT && c.y1 - c.y0 >= MIN_EXTENT;

export function heightFor(width: number): number {
  return Math.round(Math.max(280, Math.min(460, width * 0.48)));
}

// The focused tile fills the map; everything below it is laid out at full depth.
export function layoutOf(focus: OverviewTile, width: number, height: number): Cell[] {
  if (width <= 0) return [];
  const h = d3
    .hierarchy(focus, (t) => t.children)
    .sum((t) => (t.children.length ? 0 : t.size))
    // Largest first, as squarify wants, but unassigned always last so it sits
    // in the same corner of every frame.
    .sort((a, b) => {
      const ua = a.data.kind === "unassigned" ? 1 : 0;
      const ub = b.data.kind === "unassigned" ? 1 : 0;
      return ua - ub || (b.value ?? 0) - (a.value ?? 0);
    });
  return d3
    .treemap<OverviewTile>()
    .tile(d3.treemapSquarify)
    .size([width, height])
    .paddingInner(PAD)
    .paddingOuter(PAD)
    // Only a frame big enough to hold a label gets a band; on a deep tree of
    // many small clusters a fixed band would eat the children it frames.
    .paddingTop((n) => (n.x1 - n.x0 >= 40 && n.y1 - n.y0 >= 3 * HEADER ? HEADER : PAD))(h)
    .descendants();
}

// Lines for a tile, dropped from the end until they fit.
export function tileLines(c: Cell): string[] {
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
export function headerText(c: Cell): string {
  const w = c.x1 - c.x0;
  if (c.y1 - c.y0 < 3 * HEADER || w < 40) return "";
  const t = c.data;
  const parts = [t.label];
  const name = shiftName(t);
  if (name) parts.push(name);
  parts.push(`n = ${fmtCount(t.size)}`);
  for (let k = parts.length; k > 0; k--) {
    const s = parts.slice(0, k).join(" · ");
    if (fits(s, w)) return s;
  }
  return "";
}

// What Enter does on a tile, given where the map is zoomed.
export function enterAction(t: OverviewTile, isFocus: boolean, canZoomOut: boolean): string {
  if (isFocus) return canZoomOut ? "Enter zooms the map out one level" : "This tile is the whole map";
  return isFrame(t) ? "Enter zooms the map into it" : "Enter selects it";
}

export function ariaLabel(
  t: OverviewTile,
  o: { target: TileTarget; hidden: number; isFocus: boolean; canZoomOut: boolean; selected: boolean },
): string {
  const share = t.owner ? `, ${fmtPct(t.size / t.owner.n_points)} of its parent` : "";
  const name = shiftName(t);
  const small = o.hidden ? `, ${o.hidden} tile${o.hidden === 1 ? "" : "s"} inside too small to show` : "";
  return `${t.pathLabel}, ${kindLabel(t.kind)}, ${fmtCount(t.size)} points${share}${
    name && t.kind !== "unassigned" ? `, ${name}` : ""
  }${small}${o.selected ? ", selected" : ""}. ${enterAction(t, o.isFocus, o.canZoomOut)}. E explores: ${o.target.what}.`;
}

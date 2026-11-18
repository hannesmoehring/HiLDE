// Animates the treemap between two zoom levels. When the focus changes, every
// tile drawn before and after glides from its old rect to its new one; a tile
// only one layout has slides in from (or out to) where the zoom puts it, so
// zooming in pushes the siblings off the edges and zooming out pulls them back.
// Resizes and rebuilds snap. With prefers-reduced-motion nothing moves.
import { useLayoutEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { Cell } from "./treemapLayout";

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface ExitingTile {
  cell: Cell;
  box: Box;
}

const DURATION = 280;

const boxOf = (c: Box): Box => ({ x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1 });

// The affine map that lays rect `a` onto rect `b`.
function mapping(a: Box, b: Box): (r: Box) => Box {
  const sx = (b.x1 - b.x0) / (a.x1 - a.x0 || 1);
  const sy = (b.y1 - b.y0) / (a.y1 - a.y0 || 1);
  return (r) => ({
    x0: b.x0 + (r.x0 - a.x0) * sx,
    x1: b.x0 + (r.x1 - a.x0) * sx,
    y0: b.y0 + (r.y0 - a.y0) * sy,
    y1: b.y0 + (r.y1 - a.y0) * sy,
  });
}

const lerp = (a: Box, b: Box, k: number): Box => ({
  x0: a.x0 + (b.x0 - a.x0) * k,
  y0: a.y0 + (b.y0 - a.y0) * k,
  x1: a.x1 + (b.x1 - a.x1) * k,
  y1: a.y1 + (b.y1 - a.y1) * k,
});

function reducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

interface Snapshot {
  focusKey: string;
  tree: unknown;
  cells: Cell[];
  rects: Map<string, Box>;
}

interface Tween {
  from: Map<string, Box>;
  exiting: { cell: Cell; from: Box; to: Box }[];
}

export interface TileTween {
  boxOf: (c: Cell) => Box;
  exiting: ExitingTile[];
  moving: boolean;
}

// `cells`: the drawn tiles of the current layout, the focus first.
// `tree`: identity of the whole tree; a new one means a rebuild, which snaps.
export function useTileTween(cells: Cell[], focusKey: string, tree: unknown): TileTween {
  const prev = useRef<Snapshot | null>(null);
  const [tween, setTween] = useState<Tween | null>(null);
  const [k, setK] = useState(1);

  useLayoutEffect(() => {
    const before = prev.current;
    const rects = new Map(cells.map((c) => [c.data.key, boxOf(c)]));
    prev.current = { focusKey, tree, cells, rects };
    if (!before || before.focusKey === focusKey || before.tree !== tree) return;
    if (!cells.length || !before.cells.length || reducedMotion()) return;

    // Zooming in: the new focus was drawn before, and grows to fill the map.
    // Zooming out: the old focus is drawn now, and shrinks into its place.
    const grown = before.rects.get(focusKey);
    const shrunk = rects.get(before.focusKey);
    const [a, b] = grown
      ? [grown, boxOf(cells[0])]
      : shrunk
        ? [boxOf(before.cells[0]), shrunk]
        : [null, null];
    if (!a || !b) return; // unrelated layouts: nothing to carry over
    const toNew = mapping(a, b);
    const toOld = mapping(b, a);

    setTween({
      from: new Map(cells.map((c) => [c.data.key, before.rects.get(c.data.key) ?? toOld(boxOf(c))])),
      exiting: before.cells
        .filter((c) => !rects.has(c.data.key))
        .map((c) => ({ cell: c, from: boxOf(c), to: toNew(boxOf(c)) })),
    });
    setK(0);
    const start = performance.now();
    let raf = requestAnimationFrame(function step(now) {
      const t = Math.min(1, (now - start) / DURATION);
      setK(t);
      if (t < 1) raf = requestAnimationFrame(step);
      else setTween(null);
    });
    return () => {
      cancelAnimationFrame(raf);
      setTween(null);
      setK(1);
    };
  }, [cells, focusKey, tree]);

  const e = d3.easeCubicInOut(k);
  return {
    boxOf: (c) => {
      const from = tween?.from.get(c.data.key);
      return from ? lerp(from, c, e) : c;
    },
    exiting: tween ? tween.exiting.map((x) => ({ cell: x.cell, box: lerp(x.from, x.to, e) })) : [],
    moving: tween !== null,
  };
}

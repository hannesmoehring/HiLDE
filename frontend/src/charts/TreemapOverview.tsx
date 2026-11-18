// Treemap of the cluster hierarchy, area = points. It lays out one focused
// subtree (the whole tree by default) at full depth, and zooms: a click digs one
// level into the frame under the cursor, the focused frame's own header band (or
// Escape / Backspace) climbs back out. Zooming only moves the map; opening a tile
// in the layer stack is a separate act ("explore": double-click, E, or the
// details column's button). It outlines where the stack currently is and the
// tile pinned in the details column.
//
// Props are declared here rather than in charts/props.ts: that file is the
// contract for the six charts ported from Streamlit, and this one is new.
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { OverviewTile } from "../overview";
import { isFrame } from "../overviewFocus";
import type { TileTarget } from "../overviewTarget";
import { theme } from "./theme";
import { useResize } from "../hooks/useResize";
import { ariaLabel, drawable, heightFor, layoutOf, type Cell } from "./treemapLayout";
import { Outline, TreemapTileShape } from "./TreemapTileShape";
import { useTileTween } from "./useTileTween";
import { TreemapCaption } from "./TreemapCaption";

export interface TreemapOverviewProps {
  tree: OverviewTile; // the whole tree: a new one is a rebuild
  focus: OverviewTile; // the frame the map is zoomed into
  fill: (t: OverviewTile) => string;
  pathKeys: string[]; // nodes on the current treePath, root excluded
  selectedKey: string | null; // pinned in the details column
  // What exploring a tile opens; the panel owns the layer-stack depth.
  targetOf: (t: OverviewTile) => TileTarget;
  onZoom: (t: OverviewTile) => void;
  onZoomOut: () => void;
  onSelect: (t: OverviewTile) => void;
  onExplore: (t: OverviewTile) => void;
  onInspect: (t: OverviewTile | null) => void; // hover or keyboard focus
}

export function TreemapOverview(props: TreemapOverviewProps) {
  const { tree, focus, fill, pathKeys, selectedKey, targetOf, onZoom, onZoomOut, onSelect, onExplore, onInspect } = props;
  const { ref, size } = useResize<HTMLDivElement>();
  const width = size.width;
  const height = heightFor(width);
  const hatchId = `ov-hatch-${useId().replace(/:/g, "")}`;
  const [focusKey, setFocusKey] = useState<string | null>(null); // keyboard focus
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  // The solid focus ring is for the keyboard, like :focus-visible: a click also
  // focuses the tile under the pointer, which may not be the one it acted on.
  const [ringVisible, setRingVisible] = useState(false);
  const pointerDown = useRef(false);
  const tileRefs = useRef(new Map<string, SVGGElement>());
  // Where keyboard focus goes once a zoom has re-laid the map out.
  const restore = useRef<string | null>(null);
  // The tile the first click of a double-click acted on: the second one lands
  // on whatever the zoom moved under the pointer.
  const lastClick = useRef<OverviewTile | null>(null);
  const canZoomOut = focus.key !== tree.key;

  const cells = useMemo(() => layoutOf(focus, width, height), [focus, width, height]);

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
  // Every laid-out cell, so an outline can still find a collapsed node.
  const allByKey = useMemo(() => new Map(cells.map((c) => [c.data.key, c])), [cells]);
  const byKey = useMemo(() => new Map(order.map((c) => [c.data.key, c])), [order]);
  // Descendants not drawn, per drawn frame: told in its accessible name.
  const hiddenIn = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of order)
      if (c.children) m.set(c.data.key, c.descendants().filter((d) => !byKey.has(d.data.key)).length);
    return m;
  }, [order, byKey]);
  const tween = useTileTween(order, focus.key, tree);

  // After a zoom: hand keyboard focus on, and forget hover / focus on tiles the
  // map no longer draws, or the details column would keep describing them.
  useEffect(() => {
    const stale = (x: string | null) => x !== null && !byKey.has(x);
    if (stale(hoverKey)) setHoverKey(null);
    if (stale(focusKey)) setFocusKey(null);
    if (stale(hoverKey) || stale(focusKey))
      onInspect(hoverKey && !stale(hoverKey) ? (byKey.get(hoverKey)?.data ?? null) : null);
    // Then hand keyboard focus on; its onFocus previews the tile it lands on.
    const k = restore.current;
    restore.current = null;
    if (k && byKey.has(k)) tileRefs.current.get(k)?.focus();
    // Only a new layout can strand a key; hover and focus changes are handled where they happen.
  }, [byKey]);

  // A node too small to draw hands its marks to its nearest drawn ancestor; one
  // outside the focused subtree gets none.
  const drawnAncestor = (key: string): string | null => {
    let c = allByKey.get(key);
    while (c && !byKey.has(c.data.key)) c = c.parent ?? undefined;
    return c?.data.key ?? null;
  };
  const currentKey = drawnAncestor(pathKeys[pathKeys.length - 1] ?? tree.key);
  const pinnedKey = selectedKey ? drawnAncestor(selectedKey) : null;
  // One tab stop for the whole map (roving tabindex): a deep tree has hundreds of
  // tiles, and tabbing through all of them to reach the layers would be a trap.
  const tabKey =
    focusKey && byKey.has(focusKey)
      ? focusKey
      : pinnedKey ?? (currentKey && byKey.has(currentKey) ? currentKey : focus.key);

  // A click acts one level below the focus: on the focus's child that holds the
  // tile under the pointer, so each click digs exactly one layer.
  const stepOf = (c: Cell): Cell => {
    let s = c;
    while (s.parent && s !== cells[0] && s.parent !== cells[0]) s = s.parent;
    return s;
  };

  // Enter on a tile, or a click on the tile a click resolves to.
  function act(c: Cell) {
    if (c === cells[0]) {
      if (canZoomOut) onZoomOut();
    } else if (isFrame(c.data)) onZoom(c.data);
    else onSelect(c.data);
  }

  function focusAt(i: number) {
    const c = order[Math.max(0, Math.min(order.length - 1, i))];
    if (c) tileRefs.current.get(c.data.key)?.focus();
  }

  function onKey(e: KeyboardEvent, c: Cell) {
    const i = order.indexOf(c);
    setRingVisible(true);
    const zoomOut = () => {
      if (!canZoomOut) return false;
      restore.current = focus.key; // the frame just left is a tile of the wider map
      onZoomOut();
    };
    const enter = () => {
      if (c === cells[0]) restore.current = focus.key;
      else if (isFrame(c.data)) restore.current = c.data.key;
      act(c);
    };
    const moves: Record<string, () => void | false> = {
      ArrowRight: () => focusAt(i + 1),
      ArrowDown: () => focusAt(i + 1),
      ArrowLeft: () => focusAt(i - 1),
      ArrowUp: () => focusAt(i - 1),
      Home: () => focusAt(0),
      End: () => focusAt(order.length - 1),
      Enter: enter,
      " ": enter,
      Escape: zoomOut,
      Backspace: zoomOut,
      e: () => onExplore(c.data),
      E: () => onExplore(c.data),
    };
    const move = moves[e.key];
    if (move && !e.altKey && !e.ctrlKey && !e.metaKey && move() !== false) e.preventDefault();
  }

  const inspect = (key: string | null) => {
    onInspect(key ? (byKey.get(key)?.data ?? null) : null);
  };

  const hover = hoverKey ? byKey.get(hoverKey) : undefined;
  // The hover ring marks what a click acts on, not the (deeper) tile described.
  const hoverStep = hover ? stepOf(hover) : undefined;
  const keyFocus = focusKey && ringVisible ? byKey.get(focusKey) : undefined;
  const labels = !tween.moving;
  const box = tween.boxOf;

  return (
    <div className="treemap-wrap">
      <div ref={ref} className="treemap" style={{ height }}>
        {width > 0 && (
          <svg
            width={width}
            height={height}
            role="group"
            aria-label={`Treemap of ${focus.pathLabel}${canZoomOut ? ", zoomed in" : ", the whole hierarchy"}. Arrow keys move between tiles; Enter zooms into a frame or selects a tile; E explores the tile in the layers below; Escape zooms out.`}
            onMouseLeave={() => {
              setHoverKey(null);
              // A tile focused by a click is not being looked at once the pointer leaves.
              inspect(ringVisible ? focusKey : null);
            }}
            onPointerDown={() => {
              pointerDown.current = true;
            }}
            onDoubleClick={() => {
              if (lastClick.current) onExplore(lastClick.current);
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

            {/* Tiles the zoom pushes off the map, behind the ones it keeps. */}
            <g pointerEvents="none" aria-hidden="true">
              {tween.exiting.map((x) => (
                <TreemapTileShape key={x.cell.data.key} cell={x.cell} box={x.box} fill={fill} hatchId={hatchId} labels={false} />
              ))}
            </g>

            {order.map((c) => {
              const t = c.data;
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
                  aria-label={ariaLabel(t, {
                    target: targetOf(t),
                    hidden: hiddenIn.get(t.key) ?? 0,
                    isFocus: c === cells[0],
                    canZoomOut,
                    selected: t.key === selectedKey,
                  })}
                  aria-current={t.key === currentKey && t.kind !== "unassigned" ? "true" : undefined}
                  onClick={(e) => {
                    if (e.detail > 1) return; // the second half of a double-click
                    const s = stepOf(c);
                    lastClick.current = s.data;
                    act(s);
                  }}
                  onKeyDown={(e) => onKey(e, c)}
                  onMouseEnter={() => {
                    setHoverKey(t.key);
                    inspect(t.key);
                  }}
                  onFocus={() => {
                    setRingVisible(!pointerDown.current);
                    pointerDown.current = false;
                    setFocusKey(t.key);
                    inspect(t.key);
                  }}
                  onBlur={() => {
                    setFocusKey((k) => (k === t.key ? null : k));
                    if (!hoverKey) inspect(null);
                  }}
                >
                  <TreemapTileShape cell={c} box={box(c)} fill={fill} hatchId={hatchId} labels={labels} />
                </g>
              );
            })}

            <g pointerEvents="none">
              {/* Where the layer stack is: ancestors thin, the current node heavy. */}
              {[...new Set(pathKeys.map(drawnAncestor))].map((k) => {
                const c = k ? byKey.get(k) : undefined;
                if (!c || c === cells[0]) return null;
                return <Outline key={k} box={box(c)} width={k === currentKey ? 2.5 : 1.25} />;
              })}
              {/* The pinned tile, in the one hue the chrome allows for "picked". */}
              {pinnedKey && byKey.get(pinnedKey) && pinnedKey !== focus.key && (
                <Outline box={box(byKey.get(pinnedKey)!)} width={2.5} color={theme.selection} inset={1.5} />
              )}
              {/* Hover ring dashed, keyboard focus solid; both inset so they never
                  hide the outlines above, or drawn around a tile too small to inset in. */}
              {hoverStep && hoverStep !== keyFocus && (hoverStep !== cells[0] || canZoomOut) && (
                <Outline box={box(hoverStep)} width={2} dashed inset={2} />
              )}
              {keyFocus && <Outline box={box(keyFocus)} width={2} inset={2} />}
            </g>
          </svg>
        )}
      </div>
      <TreemapCaption step={hoverStep?.data ?? null} isFocus={hoverStep === cells[0]} canZoomOut={canZoomOut} />
    </div>
  );
}

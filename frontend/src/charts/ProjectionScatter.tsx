// C — Projection scatter. Replaces src/ui/visualization.py::make_scatter_fig.
// 2D scatter of a node's embedding (equal aspect) with lasso + box selection,
// zoom/pan, and optional coloring by cluster label or interactive-filter membership.
//
// Zoom is semantic: positions spread apart, marks keep their on-screen size, so
// zooming into a dense region actually separates it. The wheel zooms in every
// mode; a drag belongs to the active tool (lasso, box, or pan).
import { useEffect, useId, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import type { ProjectionScatterProps } from "./props";
import { useResize } from "../hooks/useResize";
import { usePlotZoom } from "../hooks/usePlotZoom";
import { theme } from "./theme";

const HEIGHT = 480;
const MARGIN = { top: 16, right: 16, bottom: 40, left: 48 };
const ACCENT = theme.accent;
const SELECTION = theme.selection;
const OTHER = theme.other;
const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;

type Mode = "lasso" | "box" | "pan";

interface Plotted {
  li: number; // index into props.points
  px: number; // pixel x within plot area
  py: number; // pixel y within plot area
  color: string;
}

function colorFor(
  clusterLabels: string[] | null | undefined,
  interactiveGroup: string[] | null | undefined,
): (i: number) => string {
  if (clusterLabels) {
    const uniques = Array.from(new Set(clusterLabels));
    const scheme = theme.categorical;
    const map = new Map<string, string>();
    uniques.forEach((label, idx) => map.set(label, scheme[idx % scheme.length]));
    return (i) => map.get(clusterLabels[i]) ?? ACCENT;
  }
  if (interactiveGroup) {
    return (i) => (interactiveGroup[i] === "Matches filters" ? ACCENT : OTHER);
  }
  return () => theme.textSecondary;
}

export function ProjectionScatter({
  points,
  method,
  clusterLabels,
  interactiveGroup,
  onSelect,
  selected,
  toolbarExtra,
}: ProjectionScatterProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>("lasso");

  const gRef = useRef<SVGGElement>(null);
  const brushRef = useRef<SVGGElement>(null);
  const brushApiRef = useRef<d3.BrushBehavior<unknown> | null>(null);
  const clipId = useId();
  const lassoRectRef = useRef<SVGRectElement>(null);
  const lassoPathRef = useRef<SVGPathElement>(null);

  // Latest onSelect without re-subscribing D3 listeners on every parent render.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const W = Math.max(0, size.width - MARGIN.left - MARGIN.right);
  const H = Math.max(0, HEIGHT - MARGIN.top - MARGIN.bottom);

  const geom = useMemo(() => {
    if (W <= 0 || H <= 0) return null;

    const valid: number[] = [];
    const xs: number[] = [];
    const ys: number[] = [];
    points.forEach((p, i) => {
      const [x, y] = p;
      if (x == null || y == null) return;
      valid.push(i);
      xs.push(x);
      ys.push(y);
    });
    if (valid.length === 0) return null;

    let [xmin, xmax] = d3.extent(xs) as [number, number];
    let [ymin, ymax] = d3.extent(ys) as [number, number];
    const dx0 = xmax - xmin || 1;
    const dy0 = ymax - ymin || 1;
    xmin -= dx0 * 0.05;
    xmax += dx0 * 0.05;
    ymin -= dy0 * 0.05;
    ymax += dy0 * 0.05;

    // Equal aspect: same px-per-unit on both axes. Center each domain and expand
    // the shorter one so nothing is squished.
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    const dx = xmax - xmin;
    const dy = ymax - ymin;
    const scale = Math.min(W / dx, H / dy);
    const halfW = W / (2 * scale);
    const halfH = H / (2 * scale);

    const xScale = d3.scaleLinear().domain([cx - halfW, cx + halfW]).range([0, W]);
    const yScale = d3.scaleLinear().domain([cy - halfH, cy + halfH]).range([H, 0]);

    const color = colorFor(clusterLabels, interactiveGroup);
    const plotted: Plotted[] = valid.map((li) => {
      const [x, y] = points[li] as [number, number];
      return { li, px: xScale(x), py: yScale(y), color: color(li) };
    });

    return { xScale, yScale, plotted };
  }, [points, clusterLabels, interactiveGroup, W, H]);

  // A new node (a new `points` array) starts from the fitted view again.
  const { transform, zoomed, zoomIn, zoomOut, reset } = usePlotZoom(gRef, W, H, mode === "pan", points);
  // The selection tools hit-test where the points are drawn *now*; a ref keeps
  // their listeners from being rebound on every zoom tick.
  const transformRef = useRef(transform);
  transformRef.current = transform;

  // Box-select (d3.brush).
  useEffect(() => {
    if (mode !== "box" || !geom || !brushRef.current) return;
    const g = d3.select(brushRef.current);
    const brush = d3
      .brush()
      .extent([
        [0, 0],
        [W, H],
      ])
      .on("end", (event: d3.D3BrushEvent<unknown>) => {
        // Programmatic clear (the view moved under the box) — keep the selection.
        if (!event.sourceEvent) return;
        const sel = event.selection as [[number, number], [number, number]] | null;
        if (!sel) {
          onSelectRef.current([]);
          return;
        }
        const [[x0, y0], [x1, y1]] = sel;
        const t = transformRef.current;
        const idx = geom.plotted
          .filter((p) => {
            const sx = t.applyX(p.px);
            const sy = t.applyY(p.py);
            return sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1;
          })
          .map((p) => p.li);
        onSelectRef.current(idx);
      });
    g.call(brush);
    brushApiRef.current = brush;
    return () => {
      brushApiRef.current = null;
      g.on(".brush", null);
      g.selectAll("*").remove();
    };
  }, [mode, geom, W, H]);

  // The drawn box is in screen space; once the view moves it no longer frames the
  // points it selected, so drop the rectangle (the selection itself stays).
  useEffect(() => {
    if (brushRef.current && brushApiRef.current) {
      d3.select(brushRef.current).call(brushApiRef.current.move, null);
    }
  }, [transform]);

  // Lasso-select (freeform polygon over an overlay rect).
  useEffect(() => {
    if (mode !== "lasso" || !geom || !gRef.current || !lassoRectRef.current || !lassoPathRef.current)
      return;
    const gNode = gRef.current;
    const path = d3.select(lassoPathRef.current);
    const line = d3
      .line<[number, number]>()
      .x((d) => d[0])
      .y((d) => d[1]);
    let poly: [number, number][] = [];

    const drag = d3
      .drag<SVGRectElement, unknown>()
      .container(gNode)
      .on("start", (event: d3.D3DragEvent<SVGRectElement, unknown, unknown>) => {
        poly = [[event.x, event.y]];
      })
      .on("drag", (event: d3.D3DragEvent<SVGRectElement, unknown, unknown>) => {
        poly.push([event.x, event.y]);
        path.attr("d", `${line(poly) ?? ""}Z`);
      })
      .on("end", () => {
        path.attr("d", null);
        if (poly.length < 3) {
          onSelectRef.current([]); // click on empty space clears
          poly = [];
          return;
        }
        const t = transformRef.current;
        const idx = geom.plotted
          .filter((p) => d3.polygonContains(poly, [t.applyX(p.px), t.applyY(p.py)]))
          .map((p) => p.li);
        onSelectRef.current(idx);
        poly = [];
      });

    const rect = d3.select(lassoRectRef.current);
    rect.call(drag);
    return () => {
      rect.on(".drag", null);
    };
  }, [mode, geom]);

  const selectedSet = useMemo(() => new Set(selected ?? []), [selected]);

  // Axes follow the view.
  const zx = useMemo(() => (geom ? transform.rescaleX(geom.xScale) : null), [geom, transform]);
  const zy = useMemo(() => (geom ? transform.rescaleY(geom.yScale) : null), [geom, transform]);

  // Marks live in fitted-view coordinates under one transformed <g>, so a pan
  // only rewrites that group's transform. Radii and strokes are divided by the
  // scale to stay constant on screen, so this subtree re-renders on zoom only.
  const k = transform.k;
  const pointsLayer = useMemo(() => {
    if (!geom) return null;
    return (
      <>
        {/* Two passes: a selected point in a dense region would otherwise be
            painted over by whichever unselected neighbour comes later in plot
            order, which is exactly where the selection matters. */}
        {geom.plotted
          .filter((p) => !selectedSet.has(p.li))
          .map((p) => (
            <circle key={p.li} cx={p.px} cy={p.py} r={4 / k} fill={p.color} fillOpacity={0.85} />
          ))}
        {/* Selected: the selection hue, over a surface halo that separates it
            from whatever it overlaps, under an ink ring. */}
        {geom.plotted
          .filter((p) => selectedSet.has(p.li))
          .map((p) => (
            <g key={p.li}>
              <circle cx={p.px} cy={p.py} r={6 / k} fill="none" stroke={BG} strokeWidth={4 / k} />
              <circle
                cx={p.px}
                cy={p.py}
                r={6 / k}
                fill={SELECTION}
                fillOpacity={0.95}
                stroke={TEXT}
                strokeWidth={1.5 / k}
              />
            </g>
          ))}
      </>
    );
  }, [geom, selectedSet, k]);

  const xLabel = method === "PCA" ? "PC1" : `${method} 1`;
  const yLabel = method === "PCA" ? "PC2" : `${method} 2`;

  const btn = (m: Mode) => ({
    background: mode === m ? ACCENT : "transparent",
    color: mode === m ? theme.accentInk : TEXT,
    border: `1px solid ${mode === m ? ACCENT : theme.borderStrong}`,
    borderRadius: 0,
    padding: "3px 12px",
    cursor: "pointer",
    fontSize: 13,
  });
  const ctrlBtn = {
    background: "transparent",
    color: TEXT,
    border: `1px solid ${theme.borderStrong}`,
    borderRadius: 0,
    padding: "3px 10px",
    cursor: "pointer",
    fontSize: 13,
  };

  return (
    <div ref={ref} style={{ width: "100%", color: TEXT }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 8,
          marginBottom: 8,
        }}
      >
        <button type="button" style={btn("lasso")} onClick={() => setMode("lasso")}>
          Lasso
        </button>
        <button type="button" style={btn("box")} onClick={() => setMode("box")}>
          Box
        </button>
        <button
          type="button"
          style={btn("pan")}
          onClick={() => setMode("pan")}
          title="Drag to pan. Scrolling zooms in every mode."
        >
          Pan
        </button>
        <span style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 4 }}>
          <button type="button" style={ctrlBtn} onClick={zoomOut} title="Zoom out" aria-label="Zoom out">
            −
          </button>
          <button type="button" style={ctrlBtn} onClick={zoomIn} title="Zoom in" aria-label="Zoom in">
            +
          </button>
          <button
            type="button"
            style={{ ...ctrlBtn, opacity: zoomed ? 1 : 0.5, cursor: zoomed ? "pointer" : "default" }}
            onClick={reset}
            disabled={!zoomed}
          >
            Reset
          </button>
        </span>
        {toolbarExtra}
      </div>

      <svg width={size.width} height={HEIGHT} style={{ background: BG, border: `1px solid ${theme.border}`, display: "block" }}>
        <g ref={gRef} transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {geom && zx && zy && (
            <>
              <defs>
                <clipPath id={clipId}>
                  <rect x={0} y={0} width={W} height={H} />
                </clipPath>
              </defs>
              {/* axes */}
              <line x1={0} y1={H} x2={W} y2={H} stroke={MUTED} strokeWidth={1} />
              <line x1={0} y1={0} x2={0} y2={H} stroke={MUTED} strokeWidth={1} />
              {zx.ticks(6).map((t) => (
                <g key={`x${t}`} transform={`translate(${zx(t)},${H})`}>
                  <line y2={5} stroke={MUTED} />
                  <text y={18} textAnchor="middle" fill={MUTED} fontSize={11}>
                    {zx.tickFormat(6)(t)}
                  </text>
                </g>
              ))}
              {zy.ticks(6).map((t) => (
                <g key={`y${t}`} transform={`translate(0,${zy(t)})`}>
                  <line x2={-5} stroke={MUTED} />
                  <text x={-8} dy="0.32em" textAnchor="end" fill={MUTED} fontSize={11}>
                    {zy.tickFormat(6)(t)}
                  </text>
                </g>
              ))}
              <text x={W / 2} y={H + 34} textAnchor="middle" fill={TEXT} fontSize={13}>
                {xLabel}
              </text>
              <text
                transform={`translate(${-36},${H / 2}) rotate(-90)`}
                textAnchor="middle"
                fill={TEXT}
                fontSize={13}
              >
                {yLabel}
              </text>

              <g clipPath={`url(#${clipId})`}>
                <g transform={transform.toString()}>{pointsLayer}</g>
              </g>

              {/* interaction layer — exactly one active at a time */}
              {mode === "box" ? (
                <g ref={brushRef} />
              ) : mode === "pan" ? (
                <rect
                  className="plot-pan"
                  x={0}
                  y={0}
                  width={W}
                  height={H}
                  fill="transparent"
                  style={{ pointerEvents: "all" }}
                />
              ) : (
                <>
                  <rect
                    ref={lassoRectRef}
                    x={0}
                    y={0}
                    width={W}
                    height={H}
                    fill="transparent"
                    style={{ cursor: "crosshair", pointerEvents: "all" }}
                  />
                  <path
                    ref={lassoPathRef}
                    fill="rgba(22,22,20,0.07)"
                    stroke={ACCENT}
                    strokeWidth={1}
                    style={{ pointerEvents: "none" }}
                  />
                </>
              )}
            </>
          )}
        </g>
      </svg>
    </div>
  );
}

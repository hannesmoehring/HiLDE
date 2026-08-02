// C — Projection scatter. Replaces src/ui/visualization.py::make_scatter_fig.
// 2D scatter of a node's embedding (equal aspect) with lasso + box selection and
// optional coloring by cluster label or interactive-filter membership.
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import type { ProjectionScatterProps } from "./props";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";

const HEIGHT = 480;
const MARGIN = { top: 16, right: 16, bottom: 40, left: 48 };
const ACCENT = theme.accent;
const OTHER = theme.other;
const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;

type Mode = "lasso" | "box";

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
}: ProjectionScatterProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [mode, setMode] = useState<Mode>("lasso");

  const gRef = useRef<SVGGElement>(null);
  const brushRef = useRef<SVGGElement>(null);
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
        const sel = event.selection as [[number, number], [number, number]] | null;
        if (!sel) {
          onSelectRef.current([]);
          return;
        }
        const [[x0, y0], [x1, y1]] = sel;
        const idx = geom.plotted
          .filter((p) => p.px >= x0 && p.px <= x1 && p.py >= y0 && p.py <= y1)
          .map((p) => p.li);
        onSelectRef.current(idx);
      });
    g.call(brush);
    return () => {
      g.on(".brush", null);
      g.selectAll("*").remove();
    };
  }, [mode, geom, W, H]);

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
        const idx = geom.plotted
          .filter((p) => d3.polygonContains(poly, [p.px, p.py]))
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

  return (
    <div ref={ref} style={{ width: "100%", color: TEXT }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button type="button" style={btn("lasso")} onClick={() => setMode("lasso")}>
          Lasso
        </button>
        <button type="button" style={btn("box")} onClick={() => setMode("box")}>
          Box
        </button>
      </div>

      <svg width={size.width} height={HEIGHT} style={{ background: BG, border: `1px solid ${theme.border}`, display: "block" }}>
        <g ref={gRef} transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {geom && (
            <>
              {/* axes */}
              <line x1={0} y1={H} x2={W} y2={H} stroke={MUTED} strokeWidth={1} />
              <line x1={0} y1={0} x2={0} y2={H} stroke={MUTED} strokeWidth={1} />
              {geom.xScale.ticks(6).map((t) => (
                <g key={`x${t}`} transform={`translate(${geom.xScale(t)},${H})`}>
                  <line y2={5} stroke={MUTED} />
                  <text y={18} textAnchor="middle" fill={MUTED} fontSize={11}>
                    {geom.xScale.tickFormat(6)(t)}
                  </text>
                </g>
              ))}
              {geom.yScale.ticks(6).map((t) => (
                <g key={`y${t}`} transform={`translate(0,${geom.yScale(t)})`}>
                  <line x2={-5} stroke={MUTED} />
                  <text x={-8} dy="0.32em" textAnchor="end" fill={MUTED} fontSize={11}>
                    {geom.yScale.tickFormat(6)(t)}
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

              {/* points — selected ones carry a surface halo under the ink ring so
                  the selection reads whatever the fill is, including near-black */}
              {geom.plotted.map((p) => {
                const sel = selectedSet.has(p.li);
                return (
                  <g key={p.li}>
                    {sel && <circle cx={p.px} cy={p.py} r={6} fill="none" stroke={BG} strokeWidth={4} />}
                    <circle
                      cx={p.px}
                      cy={p.py}
                      r={sel ? 6 : 4}
                      fill={p.color}
                      fillOpacity={0.85}
                      stroke={sel ? TEXT : "none"}
                      strokeWidth={sel ? 1.5 : 0}
                    />
                  </g>
                );
              })}

              {/* interaction layer — exactly one active at a time */}
              {mode === "box" ? (
                <g ref={brushRef} />
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

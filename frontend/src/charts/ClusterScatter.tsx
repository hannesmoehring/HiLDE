// A — Cluster projection scatter. Replaces the KDE topography: draws the parent
// node's own 2D embedding with points colored by child cluster, HDBSCAN noise
// (rows in no child) as grey ×, and a clickable legend. Clicking a point, a
// legend chip, or a centroid label selects that child — the same drill-down the
// KDE centroids triggered.
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactElement } from "react";
import * as d3 from "d3";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { TreeNode } from "../types";
import type { ClusterScatterProps } from "./props";

const PLOT_HEIGHT = 384;
const MARGIN = 12;

const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
const ACCENT = theme.accent;

const clusterColor = (ci: number) => theme.categorical[ci % theme.categorical.length];

interface PlottedPoint {
  px: number;
  py: number;
  child: number; // index into node.children; -1 = noise (in no child cluster)
  row: number; // source dataframe row id, for the outlier-table highlight ring
}

export function ClusterScatter({ node, onSelectCluster, selectedChild, title, highlightRow }: ClusterScatterProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const children: TreeNode[] = node.children ?? [];
  const width = size.width;

  // Latest onSelectCluster without invalidating the memoized points layer
  // (App passes a fresh closure every render).
  const onSelectRef = useRef(onSelectCluster);
  onSelectRef.current = onSelectCluster;

  // Screen-space geometry: child membership per point (via row index), equal-
  // aspect projection of embedding_original into the plot area, centroid label
  // anchors, and the outlier count for the legend.
  const { plotted, centroids, outlierCount } = useMemo(() => {
    const byRow = new Map<number, number>();
    children.forEach((c, ci) => c.row_indices.forEach((r) => byRow.set(r, ci)));
    const membership = node.row_indices.map((r) => byRow.get(r) ?? -1);
    const outlierCount = membership.filter((m) => m === -1).length;

    const valid: { x: number; y: number; child: number; row: number }[] = [];
    (node.embedding_original ?? []).forEach(([x, y], i) => {
      if (x == null || y == null) return;
      valid.push({ x, y, child: membership[i] ?? -1, row: node.row_indices[i] });
    });
    if (valid.length === 0 || width <= 0) {
      return { plotted: [] as PlottedPoint[], centroids: [], outlierCount };
    }

    let [xmin, xmax] = d3.extent(valid, (p) => p.x) as [number, number];
    let [ymin, ymax] = d3.extent(valid, (p) => p.y) as [number, number];
    const dx0 = xmax - xmin || 1;
    const dy0 = ymax - ymin || 1;
    xmin -= dx0 * 0.05;
    xmax += dx0 * 0.05;
    ymin -= dy0 * 0.05;
    ymax += dy0 * 0.05;

    // Equal aspect: one px-per-unit factor for both axes (y flipped for screen).
    const pw = Math.max(1, width - 2 * MARGIN);
    const ph = PLOT_HEIGHT - 2 * MARGIN;
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    const k = Math.min(pw / (xmax - xmin), ph / (ymax - ymin));
    const x0 = MARGIN + pw / 2;
    const y0 = MARGIN + ph / 2;

    const plotted: PlottedPoint[] = valid.map((p) => ({
      px: x0 + (p.x - cx) * k,
      py: y0 - (p.y - cy) * k,
      child: p.child,
      row: p.row,
    }));

    const centroids = children.flatMap((_, ci) => {
      const own = plotted.filter((p) => p.child === ci);
      if (own.length === 0) return [];
      return [{
        index: ci,
        bx: d3.mean(own, (p) => p.px) as number,
        by: d3.mean(own, (p) => p.py) as number,
      }];
    });

    return { plotted, centroids, outlierCount };
  }, [node, children, width]);

  // Point size shrinks with density so large layers stay readable.
  const R = plotted.length > 4000 ? 2 : plotted.length > 1500 ? 2.5 : plotted.length > 500 ? 3 : 4;

  // Memoized marks: zooming only touches the wrapper <g> transform, never this
  // subtree. Draw order: noise ×, unselected clusters, selected cluster on top.
  const pointsLayer = useMemo(() => {
    const dim = selectedChild != null;
    let cross = "";
    for (const p of plotted) {
      if (p.child >= 0) continue;
      cross += `M${(p.px - R).toFixed(1)} ${(p.py - R).toFixed(1)}L${(p.px + R).toFixed(1)} ${(p.py + R).toFixed(1)}`;
      cross += `M${(p.px - R).toFixed(1)} ${(p.py + R).toFixed(1)}L${(p.px + R).toFixed(1)} ${(p.py - R).toFixed(1)}`;
    }

    const circle = (p: PlottedPoint, i: number, selected: boolean): ReactElement => (
      <circle
        key={`${p.child}-${i}`}
        cx={p.px.toFixed(1)}
        cy={p.py.toFixed(1)}
        r={R}
        fill={clusterColor(p.child)}
        fillOpacity={dim && !selected ? 0.18 : 0.85}
        style={{ cursor: "pointer" }}
        onClick={() => onSelectRef.current(p.child)}
      >
        <title>C{p.child} — click to explore</title>
      </circle>
    );

    return (
      <g>
        {cross && (
          <path
            d={cross}
            stroke={MUTED}
            strokeWidth={1.2}
            strokeOpacity={dim ? 0.18 : 0.5}
            fill="none"
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        )}
        {plotted.filter((p) => p.child >= 0 && p.child !== selectedChild).map((p, i) => circle(p, i, false))}
        {dim && plotted.filter((p) => p.child === selectedChild).map((p, i) => circle(p, i, true))}
      </g>
    );
  }, [plotted, R, selectedChild]);

  const hasChart = plotted.length > 0;
  const highlighted = highlightRow == null ? null : plotted.find((p) => p.row === highlightRow);

  // Zoom/pan (same interaction as the old topography): scroll/pinch to zoom,
  // drag to pan. Marks scale geometrically; centroid labels stay constant size.
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 12])
      .on("zoom", (event) => setTransform(event.transform));
    zoomRef.current = zoom;
    const sel = d3.select(svg);
    sel.call(zoom);
    return () => {
      sel.on(".zoom", null);
      zoomRef.current = null;
    };
  }, [hasChart]);

  // Reset the view when we navigate to a different node (drill in/out).
  useEffect(() => {
    const svg = svgRef.current;
    if (svg && zoomRef.current) d3.select(svg).call(zoomRef.current.transform, d3.zoomIdentity);
    else setTransform(d3.zoomIdentity);
  }, [node]);

  const scaleBy = (factor: number) => {
    const svg = svgRef.current;
    if (svg && zoomRef.current) d3.select(svg).transition().duration(180).call(zoomRef.current.scaleBy, factor);
  };
  const resetZoom = () => {
    const svg = svgRef.current;
    if (svg && zoomRef.current) d3.select(svg).transition().duration(180).call(zoomRef.current.transform, d3.zoomIdentity);
  };

  const ctrlBtn: CSSProperties = {
    background: "transparent",
    color: TEXT,
    border: `1px solid ${theme.borderStrong}`,
    borderRadius: 6,
    padding: "2px 9px",
    cursor: "pointer",
    fontSize: 13,
    lineHeight: 1.2,
  };
  const zoomed = transform.k !== 1 || transform.x !== 0 || transform.y !== 0;

  return (
    <div
      ref={ref}
      style={{
        width: "100%",
        boxSizing: "border-box",
        background: BG,
        color: TEXT,
        borderRadius: 8,
        padding: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ fontSize: 14, fontWeight: 600 }}>{title ?? "Cluster projection"}</div>
        {hasChart && (
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
            <button type="button" style={ctrlBtn} onClick={() => scaleBy(1 / 1.4)} title="Zoom out" aria-label="Zoom out">
              −
            </button>
            <button type="button" style={ctrlBtn} onClick={() => scaleBy(1.4)} title="Zoom in" aria-label="Zoom in">
              +
            </button>
            <button
              type="button"
              style={{ ...ctrlBtn, opacity: zoomed ? 1 : 0.5, cursor: zoomed ? "pointer" : "default" }}
              onClick={resetZoom}
              disabled={!zoomed}
            >
              Reset
            </button>
          </div>
        )}
      </div>

      {/* Legend: one chip per child cluster (click = same drill-down as the
          points — the reliable target when a cluster is buried under others). */}
      {children.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6, marginBottom: 8 }}>
          {children.map((c, ci) => {
            const sel = selectedChild === ci;
            return (
              <button
                key={ci}
                type="button"
                onClick={() => onSelectRef.current(ci)}
                title={`Explore C${ci}`}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  background: sel ? theme.accentSoft : "transparent",
                  border: `1px solid ${sel ? ACCENT : theme.borderStrong}`,
                  borderRadius: 999,
                  padding: "2px 10px 2px 7px",
                  color: TEXT,
                  fontSize: 12,
                  lineHeight: 1.6,
                  cursor: "pointer",
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: clusterColor(ci),
                    flex: "0 0 auto",
                  }}
                />
                <span style={{ fontWeight: sel ? 600 : 500 }}>C{ci}</span>
                <span style={{ color: theme.textSecondary }}>{c.n_points}</span>
              </button>
            );
          })}
          {outlierCount > 0 && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "2px 10px 2px 7px",
                fontSize: 12,
                lineHeight: 1.6,
                color: theme.textSecondary,
              }}
            >
              <svg width={10} height={10} aria-hidden="true">
                <path d="M2 2L8 8M2 8L8 2" stroke={MUTED} strokeWidth={1.5} strokeOpacity={0.7} />
              </svg>
              Outliers {outlierCount}
            </span>
          )}
        </div>
      )}

      {children.length === 0 ? (
        <div style={{ color: MUTED, fontSize: 13, padding: "24px 0" }}>No child clusters.</div>
      ) : width > 0 ? (
        <svg
          ref={svgRef}
          width={width}
          height={PLOT_HEIGHT}
          style={{ display: "block", cursor: "grab", touchAction: "none" }}
        >
          <g transform={transform.toString()}>{pointsLayer}</g>

          {/* Ring on the point picked in the GLOSH outlier table. Constant screen
              size, and a surface halo underneath so it reads over dense marks. */}
          {highlighted && (
            <g pointerEvents="none">
              <circle
                cx={transform.applyX(highlighted.px).toFixed(1)}
                cy={transform.applyY(highlighted.py).toFixed(1)}
                r={9}
                fill="none"
                stroke={BG}
                strokeWidth={4}
              />
              <circle
                cx={transform.applyX(highlighted.px).toFixed(1)}
                cy={transform.applyY(highlighted.py).toFixed(1)}
                r={9}
                fill="none"
                stroke={TEXT}
                strokeWidth={2}
              />
            </g>
          )}

          {/* Centroid labels: constant screen size, clickable like the points. */}
          {centroids.map((c) => {
            const sel = selectedChild === c.index;
            return (
              <text
                key={`label-${c.index}`}
                x={transform.applyX(c.bx).toFixed(1)}
                y={transform.applyY(c.by).toFixed(1)}
                dy="0.35em"
                textAnchor="middle"
                fontSize={12}
                fontWeight={700}
                fill={sel ? ACCENT : TEXT}
                stroke={BG}
                strokeWidth={3.5}
                strokeLinejoin="round"
                paintOrder="stroke"
                style={{ cursor: "pointer" }}
                onClick={() => onSelectRef.current(c.index)}
              >
                C{c.index}
              </text>
            );
          })}
        </svg>
      ) : (
        <div style={{ height: PLOT_HEIGHT }} />
      )}
    </div>
  );
}

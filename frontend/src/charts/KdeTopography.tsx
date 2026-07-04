// A — KDE cluster topography. Ports src/ui/visualization.py::cluster_gauss_kde.
// Small-multiples of per-child 2D KDE density contours, positioned by each child's
// MDS rel_position and size-scaled by n_points, with clickable centroids C0..Cn.
import { useMemo } from "react";
import * as d3 from "d3";
import type { ContourMultiPolygon } from "d3";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { TreeNode } from "../types";
import type { KdeTopographyProps } from "./props";

const RES = 60; // KDE grid resolution (60x60), extent [-0.5, 0.5]
const PLOT_HEIGHT = 384;
const MARGIN = { top: 16, right: 16, bottom: 16, left: 16 };

const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
const ACCENT = theme.accent;

// Sequential density ramp for the light surface: pale green (low) → deep blue
// (high). Unlike viridis, its high end stays dark so hot cores pop on white.
const density = (ratio: number) => d3.interpolateYlGnBu(0.25 + 0.7 * ratio);

interface ChildField {
  index: number;
  cx: number;
  cy: number;
  targetSize: number;
  gridMax: number;
  hasKde: boolean;
  bands: ContourMultiPolygon[]; // ascending threshold order, coords in grid-index space
}

// Normalised grid index (0..RES-1) -> data coordinate, scaled by the child's
// target_size and translated to its rel_position centroid (mirrors the Streamlit
// `_KDE_GRID * target_size + c` positioning).
function gridToData(g: number, targetSize: number, center: number): number {
  return (g / (RES - 1) - 0.5) * targetSize + center;
}

// Separable box blur over an n×n grid (edge-clamped). Softens the 60×60 marching-
// squares stair-stepping so stacked contour bands read as a continuous field.
function boxBlur(src: number[], n: number, r: number): number[] {
  if (r <= 0) return src;
  const tmp = new Array<number>(n * n);
  const out = new Array<number>(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      let c = 0;
      for (let k = -r; k <= r; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < n) {
          s += src[y * n + xx];
          c++;
        }
      }
      tmp[y * n + x] = s / c;
    }
  }
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      let c = 0;
      for (let k = -r; k <= r; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < n) {
          s += tmp[yy * n + x];
          c++;
        }
      }
      out[y * n + x] = s / c;
    }
  }
  return out;
}

function bandPath(
  coords: number[][][][],
  field: ChildField,
  fx: (x: number) => number,
  fy: (y: number) => number,
): string {
  let d = "";
  for (const polygon of coords) {
    for (const ring of polygon) {
      for (let i = 0; i < ring.length; i++) {
        const sx = fx(gridToData(ring[i][0], field.targetSize, field.cx));
        const sy = fy(gridToData(ring[i][1], field.targetSize, field.cy));
        d += (i === 0 ? "M" : "L") + sx.toFixed(2) + " " + sy.toFixed(2) + " ";
      }
      d += "Z";
    }
  }
  return d;
}

export function KdeTopography({ node, onSelectCluster, selectedChild, title }: KdeTopographyProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const children: TreeNode[] = node.children ?? [];

  // Contours + data-space extent depend only on the node, not on pixel size.
  const { fields, extent } = useMemo(() => {
    const sizes = children.map((c) => c.n_points);
    const maxSize = sizes.length ? Math.max(...sizes) : 1;

    // Pass 1: contours + raw centroid positions (MDS units), no layout scaling yet.
    const raw = children.map((child, index) => {
      const cx = child.rel_position?.[0] ?? 0;
      const cy = child.rel_position?.[1] ?? 0;
      const targetSize = 0.8 * Math.sqrt((child.n_points || 0) / maxSize) + 0.3;

      if (!child.kde) {
        return { index, cx, cy, targetSize, gridMax: 0, hasKde: false, bands: [] as ContourMultiPolygon[] };
      }

      // Flatten grid[y][x] (nulls -> 0) into d3.contours layout values[y*RES + x].
      const grid = child.kde.grid;
      const values = new Array<number>(RES * RES);
      for (let y = 0; y < RES; y++) {
        const row = grid[y] ?? [];
        for (let x = 0; x < RES; x++) values[y * RES + x] = row[x] ?? 0;
      }

      // Light blur + many levels => a smooth continuous density field instead of
      // the blocky few-band look of raw 60×60 marching squares.
      const blurred = boxBlur(values, RES, 1);
      let gridMax = 0;
      for (const v of blurred) if (v > gridMax) gridMax = v;
      if (gridMax <= 0) {
        return { index, cx, cy, targetSize, gridMax, hasKde: true, bands: [] as ContourMultiPolygon[] };
      }
      const LEVELS = 16;
      const thresholds: number[] = [];
      for (let i = 1; i <= LEVELS; i++) thresholds.push((gridMax * i) / (LEVELS + 1));
      const bands = d3.contours().size([RES, RES]).thresholds(thresholds)(blurred);
      return { index, cx, cy, targetSize, gridMax, hasKde: true, bands };
    });

    // Compress the raw MDS layout so each ~1-unit density field reads against the
    // gaps between centroids (raw MDS spread otherwise dwarfs every blob to a dot).
    const tsSorted = raw.map((f) => f.targetSize).sort((a, b) => a - b);
    const medSize = tsSorted.length ? tsSorted[Math.floor(tsSorted.length / 2)] : 1;
    // Compress toward a spread of ~K median-blob-widths so every field is visible,
    // regardless of the arbitrary MDS unit scale. Uses the overall span (robust to
    // close pairs, unlike a min nearest-neighbour distance). Only ever compresses.
    let s = 1;
    if (raw.length > 1) {
      const cxs = raw.map((f) => f.cx);
      const cys = raw.map((f) => f.cy);
      const rawSpan = Math.max(
        Math.max(...cxs) - Math.min(...cxs),
        Math.max(...cys) - Math.min(...cys),
        1e-9,
      );
      const K = 2.2 * Math.sqrt(raw.length) + 1;
      s = Math.min(1, (K * medSize) / rawSpan);
    }

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    const grow = (x: number, y: number) => {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    };

    const fields: ChildField[] = raw.map((f) => {
      const cx = f.cx * s;
      const cy = f.cy * s;
      grow(cx - 0.5 * f.targetSize, cy - 0.5 * f.targetSize);
      grow(cx + 0.5 * f.targetSize, cy + 0.5 * f.targetSize);
      return { index: f.index, cx, cy, targetSize: f.targetSize, gridMax: f.gridMax, hasKde: f.hasKde, bands: f.bands };
    });

    if (!isFinite(xMin)) {
      xMin = -0.5;
      xMax = 0.5;
      yMin = -0.5;
      yMax = 0.5;
    }

    return { fields, extent: { xMin, xMax, yMin, yMax } };
  }, [children]);

  const width = size.width;
  const pw = Math.max(1, width - MARGIN.left - MARGIN.right);
  const ph = Math.max(1, PLOT_HEIGHT - MARGIN.top - MARGIN.bottom);

  // Single shared scale factor k for both axes => equal aspect ratio everywhere.
  const dx = extent.xMax - extent.xMin || 1;
  const dy = extent.yMax - extent.yMin || 1;
  const pad = 1.06; // small breathing room
  const k = Math.min(pw / (dx * pad), ph / (dy * pad));
  const xMid = (extent.xMin + extent.xMax) / 2;
  const yMid = (extent.yMin + extent.yMax) / 2;
  const cx0 = MARGIN.left + pw / 2;
  const cy0 = MARGIN.top + ph / 2;
  const fx = (x: number) => cx0 + (x - xMid) * k;
  const fy = (y: number) => cy0 - (y - yMid) * k; // flip: data y up, screen y down

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
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
        {title ?? "Cluster topography"}
      </div>

      {children.length === 0 ? (
        <div style={{ color: MUTED, fontSize: 13, padding: "24px 0" }}>No child clusters.</div>
      ) : width > 0 ? (
        <svg width={width} height={PLOT_HEIGHT} style={{ display: "block" }}>
          {/* Density fields: lowest level first so higher levels overlay it. */}
          {fields.map((field) =>
            field.bands.map((band, bi) => (
              <path
                key={`f${field.index}-b${bi}`}
                d={bandPath(band.coordinates as number[][][][], field, fx, fy)}
                fill={density(band.value / field.gridMax)}
                fillRule="evenodd"
                stroke="none"
                pointerEvents="none"
              />
            )),
          )}

          {/* Clickable centroid markers C0..Cn (index -> children[index]). */}
          {fields.map((field) => {
            const px = fx(field.cx);
            const py = fy(field.cy);
            const selected = selectedChild === field.index;
            return (
              <g
                key={`c${field.index}`}
                transform={`translate(${px.toFixed(2)},${py.toFixed(2)})`}
                style={{ cursor: "pointer" }}
                onClick={() => onSelectCluster(field.index)}
              >
                {selected && (
                  <circle r={13} fill="none" stroke={ACCENT} strokeWidth={2.5} opacity={0.9} />
                )}
                {selected && (
                  <circle r={13} fill={ACCENT} opacity={0.18} />
                )}
                <circle
                  r={7}
                  fill="#ffffff"
                  fillOpacity={0.95}
                  stroke={theme.textPrimary}
                  strokeWidth={2}
                />
                <text
                  y={-14}
                  textAnchor="middle"
                  fontSize={12}
                  fontWeight={600}
                  fill={selected ? ACCENT : TEXT}
                  pointerEvents="none"
                >
                  C{field.index}
                </text>
              </g>
            );
          })}
        </svg>
      ) : (
        <div style={{ height: PLOT_HEIGHT }} />
      )}
    </div>
  );
}

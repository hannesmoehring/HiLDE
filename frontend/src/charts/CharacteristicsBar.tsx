// B — Cluster characteristics bar. Ports src/ui/visualization.py::cluster_characteristics_fig.
// Vertical z-score bars per feature with ±z_std error bars, a dotted zero line, and
// sign-based coloring. NOTE: the Streamlit source (visualization.py L106) colors
// negative z_mean crimson and positive steelblue; we mirror that encoding here.
import { useState } from "react";
import { scaleBand, scaleLinear, format } from "d3";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { CharacteristicsBarProps } from "./props";
import type { Characteristic } from "../types";

const CRIMSON = theme.divNeg; // negative z_mean
const STEELBLUE = theme.divPos; // positive (>= 0) z_mean
const NONFEAT = theme.nonFeature; // columns not selected as a feature
const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
const GRID = theme.grid;
const ERR = "rgba(27,31,40,0.45)";

const SVG_HEIGHT = 320;
const MARGIN = { top: 12, right: 16, bottom: 92, left: 48 };
const SCROLL_THRESHOLD = 60; // beyond this, scroll instead of cramming
const SCROLL_BAND = 24; // fixed px per feature in scroll mode
const TICK_FONT = 11;
const LABEL_OFFSET = 12; // gap between the axis and the start of a tick label

const fmt3 = format(".3~g"); // ~3 significant digits, trailing zeros trimmed
const fmtNum = (v: number | null): string => (v == null ? "—" : fmt3(v));

// A rotated label runs `sin(rot)` of its length downwards, so how many characters
// fit is a function of the rotation and the bottom margin — derive it rather than
// hard-coding a budget, or the vertical (-90°) case overflows the fixed SVG height.
function maxLabelChars(rot: number): number {
  const drop = MARGIN.bottom - LABEL_OFFSET - 4; // px of vertical room for the label
  const usable = drop / Math.abs(Math.sin((rot * Math.PI) / 180));
  return Math.max(4, Math.floor(usable / (TICK_FONT * 0.58))); // 0.58em ≈ mean advance
}
const truncate = (s: string, max: number) => (s.length > max ? s.slice(0, max - 1) + "…" : s);

const SHELL_PAD = 8; // `shell` padding: hover x/y are border-box, CSS insets are padding-box
const TIP_CHAR_W = 7; // ≈ mean advance of the 13px tooltip face

// Rough width of the tooltip about to be drawn, from its longest line. This only
// picks which SIDE the tooltip opens on — the flipped one is anchored by its right
// edge, so a poor estimate costs an early or late flip, never a clipped tooltip.
function tipWidth(d: Characteristic): number {
  const widest = Math.max(
    d.feature.length + (d.is_feature === false ? " · not a feature".length : 0),
    `z-mean: ${fmtNum(d.z_mean)}`.length,
    `z-std: ${fmtNum(d.z_std)}`.length,
    `raw mean: ${fmtNum(d.raw_mean)}`.length,
  );
  return widest * TIP_CHAR_W + 18; // + horizontal padding and border
}

interface Hover {
  i: number;
  x: number;
  y: number;
}

export function CharacteristicsBar({ data, title, nonFeatureOnly = false }: CharacteristicsBarProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [hover, setHover] = useState<Hover | null>(null);

  const heading = title ?? "Cluster characteristics";
  const shown = nonFeatureOnly ? data.filter((d) => d.is_feature === false) : data;
  const n = shown.length;

  const shell = {
    position: "relative" as const,
    width: "100%",
    background: BG,
    color: TEXT,
    borderRadius: 0,
    padding: 8,
    boxSizing: "border-box" as const,
  };
  const titleStyle = {
    fontSize: 14,
    fontWeight: 600,
    color: TEXT,
    padding: "2px 4px 6px",
  };

  // Empty state — nothing to plot.
  if (n === 0) {
    return (
      <div ref={ref} style={shell}>
        <div style={titleStyle}>{heading}</div>
        <div style={{ color: MUTED, fontSize: 14, padding: "24px 8px" }}>
          {nonFeatureOnly ? "No non-feature columns to display." : "No characteristics to display."}
        </div>
      </div>
    );
  }

  // Null z_mean / z_std draw as 0.
  const rows = shown.map((d) => ({
    feature: d.feature,
    zm: d.z_mean ?? 0,
    zs: d.z_std ?? 0,
    isFeature: d.is_feature !== false,
  }));
  // Legend only helps when both kinds of bar are on screen.
  const showLegend = rows.some((r) => r.isFeature) && rows.some((r) => !r.isFeature);

  const scroll = n > SCROLL_THRESHOLD;
  const availW = Math.max(size.width, 320);
  const innerW = scroll ? n * SCROLL_BAND : Math.max(availW - MARGIN.left - MARGIN.right, 40);
  const svgW = MARGIN.left + innerW + MARGIN.right;
  const innerH = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  const x = scaleBand<string>()
    .domain(rows.map((r) => r.feature))
    .range([0, innerW])
    .padding(0.2);

  // y domain covers min/max including error whiskers, always includes 0.
  let lo = 0;
  let hi = 0;
  for (const r of rows) {
    lo = Math.min(lo, r.zm - r.zs);
    hi = Math.max(hi, r.zm + r.zs);
  }
  if (lo === hi) {
    lo = -1;
    hi = 1;
  }
  const pad = (hi - lo) * 0.08;
  const y = scaleLinear().domain([lo - pad, hi + pad]).range([innerH, 0]).nice();

  const bw = x.bandwidth();
  // Tick labels go vertical once the bands are too narrow for a 40° slant.
  const step = x.step();
  const rot = step < 28 ? -90 : -40;
  const labelChars = maxLabelChars(rot);
  const cap = Math.min(bw * 0.4, 6); // error-bar cap half-width
  const y0 = y(0);
  const yTicks = y.ticks(5);

  const onMove = (i: number) => (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ i, x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  // The tooltip opens to the right of the cursor, and to its left when that would
  // run past the panel edge — the column clips with overflow: hidden, so the
  // rightmost features would otherwise lose half their tooltip.
  // `hover` indexes into `shown`, which shrinks when `nonFeatureOnly` flips — and
  // React fires no mouseleave when the hit rect unmounts under a stationary pointer,
  // so the index can outlive its bar. Resolve it once, and draw nothing if it is gone.
  const hovered = hover != null ? shown[hover.i] : undefined;
  const outerW = size.width + SHELL_PAD * 2;
  const flipTip =
    hover != null && hovered != null && hover.x + 12 + SHELL_PAD + tipWidth(hovered) > outerW;

  return (
    <div ref={ref} style={shell}>
      <div style={titleStyle}>{heading}</div>
      {showLegend && (
        <div
          style={{
            display: "flex",
            gap: 14,
            padding: "0 4px 6px",
            fontSize: 12,
            color: MUTED,
          }}
        >
          <LegendSwatch color={STEELBLUE} label="Selected feature" />
          <LegendSwatch color={NONFEAT} label="Not a feature" />
        </div>
      )}
      <div style={{ overflowX: scroll ? "auto" : "hidden", width: "100%" }}>
        <svg width={svgW} height={SVG_HEIGHT} style={{ display: "block" }}>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {/* gridlines + y tick labels */}
            {yTicks.map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                {t !== 0 && <line x1={0} x2={innerW} stroke={GRID} />}
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill={MUTED}>
                  {fmt3(t)}
                </text>
              </g>
            ))}

            {/* dotted zero baseline */}
            <line
              x1={0}
              x2={innerW}
              y1={y0}
              y2={y0}
              stroke={MUTED}
              strokeDasharray="2,3"
              strokeWidth={1}
              opacity={0.75}
            />

            {/* y-axis label */}
            <text
              transform={`translate(${-MARGIN.left + 12},${innerH / 2}) rotate(-90)`}
              textAnchor="middle"
              fontSize={12}
              fill={MUTED}
            >
              z-score
            </text>

            {/* bars + error bars */}
            {rows.map((r, i) => {
              const bx = x(r.feature)!;
              const cx = bx + bw / 2;
              const top = Math.min(y0, y(r.zm));
              const h = Math.abs(y(r.zm) - y0);
              const color = !r.isFeature ? NONFEAT : r.zm < 0 ? CRIMSON : STEELBLUE;
              const yHi = y(r.zm + r.zs);
              const yLo = y(r.zm - r.zs);
              const active = hover?.i === i;
              return (
                <g key={r.feature}>
                  <rect
                    x={bx}
                    y={top}
                    width={bw}
                    height={Math.max(h, 0.5)}
                    fill={color}
                    opacity={active ? 1 : 0.9}
                    stroke={active ? TEXT : "none"}
                    strokeWidth={active ? 1.5 : 0}
                  />
                  {r.zs > 0 && (
                    <g stroke={ERR} strokeWidth={1.25}>
                      <line x1={cx} x2={cx} y1={yHi} y2={yLo} />
                      <line x1={cx - cap} x2={cx + cap} y1={yHi} y2={yHi} />
                      <line x1={cx - cap} x2={cx + cap} y1={yLo} y2={yLo} />
                    </g>
                  )}
                  {/* full-column transparent hit area for hover */}
                  <rect
                    x={bx}
                    y={0}
                    width={bw}
                    height={innerH}
                    fill="transparent"
                    onMouseMove={onMove(i)}
                    onMouseLeave={() => setHover(null)}
                  />
                </g>
              );
            })}

            {/* rotated x tick labels */}
            {rows.map((r, i) => {
              if (step < 14 && i % 2 === 1) return null;
              const lx = x(r.feature)! + bw / 2;
              const ly = innerH + LABEL_OFFSET;
              return (
                <text
                  key={r.feature}
                  x={lx}
                  y={ly}
                  transform={`rotate(${rot},${lx},${ly})`}
                  textAnchor="end"
                  fontSize={TICK_FONT}
                  fill={MUTED}
                >
                  {truncate(r.feature, labelChars)}
                  <title>{r.feature}</title>
                </text>
              );
            })}
          </g>
        </svg>
      </div>

      {hover && hovered && (
        <div
          style={{
            position: "absolute",
            // Flipped: anchor the tooltip's RIGHT edge 12px left of the cursor, so
            // it can never spill past the panel however wide it turns out to be.
            ...(flipTip ? { right: outerW - hover.x + 12 } : { left: hover.x + 12 }),
            top: hover.y + 12,
            background: theme.surface,
            border: `1px solid ${theme.textPrimary}`,
            borderRadius: 0,
            padding: "5px 8px",
            fontSize: 13,
            color: TEXT,
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 2 }}>
            {hovered.feature}
            {hovered.is_feature === false && (
              <span style={{ color: NONFEAT, fontWeight: 500 }}> · not a feature</span>
            )}
          </div>
          <div style={{ color: MUTED }}>
            z-mean: <span style={{ color: TEXT }}>{fmtNum(hovered.z_mean)}</span>
          </div>
          <div style={{ color: MUTED }}>
            z-std: <span style={{ color: TEXT }}>{fmtNum(hovered.z_std)}</span>
          </div>
          <div style={{ color: MUTED }}>
            raw mean: <span style={{ color: TEXT }}>{fmtNum(hovered.raw_mean)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 8, height: 8, background: color, display: "inline-block" }} />
      {label}
    </span>
  );
}

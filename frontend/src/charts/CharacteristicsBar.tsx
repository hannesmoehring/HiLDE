// B — Cluster characteristics bar. Ports src/ui/visualization.py::cluster_characteristics_fig.
// Vertical z-score bars per feature with ±z_std error bars, a dotted zero line, and
// sign-based coloring. NOTE: the Streamlit source (visualization.py L106) colors
// negative z_mean crimson and positive steelblue; we mirror that encoding here.
import { useState } from "react";
import { scaleBand, scaleLinear, format } from "d3";
import { useResize } from "../hooks/useResize";
import type { CharacteristicsBarProps } from "./props";

const CRIMSON = "#d64550"; // negative z_mean
const STEELBLUE = "#4682b4"; // positive (>= 0) z_mean
const BG = "#1e222b";
const TEXT = "#e6e8ec";
const MUTED = "#9aa2af";
const GRID = "rgba(154,162,175,0.15)";
const ERR = "rgba(230,232,236,0.6)";

const SVG_HEIGHT = 320;
const MARGIN = { top: 12, right: 16, bottom: 78, left: 48 };
const SCROLL_THRESHOLD = 60; // beyond this, scroll instead of cramming
const SCROLL_BAND = 24; // fixed px per feature in scroll mode
const MAX_LABEL = 14; // chars before truncating a tick label

const fmt3 = format(".3~g"); // ~3 significant digits, trailing zeros trimmed
const fmtNum = (v: number | null): string => (v == null ? "—" : fmt3(v));
const truncate = (s: string) => (s.length > MAX_LABEL ? s.slice(0, MAX_LABEL - 1) + "…" : s);

interface Hover {
  i: number;
  x: number;
  y: number;
}

export function CharacteristicsBar({ data, title }: CharacteristicsBarProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [hover, setHover] = useState<Hover | null>(null);

  const heading = title ?? "Cluster characteristics";
  const n = data.length;

  const shell = {
    position: "relative" as const,
    width: "100%",
    background: BG,
    color: TEXT,
    borderRadius: 6,
    padding: 8,
    boxSizing: "border-box" as const,
    fontFamily: "system-ui, sans-serif",
  };
  const titleStyle = {
    fontSize: 13,
    fontWeight: 600,
    color: TEXT,
    padding: "2px 4px 6px",
  };

  // Empty state — nothing to plot.
  if (n === 0) {
    return (
      <div ref={ref} style={shell}>
        <div style={titleStyle}>{heading}</div>
        <div style={{ color: MUTED, fontSize: 13, padding: "24px 8px" }}>
          No characteristics to display.
        </div>
      </div>
    );
  }

  // Null z_mean / z_std draw as 0.
  const rows = data.map((d) => ({
    feature: d.feature,
    zm: d.z_mean ?? 0,
    zs: d.z_std ?? 0,
  }));

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
  const cap = Math.min(bw * 0.4, 6); // error-bar cap half-width
  const y0 = y(0);
  const yTicks = y.ticks(5);

  const onMove = (i: number) => (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ i, x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div ref={ref} style={shell}>
      <div style={titleStyle}>{heading}</div>
      <div style={{ overflowX: scroll ? "auto" : "hidden", width: "100%" }}>
        <svg width={svgW} height={SVG_HEIGHT} style={{ display: "block" }}>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {/* gridlines + y tick labels */}
            {yTicks.map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                {t !== 0 && <line x1={0} x2={innerW} stroke={GRID} />}
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={10} fill={MUTED}>
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
              fontSize={11}
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
              const color = r.zm < 0 ? CRIMSON : STEELBLUE;
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
            {rows.map((r) => {
              const lx = x(r.feature)! + bw / 2;
              const ly = innerH + 12;
              return (
                <text
                  key={r.feature}
                  x={lx}
                  y={ly}
                  transform={`rotate(-40,${lx},${ly})`}
                  textAnchor="end"
                  fontSize={10}
                  fill={MUTED}
                >
                  {truncate(r.feature)}
                  <title>{r.feature}</title>
                </text>
              );
            })}
          </g>
        </svg>
      </div>

      {hover && (
        <div
          style={{
            position: "absolute",
            left: hover.x + 12,
            top: hover.y + 12,
            background: "#2a2f3a",
            border: "1px solid #3a4150",
            borderRadius: 4,
            padding: "6px 8px",
            fontSize: 12,
            color: TEXT,
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
            boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{data[hover.i].feature}</div>
          <div style={{ color: MUTED }}>
            z-mean: <span style={{ color: TEXT }}>{fmtNum(data[hover.i].z_mean)}</span>
          </div>
          <div style={{ color: MUTED }}>
            z-std: <span style={{ color: TEXT }}>{fmtNum(data[hover.i].z_std)}</span>
          </div>
          <div style={{ color: MUTED }}>
            raw mean: <span style={{ color: TEXT }}>{fmtNum(data[hover.i].raw_mean)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

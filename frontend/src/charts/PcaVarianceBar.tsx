// D — PCA explained-variance bar. Ports src/ui/visualization.py::make_pca_variance_fig.
// A single horizontal stacked bar: one segment per principal component (width ∝ its
// variance ratio), sequential blue shading, an optional muted "unexplained" tail when
// the ratios sum to < 1, per-segment hover tooltip, and a 0–100% tick row underneath.
import { useState } from "react";
import * as d3 from "d3";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { PcaVarianceBarProps } from "./props";

const HEIGHT = 150;
const PAD_X = 10;
const BAR_Y = 30;
const BAR_H = 58;
const BAR_BOTTOM = BAR_Y + BAR_H;

const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
// The tail is an ABSENCE, so it stays near-surface — a full-bar outline (drawn
// below, outside the clip) is what makes it legible; without one it is
// indistinguishable from the white card and 62%-explained reads as 100%.
const UNEXPLAINED = theme.track;
// Components are ordinal, not categorical — an ink ramp, no hue.
const pcRamp = d3.interpolateLab("#2a2925", "#c9c6be");

interface Segment {
  key: string;
  label: string; // "PC1", …, "Unexplained"
  pct: number; // share of total variance, in %
  cumulativePct: number; // cumulative % up to and including this segment
  x: number; // left edge (px, within svg)
  w: number; // width (px)
  fill: string;
  textFill: string;
  isUnexplained: boolean;
}

// Pick a readable label color for a given fill by its perceived luminance:
// dark ink on light segments, white on the darker (high-index) blue segments.
function labelColor(fill: string): string {
  const c = d3.rgb(fill);
  const lum = (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255;
  return lum > 0.6 ? theme.textPrimary : "#ffffff";
}

export function PcaVarianceBar({ explainedVariance }: PcaVarianceBarProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [hovered, setHovered] = useState<number | null>(null);

  if (explainedVariance.length === 0) return null;

  const width = size.width;
  const innerWidth = Math.max(0, width - PAD_X * 2);

  let segments: Segment[] = [];
  if (innerWidth > 0) {
    const n = explainedVariance.length;
    const x = d3.scaleLinear().domain([0, 1]).range([0, innerWidth]);
    const total = d3.sum(explainedVariance);

    let cum = 0;
    segments = explainedVariance.map((ratio, i) => {
      const start = cum;
      cum += ratio;
      const fill = pcRamp(n > 1 ? i / (n - 1) : 0.25);
      return {
        key: `pc${i}`,
        label: `PC${i + 1}`,
        pct: ratio * 100,
        cumulativePct: cum * 100,
        x: PAD_X + x(start),
        w: x(cum) - x(start),
        fill,
        textFill: labelColor(fill),
        isUnexplained: false,
      };
    });

    const remainder = 1 - total;
    if (remainder > 0.001) {
      segments.push({
        key: "unexplained",
        label: "Unexplained",
        pct: remainder * 100,
        cumulativePct: 100,
        x: PAD_X + x(total),
        w: x(1) - x(total),
        fill: UNEXPLAINED,
        textFill: theme.textSecondary,
        isUnexplained: true,
      });
    }
  }

  const ticks = [0, 25, 50, 75, 100];
  const tickX = (t: number) => PAD_X + (innerWidth * t) / 100;
  const active = hovered != null ? segments[hovered] : null;

  return (
    <div
      ref={ref}
      style={{ position: "relative", width: "100%", height: HEIGHT, background: BG }}
    >
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="PCA explained variance">
          <defs>
            <clipPath id="pca-variance-clip">
              <rect x={PAD_X} y={BAR_Y} width={innerWidth} height={BAR_H} />
            </clipPath>
          </defs>

          <g clipPath="url(#pca-variance-clip)">
            {segments.map((s, i) => (
              <g
                key={s.key}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
                style={{ cursor: "default" }}
              >
                <rect
                  x={s.x}
                  y={BAR_Y}
                  width={Math.max(0, s.w)}
                  height={BAR_H}
                  fill={s.fill}
                  stroke={BG}
                  strokeWidth={1}
                  opacity={active && active !== s ? 0.72 : 1}
                />
                {s.w >= 46 ? (
                  <text
                    x={s.x + s.w / 2}
                    y={BAR_Y + BAR_H / 2}
                    textAnchor="middle"
                    fill={s.textFill}
                    style={{ pointerEvents: "none" }}
                  >
                    <tspan x={s.x + s.w / 2} dy="-0.2em" fontSize={12} fontWeight={600}>
                      {s.label}
                    </tspan>
                    <tspan x={s.x + s.w / 2} dy="1.3em" fontSize={11}>
                      {s.pct.toFixed(1)}%
                    </tspan>
                  </text>
                ) : s.w >= 22 && !s.isUnexplained ? (
                  <text
                    x={s.x + s.w / 2}
                    y={BAR_Y + BAR_H / 2}
                    dy="0.32em"
                    textAnchor="middle"
                    fontSize={11}
                    fontWeight={600}
                    fill={s.textFill}
                    style={{ pointerEvents: "none" }}
                  >
                    {s.label}
                  </text>
                ) : null}
              </g>
            ))}
          </g>

          {/* Bar outline — bounds the 0–100% extent so an unexplained tail reads
              as "missing variance" rather than as blank card. */}
          <rect
            x={PAD_X + 0.5}
            y={BAR_Y + 0.5}
            width={Math.max(0, innerWidth - 1)}
            height={BAR_H - 1}
            fill="none"
            stroke={theme.borderStrong}
            strokeWidth={1}
          />

          {/* 0–100% axis tick row */}
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={tickX(t)}
                x2={tickX(t)}
                y1={BAR_BOTTOM + 2}
                y2={BAR_BOTTOM + 7}
                stroke={MUTED}
                strokeWidth={1}
              />
              <text
                x={tickX(t)}
                y={BAR_BOTTOM + 19}
                textAnchor="middle"
                fontSize={11}
                fill={MUTED}
              >
                {t}%
              </text>
            </g>
          ))}

          <text
            x={PAD_X + innerWidth / 2}
            y={HEIGHT - 6}
            textAnchor="middle"
            fontSize={12}
            fill={MUTED}
          >
            Share of total variance (%)
          </text>
        </svg>
      )}

      {active && (
        <div
          style={{
            position: "absolute",
            left: active.x + active.w / 2,
            top: BAR_Y - 8,
            transform: "translate(-50%, -100%)",
            background: theme.surface,
            border: `1px solid ${theme.textPrimary}`,
            borderRadius: 0,
            padding: "5px 8px",
            color: TEXT,
            fontSize: 13,
            lineHeight: 1.3,
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 1,
          }}
        >
          {active.isUnexplained
            ? `Unexplained: ${active.pct.toFixed(1)}%`
            : `${active.label}: ${active.pct.toFixed(1)}% (cumulative ${active.cumulativePct.toFixed(1)}%)`}
        </div>
      )}
    </div>
  );
}

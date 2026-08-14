// D — PCA explained-variance strip. Ports src/ui/visualization.py::make_pca_variance_fig.
// A compact stacked bar that rides in the projection toolbar, right-bound to the
// scatter's edge: one segment per principal component (width ∝ its variance ratio),
// grey ramp, a near-surface "unexplained" tail when the ratios sum to < 1, and the
// running total in text. At this size nothing fits inside the segments, so the
// per-component numbers live in the hover tooltip.
import { useState } from "react";
import * as d3 from "d3";
import { theme } from "./theme";
import type { PcaVarianceBarProps } from "./props";

const BAR_W = 200;
const BAR_H = 10;

const BG = theme.surface;
// The tail is an ABSENCE, so it stays near-surface — the full-bar outline (drawn
// below, over the segments) is what makes it legible; without one it is
// indistinguishable from the white card and 62%-explained reads as 100%.
const UNEXPLAINED = theme.track;
// Components are ordinal, not categorical — a grey ramp, no hue, kept in the light
// half so the strip reads as chrome-weight next to the Lasso/Box buttons.
const pcRamp = d3.interpolateLab("#8a8780", "#dedbd4");

interface Segment {
  key: string;
  label: string; // "PC1", …, "Unexplained"
  pct: number; // share of total variance, in %
  cumulativePct: number; // cumulative % up to and including this segment
  x: number; // left edge (px, within svg)
  w: number; // width (px)
  fill: string;
  isUnexplained: boolean;
}

export function PcaVarianceBar({ explainedVariance }: PcaVarianceBarProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  if (explainedVariance.length === 0) return null;

  const n = explainedVariance.length;
  const x = d3.scaleLinear().domain([0, 1]).range([0, BAR_W]);
  const total = d3.sum(explainedVariance);

  let cum = 0;
  const segments: Segment[] = explainedVariance.map((ratio, i) => {
    const start = cum;
    cum += ratio;
    return {
      key: `pc${i}`,
      label: `PC${i + 1}`,
      pct: ratio * 100,
      cumulativePct: cum * 100,
      x: x(start),
      w: x(cum) - x(start),
      fill: pcRamp(n > 1 ? i / (n - 1) : 0.25),
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
      x: x(total),
      w: x(1) - x(total),
      fill: UNEXPLAINED,
      isUnexplained: true,
    });
  }

  const active = hovered != null ? segments[hovered] : null;

  return (
    <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 10.5, color: theme.faint }}>Explained variance</span>

      <span style={{ position: "relative", display: "inline-flex" }}>
        <svg
          width={BAR_W}
          height={BAR_H}
          style={{ display: "block" }}
          role="img"
          aria-label={`PCA explained variance: ${(total * 100).toFixed(1)}% over ${n} components`}
        >
          {segments.map((s, i) => (
            <rect
              key={s.key}
              x={s.x}
              y={0}
              width={Math.max(0, s.w)}
              height={BAR_H}
              fill={s.fill}
              stroke={BG}
              strokeWidth={1}
              opacity={active && active !== s ? 0.72 : 1}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
            />
          ))}

          {/* Bar outline — bounds the 0–100% extent so an unexplained tail reads
              as "missing variance" rather than as blank card. */}
          <rect
            x={0.5}
            y={0.5}
            width={BAR_W - 1}
            height={BAR_H - 1}
            fill="none"
            stroke={theme.borderStrong}
            strokeWidth={1}
            pointerEvents="none"
          />
        </svg>

        {/* Anchored to the strip's right edge, not centred on the segment: the strip
            itself is flush with the panel border, so a centred tooltip would spill
            past it. Dropping below keeps it clear of the toolbar. */}
        {active && (
          <div
            style={{
              position: "absolute",
              right: 0,
              top: BAR_H + 6,
              background: theme.surface,
              border: `1px solid ${theme.textPrimary}`,
              borderRadius: 0,
              padding: "5px 8px",
              color: theme.textPrimary,
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
      </span>

      <span
        style={{
          fontSize: 11,
          color: theme.textSecondary,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {(total * 100).toFixed(1)}%
      </span>
    </div>
  );
}

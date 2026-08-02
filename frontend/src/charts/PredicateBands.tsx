// E — Predicate feature-range bands. Replaces src/ui/visualization.py::make_feature_range_fig.
// One horizontal row per feature (from props.full). Each feature is normalized to its
// OWN [global_min, global_max] range: a faint global track, a translucent "Full range"
// band (RCM 1.0), and a solid inner "Core range" band (RCM 0.9, matched from props.trimmed
// by feature name). Predicate-clause features are indigo and sorted to the top.
import { scaleLinear } from "d3";
import { useMemo, useState } from "react";
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { PredicateRow } from "../types";
import type { PredicateBandsProps } from "./props";

// Light theme + predicate palette (indigo clause / neutral non-clause).
const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
const INDIGO = theme.indigo;
const GREY = theme.neutral;
const TRACK = theme.track;
const GRID = theme.grid;

// Layout constants.
const LEFT = 129; // label gutter — scales with the tick font, or long names clip
const RIGHT = 24;
const TOP = 8;
const BOTTOM = 26; // x-axis (0% / 50% / 100%)
const PITCH = 22; // vertical distance between rows
const BAND_H = 13;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const fmt = (v: number) => v.toFixed(2);

export function PredicateBands({ full, trimmed }: PredicateBandsProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [tip, setTip] = useState<{ row: PredicateRow; x: number; y: number } | null>(null);

  // Predicate clauses on top, then clause_f1 descending (mirrors the Streamlit lexsort).
  const rows = useMemo(
    () =>
      [...full].sort((a, b) => {
        if (a.in_predicate !== b.in_predicate) return a.in_predicate ? -1 : 1;
        return b.clause_f1 - a.clause_f1;
      }),
    [full],
  );
  const coreByFeature = useMemo(
    () => new Map(trimmed.map((r) => [r.feature, r])),
    [trimmed],
  );

  if (full.length === 0) return null;

  const width = size.width > 0 ? size.width : 680;
  const trackWidth = Math.max(width - LEFT - RIGHT, 10);
  const height = TOP + rows.length * PITCH + BOTTOM;

  const handleMove = (e: ReactMouseEvent<SVGRectElement>, row: PredicateRow) => {
    const rect = ref.current?.getBoundingClientRect();
    setTip({ row, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) });
  };

  return (
    <div
      ref={ref}
      style={{
        position: "relative",
        width: "100%",
        background: BG,
        color: TEXT,
        fontFamily: "inherit",
        fontSize: 13,
      }}
    >
      {/* Legend */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 16,
          padding: "6px 4px 8px",
        }}
      >
        <LegendItem swatch={<Swatch fill={TRACK} border />} label="Global range" />
        <LegendItem swatch={<Swatch fill={INDIGO} opacity={0.22} />} label="Full range (RCM 1.0)" />
        <LegendItem swatch={<Swatch fill={INDIGO} opacity={0.95} />} label="Core range (RCM 0.9)" />
        <span style={{ marginLeft: "auto", color: MUTED }}>
          Position within each feature&apos;s global range
        </span>
      </div>

      <svg width={width} height={height} role="img" aria-label="Predicate feature range bands">
        {/* Gridlines at 0% / 50% / 100% of every feature's normalized range. */}
        <g transform={`translate(${LEFT},${TOP})`}>
          {[0, 0.5, 1].map((f) => (
            <line
              key={f}
              x1={f * trackWidth}
              x2={f * trackWidth}
              y1={0}
              y2={rows.length * PITCH}
              stroke={GRID}
              strokeWidth={1}
            />
          ))}

          {rows.map((row, i) => {
            const color = row.in_predicate ? INDIGO : GREY;
            const span = row.global_max - row.global_min || 1;
            const x = scaleLinear().domain([row.global_min, row.global_min + span]).range([0, trackWidth]);
            const rowTop = i * PITCH;
            const bandTop = rowTop + (PITCH - BAND_H) / 2;

            const fullLo = clamp(x(row.sel_min), 0, trackWidth);
            const fullHi = clamp(x(row.sel_max), 0, trackWidth);

            const core = coreByFeature.get(row.feature);
            const coreLo = core ? clamp(x(core.sel_min), 0, trackWidth) : 0;
            const coreHi = core ? clamp(x(core.sel_max), 0, trackWidth) : 0;

            return (
              <g key={row.feature}>
                {/* Faint global track */}
                <rect x={0} y={bandTop} width={trackWidth} height={BAND_H} fill={TRACK} />
                {/* Full range (RCM 1.0) */}
                <rect
                  x={fullLo}
                  y={bandTop}
                  width={Math.max(fullHi - fullLo, 0)}
                  height={BAND_H}
                  fill={color}
                  opacity={row.in_predicate ? 0.22 : 0.14}
                />
                {/* Core range (RCM 0.9) — omitted when the feature has no trimmed match */}
                {core && (
                  <rect
                    x={coreLo}
                    y={bandTop}
                    width={Math.max(coreHi - coreLo, 0)}
                    height={BAND_H}
                    fill={color}
                    opacity={row.in_predicate ? 1 : 0.45}
                  />
                )}
                {/* Transparent hover target across the full row */}
                <rect
                  x={0}
                  y={rowTop}
                  width={trackWidth}
                  height={PITCH}
                  fill="transparent"
                  onMouseMove={(e) => handleMove(e, row)}
                  onMouseLeave={() => setTip(null)}
                />
              </g>
            );
          })}
        </g>

        {/* Feature labels in the left gutter */}
        {rows.map((row, i) => (
          <text
            key={row.feature}
            x={LEFT - 8}
            y={TOP + i * PITCH + PITCH / 2}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={11.5}
            fill={row.in_predicate ? TEXT : MUTED}
            fontWeight={row.in_predicate ? 600 : 400}
          >
            {row.feature}
          </text>
        ))}

        {/* X-axis ticks */}
        <g transform={`translate(${LEFT},${TOP + rows.length * PITCH + 6})`}>
          {[0, 0.5, 1].map((f, idx) => (
            <text
              key={f}
              x={f * trackWidth}
              y={12}
              textAnchor={idx === 0 ? "start" : idx === 1 ? "middle" : "end"}
              fill={MUTED}
            >
              {f * 100}%
            </text>
          ))}
        </g>
      </svg>

      {/* Tooltip */}
      {tip && (
        <div
          style={{
            position: "absolute",
            left: tip.x + 14,
            top: tip.y + 14,
            pointerEvents: "none",
            background: theme.surface,
            border: `1px solid ${theme.textPrimary}`,
            borderRadius: 0,
            padding: "5px 8px",
            color: TEXT,
            fontSize: 13,
            lineHeight: 1.4,
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 2 }}>{tip.row.feature}</div>
          <div>
            Selection: {fmt(tip.row.sel_min)} – {fmt(tip.row.sel_max)}
          </div>
          <div style={{ color: MUTED }}>
            Global: {fmt(tip.row.global_min)} – {fmt(tip.row.global_max)}
          </div>
          <div>Clause F1: {fmt(tip.row.clause_f1)}</div>
          <div style={{ color: tip.row.in_predicate ? INDIGO : MUTED }}>
            {tip.row.in_predicate ? "In predicate" : "Not in predicate"}
          </div>
        </div>
      )}
    </div>
  );
}

function LegendItem({ swatch, label }: { swatch: ReactNode; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      {swatch}
      <span style={{ color: MUTED }}>{label}</span>
    </span>
  );
}

function Swatch({ fill, opacity = 1, border = false }: { fill: string; opacity?: number; border?: boolean }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 16,
        height: 12,
        borderRadius: 0,
        background: fill,
        opacity,
        border: border ? `1px solid ${theme.border}` : "none",
      }}
    />
  );
}

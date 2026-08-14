// E2 — Target-value bands. Borrows E's (PredicateBands) row geometry for the
// `target_*` label columns, which are held out of the feature space and therefore
// out of the predicate. Everything here is drawn in theme.target so the two stacks
// never read as one: an indigo/ink band is a predicate clause, a teal band is not.
//
// Two row kinds share the track (each feature normalized to its own global range):
//   - one-hot label (`is_boolean`): a bar from 0 to the selection's class share.
//   - continuous label: the selection's [min, max] band with a mean tick.
// Both carry a dashed dataset-mean reference so a selection reads against the whole.
import { format, scaleLinear } from "d3";
import { useState } from "react";
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";
import type { TargetStat } from "../types";
import type { TargetBandsProps } from "./props";

const BG = theme.surface;
const TEXT = theme.textPrimary;
const MUTED = theme.muted;
const TARGET = theme.target;
const TRACK = theme.track;
const GRID = theme.grid;

// Layout — LEFT matches PredicateBands so stacked labels line up.
const LEFT = 129;
// Value readout gutter — the numbers are the point, not a hover reward. Wide
// enough for the longest form ("12.3% · 152/1234"); the svg would clip it.
const RIGHT = 118;
const TOP = 8;
const BOTTOM = 26;
const PITCH = 22;
const BAND_H = 13;

const fmt3 = format(".3~g");
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const pct = (v: number) => `${(v * 100).toFixed(v * 100 < 10 && v > 0 ? 1 : 0)}%`;

/** Right-gutter readout: a share for one-hot labels, a range for continuous ones. */
function readout(t: TargetStat, nSelected: number): string {
  if (t.sel_mean == null) return "—";
  if (t.is_boolean) return `${pct(t.sel_mean)} · ${Math.round(t.sel_mean * nSelected)}/${nSelected}`;
  if (t.sel_min === t.sel_max) return fmt3(t.sel_min ?? 0);
  return `${fmt3(t.sel_min ?? 0)} – ${fmt3(t.sel_max ?? 0)}`;
}

export function TargetBands({ targets, nSelected }: TargetBandsProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [tip, setTip] = useState<{ row: TargetStat; x: number; y: number } | null>(null);

  if (targets.length === 0) return null;

  const width = size.width > 0 ? size.width : 680;
  const trackWidth = Math.max(width - LEFT - RIGHT, 10);
  const height = TOP + targets.length * PITCH + BOTTOM;

  const handleMove = (e: ReactMouseEvent<SVGRectElement>, row: TargetStat) => {
    const rect = ref.current?.getBoundingClientRect();
    setTip({ row, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) });
  };

  return (
    <div
      ref={ref}
      style={{ position: "relative", width: "100%", background: BG, color: TEXT, fontFamily: "inherit", fontSize: 13 }}
    >
      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 16, padding: "6px 4px 8px" }}>
        <LegendItem swatch={<Swatch fill={TRACK} border />} label="Global range" />
        <LegendItem swatch={<Swatch fill={TARGET} opacity={0.28} />} label="Selection" />
        <LegendItem swatch={<Tick dashed />} label="Dataset mean" />
        <span style={{ marginLeft: "auto", color: MUTED }}>Labels — excluded from the predicate</span>
      </div>

      <svg width={width} height={height} role="img" aria-label="Target value bands for the selection">
        <g transform={`translate(${LEFT},${TOP})`}>
          {[0, 0.5, 1].map((f) => (
            <line
              key={f}
              x1={f * trackWidth}
              x2={f * trackWidth}
              y1={0}
              y2={targets.length * PITCH}
              stroke={GRID}
              strokeWidth={1}
            />
          ))}

          {targets.map((row, i) => {
            const lo = row.global_min ?? 0;
            const span = (row.global_max ?? 1) - lo || 1;
            const x = scaleLinear().domain([lo, lo + span]).range([0, trackWidth]);
            const rowTop = i * PITCH;
            const bandTop = rowTop + (PITCH - BAND_H) / 2;

            // One-hot: fill 0 -> class share. Continuous: the selection's own span.
            const from = row.is_boolean ? lo : (row.sel_min ?? lo);
            const to = row.is_boolean ? (row.sel_mean ?? lo) : (row.sel_max ?? lo);
            const bandLo = clamp(x(from), 0, trackWidth);
            const bandHi = clamp(x(to), 0, trackWidth);
            const hasSel = row.sel_mean != null;

            return (
              <g key={row.feature}>
                <rect x={0} y={bandTop} width={trackWidth} height={BAND_H} fill={TRACK} />
                {hasSel && (
                  <rect
                    x={bandLo}
                    y={bandTop}
                    width={Math.max(bandHi - bandLo, row.is_boolean ? 0 : 1.5)}
                    height={BAND_H}
                    fill={TARGET}
                    opacity={0.28}
                  />
                )}
                {/* Selection mean — the solid edge of a share bar, an inner tick otherwise. */}
                {hasSel && (
                  <line
                    x1={clamp(x(row.sel_mean as number), 0, trackWidth)}
                    x2={clamp(x(row.sel_mean as number), 0, trackWidth)}
                    y1={bandTop}
                    y2={bandTop + BAND_H}
                    stroke={TARGET}
                    strokeWidth={2}
                  />
                )}
                {row.global_mean != null && (
                  <line
                    x1={clamp(x(row.global_mean), 0, trackWidth)}
                    x2={clamp(x(row.global_mean), 0, trackWidth)}
                    y1={bandTop - 2}
                    y2={bandTop + BAND_H + 2}
                    stroke={MUTED}
                    strokeWidth={1}
                    strokeDasharray="2,2"
                  />
                )}
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

        {/* Labels (left gutter) and value readout (right gutter) */}
        {targets.map((row, i) => (
          <g key={row.feature}>
            <text
              x={LEFT - 8}
              y={TOP + i * PITCH + PITCH / 2}
              textAnchor="end"
              dominantBaseline="central"
              fontSize={11.5}
              fill={TEXT}
            >
              {row.feature}
            </text>
            <text
              x={LEFT + trackWidth + 8}
              y={TOP + i * PITCH + PITCH / 2}
              dominantBaseline="central"
              fontSize={11.5}
              fill={row.sel_mean == null ? MUTED : TARGET}
            >
              {readout(row, nSelected)}
            </text>
          </g>
        ))}

        <g transform={`translate(${LEFT},${TOP + targets.length * PITCH + 6})`}>
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
          {tip.row.is_boolean ? (
            <div>
              Selection: {readout(tip.row, nSelected)} true
            </div>
          ) : (
            <>
              <div>
                Selection: {fmt3(tip.row.sel_min ?? 0)} – {fmt3(tip.row.sel_max ?? 0)}
              </div>
              <div>Selection mean: {tip.row.sel_mean == null ? "—" : fmt3(tip.row.sel_mean)}</div>
            </>
          )}
          <div style={{ color: MUTED }}>
            Dataset: {fmt3(tip.row.global_min ?? 0)} – {fmt3(tip.row.global_max ?? 0)}
            {tip.row.global_mean != null &&
              ` (mean ${tip.row.is_boolean ? pct(tip.row.global_mean) : fmt3(tip.row.global_mean)})`}
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

function Tick({ dashed = false }: { dashed?: boolean }) {
  return (
    <svg width={16} height={12} style={{ display: "block" }} aria-hidden>
      <line x1={8} x2={8} y1={0} y2={12} stroke={MUTED} strokeWidth={1} strokeDasharray={dashed ? "2,2" : undefined} />
    </svg>
  );
}

// E — Predicate feature-range bands. Replaces src/ui/visualization.py::make_feature_range_fig.
// One horizontal row per feature (from props.full). Each feature is normalized to its
// OWN [global_min, global_max] range: a faint global track, a translucent "Full range"
// band (RCM 1.0), and a solid inner "Core range" band (RCM 0.9, matched from props.trimmed
// by feature name). Predicate-clause features are indigo, sort to the top in greedy
// step order, and carry the marginal F1 gain they bought at that step.
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
const LABEL_FONT = 11.5;
const CHAR_W = LABEL_FONT * 0.6; // mean advance — the 600 weight clause labels set the width
const GUTTER_PAD = 10; // breathing room between the longest label and the track
const LEFT_MIN = 44;
const LEFT_MAX = 160; // past this, long names clip rather than eat the whole width
const RIGHT = 24;
const TOP = 8;
const HEAD = 14; // header over the ΔF1 gutter, only drawn when a clause was selected
const GAIN_W = 96; // right gutter holding "+0.214 · 47"
const BOTTOM = 26; // x-axis (0% / 50% / 100%)
const PITCH = 22; // vertical distance between rows
const BAND_H = 13;
const TIP_PAD = 14; // cursor → tooltip offset
const TIP_CHAR_W = 7; // ≈ mean advance of the 13px tooltip face

const GAIN_HEADER_TITLE =
  "Per clause, in the order the greedy search added it: the F1 the predicate gained when " +
  "the clause was added, and the number of background points the conjunction still " +
  "matched after that step (the selection is a subset of them; precision = selected ÷ matched).";

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const fmt = (v: number) => v.toFixed(2);
// The stopping tolerance is 1e-6, so an accepted clause can legitimately round to
// +0.000 here — that is why the match count is shown next to it.
const gain = (v: number) => `+${v.toFixed(3)}`;

export function PredicateBands({ full, trimmed, nBackground }: PredicateBandsProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [tip, setTip] = useState<{ row: PredicateRow; x: number; y: number } | null>(null);

  // Predicate clauses on top in the order the greedy loop added them, then the rest
  // by clause_f1 descending (mirrors the Streamlit lexsort).
  // NOTE: this order is the RCM 1.0 construction. `trimmed` is a SEPARATE greedy
  // pass — its clause membership and its step order can both differ, which is why
  // the tooltip carries its own core-run block rather than reusing these steps.
  const rows = useMemo(
    () =>
      [...full].sort((a, b) => {
        if (a.in_predicate !== b.in_predicate) return a.in_predicate ? -1 : 1;
        if (a.in_predicate && b.in_predicate)
          return (a.predicate_step ?? 0) - (b.predicate_step ?? 0);
        return b.clause_f1 - a.clause_f1;
      }),
    [full],
  );
  const coreByFeature = useMemo(
    () => new Map(trimmed.map((r) => [r.feature, r])),
    [trimmed],
  );
  // Fit the gutter to the names actually on screen. Fixed at the widest case it
  // stranded a third of a narrow column on short names like `px_54`.
  const left = useMemo(() => {
    const longest = rows.reduce((m, r) => Math.max(m, r.feature.length), 0);
    return clamp(Math.ceil(longest * CHAR_W) + GUTTER_PAD, LEFT_MIN, LEFT_MAX);
  }, [rows]);

  if (full.length === 0) return null;

  const width = size.width > 0 ? size.width : 680;
  const hasGains = rows.some((r) => r.in_predicate && r.predicate_f1_gain !== null);
  const gainW = hasGains ? GAIN_W : 0;
  const top = hasGains ? TOP + HEAD : TOP;
  const trackWidth = Math.max(width - left - RIGHT - gainW, 10);
  const height = top + rows.length * PITCH + BOTTOM;

  const handleMove = (e: ReactMouseEvent<SVGRectElement>, row: PredicateRow) => {
    const rect = ref.current?.getBoundingClientRect();
    setTip({ row, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) });
  };

  const matchedLine = (row: PredicateRow) =>
    nBackground != null
      ? `Matches ${row.predicate_n_matched} of ${nBackground} background points`
      : `Matches ${row.predicate_n_matched} background points`;

  // The tooltip opens to the right of the cursor, and to its LEFT when that would
  // run past the chart's edge — the column clips with overflow: hidden, so the
  // rightmost rows (the gain gutter) otherwise lose half their tooltip. The
  // width estimate only picks the side; the flipped tooltip is anchored by its
  // right edge, so a poor estimate costs an early flip, never a clipped tip.
  const tipWidth = (row: PredicateRow) => {
    const core = coreByFeature.get(row.feature);
    const lines = [
      row.feature,
      `Selection: ${fmt(row.sel_min)} – ${fmt(row.sel_max)}`,
      `Standalone F1: ${fmt(row.clause_f1)}`,
      `${matchedLine(row)} · P 0.00 · R 0.00`,
      core ? `Core (RCM 0.9): step 0 · +0.000 · 0.00 → 0.00` : "",
    ];
    return Math.max(...lines.map((l) => l.length)) * TIP_CHAR_W + 16;
  };
  const flipTip = tip != null && tip.x + TIP_PAD + tipWidth(tip.row) > width;

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
        {/* Header for the marginal-gain gutter. */}
        {hasGains && (
          <text
            x={width - RIGHT}
            y={TOP + 8}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={10.5}
            fill={MUTED}
          >
            <title>{GAIN_HEADER_TITLE}</title>
            ΔF1 when added · points matched
          </text>
        )}

        {/* Gridlines at 0% / 50% / 100% of every feature's normalized range. */}
        <g transform={`translate(${left},${top})`}>
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
                {/* Marginal F1 gain at the step this clause was added, plus the
                    exact count of background points the conjunction still matched
                    (counts never degenerate the way a rounded gain can). */}
                {row.in_predicate && row.predicate_f1_gain !== null && (
                  <text
                    x={trackWidth + gainW}
                    y={rowTop + PITCH / 2}
                    textAnchor="end"
                    dominantBaseline="central"
                    fontSize={LABEL_FONT}
                    fill={TEXT}
                  >
                    <title>
                      {`ΔF1 when added ${gain(row.predicate_f1_gain)} at step ${row.predicate_step}; ${matchedLine(row).toLowerCase()} after this step`}
                    </title>
                    {gain(row.predicate_f1_gain)}
                    <tspan fill={MUTED}> · {row.predicate_n_matched}</tspan>
                  </text>
                )}
                {/* Transparent hover target across the full row */}
                <rect
                  x={0}
                  y={rowTop}
                  width={trackWidth + gainW}
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
            x={left - 8}
            y={top + i * PITCH + PITCH / 2}
            textAnchor="end"
            dominantBaseline="central"
            fontSize={LABEL_FONT}
            fill={row.in_predicate ? TEXT : MUTED}
            fontWeight={row.in_predicate ? 600 : 400}
          >
            {row.feature}
          </text>
        ))}

        {/* X-axis ticks */}
        <g transform={`translate(${left},${top + rows.length * PITCH + 6})`}>
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
            // Flipped: anchor the tooltip's RIGHT edge left of the cursor, so it
            // can never spill past the chart however wide it turns out to be.
            ...(flipTip ? { right: width - tip.x + TIP_PAD } : { left: tip.x + TIP_PAD }),
            top: tip.y + TIP_PAD,
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
          <div>Standalone F1: {fmt(tip.row.clause_f1)}</div>
          {tip.row.in_predicate && tip.row.predicate_f1_gain !== null ? (
            <>
              <div style={{ color: INDIGO }}>
                Step {tip.row.predicate_step} · ΔF1 when added {gain(tip.row.predicate_f1_gain)}
              </div>
              <div style={{ color: MUTED }}>
                Predicate F1: {fmt(tip.row.predicate_f1_before ?? 0)} →{" "}
                {fmt(tip.row.predicate_f1_after ?? 0)}
              </div>
              <div style={{ color: MUTED }}>
                {matchedLine(tip.row)} · P {fmt(tip.row.predicate_precision ?? 0)} · R{" "}
                {fmt(tip.row.predicate_recall ?? 0)}
              </div>
            </>
          ) : (
            <div style={{ color: MUTED }}>Not selected</div>
          )}
          {/* The core (RCM 0.9) run is its own greedy construction — different
              membership and a different step order, so it gets its own block. */}
          {(() => {
            const core = coreByFeature.get(tip.row.feature);
            if (!core?.in_predicate || core.predicate_f1_gain === null) return null;
            return (
              <div style={{ color: MUTED, marginTop: 2 }}>
                Core (RCM 0.9): step {core.predicate_step} · {gain(core.predicate_f1_gain)} ·{" "}
                {fmt(core.predicate_f1_before ?? 0)} → {fmt(core.predicate_f1_after ?? 0)}
              </div>
            );
          })()}
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

// GLOSH outlier-score distribution for one layer's points.
// Single-series magnitude histogram over the fixed GLOSH range [0, 1], so the
// shape is comparable between layers. One hue (bar length already encodes the
// count — no value-ramp), hairline baseline, per-bar hover tooltip.
//
// Props are declared here rather than in charts/props.ts: that file is the
// contract for the six charts ported from Streamlit, and this one is new.
import { useMemo, useState } from "react";
import * as d3 from "d3";
import { useResize } from "../hooks/useResize";
import { theme } from "./theme";

export interface OutlierHistogramProps {
  scores: number[]; // finite GLOSH scores, one per point
}

const HEIGHT = 116;
const PAD_X = 8;
const PLOT_TOP = 8;
const PLOT_BOTTOM = HEIGHT - 26; // room for the tick row
const BIN_COUNT = 24;
const BAR_GAP = 2; // surface gap between adjacent bars

const BG = theme.surface;
const MUTED = theme.muted;
const BAR = theme.accent;

interface Bar {
  x0: number;
  x1: number; // score bounds
  count: number;
  px: number;
  pw: number;
  py: number;
  ph: number;
}

export function OutlierHistogram({ scores }: OutlierHistogramProps) {
  const { ref, size } = useResize<HTMLDivElement>();
  const [hovered, setHovered] = useState<number | null>(null);
  const width = size.width;

  const { bars, maxCount } = useMemo(() => {
    const inner = Math.max(0, width - PAD_X * 2);
    if (inner <= 0 || scores.length === 0) return { bars: [] as Bar[], maxCount: 0 };

    const thresholds = d3.range(BIN_COUNT + 1).map((i) => i / BIN_COUNT);
    const binned = d3.bin<number, number>().domain([0, 1]).thresholds(thresholds)(scores);
    const maxCount = d3.max(binned, (b) => b.length) ?? 0;

    const x = d3.scaleLinear().domain([0, 1]).range([PAD_X, PAD_X + inner]);
    const y = d3.scaleLinear().domain([0, maxCount || 1]).range([PLOT_BOTTOM, PLOT_TOP]);

    const bars = binned.map((b) => {
      const x0 = b.x0 ?? 0;
      const x1 = b.x1 ?? 0;
      const px = x(x0);
      const pw = Math.max(1, x(x1) - px - BAR_GAP);
      const py = y(b.length);
      return { x0, x1, count: b.length, px, pw, py, ph: Math.max(0, PLOT_BOTTOM - py) };
    });
    return { bars, maxCount };
  }, [scores, width]);

  const active = hovered != null ? bars[hovered] : null;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const tickX = (t: number) => PAD_X + (Math.max(0, width - PAD_X * 2) * t);

  return (
    <div ref={ref} style={{ position: "relative", width: "100%", height: HEIGHT, background: BG }}>
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="Distribution of GLOSH outlier scores">
          {bars.map((b, i) => (
            <g key={i} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered((h) => (h === i ? null : h))}>
              {/* Full-height hit area so thin/empty bars are still hoverable. */}
              <rect x={b.px} y={PLOT_TOP} width={b.pw} height={PLOT_BOTTOM - PLOT_TOP} fill="transparent" />
              {b.count > 0 && (
                <rect
                  x={b.px}
                  y={b.py}
                  width={b.pw}
                  height={b.ph}
                  rx={Math.min(3, b.pw / 2)}
                  fill={BAR}
                  opacity={active && active !== b ? 0.55 : 1}
                />
              )}
            </g>
          ))}

          <line
            x1={PAD_X}
            x2={Math.max(PAD_X, width - PAD_X)}
            y1={PLOT_BOTTOM + 0.5}
            y2={PLOT_BOTTOM + 0.5}
            stroke={theme.baseline}
            strokeWidth={1}
          />

          {ticks.map((t) => (
            <text key={t} x={tickX(t)} y={PLOT_BOTTOM + 15} textAnchor="middle" fontSize={10} fill={MUTED}>
              {t}
            </text>
          ))}
          <text x={Math.max(PAD_X, width - PAD_X)} y={PLOT_TOP + 9} textAnchor="end" fontSize={10} fill={MUTED}>
            peak {maxCount} pts
          </text>
        </svg>
      )}

      {active && (
        <div
          style={{
            position: "absolute",
            left: active.px + active.pw / 2,
            top: Math.max(0, active.py - 8),
            transform: "translate(-50%, -100%)",
            background: theme.surface,
            border: `1px solid ${theme.borderStrong}`,
            borderRadius: 6,
            padding: "5px 8px",
            color: theme.textPrimary,
            fontSize: 12,
            lineHeight: 1.3,
            whiteSpace: "nowrap",
            pointerEvents: "none",
            boxShadow: "0 8px 20px -6px rgba(16,24,40,0.28)",
            zIndex: 1,
          }}
        >
          {active.x0.toFixed(2)}–{active.x1.toFixed(2)}: {active.count} {active.count === 1 ? "point" : "points"}
        </div>
      )}
    </div>
  );
}

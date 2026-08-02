import type { CSSProperties } from "react";
import type { ScoreTilesProps } from "./props";
import { qualityColor, theme } from "./theme";

// F — DR-quality score tiles. Replaces src/ui/components/scores.py::render_node_scores.
// Square, hairline-separated tiles: the value stays ink, a thin bar underneath
// carries the quality reading so the number itself never has to be colored.
//
// The bar is a QUALITY reading, not the raw value: longer and cooler is better on
// every tile, so the row scans as one encoding. Trustworthiness and continuity are
// higher-is-better and map straight through; stress and CADI are distortion
// measures (lower is better) and are inverted before they reach a bar.

const fmt = (value: number | null | undefined): string =>
  value == null ? "—" : value.toFixed(3);

const cardStyle: CSSProperties = {
  color: theme.textPrimary,
  marginBottom: 12,
};

const headStyle: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 8,
  marginBottom: 8,
};

const titleStyle: CSSProperties = {
  fontSize: 12.5,
  fontWeight: 600,
};

const captionStyle: CSSProperties = {
  marginLeft: "auto",
  fontSize: 11.5,
  color: theme.faint,
};

const gridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
  gap: 1,
  background: theme.border,
  border: `1px solid ${theme.border}`,
};

const tileStyle: CSSProperties = {
  background: theme.surface,
  padding: "7px 9px 8px",
  minWidth: 0,
};

const barTitle = "Quality — longer is better";

const labelStyle: CSSProperties = {
  fontSize: 10.5,
  letterSpacing: 0.9,
  textTransform: "uppercase",
  color: theme.muted,
};

const valueStyle: CSSProperties = {
  fontSize: 18,
  fontWeight: 500,
  lineHeight: 1.25,
  fontVariantNumeric: "tabular-nums",
};

// `quality` is 0..1 with 1 = best, whichever direction the raw metric runs.
function Tile({ label, value, quality }: { label: string; value: number | null; quality: number | null }) {
  // A scored-but-terrible metric keeps a visible stub: an empty track would read
  // as "no score", which is what the null case (an em-dash value) already means.
  const t = quality == null ? 0 : Math.max(0.03, Math.min(1, quality));
  return (
    <div style={tileStyle}>
      <div style={labelStyle}>{label}</div>
      <div style={valueStyle}>{fmt(value)}</div>
      <div style={{ height: 3, background: theme.grid, marginTop: 3 }} title={barTitle}>
        <div style={{ height: 3, width: `${t * 100}%`, background: qualityColor(quality) ?? "transparent" }} />
      </div>
    </div>
  );
}

export function ScoreTiles({ scores, title }: ScoreTilesProps) {
  if (!scores) return null;

  const captionParts: string[] = [];
  if (scores.mrre_false != null) captionParts.push(`MRRE false ${fmt(scores.mrre_false)}`);
  if (scores.mrre_missing != null) captionParts.push(`missing ${fmt(scores.mrre_missing)}`);
  captionParts.push(`n=${scores.n_points}`);
  if (scores.k != null) captionParts.push(`k=${scores.k}`);

  // Stress is lower-is-better and unbounded in practice; 0.4 reads as "poor".
  // CADI (ZADU's Class Angular Distortion Index) is a distortion in [0, 1] —
  // also lower-is-better. Both become "1 = best" before they reach a bar.
  const stressQuality = scores.stress == null ? null : 1 - Math.min(1, scores.stress / 0.4);
  const cadiQuality = scores.cadi == null ? null : 1 - Math.max(0, Math.min(1, scores.cadi));

  return (
    <div style={cardStyle}>
      <div style={headStyle}>
        <span style={titleStyle}>{title ?? "DR quality"}</span>
        <span style={captionStyle}>{captionParts.join(" · ")}</span>
      </div>
      <div style={gridStyle}>
        <Tile label="Trustworth." value={scores.trustworthiness} quality={scores.trustworthiness} />
        <Tile label="Continuity" value={scores.continuity} quality={scores.continuity} />
        <Tile label="Stress" value={scores.stress} quality={stressQuality} />
        <Tile label="CADI" value={scores.cadi} quality={cadiQuality} />
      </div>
    </div>
  );
}

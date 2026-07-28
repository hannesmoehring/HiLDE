import type { CSSProperties } from "react";
import type { ScoreTilesProps } from "./props";
import { qualityColor, theme } from "./theme";

// F — DR-quality score tiles. Replaces src/ui/components/scores.py::render_node_scores.
// Tiles for Trustworthiness / Continuity / Stress / CADI + an MRRE/context caption.

const fmt = (value: number | null | undefined): string =>
  value == null ? "—" : value.toFixed(3);

const cardStyle: CSSProperties = {
  background: theme.surface,
  border: `1px solid ${theme.border}`,
  borderRadius: 10,
  padding: 14,
  color: theme.textPrimary,
};

const titleStyle: CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: theme.textPrimary,
  marginBottom: 10,
};

const gridStyle: CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 10,
};

const tileStyle: CSSProperties = {
  flex: "1 1 120px",
  minWidth: 120,
  background: theme.surfaceInset,
  border: `1px solid ${theme.border}`,
  borderRadius: 8,
  padding: "10px 12px",
};

const labelStyle: CSSProperties = {
  fontSize: 11,
  color: theme.textSecondary,
  textTransform: "uppercase",
  letterSpacing: 0.4,
  marginBottom: 4,
};

const valueStyle: CSSProperties = {
  fontSize: 22,
  fontWeight: 600,
  lineHeight: 1.1,
  fontVariantNumeric: "tabular-nums",
};

const captionStyle: CSSProperties = {
  fontSize: 11,
  color: theme.muted,
  marginTop: 10,
};

function Tile({ label, value, color }: { label: string; value: number | null; color?: string }) {
  return (
    <div style={tileStyle}>
      <div style={labelStyle}>{label}</div>
      <div style={{ ...valueStyle, color: color ?? theme.textPrimary }}>{fmt(value)}</div>
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

  return (
    <div style={cardStyle}>
      <div style={titleStyle}>{title ?? "DR quality"}</div>
      <div style={gridStyle}>
        <Tile label="Trustworthiness" value={scores.trustworthiness} color={qualityColor(scores.trustworthiness)} />
        <Tile label="Continuity" value={scores.continuity} color={qualityColor(scores.continuity)} />
        <Tile label="Stress" value={scores.stress} color={qualityColor(scores.stress, true)} />
        <Tile label="CADI" value={scores.cadi} />
      </div>
      <div style={captionStyle}>{captionParts.join(" · ")}</div>
    </div>
  );
}

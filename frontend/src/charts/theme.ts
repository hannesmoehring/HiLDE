// Shared light-theme design tokens for the D3 / SVG charts.
// Single source of truth for chart colors — mirrors the CSS custom properties in
// styles.css, so keep the two in sync. Categorical order is CVD-validated
// (worst adjacent ΔE 24.2 on a white surface); do not reorder or re-hue slots
// without re-running the palette validator.
export const theme = {
  // Surfaces & chrome
  surface: "#ffffff", // chart card background
  surfaceInset: "#f2f5fa", // nested tiles / tracks
  border: "#e4e8f0",
  borderStrong: "#d3d9e4",

  // Ink
  textPrimary: "#1b1f28",
  textSecondary: "#4a5262",
  muted: "#8a93a3", // axis ticks & labels

  // Hairlines
  grid: "rgba(24,32,48,0.08)",
  gridStrong: "rgba(24,32,48,0.14)",
  baseline: "#c7cedb",

  // Brand accent (indigo)
  accent: "#4f46e5",
  accentSoft: "#eef0fe",
  accentInk: "#ffffff",

  // Diverging pair for z-score characteristics (blue ↔ red, gray midpoint)
  divPos: "#2a78d6", // z_mean >= 0
  divNeg: "#e0483d", // z_mean < 0
  nonFeature: "#7c5cd4", // columns not selected as a feature (sign shown by bar direction)

  // Predicate bands
  indigo: "#4f46e5", // in-predicate clause
  neutral: "#94a0b5", // not in predicate

  // Recessive "other" for scatter non-matches (visible but quiet on white)
  other: "#c4cbd6",

  // Categorical cluster palette — fixed, CVD-safe order (do not cycle)
  categorical: [
    "#2a78d6", // blue
    "#1baf7a", // aqua
    "#eda100", // yellow
    "#008300", // green
    "#4a3aa7", // violet
    "#e34948", // red
    "#e87ba4", // magenta
    "#eb6834", // orange
  ] as const,
};

// Sequential quality color for higher-is-better metrics (red → amber → green),
// tuned dark enough to read as text on a light tile. `invert` for lower-is-better.
export function qualityColor(value: number | null | undefined, invert = false): string | undefined {
  if (value == null) return undefined;
  const clamped = Math.max(0, Math.min(1, value));
  const t = invert ? 1 - clamped : clamped;
  const hue = 120 * t; // 0 red → 120 green
  return `hsl(${hue.toFixed(0)}, 70%, 40%)`;
}

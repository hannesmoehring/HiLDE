// Shared design tokens for the D3 / SVG charts.
// Neutral paper-white shell, ink type, hairline rules — the chrome carries no hue.
// Color is reserved for DATA: the categorical cluster palette and the diverging
// characteristics pair. Keep in sync with the custom properties in styles.css.
export const theme = {
  // Surfaces & chrome
  surface: "#ffffff",
  border: "#e4e1da",
  borderStrong: "#c9c6be",

  // Ink. Every step used for text clears WCAG AA (4.5:1) on the white surface:
  // 18.1 / 9.1 / 6.0 / 4.8.
  textPrimary: "#161614",
  textSecondary: "#4a4843",
  muted: "#66635b",
  faint: "#757269",

  // Hairlines
  grid: "#f1efea",
  gridStrong: "#e4e1da",
  baseline: "#c9c6be",

  // UI accent is ink — selected states invert to black on white.
  accent: "#161614",
  accentInk: "#ffffff",

  // Diverging pair for z-score characteristics (cool positive / warm negative),
  // plus the neutral used for the binary "column is not a feature" flag. The
  // three co-occur in one chart, so they are validated as a set: worst pair
  // CVD ΔE 13.1, tritan 15.3, normal-vision 17.4, all >= 6:1 on white.
  divPos: "#1f6699",
  divNeg: "#a8431f",
  nonFeature: "#8a8780", // columns not selected as a feature — neutral, no polarity

  // Predicate bands: clause membership is binary, so it reads in ink and grey.
  indigo: "#161614", // in-predicate clause
  neutral: "#a9a69e", // not in predicate
  track: "#f3f1ec", // faint global range track

  // Recessive "other" for scatter non-matches
  other: "#c9c6be",

  // Categorical cluster palette — CVD-validated on the white surface with
  // scripts/validate_palette.js --pairs all (scatter semantics: every cluster is
  // adjacent to every other). All five checks PASS for the full eight slots:
  // worst all-pairs CVD ΔE 9.6, tritan 9.9, normal-vision 17.9, contrast >= 3:1.
  // Slot ORDER is part of the safety mechanism, not cosmetic — a k-cluster layer
  // uses slots 0..k-1, and the order was chosen by farthest-point insertion so
  // small layers get the most separable prefix (k=2: CVD 23.2, k=4: 10.1).
  // Do not reorder or re-hue slots without re-running the palette validator.
  categorical: [
    "#8e1b5c", // plum
    "#269cdb", // sky
    "#a6930b", // gold
    "#06568b", // navy
    "#e1699f", // pink
    "#8758c1", // violet
    "#00835c", // green
    "#ba4900", // burnt orange
  ] as const,
};

// Quality accent for the bar under a score tile — never for the number itself,
// which stays ink. Quality is a POLARITY (good <-> poor), so it reads on the
// diverging pair with a neutral midpoint, deliberately not on the categorical
// palette: those hues mean cluster identity elsewhere in the same viewport.
// `value` is always normalised so that 1 = best, 0 = worst.
export function qualityColor(value: number | null | undefined): string | undefined {
  if (value == null) return undefined;
  const t = Math.max(0, Math.min(1, value));
  return t > 0.75 ? theme.divPos : t > 0.5 ? theme.nonFeature : theme.divNeg;
}

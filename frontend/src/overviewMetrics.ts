// What the overview's tiles can be coloured by, and the colours themselves.
// Pure, like overview.ts; the panel renders the select and legend from this.
import * as d3 from "d3";
import { theme } from "./charts/theme";
import {
  flatten,
  fmtPct,
  fmtValue,
  unassignedShare,
  type OverviewTile,
  type TileKind,
} from "./overview";

export interface OverviewMetric {
  key: string;
  group: "Structure" | "Projection quality" | "Column means";
  label: string;
  // null = this tile has no value under the metric; it stays neutral.
  value: ((t: OverviewTile) => number | null) | null; // null = categorical (node type)
  format: (v: number) => string;
  fixed?: [number, number];
}

// Sequential single-hue ramp for every magnitude. A slate indigo, because each
// other hue in this app already means something: the categorical palette is
// cluster identity (and the scatter right below uses it), blue/rust is z-score
// polarity, teal is target columns, orange is a lasso selection.
// Quantized rather than continuous: tile labels flip from ink to white partway
// along the ramp, and around that luminance NEITHER clears 4.5:1, so the steps are
// chosen to stay out of that band. Worst label contrast per step, light to dark:
// ink 14.7 / 10.8 / 7.6 / 5.2, white 6.5 / 12.5.
export const RAMP = ["#e4e7f5", "#bfc6e8", "#9ca5d7", "#7b85c5", "#4e58a4", "#262d6c"] as const;

// Node-type fills stay hueless: the type is also carried by the frame (internal
// nodes hold their children) and the hatch (unassigned), so it never rests on
// colour alone.
export const KIND_FILL: Record<TileKind, string> = {
  root: "#f3f1ec",
  internal: "#ebe8e1",
  leaf: theme.surface,
  unassigned: "#f6f5f1",
};

export const NO_VALUE_FILL = theme.surface;

const nodeScore = (k: "trustworthiness" | "continuity") => (t: OverviewTile) =>
  t.node?.scores?.[k] ?? null;

export function metricsFor(root: OverviewTile): OverviewMetric[] {
  const all = flatten(root);
  // Features first, then the extra columns (target_* and the like), each in the
  // order the payload lists them.
  const features: string[] = [];
  const extras: string[] = [];
  for (const t of all)
    for (const c of t.node?.rel_characteristics ?? []) {
      const list = c.is_feature === false ? extras : features;
      if (!features.includes(c.feature) && !extras.includes(c.feature)) list.push(c.feature);
    }
  const hasScores = all.some((t) => t.node?.scores != null);
  const out: OverviewMetric[] = [
    { key: "kind", group: "Structure", label: "Node type", value: null, format: String },
    {
      key: "depth",
      group: "Structure",
      label: "Depth",
      value: (t) => (t.kind === "root" ? null : t.depth),
      format: (v) => `depth ${v}`,
    },
    {
      key: "unassigned",
      group: "Structure",
      label: "Unassigned share when split",
      value: unassignedShare,
      format: fmtPct,
      fixed: [0, 1],
    },
  ];
  if (hasScores)
    out.push(
      {
        key: "trustworthiness",
        group: "Projection quality",
        label: "Trustworthiness",
        value: nodeScore("trustworthiness"),
        format: (v) => v.toFixed(3),
      },
      {
        key: "continuity",
        group: "Projection quality",
        label: "Continuity",
        value: nodeScore("continuity"),
        format: (v) => v.toFixed(3),
      },
    );
  for (const col of [...features, ...extras])
    out.push({
      key: `col:${col}`,
      group: "Column means",
      label: col,
      value: (t) => t.means.get(col) ?? null,
      format: fmtValue,
    });
  return out;
}

export interface MetricScale {
  domain: [number, number] | null; // null = no tile has a value
  // One value on every valued tile: the legend shows a single swatch, not a range.
  constant: boolean;
  // How many unassigned tiles carry a value: they have no node of their own, so
  // node-level metrics (scores, unassigned share) never reach them.
  unassigned: "all" | "some" | "none" | "absent";
  fill: (t: OverviewTile) => string;
}

export const CONSTANT_FILL = RAMP[Math.floor(RAMP.length / 2)];

export function scaleFor(metric: OverviewMetric, root: OverviewTile): MetricScale {
  const value = metric.value;
  const all = flatten(root);
  const holes = all.filter((t) => t.kind === "unassigned");
  if (!value)
    return { domain: null, constant: false, unassigned: holes.length ? "all" : "absent", fill: (t) => KIND_FILL[t.kind] };
  const valued = (t: OverviewTile) => {
    const v = value(t);
    return v != null && Number.isFinite(v) ? v : null;
  };
  const nHoles = holes.filter((t) => valued(t) != null).length;
  const unassigned = !holes.length ? "absent" : nHoles === holes.length ? "all" : nHoles ? "some" : "none";
  // The root is the frame around everything; letting it into the extent would
  // only compress the scale for the tiles that are actually compared.
  const vals = all
    .filter((t) => t.kind !== "root")
    .map(valued)
    .filter((v): v is number => v != null);
  // A fixed scale is still "no value" when nothing has one (a leaf root).
  const extent = vals.length ? (metric.fixed ?? (d3.extent(vals) as [number, number])) : null;
  if (!extent) return { domain: null, constant: false, unassigned, fill: () => NO_VALUE_FILL };
  const [lo, hi] = extent;
  const constant = !(hi > lo);
  const q = d3.scaleQuantize<string>().domain([lo, constant ? lo + 1 : hi]).range([...RAMP]);
  return {
    domain: [lo, hi],
    constant,
    unassigned,
    fill: (t) => {
      // Left out of the extent, so painting it would clamp its value to an end
      // of a legend that does not contain it.
      if (t.kind === "root") return KIND_FILL.root;
      const v = valued(t);
      if (v == null) return NO_VALUE_FILL;
      return constant ? CONSTANT_FILL : q(Math.max(lo, Math.min(hi, v)));
    },
  };
}

// Label ink for a fill: whichever of ink or white reads better. Every fill used
// on a tile is one of the steps above, all of which clear 4.5:1 with this pick.
export function textOn(fill: string): string {
  const c = d3.rgb(fill);
  const lin = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  const L = 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const inkL = 0.0083; // theme.textPrimary
  return (L + 0.05) / (inkL + 0.05) >= 1.05 / (L + 0.05) ? theme.textPrimary : "#ffffff";
}

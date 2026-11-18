// The whole cluster tree as one nested structure for the treemap overview.
// Pure: derived from the already-fetched AnalysisResponse.tree, no requests.
import type { Characteristic, TreeNode } from "./types";

export type TileKind = "root" | "internal" | "leaf" | "unassigned";

// Mean shift of one feature relative to the node's parent, in SDs of the parent.
export interface Shift {
  feature: string;
  sd: number;
}

export interface OverviewTile {
  key: string; // TreeNode.id, or "<parent id>/unassigned"
  kind: TileKind;
  // Where activating the tile navigates. An unassigned tile has no node of its
  // own: its points are drawn as noise in the PARENT's layer scatter, so that is
  // where it leads.
  path: number[];
  node: TreeNode | null; // null for unassigned tiles
  owner: TreeNode | null; // the node this tile is a part of (null for the root)
  depth: number;
  size: number;
  label: string; // "root", "C2", "unassigned"
  pathLabel: string; // "C0 → C2", same naming as the exploration panel
  shifts: Shift[]; // largest |sd| first; empty for the root
  means: Map<string, number>; // raw column means, where known
  children: OverviewTile[];
}

// Below this a node's largest move is too small to name it by.
export const CLOSE_TO_PARENT_SD = 0.3;

export const pathLabelOf = (path: number[]): string =>
  path.length ? path.map((c) => `C${c}`).join(" → ") : "root";

// Feature rows only: the non-feature (target_*) z-scores are contrasted against
// the space the cluster was selected out of, not the whole-data scaler, so they
// cannot be differenced along a path.
function featureRows(chars: Characteristic[]): Map<string, Characteristic> {
  const out = new Map<string, Characteristic>();
  for (const c of chars) if (c.is_feature !== false) out.set(c.feature, c);
  return out;
}

// Feature z_means are whole-dataset relative at every depth (one scaler fit at
// the root), so the root's own mean is 0 and its SD 1 by construction even though
// it carries no rel_characteristics. A move relative to the parent is then the
// difference of z_means, rescaled by the parent's own spread.
interface FeatureStats {
  z: Map<string, number>;
  sd: Map<string, number>;
}

function statsOf(node: TreeNode, isRoot: boolean): FeatureStats | null {
  if (isRoot) return null; // stands for "mean 0, SD 1 on every feature"
  const z = new Map<string, number>();
  const sd = new Map<string, number>();
  for (const [f, c] of featureRows(node.rel_characteristics)) {
    if (c.z_mean != null) z.set(f, c.z_mean);
    if (c.z_std != null) sd.set(f, c.z_std);
  }
  return z.size ? { z, sd } : null;
}

function shiftsFrom(
  z: Map<string, number>,
  parent: FeatureStats | null,
  parentIsRoot: boolean,
): Shift[] {
  if (!parentIsRoot && !parent) return [];
  const out: Shift[] = [];
  for (const [f, v] of z) {
    const pz = parentIsRoot ? 0 : parent!.z.get(f);
    if (pz == null) continue;
    const psd = parentIsRoot ? 1 : parent!.sd.get(f);
    // A parent with no spread on a feature cannot have a child that differs on it.
    const scale = psd != null && psd > 1e-9 ? psd : 1;
    out.push({ feature: f, sd: (v - pz) / scale });
  }
  return out.sort((a, b) => Math.abs(b.sd) - Math.abs(a.sd));
}

function rawMeans(node: TreeNode): Map<string, number> {
  const m = new Map<string, number>();
  for (const c of node.rel_characteristics)
    if (c.raw_mean != null && Number.isFinite(c.raw_mean)) m.set(c.feature, c.raw_mean);
  return m;
}

// The noise points' mean is what is left of the parent's total once the
// children's are taken out. Only possible where the parent's own mean is known.
function residualMeans(
  parentN: number,
  parentMeans: Map<string, number>,
  kids: TreeNode[],
  kidMeans: Map<string, number>[],
  n: number,
): Map<string, number> {
  const out = new Map<string, number>();
  for (const [f, pm] of parentMeans) {
    let sum = parentN * pm;
    let ok = true;
    kids.forEach((k, i) => {
      const v = kidMeans[i].get(f);
      if (v == null) ok = false;
      else sum -= k.n_points * v;
    });
    if (ok) out.set(f, sum / n);
  }
  return out;
}

function residualZ(
  parentN: number,
  parent: FeatureStats | null,
  parentIsRoot: boolean,
  kids: TreeNode[],
  n: number,
): Map<string, number> {
  const out = new Map<string, number>();
  if (!parentIsRoot && !parent) return out;
  const kidZ = kids.map((k) => statsOf(k, false));
  const features = parentIsRoot
    ? new Set(kidZ.flatMap((s) => (s ? [...s.z.keys()] : [])))
    : new Set(parent!.z.keys());
  for (const f of features) {
    let sum = parentN * (parentIsRoot ? 0 : parent!.z.get(f)!);
    let ok = true;
    kids.forEach((k, i) => {
      const v = kidZ[i]?.z.get(f);
      if (v == null) ok = false;
      else sum -= k.n_points * v;
    });
    if (ok) out.set(f, sum / n);
  }
  return out;
}

export function buildOverview(root: TreeNode): OverviewTile {
  function visit(
    node: TreeNode,
    path: number[],
    parent: TreeNode | null,
  ): OverviewTile {
    const isRoot = parent === null;
    const kind: TileKind = isRoot ? "root" : node.is_leaf ? "leaf" : "internal";
    const own = statsOf(node, isRoot);
    const tile: OverviewTile = {
      key: node.id,
      kind,
      path,
      node,
      owner: parent,
      depth: path.length,
      size: node.n_points,
      label: isRoot ? "root" : `C${path[path.length - 1]}`,
      pathLabel: pathLabelOf(path),
      shifts:
        isRoot || !own
          ? []
          : shiftsFrom(own.z, statsOf(parent!, parent === root), parent === root),
      means: rawMeans(node),
      children: [],
    };
    const kids = node.is_leaf ? [] : (node.children ?? []);
    tile.children = kids.map((k, i) => visit(k, [...path, i], node));
    if (!node.is_leaf) {
      const assigned = kids.reduce((s, k) => s + k.n_points, 0);
      const n = node.n_points - assigned;
      if (n > 0) {
        const z = residualZ(node.n_points, own, isRoot, kids, n);
        tile.children.push({
          key: `${node.id}/unassigned`,
          kind: "unassigned",
          path,
          node: null,
          owner: node,
          depth: path.length + 1,
          size: n,
          label: "unassigned",
          pathLabel: `${pathLabelOf(path)} → unassigned`,
          shifts: shiftsFrom(z, own, isRoot),
          means: residualMeans(
            node.n_points,
            tile.means,
            kids,
            tile.children.map((c) => c.means),
            n,
          ),
          children: [],
        });
      }
    }
    return tile;
  }
  return visit(root, [], null);
}

// Second tile line: the feature whose mean moves most relative to the parent.
export function shiftName(tile: OverviewTile): string | null {
  const top = tile.shifts[0];
  if (!top) return null;
  if (Math.abs(top.sd) < CLOSE_TO_PARENT_SD) return "close to parent";
  return `${top.sd > 0 ? "↑" : "↓"} ${top.feature}`;
}

export function kindLabel(kind: TileKind): string {
  return kind === "root" ? "root" : kind;
}

// Every tile in pre-order (parent before its children): the keyboard order.
export function flatten(tile: OverviewTile): OverviewTile[] {
  const out: OverviewTile[] = [];
  const walk = (t: OverviewTile) => {
    out.push(t);
    t.children.forEach(walk);
  };
  walk(tile);
  return out;
}

// Keys of the nodes the current treePath runs through, root excluded.
export function pathKeys(root: TreeNode, treePath: number[]): string[] {
  const keys: string[] = [];
  let node = root;
  for (const idx of treePath) {
    const next = node.is_leaf ? undefined : node.children?.[idx];
    if (!next) break;
    keys.push(next.id);
    node = next;
  }
  return keys;
}

// Unassigned share of a split node; null where nothing was split.
export function unassignedShare(tile: OverviewTile): number | null {
  if (tile.kind !== "internal" && tile.kind !== "root") return null;
  if (!tile.children.length && tile.node?.is_leaf) return null;
  const u = tile.children.find((c) => c.kind === "unassigned");
  return tile.size > 0 ? (u?.size ?? 0) / tile.size : null;
}

export const fmtCount = (n: number): string => n.toLocaleString("en-US");

export const fmtPct = (v: number): string =>
  `${(100 * v).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;

export function fmtValue(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

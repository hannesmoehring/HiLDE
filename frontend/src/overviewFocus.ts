// Tree lookups the overview map's zoom needs: which tile a key names, whose
// child it is, and how two tiles relate. Pure, so the zoom rules stay testable
// without a DOM.
import type { OverviewTile } from "./overview";

export interface TileIndex {
  root: OverviewTile;
  byKey: Map<string, OverviewTile>;
  parentOf: Map<string, OverviewTile | null>;
}

export function indexTree(root: OverviewTile): TileIndex {
  const byKey = new Map<string, OverviewTile>();
  const parentOf = new Map<string, OverviewTile | null>();
  const walk = (t: OverviewTile, parent: OverviewTile | null) => {
    byKey.set(t.key, t);
    parentOf.set(t.key, parent);
    t.children.forEach((c) => walk(c, t));
  };
  walk(root, null);
  return { root, byKey, parentOf };
}

// Root first, the tile itself last; empty for a key not in the tree.
export function chainTo(index: TileIndex, key: string): OverviewTile[] {
  const out: OverviewTile[] = [];
  let t = index.byKey.get(key) ?? null;
  while (t) {
    out.unshift(t);
    t = index.parentOf.get(t.key) ?? null;
  }
  return out;
}

// True when `key` is `ancestor` itself or lies anywhere below it.
export function isWithin(index: TileIndex, ancestor: string, key: string): boolean {
  return chainTo(index, key).some((t) => t.key === ancestor);
}

// The deepest tile both keys lie within: what the map widens to so it shows both.
export function commonAncestor(index: TileIndex, a: string, b: string): OverviewTile {
  const ca = chainTo(index, a);
  const cb = chainTo(index, b);
  let i = 0;
  while (i < ca.length && i < cb.length && ca[i].key === cb[i].key) i++;
  return ca[i - 1] ?? index.root;
}

export const isFrame = (t: OverviewTile) => t.children.length > 0;

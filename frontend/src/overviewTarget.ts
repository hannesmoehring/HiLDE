// Where activating an overview tile actually leads, given how deep the layer
// stack below currently renders. The rail can lower hierarchical_layers without a
// rebuild, so the tree may be deeper than the stack: the map has to announce the
// node that will open, not the one that was clicked.
import { pathLabelOf, type OverviewTile } from "./overview";

export interface TileTarget {
  path: number[];
  // For the accessible name: "Activate: <what>."
  what: string;
}

const layersWord = (n: number) => `${n} layer${n === 1 ? "" : "s"}`;

export function targetOf(t: OverviewTile, nLayers: number): TileTarget {
  const depth = Math.max(0, nLayers);
  const path = t.path.slice(0, depth);
  const limit = `the layer stack is limited to ${layersWord(depth)}`;
  if (t.kind === "unassigned") {
    // Layer L draws the node at a path of length L - 1, noise included, so the
    // parent's scatter exists only while its path is shorter than the stack.
    if (t.path.length < depth) return { path, what: "opens the layer that shows these points as noise" };
    return {
      path,
      what: `opens ${pathLabelOf(path)} in the exploration panel; ${limit}, so these points are not marked as noise`,
    };
  }
  if (t.path.length <= depth) return { path, what: "opens it in the layers below" };
  return {
    path,
    what: path.length
      ? `opens its ancestor ${pathLabelOf(path)}; ${limit}`
      : `returns to the root; ${limit}`,
  };
}

// Deepest cluster path in the tree, to tell whether the stack cuts anything off.
export function maxClusterDepth(tiles: OverviewTile[]): number {
  return tiles.reduce((m, t) => (t.kind === "unassigned" ? m : Math.max(m, t.path.length)), 0);
}

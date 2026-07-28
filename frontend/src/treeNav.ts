// Client-side tree navigation — port of src/ui/tree_nav.py.
// Drill-down happens entirely in the browser on the already-fetched tree.
import type { TreeNode } from "./types";

export function getNodeAtPath(root: TreeNode, path: number[]): TreeNode {
  let node = root;
  for (const idx of path) {
    if (node.is_leaf || !node.children) break;
    node = node.children[idx];
  }
  return node;
}

export function childSize(child: TreeNode): number {
  return child.n_points;
}

// The layer index (1-based) at which a given path terminates in a leaf, or null
// if the path is still on internal nodes. Used to decide when to show the
// exploration panel vs another topography layer.
export function isLeafAtPath(root: TreeNode, path: number[]): boolean {
  return getNodeAtPath(root, path).is_leaf;
}

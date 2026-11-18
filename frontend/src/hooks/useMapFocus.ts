// The overview map's own state, kept apart from the app path: which frame the
// map is zoomed into (`focus`) and which tile is pinned in the details column
// (`selected`). Browsing the map never moves the layer stack; only exploring
// does, and the panel does that through the same navigate as everything else.
import { useEffect, useMemo, useRef, useState } from "react";
import type { OverviewTile } from "../overview";
import { chainTo, commonAncestor, indexTree, isFrame, isWithin } from "../overviewFocus";

export interface MapFocus {
  focus: OverviewTile; // always a frame; the root by default
  selected: OverviewTile | null;
  chain: OverviewTile[]; // root → focus, for the map breadcrumb
  zoomTo: (t: OverviewTile) => void;
  zoomOut: () => void;
  select: (t: OverviewTile) => void;
  // The map itself sent the app to `landingKey` while exploring `t`.
  explored: (t: OverviewTile, landingKey: string) => void;
}

// `currentKey`: the node the layer stack is on (the tree root when at the root).
export function useMapFocus(tree: OverviewTile, currentKey: string): MapFocus {
  const index = useMemo(() => indexTree(tree), [tree]);
  const [focusRaw, setFocusRaw] = useState(tree.key);
  const [selectedRaw, setSelectedRaw] = useState<string | null>(currentKey === tree.key ? null : currentKey);
  // A path change the map caused itself: it must not re-pin or widen.
  const ownLanding = useRef<string | null>(null);
  const lastCurrent = useRef(currentKey);

  // A rebuild swaps the tree: a key it no longer has falls back, never sticks.
  const focusTile = index.byKey.get(focusRaw);
  const focus = focusTile && isFrame(focusTile) ? focusTile : tree;
  const selected = (selectedRaw && index.byKey.get(selectedRaw)) || null;

  useEffect(() => {
    if (!index.byKey.has(focusRaw)) setFocusRaw(tree.key);
    if (selectedRaw && !index.byKey.has(selectedRaw)) setSelectedRaw(null);
  }, [index, tree.key, focusRaw, selectedRaw]);

  // The stack moved. From elsewhere (a layer drill-down, the path crumbs), pin the
  // node it is on and widen the map until it shows that node; browsing deeper
  // than the stack is fine, hiding where the stack is would not be.
  useEffect(() => {
    if (lastCurrent.current === currentKey) return;
    lastCurrent.current = currentKey;
    if (ownLanding.current === currentKey) {
      ownLanding.current = null;
      return;
    }
    ownLanding.current = null;
    if (!index.byKey.has(currentKey)) return;
    setFocusRaw((f) => (isWithin(index, f, currentKey) ? f : commonAncestor(index, f, currentKey).key));
    setSelectedRaw(currentKey === tree.key ? null : currentKey);
  }, [currentKey, index, tree.key]);

  // The landing an explore predicts is good for the render its navigate causes
  // (App sets the path synchronously, batched with `explored`), not beyond: if
  // that render did not land there, a later navigation from elsewhere to the
  // same key must still pin and widen. Runs after the effect above, every render.
  useEffect(() => {
    ownLanding.current = null;
  });

  function zoomTo(t: OverviewTile) {
    if (!isFrame(t)) return;
    setFocusRaw(t.key);
    // A pinned tile the map no longer shows would describe something off screen.
    if (selected && !isWithin(index, t.key, selected.key)) setSelectedRaw(null);
  }

  function zoomOut() {
    const parent = index.parentOf.get(focus.key);
    if (parent) setFocusRaw(parent.key);
  }

  function explored(t: OverviewTile, landingKey: string) {
    setSelectedRaw(t.key);
    ownLanding.current = landingKey === currentKey ? null : landingKey;
  }

  return {
    focus,
    selected,
    chain: chainTo(index, focus.key),
    zoomTo,
    zoomOut,
    select: (t) => setSelectedRaw(t.key),
    explored,
  };
}

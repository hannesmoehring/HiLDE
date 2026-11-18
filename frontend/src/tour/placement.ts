// Where the tour card goes relative to the spotlighted element: below it if it
// fits, else above, else beside it, and always clamped inside the viewport.
export interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

const GAP = 12;
const MARGIN = 12;

export function placeCard(target: Box | null, card: { width: number; height: number }) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const clampX = (x: number) => Math.min(Math.max(MARGIN, x), vw - card.width - MARGIN);
  const clampY = (y: number) => Math.min(Math.max(MARGIN, y), vh - card.height - MARGIN);
  if (!target) return { top: clampY((vh - card.height) / 2), left: clampX((vw - card.width) / 2) };

  const below = target.top + target.height + GAP;
  const above = target.top - GAP - card.height;
  const right = target.left + target.width + GAP;
  const left = target.left - GAP - card.width;
  if (below + card.height <= vh - MARGIN) return { top: below, left: clampX(target.left) };
  if (above >= MARGIN) return { top: above, left: clampX(target.left) };
  if (right + card.width <= vw - MARGIN) return { top: clampY(target.top), left: right };
  if (left >= MARGIN) return { top: clampY(target.top), left };
  // A target taller and wider than the viewport: sit in its lower-right corner.
  return { top: vh - card.height - MARGIN, left: vw - card.width - MARGIN };
}

/** The spotlight hugs the visible part of the target only: a panel taller than
 *  the viewport would otherwise put its outline off screen. */
export function visibleBox(r: DOMRect, pad = 6): Box {
  const top = Math.max(r.top - pad, 4);
  const left = Math.max(r.left - pad, 4);
  const bottom = Math.min(r.bottom + pad, window.innerHeight - 4);
  const right = Math.min(r.right + pad, window.innerWidth - 4);
  return { top, left, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

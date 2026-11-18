// The line under the treemap: what a click on the hovered tile would do, since
// a click acts one level below the focus rather than on the tile under the
// pointer. Sighted-pointer help only; each tile's accessible name says the same.
import type { OverviewTile } from "../overview";
import { isFrame } from "../overviewFocus";

interface Props {
  step: OverviewTile | null; // the tile a click would act on
  isFocus: boolean; // that tile is the focused frame itself (its header band)
  canZoomOut: boolean;
}

export function TreemapCaption({ step, isFocus, canZoomOut }: Props) {
  let text: string;
  if (!step)
    text = `Click a tile to zoom in one level · double-click or E to explore it in the layers${
      canZoomOut ? " · Esc or the top band zooms out" : ""
    }`;
  else if (isFocus)
    text = canZoomOut
      ? `Click: zoom out one level · double-click: explore ${step.pathLabel}`
      : "The whole hierarchy · click a tile inside to zoom in";
  else
    text = `Click: ${isFrame(step) ? "zoom into" : "select"} ${step.pathLabel} · double-click: explore it`;
  return (
    <p className="treemap__caption" aria-hidden="true">
      {text}
    </p>
  );
}

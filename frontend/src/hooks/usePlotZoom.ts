// Zoom/pan for a plot area drawn inside a margin-translated <g>. The behaviour is
// bound to that <g> rather than the <svg>, so the transform lives in plot-area
// pixels — the same space the caller's scales and selection tools already use.
import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import * as d3 from "d3";

const MAX_ZOOM = 40;
const STEP = 1.4;
const TRANSITION_MS = 180;

export interface PlotZoom {
  transform: d3.ZoomTransform;
  zoomed: boolean;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
}

/**
 * @param gRef     the plot-area group; must contain something that takes pointer events
 * @param width    plot-area width in px (0 while unmeasured — the zoom stays unbound)
 * @param height   plot-area height in px
 * @param panning  whether a drag pans. The wheel always zooms; drags are only claimed
 *                 while this is true, so a selection tool can own them otherwise.
 * @param resetKey the view snaps back to identity whenever this changes identity
 * @param fitScale px per data unit of the caller's fitted (unzoomed) layout, which
 *                 maps data to pixels about the plot centre. Lets a resize keep the
 *                 same data under the view centre instead of jumping.
 */
export function usePlotZoom(
  gRef: RefObject<SVGGElement>,
  width: number,
  height: number,
  panning: boolean,
  resetKey: unknown,
  fitScale = 1,
): PlotZoom {
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);
  const zoomRef = useRef<d3.ZoomBehavior<SVGGElement, unknown> | null>(null);

  // Read inside the d3 filter, so toggling pan mode never rebinds the behaviour.
  const panningRef = useRef(panning);
  panningRef.current = panning;
  // The layout the node's stored transform was expressed in.
  const layoutRef = useRef<{ width: number; height: number; fitScale: number } | null>(null);

  useEffect(() => {
    const g = gRef.current;
    if (!g || width <= 0 || height <= 0) return;
    const bounds: [[number, number], [number, number]] = [
      [0, 0],
      [width, height],
    ];
    const zoom = d3
      .zoom<SVGGElement, unknown>()
      .scaleExtent([1, MAX_ZOOM])
      .extent(bounds)
      // With a minimum scale of 1 this keeps the fitted view as the outer limit:
      // the data can never be dragged out of the frame.
      .translateExtent(bounds)
      .filter((event: Event) => {
        if (event.type === "wheel") return true;
        const e = event as MouseEvent;
        return panningRef.current && !e.ctrlKey && !e.button;
      })
      .on("zoom", (event: d3.D3ZoomEvent<SVGGElement, unknown>) => setTransform(event.transform));
    zoomRef.current = zoom;
    const sel = d3.select(g);
    sel.call(zoom);
    // The node keeps its transform across a rebind (resize), but the fitted
    // layout under it has moved: re-express the transform so the data point at
    // the old view centre sits at the new one, then constrain to the new bounds.
    const prev = layoutRef.current;
    const t = d3.zoomTransform(g);
    if (prev && t.k !== 1 && fitScale > 0 && prev.fitScale > 0) {
      const ratio = fitScale / prev.fitScale;
      const cx = ((prev.width / 2 - t.x) / t.k - prev.width / 2) * ratio + width / 2;
      const cy = ((prev.height / 2 - t.y) / t.k - prev.height / 2) * ratio + height / 2;
      sel.call(zoom.transform, d3.zoomIdentity.translate(width / 2 - t.k * cx, height / 2 - t.k * cy).scale(t.k));
    }
    layoutRef.current = { width, height, fitScale };
    sel.call(zoom.translateBy, 0, 0);
    return () => {
      sel.on(".zoom", null);
      zoomRef.current = null;
    };
  }, [gRef, width, height, fitScale]);

  const apply = useCallback(
    (fn: (sel: d3.Selection<SVGGElement, unknown, null, undefined>, zoom: d3.ZoomBehavior<SVGGElement, unknown>) => void) => {
      const g = gRef.current;
      const zoom = zoomRef.current;
      if (g && zoom) fn(d3.select<SVGGElement, unknown>(g), zoom);
    },
    [gRef],
  );

  useEffect(() => {
    const g = gRef.current;
    if (g && zoomRef.current) d3.select<SVGGElement, unknown>(g).call(zoomRef.current.transform, d3.zoomIdentity);
    else setTransform(d3.zoomIdentity);
  }, [gRef, resetKey]);

  const zoomIn = useCallback(
    () => apply((sel, zoom) => sel.transition().duration(TRANSITION_MS).call(zoom.scaleBy, STEP)),
    [apply],
  );
  const zoomOut = useCallback(
    () => apply((sel, zoom) => sel.transition().duration(TRANSITION_MS).call(zoom.scaleBy, 1 / STEP)),
    [apply],
  );
  const reset = useCallback(
    () => apply((sel, zoom) => sel.transition().duration(TRANSITION_MS).call(zoom.transform, d3.zoomIdentity)),
    [apply],
  );

  const zoomed = transform.k !== 1 || transform.x !== 0 || transform.y !== 0;
  return { transform, zoomed, zoomIn, zoomOut, reset };
}

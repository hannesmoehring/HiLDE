// Cluster-movement client state, plus the pure helpers ClusterScatter needs.
//
// The original plan put this state in a `Navigation` struct in App.tsx. There is
// no such struct — App.tsx holds flat `useState`, and `Navigation` there is a
// component, not a state container — so the machine typed as `MovementState` in
// types.ts lives here as its own hook.
//
// Nothing in this file touches the DOM or d3: the hook is plain state plus
// `runMovement`, and the geometry/resolution helpers are ordinary functions, so
// both can be exercised without rendering a chart.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { runMovement } from "./api";
import type {
  AnalysisResponse,
  MovementRequest,
  MovementResponse,
  MovementState,
  MovementTarget,
} from "./types";

// ── Geometry ────────────────────────────────────────────────────────────────

/** The equal-aspect mapping ClusterScatter's `plotted` memo applies to
 *  `embedding_original`:  px = x0 + (x − cx)·k,  py = y0 − (y − cy)·k.
 *  y is flipped because screen y grows downwards. */
export interface PlotFrame {
  k: number; // px per embedding unit, the same on both axes
  x0: number; // plot-area centre, screen x
  y0: number; // plot-area centre, screen y
  cx: number; // embedding-space centre, x
  cy: number; // embedding-space centre, y
}

/** The fields of a `d3.ZoomTransform` used here. A real transform satisfies this
 *  structurally, so the chart passes its own straight in. */
export interface ZoomView {
  k: number;
  x: number;
  y: number;
}

export interface EmbeddingPoint {
  x: number;
  y: number;
}

export interface PlotPoint {
  px: number;
  py: number;
}

/** SVG-local screen point → `embedding_original` units, which is what a
 *  `MovementTarget` carries. Two inversions, in this order: the current d3 zoom
 *  transform (this is exactly `transform.invert([sx, sy])`), then the
 *  equal-aspect plot transform. Getting both right is what makes a click resolve
 *  correctly while the plot is zoomed AND panned. */
export function screenToEmbedding(
  sx: number,
  sy: number,
  zoom: ZoomView,
  frame: PlotFrame,
): EmbeddingPoint {
  const px = (sx - zoom.x) / zoom.k;
  const py = (sy - zoom.y) / zoom.k;
  return {
    x: frame.cx + (px - frame.x0) / frame.k,
    y: frame.cy - (py - frame.y0) / frame.k,
  };
}

/** `embedding_original` units → plot space (before the zoom transform): the
 *  inverse of the second step above. Places ghosts, centroids and the target
 *  marker on the same grid the points were laid out on. */
export function embeddingToPlot(p: EmbeddingPoint, frame: PlotFrame): PlotPoint {
  return {
    px: frame.x0 + (p.x - frame.cx) * frame.k,
    py: frame.y0 - (p.y - frame.cy) * frame.k,
  };
}

// ── Target resolution ───────────────────────────────────────────────────────

/** One drawn point of the parent embedding, as ClusterScatter lays it out. */
export interface ScatterPoint {
  px: number; // plot space, before the zoom transform
  py: number;
  x: number; // embedding_original units
  y: number;
  child: number; // index into node.children; -1 = HDBSCAN noise
  row: number; // source dataframe row id
}

/** What the user hit. The chart classifies the DOM event; this module decides
 *  what it means. */
export type MovementClick =
  | { kind: "cluster"; childIndex: number } // a centroid label or a legend chip
  | { kind: "point"; point: ScatterPoint } // a drawn point (cluster or noise)
  | { kind: "background"; sx: number; sy: number }; // empty space, SVG-local screen px

export interface ResolveContext {
  points: readonly ScatterPoint[];
  sourceChildIndex: number; // the cluster being moved; clicks on it resolve to nothing
  zoom: ZoomView;
  frame: PlotFrame;
  snapRadiusPx?: number;
}

/** A resolution either produces a target or explains why it produced none. */
export type TargetResolution =
  | { target: MovementTarget; hint: null }
  | { target: null; hint: string };

/** Screen-space snap radius for background clicks, measured AFTER the zoom
 *  transform so it stays 10 physical pixels at every zoom level. */
export const SNAP_RADIUS_PX = 10;

/** How long the strength slider must settle before a UMAP re-request goes out.
 *  Long enough that dragging across the track sends one request rather than
 *  thirty, short enough that letting go feels immediate. PCA never waits: it
 *  issues no request at all. */
const STRENGTH_DEBOUNCE_MS = 350;

const SOURCE_HINT = "That is the cluster being moved — pick a different destination.";

/** Resolve a click into a movement target, in the precedence the spec fixes:
 *
 *   1. a centroid label / legend entry of another cluster → that cluster
 *   2. a point belonging to another cluster                → that cluster
 *   3. the nearest other-cluster point within `SNAP_RADIUS_PX` screen pixels of
 *      a background click (measured after the zoom transform) → that cluster
 *   4. otherwise                                            → a free point
 *
 *  Anything belonging to the source cluster resolves to nothing: no target, and
 *  a hint the chart can show instead of silently doing nothing. Noise belongs to
 *  no child, so a noise point can only ever be a free point — at its own
 *  projected coordinate, not at the click. */
export function resolveMovementTarget(
  click: MovementClick,
  ctx: ResolveContext,
): TargetResolution {
  const snap = ctx.snapRadiusPx ?? SNAP_RADIUS_PX;

  // 1 — a label or legend chip names its cluster outright.
  if (click.kind === "cluster") {
    if (click.childIndex === ctx.sourceChildIndex) return { target: null, hint: SOURCE_HINT };
    return { target: { kind: "cluster", child_index: click.childIndex }, hint: null };
  }

  // 2 — a point carries its own membership.
  if (click.kind === "point") {
    const p = click.point;
    if (p.child === ctx.sourceChildIndex) return { target: null, hint: SOURCE_HINT };
    if (p.child >= 0) return { target: { kind: "cluster", child_index: p.child }, hint: null };
    // Noise: no child to name, so the destination is where it sits.
    return { target: { kind: "point", x: p.x, y: p.y }, hint: null };
  }

  // 3 — background click: snap to the nearest OTHER-cluster point inside the
  //     radius. Distances are taken on screen, after the zoom transform, so the
  //     radius means the same thing however far the user has zoomed in.
  let nearest: ScatterPoint | null = null;
  let nearestDist = Infinity;
  for (const p of ctx.points) {
    if (p.child < 0 || p.child === ctx.sourceChildIndex) continue;
    const zx = p.px * ctx.zoom.k + ctx.zoom.x;
    const zy = p.py * ctx.zoom.k + ctx.zoom.y;
    const d = Math.hypot(zx - click.sx, zy - click.sy);
    if (d < nearestDist) {
      nearestDist = d;
      nearest = p;
    }
  }
  if (nearest && nearestDist <= snap) {
    return { target: { kind: "cluster", child_index: nearest.child }, hint: null };
  }

  // 4 — free point at the inverted coordinate.
  const e = screenToEmbedding(click.sx, click.sy, ctx.zoom, ctx.frame);
  if (!Number.isFinite(e.x) || !Number.isFinite(e.y)) {
    return { target: null, hint: "That position could not be converted into projection coordinates." };
  }
  return { target: { kind: "point", x: e.x, y: e.y }, hint: null };
}

// ── State machine ───────────────────────────────────────────────────────────

export function sameTarget(a: MovementTarget | null, b: MovementTarget | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind === "cluster" && b.kind === "cluster") return a.child_index === b.child_index;
  if (a.kind === "point" && b.kind === "point") return a.x === b.x && a.y === b.y;
  return false;
}

/** `TreeNode.id` for a navigation path: [] → "root", [2, 0] → "root/2/0". */
export function nodeIdForPath(path: readonly number[]): string {
  return ["root", ...path].join("/");
}

/** True while `nodeId` is the node the user stands on or an ancestor of it —
 *  i.e. they have not navigated AWAY from the movement's parent. Drilling deeper
 *  leaves the parent projection on screen, so it keeps the movement alive;
 *  stepping to a sibling or back up removes it, and clears. */
export function nodeIsOnPath(nodeId: string, path: readonly number[]): boolean {
  const own = nodeId.split("/").slice(1).map(Number);
  if (own.length > path.length) return false;
  return own.every((c, i) => c === path[i]);
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** The response re-expressed at a different slider strength.
 *
 *  Feature deltas are linear in α for every method (`X' = X + αΔ` by
 *  construction), so `applied_delta_raw` always rescales. The GHOST positions
 *  only interpolate for PCA: the backend hands back `visible_displacement`, the
 *  full-strength projected displacement already in visible coordinates, and
 *  `ghost = Y_source + α·d_vis` is exact for a linear reducer. UMAP's
 *  out-of-sample transform is not linear, so its preview points are left exactly
 *  as the server drew them until a re-request replaces them.
 *
 *  `metrics` are NOT recomputed — they describe `response.applied_strength`. */
export function previewAtStrength(res: MovementResponse, strength: number): MovementResponse {
  if (strength === res.applied_strength) return res;
  const shift = strength - res.applied_strength;
  const [dx, dy] = res.visible_displacement;
  const linear = res.method === "PCA";
  return {
    ...res,
    applied_strength: strength,
    feature_changes: res.feature_changes.map((f) => ({
      ...f,
      applied_delta_raw:
        f.recommended_delta_raw == null ? f.applied_delta_raw : f.recommended_delta_raw * strength,
    })),
    preview_points: linear
      ? res.preview_points.map((p) => ({ row_id: p.row_id, x: p.x + shift * dx, y: p.y + shift * dy }))
      : res.preview_points,
  };
}

/** Everything the stale-response guard compares against, as it stands at the
 *  moment a response settles — never as it was when the request went out. */
export interface StaleGuardContext {
  generation: number; // the hook's request counter right now
  state: MovementState; // the state right now
  analysisId: string | null; // the analysis id of the run on screen right now
}

/** The stale-response guard.
 *
 *  A response is committed only when EVERY check below still holds. Anything
 *  else — a superseded request, a cancel, a navigation away, a rebuild, or two
 *  requests answered out of order — is dropped without touching state. Pass
 *  `res = null` for a rejected request, which is checked the same way minus the
 *  response-side comparisons. */
export function isCurrentResponse(
  gen: number,
  req: MovementRequest,
  res: MovementResponse | null,
  ctx: StaleGuardContext,
): boolean {
  // 1. generation — a newer request, or a cancel, has been issued since.
  if (gen !== ctx.generation) return false;
  const s = ctx.state;
  // 2. we must still be asking exactly this question. `loading` is a first
  //    request; `ready` is a UMAP strength re-request, which deliberately keeps
  //    the existing preview on screen instead of blanking it, so it never
  //    passes through `loading`. Ordering is enforced by the generation check
  //    above, not by the phase — `idle` / `choosing` / `error` still reject.
  if (s.phase !== "loading" && s.phase !== "ready") return false;
  // 3. node.
  if (s.nodeId !== req.node_id) return false;
  // 4. source child.
  if (s.sourceChildIndex !== req.source_child_index) return false;
  // 5. target.
  if (!sameTarget(s.target, req.target)) return false;
  // 6. analysis id — the run on screen must still be the run we asked about.
  if (ctx.analysisId !== req.analysis_id) return false;
  if (res) {
    // And the server must be answering the question we asked. Point-target x/y
    // are deliberately NOT compared: the backend reports the REACHABLE
    // destination there, which may legitimately differ from the click.
    if (res.analysis_id !== req.analysis_id) return false;
    if (res.node_id !== req.node_id) return false;
    if (res.source_child_index !== req.source_child_index) return false;
    if (res.target.kind !== req.target.kind) return false;
    if (req.target.kind === "cluster" && res.target.child_index !== req.target.child_index) {
      return false;
    }
  }
  return true;
}

export interface UseMovementArgs {
  /** The run on screen. A NEW object here (a rebuild, a dataset swap) clears any
   *  movement in flight: a movement describes one (run, node, source) and none
   *  of the three survive a new analysis. Pass a stable reference. */
  analysis: AnalysisResponse | null;
  /** Only a clear-condition. Everything the request needs to IDENTIFY the run —
   *  dataset, feature columns, config — is read from `analysis.meta`, because
   *  the config rail describes the next build rather than the one on screen. */
  datasetKey: string;
  treePath: readonly number[];
}

export interface MovementController {
  state: MovementState;
  /** `state.response` re-expressed at the slider's strength — what the chart
   *  should draw as ghosts. Null unless `state.phase === "ready"`. */
  preview: MovementResponse | null;
  /** True only when the preview on screen is EXACTLY what the server would
   *  write on Apply: phase `ready`, no strength re-request pending or in
   *  flight, the response answered at the slider's own strength, and the
   *  response belongs to the run on screen. `ready` alone is not enough — the
   *  hook stays `ready` while a UMAP refinement is pending or after one failed. */
  settled: boolean;
  /** Enter movement mode for one child of one node. Re-calling it with a
   *  different source replaces the movement (and drops anything in flight). */
  start: (nodeId: string, sourceChildIndex: number) => void;
  /** Commit a resolved destination and request the recommendation. */
  setTarget: (target: MovementTarget) => void;
  /** Move the strength slider. PCA updates client-side; no request is issued. */
  setStrength: (strength: number) => void;
  /** Leave movement mode and drop any response still in flight. */
  cancel: () => void;
}

const IDLE: MovementState = { phase: "idle" };

export function useMovement(args: UseMovementArgs): MovementController {
  const { analysis, datasetKey, treePath } = args;

  const [state, setState] = useState<MovementState>(IDLE);

  // Mirrors so the actions below can stay identity-stable while still reading
  // the newest props — the same pattern ClusterScatter uses for onSelectCluster.
  const stateRef = useRef(state);
  stateRef.current = state;
  const analysisRef = useRef(analysis);
  analysisRef.current = analysis;

  // Monotonic request generation. Every start / target / cancel bumps it, so a
  // response that belongs to a superseded question can be recognised as such.
  const genRef = useRef(0);

  // A UMAP strength re-request waiting for the slider to settle. Mirrored in
  // state (`pending`, `inFlight`) so `settled` re-renders when they change; the
  // refs are what the callbacks read.
  const pendingRef = useRef<number | null>(null);
  const [pending, setPending] = useState(false);
  const [inFlight, setInFlight] = useState(0);
  const clearPending = useCallback(() => {
    if (pendingRef.current !== null) {
      window.clearTimeout(pendingRef.current);
      pendingRef.current = null;
    }
    setPending(false);
  }, []);

  const cancel = useCallback(() => {
    genRef.current += 1;
    clearPending();
    setState((s) => (s.phase === "idle" ? s : IDLE));
  }, [clearPending]);

  // A timer that outlives the component would call into a dead hook.
  useEffect(() => clearPending, [clearPending]);

  const start = useCallback((nodeId: string, sourceChildIndex: number) => {
    genRef.current += 1; // a source change drops whatever was in flight
    setState({ phase: "choosing", nodeId, sourceChildIndex });
  }, []);

  // Delegates to the pure guard with the values as they stand at the moment the
  // response settles — never the ones captured when the request went out. Only
  // refs are read, so the closure never goes stale.
  const isCurrent = useCallback(
    (gen: number, req: MovementRequest, res: MovementResponse | null) =>
      isCurrentResponse(gen, req, res, {
        generation: genRef.current,
        state: stateRef.current,
        analysisId: analysisRef.current?.meta.analysis_id ?? null,
      }),
    [],
  );

  /** Issue one movement request and commit it if it is still the current one.
   *
   *  `showLoading` is false for a strength re-request: the answer only refines a
   *  preview that is already on screen, and blanking the panel on every slider
   *  nudge would make the ghosts flicker away exactly while they are being
   *  compared. A first request has nothing to keep, so it does show loading. */
  const issue = useCallback(
    (target: MovementTarget, strength: number | null, showLoading: boolean) => {
      const s = stateRef.current;
      if (s.phase === "idle") return; // no source selected: nothing to move
      const { nodeId, sourceChildIndex } = s;

      const meta = analysisRef.current?.meta ?? null;
      const analysisId = meta?.analysis_id ?? null;
      if (!meta || !analysisId) {
        setState({
          phase: "error",
          nodeId,
          sourceChildIndex,
          target,
          message:
            "This run was built before cluster movement existed and carries no analysis id. Rebuild the analysis to move a cluster.",
        });
        return;
      }

      const gen = ++genRef.current;
      // Every identifying field comes from the RUN, never from the config rail.
      // The rail describes the *next* build: the user can retick a feature or
      // nudge a knob without rebuilding, and the tree on screen is unaffected.
      // The server looks the payload up by (dataset, feature_cols, config), so
      // sending rail values would miss the very run being looked at and answer
      // 409 "no longer on the server" about a run that is still cached.
      const req: MovementRequest = {
        analysis_id: analysisId,
        dataset: meta.dataset,
        feature_cols: meta.feature_cols,
        config: meta.config,
        node_id: nodeId,
        source_child_index: sourceChildIndex,
        target,
        strength, // null = let the server pick its recommended strength
      };
      if (showLoading) setState({ phase: "loading", nodeId, sourceChildIndex, target });

      setInFlight((n) => n + 1);
      runMovement(req)
        .finally(() => setInFlight((n) => n - 1))
        .then(
        (res) => {
          if (!isCurrent(gen, req, res)) return;
          setState((prev) => ({
            phase: "ready",
            nodeId,
            sourceChildIndex,
            target,
            response: res,
            // A re-request already waiting means the slider has moved on since
            // this answer was asked for. Adopting `res.applied_strength` would
            // yank the slider backwards under the user's cursor AND make the
            // pending request ask for the position they just left.
            strength:
              pendingRef.current !== null && prev.phase === "ready"
                ? prev.strength
                : res.applied_strength,
          }));
        },
        (err: unknown) => {
          if (!isCurrent(gen, req, null)) return;
          // A failed REFINEMENT keeps what is already on screen. Only a first
          // request has nothing to lose; throwing away a completed
          // recommendation because an optional strength re-request hiccuped
          // would cost the user the very thing they were reading. The panel
          // already says the metrics describe the server's strength, and that
          // stays true — the server simply never answered at the new one.
          if (!showLoading) return;
          setState({ phase: "error", nodeId, sourceChildIndex, target, message: String(err) });
        },
      );
    },
    [isCurrent],
  );

  const setTarget = useCallback(
    (target: MovementTarget) => {
      clearPending();
      issue(target, null, true);
    },
    [issue, clearPending],
  );

  const setStrength = useCallback(
    (strength: number) => {
      const a = clamp01(strength);
      // The slider itself is never delayed: feature deltas are linear in α for
      // every method, so `previewAtStrength` rescales them immediately.
      setState((s) => (s.phase === "ready" ? { ...s, strength: a } : s));

      const s = stateRef.current;
      if (s.phase !== "ready") return;
      // PCA interpolates its ghosts exactly on the client, so the slider issues
      // no request at all. UMAP's out-of-sample transform is not linear — its
      // ghosts and its distance metrics are only right at a strength the server
      // actually projected, so ask again once the slider settles.
      if (s.response.method === "PCA") return;
      clearPending();
      setPending(true);
      pendingRef.current = window.setTimeout(() => {
        pendingRef.current = null;
        setPending(false);
        const now = stateRef.current;
        if (now.phase !== "ready" || now.response.method === "PCA") return;
        issue(now.target, now.strength, false);
      }, STRENGTH_DEBOUNCE_MS);
    },
    [issue, clearPending],
  );

  // ── Clearing ──────────────────────────────────────────────────────────────
  // Dataset change and new analysis. Each of these invalidates the (run, node,
  // source) the movement was defined against.
  useEffect(() => {
    cancel();
  }, [datasetKey, analysis, cancel]);

  // Navigation away from the movement's parent node. Drilling DEEPER keeps that
  // projection on screen and so keeps the movement; a sibling or a step back up
  // does not.
  const pathKey = treePath.join(",");
  useEffect(() => {
    const s = stateRef.current;
    if (s.phase === "idle") return;
    const path = pathKey === "" ? [] : pathKey.split(",").map(Number);
    if (!nodeIsOnPath(s.nodeId, path)) cancel();
  }, [pathKey, cancel]);

  // Escape cancels. It lives here rather than in the chart because
  // `ClusterScatterProps` is frozen and carries no cancel callback — and because
  // the key should work wherever focus happens to be, including in the panel.
  useEffect(() => {
    if (state.phase === "idle") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") cancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.phase, cancel]);

  const preview = useMemo(
    () => (state.phase === "ready" ? previewAtStrength(state.response, state.strength) : null),
    [state],
  );

  // The Apply-enablement condition, exactly. PCA's local re-expression is exact,
  // so `applied_strength === strength` holds by construction there; for UMAP it
  // means the last response really is for the current slider value.
  const settled =
    state.phase === "ready" &&
    !pending &&
    inFlight === 0 &&
    state.response.applied_strength === state.strength &&
    state.response.analysis_id === (analysis?.meta.analysis_id ?? null);

  return { state, preview, settled, start, setTarget, setStrength, cancel };
}

/** The five movement props one `ClusterScatter` needs, for the node it draws.
 *  Every other node in the layer stack gets the inert set, so only the projection
 *  the movement was started on leaves normal mode. */
export function movementPropsFor(
  m: MovementController,
  nodeId: string,
): {
  movementMode: boolean;
  movementSource: number | null;
  movementTarget: MovementTarget | null;
  movementPreview: MovementResponse | null;
  onMovementTarget: (target: MovementTarget) => void;
} {
  const s = m.state;
  if (s.phase === "idle" || s.nodeId !== nodeId) {
    return {
      movementMode: false,
      movementSource: null,
      movementTarget: null,
      movementPreview: null,
      onMovementTarget: m.setTarget,
    };
  }
  return {
    movementMode: true,
    movementSource: s.sourceChildIndex,
    movementTarget: s.phase === "choosing" ? null : s.target,
    movementPreview: m.preview,
    onMovementTarget: m.setTarget,
  };
}

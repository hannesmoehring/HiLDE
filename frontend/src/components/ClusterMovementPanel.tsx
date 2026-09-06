// Cluster movement panel — the reading half of the counterfactual. The chart draws
// the ghosts; this panel says what the shared feature-space translation actually is,
// how far it got, and everything about it that is estimated rather than measured.
//
// It owns no state: `state` and `preview` come straight from `useMovement`, and the
// three callbacks are the hook's own. That is deliberate — the slider position lives
// on the `ready` variant of the frozen `MovementState`, so a local copy here could
// only ever disagree with it.
//
// STALE METRICS (the one honesty trap): `preview` is `response` re-expressed at the
// slider, but `previewAtStrength` rescales feature deltas and ghost points ONLY —
// `metrics` still describe `response.applied_strength`. So this panel reads
// `feature_changes` off `preview` (live) and `metrics` off `state.response` (frozen),
// and labels the metrics block with the strength they were computed at. Nothing here
// presents a distance as if it tracked the slider.
import { useMemo } from "react";
import type {
  MovementFeatureChange,
  MovementResponse,
  MovementState,
  MovementStrengthPoint,
} from "../types";

export interface ClusterMovementPanelProps {
  /** The machine, straight from `useMovement().state`. */
  state: MovementState;
  /** `useMovement().preview` — the response re-expressed at the slider position.
   *  Non-null exactly while `state.phase === "ready"`. */
  preview: MovementResponse | null;
  /** The node this panel belongs to (`TreeNode.id`). Only a movement whose
   *  `state.nodeId` matches is shown here; every other layer's panel stays on its
   *  start form, so two panels can never claim the same movement. */
  nodeId: string;
  /** The child cluster selected in that node — the source a start would move.
   *  `null` = nothing selected, so there is nothing to move yet. */
  selectedChildIndex: number | null;
  /** `node.children.length`. A free-point target needs ≥ 1 child, a cluster target
   *  ≥ 2; with none there is nothing to move at all. */
  childCount: number;
  /** The method the run on screen was BUILT with (`analysis.meta.config.method`),
   *  never the live rail knob: t-SNE and MDS get the disabled state. */
  builtMethod: string;
  /** `useMovement().start` — called with (nodeId, sourceChildIndex). */
  onStart: (nodeId: string, sourceChildIndex: number) => void;
  /** `useMovement().setStrength` — 0..1. */
  onStrength: (strength: number) => void;
  /** `useMovement().cancel`. */
  onCancel: () => void;
}

// Only PCA and UMAP have what a movement needs. t-SNE has no out-of-sample
// transform at all, and MDS only ever reconstructs a configuration from a distance
// matrix — neither can project a moved point into the frozen embedding.
const SUPPORTED = ["PCA", "UMAP"];

const OUT_OF_RANGE_WARN = 0.05;

// Display threshold only — the backend owns the real support warning and sends its
// own text. Above this the ratio is styled as a caution instead of a plain metric.
const SUPPORT_CAUTION = 2;

const FROZEN_NOTE =
  "Counterfactual preview in the current frozen projection. The moved points are pushed " +
  "through the reducer that was fitted for this node; the embedding is never recomputed " +
  "afterwards, because a new fit could rotate, reflect or reshape it and the two pictures " +
  "would not be comparable. Nothing is written back to the dataset or to the hierarchy, and " +
  "the movement does not guarantee that HDBSCAN would merge the clusters after a rebuild.";

// Fallback for the amended min-norm sentence. The backend emits the norm space it
// actually used (standardized vs raw); this names both cases, so it stays true
// whichever way `normalize` was set, and is used only if no backend sentence arrived.
const MIN_NORM_FALLBACK =
  "This destination is under-determined: infinitely many feature-space changes land on the " +
  "same 2D point, and the one shown is the smallest of them — measured in the reducer's " +
  "input space, i.e. root-standardized units when the run normalized and raw units when it " +
  "did not. Other, larger changes reach the same place.";

function num(v: number | null | undefined, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function rmse(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v === 0 ? "0" : v.toExponential(1);
}

/** Descending |standardized change| — the backend deliberately returns
 *  `feature_cols` order, so the ranking is the panel's job. */
function bySalience(changes: readonly MovementFeatureChange[]): MovementFeatureChange[] {
  return [...changes].sort(
    (a, b) => Math.abs(b.standardized_magnitude ?? 0) - Math.abs(a.standardized_magnitude ?? 0),
  );
}

/** True when projected distance moves in one direction across the whole α grid.
 *  Differences under a thousandth of the range count as flat, so numerical jitter
 *  in the UMAP transform does not read as a real reversal. */
function isMonotone(points: readonly MovementStrengthPoint[]): boolean {
  if (points.length < 3) return true;
  const ys = points.map((p) => p.projected_distance);
  const span = Math.max(...ys) - Math.min(...ys);
  const eps = span * 1e-3;
  let up = false;
  let down = false;
  for (let i = 1; i < ys.length; i++) {
    const d = ys[i] - ys[i - 1];
    if (d > eps) up = true;
    else if (d < -eps) down = true;
  }
  return !(up && down);
}

function csvOf(rows: readonly MovementFeatureChange[]): string {
  const header =
    "feature,mutable,source_mean_raw,target_value_raw,recommended_delta_raw,applied_delta_raw,standardized_magnitude";
  const body = rows
    .map((r) =>
      [
        JSON.stringify(r.feature),
        r.mutable,
        r.source_mean_raw ?? "",
        r.target_value_raw ?? "",
        r.recommended_delta_raw ?? "",
        r.applied_delta_raw ?? "",
        r.standardized_magnitude ?? "",
      ].join(","),
    )
    .join("\n");
  return `${header}\n${body}\n`;
}

function download(name: string, body: string, mime: string) {
  const url = URL.createObjectURL(new Blob([body], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

/** Projected distance across the α grid. UMAP only — PCA is linear, so its
 *  trajectory is a straight line the backend does not bother returning. */
function StrengthTrajectory({
  points,
  cursor,
}: {
  points: readonly MovementStrengthPoint[];
  cursor: number;
}) {
  const W = 208;
  const H = 46;
  const P = 5;
  const xs = points.map((p) => p.strength);
  const ys = points.map((p) => p.projected_distance);
  const xMin = Math.min(...xs);
  const xSpan = Math.max(...xs) - xMin || 1;
  const yMin = Math.min(...ys);
  const ySpan = Math.max(...ys) - yMin || 1;
  const px = (s: number) => P + ((s - xMin) / xSpan) * (W - 2 * P);
  const py = (d: number) => H - P - ((d - yMin) / ySpan) * (H - 2 * P);
  const path = points
    .map((p, i) => `${i ? "L" : "M"}${px(p.strength).toFixed(1)} ${py(p.projected_distance).toFixed(1)}`)
    .join("");

  return (
    <svg
      className="movement__spark"
      width={W}
      height={H}
      role="img"
      aria-label="Projected distance against movement strength"
    >
      <path className="movement__spark-line" d={path} />
      {points.map((p) => (
        <circle
          key={p.strength}
          className="movement__spark-dot"
          cx={px(p.strength).toFixed(1)}
          cy={py(p.projected_distance).toFixed(1)}
          r={1.6}
        />
      ))}
      <line
        className="movement__spark-cursor"
        x1={px(cursor).toFixed(1)}
        x2={px(cursor).toFixed(1)}
        y1={P - 3}
        y2={H - P + 3}
      />
    </svg>
  );
}

export function ClusterMovementPanel({
  state,
  preview,
  nodeId,
  selectedChildIndex,
  childCount,
  builtMethod,
  onStart,
  onStrength,
  onCancel,
}: ClusterMovementPanelProps) {
  const supported = SUPPORTED.includes(builtMethod);
  // The movement this panel is allowed to show: only one for this node, and only
  // once it has an answer to report.
  const ready = state.phase === "ready" && state.nodeId === nodeId ? state : null;
  const res = ready?.response ?? null;
  // Feature deltas track the slider; metrics do not (see the file header).
  const live = ready ? (preview ?? ready.response) : null;
  const rows = useMemo(() => bySalience(live?.feature_changes ?? []), [live]);

  const maxMagnitude = rows.reduce(
    (m, r) => Math.max(m, Math.abs(r.standardized_magnitude ?? 0)),
    0,
  );
  const heldFixed = rows.filter((r) => !r.mutable);

  // The amended min-norm sentence is emitted by the backend as a warning; a free
  // PCA target is exactly the case it applies to. Lift it out of the list so it
  // reads as a property of the answer rather than as one more caveat, and leave
  // cluster targets (Δ = μ_t − μ_s, unique) without it.
  const underdetermined = res?.target.kind === "point" && res.target.estimator === "pca_pinv";
  const minNormFromBackend = underdetermined
    ? (res?.warnings.find((w) => /smallest/i.test(w)) ?? null)
    : null;
  const otherWarnings = (res?.warnings ?? []).filter((w) => w !== minNormFromBackend);

  const trajectory = res?.strength_trajectory ?? [];
  const trajectoryMonotone = isMonotone(trajectory);

  function exportCsv() {
    if (!live) return;
    download(
      `movement_${live.node_id.replace(/\//g, "_")}_C${live.source_child_index}.csv`,
      csvOf(rows),
      "text/csv",
    );
  }

  function exportJson() {
    if (!live) return;
    const payload = {
      analysis_id: live.analysis_id,
      node_id: live.node_id,
      method: live.method,
      source_child_index: live.source_child_index,
      target: live.target,
      recommended_strength: live.recommended_strength,
      applied_strength: live.applied_strength,
      feature_changes: rows,
    };
    download(
      `movement_${live.node_id.replace(/\//g, "_")}_C${live.source_child_index}.json`,
      JSON.stringify(payload, null, 2),
      "application/json",
    );
  }

  // ── Nothing running here: the start form, or the unsupported-method notice ──
  if (state.phase === "idle" || state.nodeId !== nodeId) {
    return (
      <div className="movement">
        <div className="movement__head">
          <span className="movement__title">Cluster movement</span>
        </div>
        {!supported ? (
          <div className="movement__disabled">
            <p>
              This projection was built with <b>{builtMethod}</b>, which cannot host a movement
              preview.
            </p>
            <p className="hint">
              <b>t-SNE</b> has no out-of-sample transform: there is no way to project a moved point
              into the embedding on screen without refitting the whole thing. <b>MDS</b> only
              reconstructs a configuration from a distance matrix, so it has no mapping from feature
              space to these coordinates either. Rebuild with PCA or UMAP to move a cluster.
            </p>
          </div>
        ) : (
          <>
            <p className="hint">
              Ask what feature values would put a cluster somewhere else in this projection. It is a
              counterfactual preview — nothing is written back.
            </p>
            <div className="movement__actions">
              <button
                disabled={selectedChildIndex == null || childCount < 1}
                onClick={() => selectedChildIndex != null && onStart(nodeId, selectedChildIndex)}
                title={
                  selectedChildIndex == null
                    ? "Select a cluster in this projection first"
                    : `Move C${selectedChildIndex} toward a destination you pick in this projection`
                }
              >
                {selectedChildIndex == null ? "Move a cluster…" : `Move C${selectedChildIndex}…`}
              </button>
              {selectedChildIndex == null && (
                <span className="hint">Select a cluster in this projection first.</span>
              )}
              {selectedChildIndex != null && childCount < 2 && (
                <span className="hint">
                  Only one cluster here — a free position is the only possible destination.
                </span>
              )}
            </div>
          </>
        )}
      </div>
    );
  }

  const source = `C${state.sourceChildIndex}`;

  // ── Active: choosing / loading / error / ready ─────────────────────────────
  return (
    <div className="movement movement--active">
      <div className="movement__head">
        <span className="movement__title">Cluster movement</span>
        {res?.metrics.preview_sampled && (
          <span className="movement__badge" title="The ghost points are a deterministic subsample">
            sampled preview
          </span>
        )}
        <button className="movement__cancel" onClick={onCancel}>
          Cancel
        </button>
      </div>

      {state.phase === "choosing" && (
        <p className="movement__prompt">
          Moving <b>{source}</b>. Click an empty position, or click another cluster to use it as the
          destination. Escape cancels.
        </p>
      )}

      {state.phase === "loading" && (
        <p className="movement__loading" role="status">
          Calculating {builtMethod} movement…
          {builtMethod === "UMAP" && (
            <span className="hint">
              {" "}
              UMAP may have to fit or restore a reducer first, which can take noticeably longer.
            </span>
          )}
        </p>
      )}

      {state.phase === "error" && (
        <p className="movement__error" role="alert">
          {state.message}
        </p>
      )}

      {ready && res && live && (
        <>
          <div className="movement__route">
            <span className="movement__route-line">
              <b>{source}</b> <span aria-hidden="true">→</span>{" "}
              <b>
                {res.target.kind === "cluster"
                  ? `C${res.target.child_index}`
                  : `point (${num(res.target.x, 2)}, ${num(res.target.y, 2)})`}
              </b>
            </span>
            <span className="hint">
              {res.source_size} points moving
              {res.target.kind === "cluster" && res.target_size != null
                ? ` toward ${res.target_size} points`
                : ""}
              {" · "}
              {res.method}
              {" · "}
              destination from {res.target.estimator.replace(/_/g, " ")}
            </span>
          </div>

          {underdetermined && (
            <p className="movement__note movement__note--minnorm">
              {minNormFromBackend ?? MIN_NORM_FALLBACK}
            </p>
          )}

          <div className="movement__strength">
            <label htmlFor="movement-strength">Strength</label>
            {/* PCA re-expresses locally (see `previewAtStrength`), so moving the
                slider issues no request. SEAM: the debounced re-request UMAP will
                need belongs on this handler. */}
            <input
              id="movement-strength"
              type="range"
              min={0}
              max={100}
              step={1}
              value={Math.round(ready.strength * 100)}
              onChange={(e) => onStrength(Number(e.currentTarget.value) / 100)}
            />
            <output htmlFor="movement-strength">{pct(ready.strength)}</output>
            {res.recommended_strength != null ? (
              <span className="hint">recommended {pct(res.recommended_strength)}</span>
            ) : (
              <span className="hint">
                No strength approached the destination, so none is recommended and
                the cluster is left where it is. Move the slider to see the
                displacement anyway.
              </span>
            )}
          </div>

          {/* Metrics describe `res.applied_strength`, NOT the slider. Labelled
              rather than hidden: they are the anchor the table is read against,
              and blanking them on every nudge would remove the only reference. */}
          <div className="movement__metrics-head">
            <span>Distances at {pct(res.applied_strength)} strength</span>
            {Math.abs(ready.strength - res.applied_strength) > 1e-9 && (
              <span className="movement__stale">
                slider is at {pct(ready.strength)} — these are not recomputed
              </span>
            )}
          </div>
          <div className="movement__metrics">
            <span>
              <b>{num(res.metrics.projected_distance_before, 2)}</b>
              <i>projected before</i>
            </span>
            <span>
              <b>{num(res.metrics.projected_distance_after, 2)}</b>
              <i>projected after</i>
            </span>
            <span>
              <b>{num(res.metrics.projected_distance_reduction, 2)}</b>
              <i>projected reduction</i>
            </span>
            <span>
              <b>{num(res.metrics.feature_centroid_distance_before, 2)}</b>
              <i>feature-space before</i>
            </span>
            {/* Cluster targets only: a point target's destination is defined as
                μ_source + Δ, so "after" would be (1−α)·‖Δ‖ by construction and the
                backend sends null instead. */}
            {res.metrics.feature_centroid_distance_after != null && (
              <span>
                <b>{num(res.metrics.feature_centroid_distance_after, 2)}</b>
                <i>feature-space after</i>
              </span>
            )}
            <span>
              <b>{rmse(res.metrics.alignment_rmse)}</b>
              <i>alignment RMSE</i>
            </span>
            <span>
              <b>{rmse(res.metrics.alignment_max_residual)}</b>
              <i>alignment max residual</i>
            </span>
          </div>
          <p className="hint">
            Projected distances are in the coordinates of this plot. Feature-space centroid distances
            are in the reducer&apos;s input space — root-standardized units when the run normalized,
            raw units when it did not. Alignment RMSE is how closely the refitted reducer reproduced
            the points already on screen (root-mean-square residual over the cloud&apos;s RMS
            radius); the server refuses anything above 0.05 outright rather than showing it. That
            bound is on the RMS only — the max residual is the single worst point and can exceed it
            — and neither says anything about the error of the new counterfactual points.
          </p>

          {trajectory.length > 0 && (
            <div className="movement__trajectory">
              <div className="movement__trajectory-head">Projected distance vs strength</div>
              <StrengthTrajectory points={trajectory} cursor={ready.strength} />
              {!trajectoryMonotone && (
                <p className="movement__warning movement__warning--caution">
                  This trajectory is not monotone. UMAP&apos;s out-of-sample transform is
                  neighbour-driven, so distance against strength can plateau and jump; the ghosts at
                  intermediate strengths are the least trustworthy part of this preview.
                </p>
              )}
            </div>
          )}

          {(otherWarnings.length > 0 ||
            (res.metrics.out_of_range_fraction ?? 0) > OUT_OF_RANGE_WARN ||
            res.metrics.support_ratio != null ||
            res.metrics.preview_sampled ||
            heldFixed.length > 0) && (
            <div className="movement__warnings">
              {otherWarnings.map((w) => (
                <p key={w} className="movement__warning">
                  {w}
                </p>
              ))}
              {(res.metrics.out_of_range_fraction ?? 0) > OUT_OF_RANGE_WARN && (
                <p className="movement__warning movement__warning--caution">
                  {pct(res.metrics.out_of_range_fraction)} of the moved points leave the range the
                  data actually covers — that is the fraction with at least one <i>mutable</i>{" "}
                  feature outside the full dataset&apos;s observed raw [min, max].
                </p>
              )}
              {res.metrics.support_ratio != null && (
                <p
                  className={
                    res.metrics.support_ratio > SUPPORT_CAUTION
                      ? "movement__warning movement__warning--caution"
                      : "movement__warning"
                  }
                >
                  Support ratio {num(res.metrics.support_ratio, 2)}: the distance from the chosen
                  position to the nearest embedded point, over the typical local neighbour distance
                  in this embedding. Large means the destination sits away from observed data and its
                  inverse is an extrapolation.
                </p>
              )}
              {res.metrics.preview_sampled && (
                <p className="movement__warning">
                  The ghost points are a deterministic subsample of {res.source_size}, not every
                  moved point. The feature centroids and the recommendation below are computed from
                  the complete cluster.
                </p>
              )}
              {heldFixed.length > 0 && (
                <p className="movement__warning">
                  {heldFixed.length} feature{heldFixed.length === 1 ? " is" : "s are"} held fixed
                  (row ids and <code>target_*</code> labels are never counterfactuals), so the
                  requested destination may not be reached exactly.
                </p>
              )}
            </div>
          )}

          <div className="movement__table-head">
            <span>Feature changes</span>
            <span className="hint">applied at {pct(live.applied_strength)}</span>
            <button onClick={exportCsv}>CSV</button>
            <button onClick={exportJson}>JSON</button>
          </div>
          <div className="table-scroll movement__table">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th className="num">Source mean</th>
                  <th className="num">Target</th>
                  <th className="num">Recommended Δ</th>
                  <th className="num">Applied Δ</th>
                  <th>|Δ| standardized</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const mag = Math.abs(r.standardized_magnitude ?? 0);
                  const width = maxMagnitude > 0 ? (mag / maxMagnitude) * 100 : 0;
                  const sign = (r.recommended_delta_raw ?? 0) < 0 ? "is-down" : "is-up";
                  return (
                    <tr key={r.feature} className={r.mutable ? undefined : "is-immutable"}>
                      <td>
                        {r.feature}
                        {!r.mutable && <span className="movement__fixed">held fixed</span>}
                      </td>
                      <td className="num">{num(r.source_mean_raw)}</td>
                      <td className="num">{num(r.target_value_raw)}</td>
                      <td className="num">{num(r.recommended_delta_raw)}</td>
                      <td className="num">{num(r.applied_delta_raw)}</td>
                      <td>
                        <span className="movement__bar">
                          <span
                            className={`movement__bar-fill ${sign}`}
                            style={{ width: `${width}%` }}
                          />
                        </span>
                        <span className="movement__bar-value">
                          {num(r.standardized_magnitude, 2)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="movement__note">{FROZEN_NOTE}</p>
        </>
      )}
    </div>
  );
}

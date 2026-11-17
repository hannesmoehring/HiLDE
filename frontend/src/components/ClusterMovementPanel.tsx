// Cluster movement — the reading half of the counterfactual, in two pieces.
//
// `MovementStartRow` is the ONLY thing that sits in a layer's side column: one
// button. `ClusterMovementPanel` is the full-width row App.tsx mounts under the
// plot and the characteristics once a movement is running: controls, stat tiles,
// the feature-change table, one-line warning chips, and a collapsed "About this
// preview" block that holds every explanatory sentence. Nothing explanatory is
// visible by default; every tile label and chip carries its sentence as a
// `title`, and the same sentences are listed in the <details> block so nothing is
// hover-only. The "click a destination" prompt lives in ClusterScatter, above the
// plot, not here.
//
// Neither component owns state: `state` and `preview` come straight from
// `useMovement`, and the callbacks are the hook's own. The slider position lives
// on the `ready` variant of the frozen `MovementState`, so a local copy here
// could only ever disagree with it.
//
// STALE METRICS (the one honesty trap): `preview` is `response` re-expressed at
// the slider, but `previewAtStrength` rescales feature deltas and ghost points
// ONLY — `metrics` still describe `response.applied_strength`. So the table reads
// `feature_changes` off `preview` (live) and the tiles read `metrics` off
// `state.response` (frozen), labelled with the strength they were computed at.
import { useMemo } from "react";
import type {
  MovementFeatureChange,
  MovementResponse,
  MovementState,
  MovementStrengthPoint,
} from "../types";

// Only PCA and UMAP have what a movement needs. t-SNE has no out-of-sample
// transform at all, and MDS only ever reconstructs a configuration from a distance
// matrix — neither can project a moved point into the frozen embedding.
const SUPPORTED = ["PCA", "UMAP"];

const OUT_OF_RANGE_WARN = 0.05;

// Display threshold only — the backend owns the real support warning (at 3) and
// sends its own text. Above this the ratio is styled as a caution.
const SUPPORT_CAUTION = 2;

/** Every explanatory sentence the panel uses, in one place: each is a tooltip on
 *  its tile / chip AND a line in the "About this preview" block. */
const ABOUT = {
  frozen:
    "Counterfactual preview in the current frozen projection. The moved points are pushed " +
    "through the reducer that was fitted for this node; the embedding is never recomputed " +
    "afterwards, because a new fit could rotate, reflect or reshape it and the two pictures " +
    "would not be comparable. Nothing is written back to the dataset or to the hierarchy, and " +
    "the movement does not guarantee that HDBSCAN would merge the clusters after a rebuild.",
  units:
    "Projected distances are in the coordinates of this plot. Feature values and deltas are in " +
    "the dataset's own units; |Δ| standardized is in the reducer's input space — " +
    "root-standardized units when the run normalized, raw units when it did not.",
  strength:
    "Fraction of the recommended displacement that is applied. PCA re-expresses ghosts and " +
    "deltas client-side, exactly. UMAP's transform is not linear, so its ghosts are re-requested " +
    "350 ms after the slider settles, and the tiles describe the strength the server last " +
    "answered at.",
  projected:
    "Distance between the destination and the moved cluster's centroid in this plot's " +
    "coordinates, before the movement and at the applied strength.",
  feature:
    "Distance between the destination and the cluster's centroid in the reducer's input space, " +
    "before and at the applied strength. Cluster targets only: a point target's destination is " +
    "defined as source + Δ, so its 'after' would be (1−α)·‖Δ‖ by construction and is not shown.",
  alignment:
    "How closely the refitted reducer reproduced the points already on screen: the " +
    "root-mean-square residual over the cloud's RMS radius (the server refuses anything above " +
    "0.05 rather than drawing it) and the single worst point, which can exceed that bound. " +
    "Neither says anything about the error of the new counterfactual points.",
  range:
    "Fraction of moved points with at least one mutable feature outside the full dataset's " +
    "observed raw [min, max]. Above 5 % the counterfactual is extrapolating beyond the data.",
  support:
    "Distance from the chosen position to the nearest embedded point, over the typical local " +
    "neighbour distance in this embedding. Large means the destination sits away from observed " +
    "data and its inverse is an extrapolation.",
  sampled:
    "The ghost points are a deterministic subsample of the cluster, not every moved point. " +
    "Centroids, metrics and the recommendation are computed from the complete cluster.",
  trajectory:
    "Projected distance to the destination at each strength of the α grid, measured through the " +
    "frozen reducer. UMAP's out-of-sample transform is neighbour-driven, so this can plateau and " +
    "jump; when it is not monotone the ghosts at intermediate strengths are the least " +
    "trustworthy part of the preview.",
  minNormStd:
    "Many different feature changes reach this position; this is the smallest standardized " +
    "change that reaches this point (smallest measured in root-standardized units — one unit is " +
    "one standard deviation over the whole dataset). Other, larger changes reach the same place.",
  minNormRaw:
    "Many different feature changes reach this position; this is the smallest raw change that " +
    "reaches this point (smallest measured in the dataset's own units, because this analysis was " +
    "built without normalization). Other, larger changes reach the same place.",
  held:
    "Row ids and target_* label columns are never counterfactuals. They are held fixed, so the " +
    "requested destination may not be reached exactly.",
  apply:
    "Apply writes this movement — the deltas at the current strength, mutable features only, " +
    "one rigid translation per row of the whole cluster — into a counterfactual copy of the " +
    "dataset and rebuilds the entire analysis on it. The original data is untouched; Undo and " +
    "Reset are in the banner above the layers. The hierarchy may change, so cluster numbers " +
    "need not correspond to this run afterwards.",
} as const;

function num(v: number | null | undefined, digits = 3): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function sci(v: number | null | undefined): string {
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

// ── Warning chips ───────────────────────────────────────────────────────────
// One short visible line per warning, the long sentence in the tooltip. Backend
// warnings are recognised by content and given a stable key, so a client-side
// chip about the same thing (out-of-range, support) never appears twice.

interface Chip {
  key: string;
  short: string;
  long: string;
  caution: boolean;
}

function chipForWarning(w: string): Chip {
  if (/smallest standardized change/i.test(w))
    return {
      key: "minnorm",
      short: "Smallest standardized change reaching this point; larger ones exist.",
      long: w,
      caution: false,
    };
  if (/smallest raw change/i.test(w))
    return {
      key: "minnorm",
      short: "Smallest raw-unit change reaching this point; larger ones exist.",
      long: w,
      caution: false,
    };
  if (/closest reachable point/i.test(w))
    return { key: "reach", short: "Destination not exactly reachable — closest point shown.", long: w, caution: true };
  if (/held fixed/i.test(w))
    return { key: "held", short: "Labels / ids held fixed; the destination may not be reached exactly.", long: w, caution: false };
  if (/outside the range/i.test(w))
    return { key: "range", short: "Moved points leave the observed data range.", long: w, caution: true };
  if (/inverse projection is approximate/i.test(w))
    return { key: "approx", short: "UMAP's inverse is approximate; the ghosts show where the points actually land.", long: w, caution: false };
  if (/estimated from the nearest observed points/i.test(w))
    return { key: "fallback", short: "Destination estimated from the nearest observed points, not inverted.", long: w, caution: true };
  if (/does not reliably approach/i.test(w))
    return { key: "norec", short: "No strength approaches the destination; none is recommended.", long: w, caution: true };
  if (/far from observed data/i.test(w))
    return { key: "support", short: "Position far from observed data; its inverse is uncertain.", long: w, caution: true };
  if (/no feature in this analysis may be changed/i.test(w))
    return { key: "nomutable", short: "No feature may change here; nothing to recommend.", long: w, caution: true };
  const first = w.split(/(?<=\.)\s/)[0];
  return { key: w, short: first.length > 90 ? `${first.slice(0, 87)}…` : first, long: w, caution: false };
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
  const W = 168;
  const H = 40;
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

function Tile({
  value,
  label,
  tip,
  caution,
}: {
  value: string;
  label: string;
  tip: string;
  caution?: boolean;
}) {
  return (
    <span className={caution ? "mv-tile mv-tile--caution" : "mv-tile"} title={tip}>
      <b>{value}</b>
      <i>
        {label} <span aria-hidden="true">ⓘ</span>
      </i>
    </span>
  );
}

// ── The idle row: one button in the side column ─────────────────────────────

export interface MovementStartRowProps {
  nodeId: string;
  /** The child cluster selected in this node — the source a start would move. */
  selectedChildIndex: number | null;
  /** The method the run on screen was BUILT with, never the live rail knob. */
  builtMethod: string;
  /** The server only serves stored runs, so it refuses the refit a UMAP movement
   *  needs. PCA movements answer without one and stay available. */
  cacheOnly: boolean;
  /** True while this node's own movement is running: the full-width row is
   *  showing and carries Cancel, so the start button steps aside. */
  active: boolean;
  onStart: (nodeId: string, sourceChildIndex: number) => void;
}

export function MovementStartRow({
  nodeId,
  selectedChildIndex,
  builtMethod,
  cacheOnly,
  active,
  onStart,
}: MovementStartRowProps) {
  if (active) return null;
  const supported = SUPPORTED.includes(builtMethod);
  const refused = cacheOnly && supported && builtMethod !== "PCA";
  const label = selectedChildIndex == null ? "Move a cluster…" : `Move C${selectedChildIndex}…`;
  const title = !supported
    ? `Not available for ${builtMethod}: no feature→2D mapping. Rebuild with PCA or UMAP.`
    : refused
      ? `Not available on this server for ${builtMethod}: it needs a refit, and only cached runs are served. Pick a PCA run.`
      : selectedChildIndex == null
        ? "Select a cluster in this projection first"
        : `Move C${selectedChildIndex} toward a destination you pick in the projection — a counterfactual preview, nothing is written back`;
  return (
    <div className="movement-start">
      <button
        disabled={!supported || refused || selectedChildIndex == null}
        title={title}
        onClick={() => selectedChildIndex != null && onStart(nodeId, selectedChildIndex)}
      >
        {label}
      </button>
    </div>
  );
}

// ── The active row: full width under the plot and the characteristics ───────

export interface ClusterMovementPanelProps {
  /** The machine, straight from `useMovement().state`. Must not be `idle`. */
  state: MovementState;
  /** `useMovement().preview` — the response re-expressed at the slider position.
   *  Non-null exactly while `state.phase === "ready"`. */
  preview: MovementResponse | null;
  /** The method the run on screen was BUILT with. */
  builtMethod: string;
  /** `useMovement().setStrength` — 0..1. */
  onStrength: (strength: number) => void;
  /** `useMovement().cancel`. */
  onCancel: () => void;
  /** Write the preview into data (App.tsx builds the request). */
  onApply: () => void;
  /** `null` = Apply is enabled; otherwise the one-line reason it is not. */
  applyDisabledReason: string | null;
  /** True while an apply or the rebuild it triggers is out. */
  applying: boolean;
}

export function ClusterMovementPanel({
  state,
  preview,
  builtMethod,
  onStrength,
  onCancel,
  onApply,
  applyDisabledReason,
  applying,
}: ClusterMovementPanelProps) {
  const ready = state.phase === "ready" ? state : null;
  const res = ready?.response ?? null;
  // Feature deltas track the slider; metrics do not (see the file header).
  const live = ready ? (preview ?? ready.response) : null;
  const rows = useMemo(() => bySalience(live?.feature_changes ?? []), [live]);
  const mutableRows = rows.filter((r) => r.mutable);
  const heldFixed = rows.filter((r) => !r.mutable);
  const maxMagnitude = mutableRows.reduce(
    (m, r) => Math.max(m, Math.abs(r.standardized_magnitude ?? 0)),
    0,
  );

  const trajectory = res?.strength_trajectory ?? [];
  const monotone = isMonotone(trajectory);

  // Chips: backend warnings first (deduplicated by key), then the client-side
  // observations about the same response.
  const chips: Chip[] = [];
  const seen = new Set<string>();
  const push = (c: Chip) => {
    if (seen.has(c.key)) return;
    seen.add(c.key);
    chips.push(c);
  };
  if (res) {
    for (const w of res.warnings) push(chipForWarning(w));
    if (!monotone)
      push({
        key: "monotone",
        short: "Trajectory not monotone: intermediate strengths are the least trustworthy.",
        long: ABOUT.trajectory,
        caution: true,
      });
    const oor = res.metrics.out_of_range_fraction ?? 0;
    if (oor > OUT_OF_RANGE_WARN)
      push({ key: "range", short: `${pct(oor)} of moved points leave the observed data range.`, long: ABOUT.range, caution: true });
    const sr = res.metrics.support_ratio;
    if (sr != null && sr > SUPPORT_CAUTION)
      push({ key: "support", short: `Support ratio ${num(sr, 1)}: destination sits away from observed data.`, long: ABOUT.support, caution: true });
    if (res.metrics.preview_sampled)
      push({ key: "sampled", short: `Ghosts are a deterministic subsample of ${res.source_size} points.`, long: ABOUT.sampled, caution: false });
  }

  const source = state.phase === "idle" ? "" : `C${state.sourceChildIndex}`;
  const destination = res
    ? res.target.kind === "cluster"
      ? `C${res.target.child_index}`
      : `point (${num(res.target.x, 2)}, ${num(res.target.y, 2)})`
    : state.phase === "choosing"
      ? "…"
      : state.phase === "idle"
        ? ""
        : state.target?.kind === "cluster"
          ? `C${state.target.child_index}`
          : state.target
            ? `point (${num(state.target.x, 2)}, ${num(state.target.y, 2)})`
            : "…";

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

  // ── Not ready yet: one line and Cancel ────────────────────────────────────
  if (!ready || !res || !live) {
    return (
      <div className="mv mv--pending">
        <span className="mv-route">
          <b>{source}</b> <span aria-hidden="true">→</span> <b>{destination}</b>
        </span>
        {state.phase === "choosing" && (
          <span className="hint">Pick a destination in the projection above. Escape cancels.</span>
        )}
        {state.phase === "loading" && (
          <span className="hint" role="status">
            Calculating {builtMethod} movement…
            {builtMethod === "UMAP" ? " UMAP may have to fit the reducer first." : ""}
          </span>
        )}
        {state.phase === "error" && (
          <span className="mv-error" role="alert">
            {state.message}
          </span>
        )}
        <button className="mv-cancel" onClick={onCancel}>
          Cancel
        </button>
      </div>
    );
  }

  const stale = Math.abs(ready.strength - res.applied_strength) > 1e-9;
  const minNormTip =
    res.warnings.find((w) => /smallest/i.test(w)) ??
    (res.method === "PCA" ? ABOUT.minNormStd : "");

  return (
    <div className="mv">
      <div className="mv-grid">
        {/* 1 — controls */}
        <div className="mv-col mv-controls">
          <div className="mv-route">
            <b>{source}</b> <span aria-hidden="true">→</span> <b>{destination}</b>
            <span className="hint">
              {" "}
              · {res.source_size} pts · {res.method} · {res.target.estimator.replace(/_/g, " ")}
            </span>
          </div>
          <div className="mv-strength">
            <label htmlFor="movement-strength" title={ABOUT.strength}>
              Strength
            </label>
            <span className="mv-strength__track">
              <input
                id="movement-strength"
                type="range"
                min={0}
                max={100}
                step={1}
                value={Math.round(ready.strength * 100)}
                onChange={(e) => onStrength(Number(e.currentTarget.value) / 100)}
              />
              {res.recommended_strength != null && (
                <span
                  className="mv-strength__rec"
                  style={{ left: `${res.recommended_strength * 100}%` }}
                  title={`Recommended strength ${pct(res.recommended_strength)}`}
                >
                  rec.
                </span>
              )}
            </span>
            <output htmlFor="movement-strength">{pct(ready.strength)}</output>
          </div>
          <div className="mv-actions">
            <button
              className="primary"
              disabled={applyDisabledReason != null || applying}
              title={applyDisabledReason ? `Apply: ${applyDisabledReason}` : ABOUT.apply}
              onClick={onApply}
            >
              {applying ? "Applying…" : "Apply"}
            </button>
            <button className="mv-cancel" onClick={onCancel} disabled={applying}>
              Cancel
            </button>
            <button onClick={exportCsv} title="Feature changes at the current strength">
              CSV
            </button>
            <button onClick={exportJson} title="Recommendation at the current strength">
              JSON
            </button>
          </div>
        </div>

        {/* 2 — metrics (at the strength the server answered at) */}
        <div className="mv-col mv-metrics">
          <div className="mv-col__head">
            <span>Distances at {pct(res.applied_strength)}</span>
            {stale && (
              <span
                className="mv-chip mv-chip--caution"
                title="The slider has moved since these were computed; they are not recomputed until the server answers at the new strength."
              >
                slider at {pct(ready.strength)}
              </span>
            )}
            {res.metrics.preview_sampled && (
              <span className="movement__badge" title={ABOUT.sampled}>
                sampled
              </span>
            )}
          </div>
          <div className="mv-tiles">
            <Tile
              value={`${num(res.metrics.projected_distance_before, 2)} → ${num(res.metrics.projected_distance_after, 2)}`}
              label="projected distance"
              tip={ABOUT.projected}
            />
            {res.metrics.feature_centroid_distance_after != null && (
              <Tile
                value={`${num(res.metrics.feature_centroid_distance_before, 2)} → ${num(res.metrics.feature_centroid_distance_after, 2)}`}
                label="feature-space distance"
                tip={ABOUT.feature}
              />
            )}
            <Tile
              value={`${sci(res.metrics.alignment_rmse)} · max ${sci(res.metrics.alignment_max_residual)}`}
              label="alignment RMSE · worst point"
              tip={ABOUT.alignment}
            />
            <Tile
              value={pct(res.metrics.out_of_range_fraction)}
              label="out of range"
              tip={ABOUT.range}
              caution={(res.metrics.out_of_range_fraction ?? 0) > OUT_OF_RANGE_WARN}
            />
            {res.metrics.support_ratio != null && (
              <Tile
                value={num(res.metrics.support_ratio, 2)}
                label="support ratio"
                tip={ABOUT.support}
                caution={res.metrics.support_ratio > SUPPORT_CAUTION}
              />
            )}
          </div>
          {trajectory.length > 0 && (
            <div className="mv-spark" title={ABOUT.trajectory}>
              <span className="mv-spark__label">distance vs strength</span>
              <StrengthTrajectory points={trajectory} cursor={ready.strength} />
            </div>
          )}
        </div>

        {/* 3 — feature changes */}
        <div className="mv-col mv-table">
          <div className="mv-col__head">
            <span>Feature changes at {pct(live.applied_strength)}</span>
          </div>
          <div className="table-scroll movement__table">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th className="num">Source mean</th>
                  <th className="num">Target</th>
                  <th className="num">Δ recommended</th>
                  <th className="num">Δ applied</th>
                  <th title={ABOUT.units}>|Δ| standardized</th>
                </tr>
              </thead>
              <tbody>
                {mutableRows.map((r) => {
                  const mag = Math.abs(r.standardized_magnitude ?? 0);
                  const width = maxMagnitude > 0 ? (mag / maxMagnitude) * 100 : 0;
                  const sign = (r.recommended_delta_raw ?? 0) < 0 ? "is-down" : "is-up";
                  return (
                    <tr key={r.feature}>
                      <td>{r.feature}</td>
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
          {heldFixed.length > 0 && (
            <div className="mv-held" title={ABOUT.held}>
              {heldFixed.length} held fixed: {heldFixed.map((r) => r.feature).join(", ")}
            </div>
          )}
        </div>
      </div>

      {chips.length > 0 && (
        <div className="mv-chips">
          {chips.map((c) => (
            <span
              key={c.key}
              className={c.caution ? "mv-chip mv-chip--caution" : "mv-chip"}
              title={c.long}
            >
              {c.short}
            </span>
          ))}
        </div>
      )}

      <details className="mv-about">
        <summary>About this preview</summary>
        <dl>
          <dt>Frozen projection</dt>
          <dd>{ABOUT.frozen}</dd>
          <dt>Units</dt>
          <dd>{ABOUT.units}</dd>
          <dt>Strength</dt>
          <dd>{ABOUT.strength}</dd>
          <dt>Projected distance</dt>
          <dd>{ABOUT.projected}</dd>
          <dt>Feature-space distance</dt>
          <dd>{ABOUT.feature}</dd>
          <dt>Alignment RMSE · worst point</dt>
          <dd>{ABOUT.alignment}</dd>
          <dt>Out of range</dt>
          <dd>{ABOUT.range}</dd>
          <dt>Support ratio</dt>
          <dd>{ABOUT.support}</dd>
          <dt>Sampled preview</dt>
          <dd>{ABOUT.sampled}</dd>
          <dt>Distance vs strength</dt>
          <dd>{ABOUT.trajectory}</dd>
          {minNormTip && (
            <>
              <dt>Under-determined destination</dt>
              <dd>{minNormTip}</dd>
            </>
          )}
          <dt>Held fixed</dt>
          <dd>{ABOUT.held}</dd>
          <dt>Apply</dt>
          <dd>{ABOUT.apply}</dd>
          {chips.map((c) => (
            <div key={`about-${c.key}`}>
              <dt>{c.short}</dt>
              <dd>{c.long}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}

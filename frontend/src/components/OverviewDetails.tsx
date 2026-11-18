// The column beside the treemap: one tile in words and numbers, so the map
// itself can stay sparse. Shows the hovered / keyboard-focused tile, else the
// pinned (selected) one, else the frame the map is zoomed into.
import type { ReactNode } from "react";
import { theme } from "../charts/theme";
import {
  fmtCount,
  fmtPct,
  kindLabel,
  unassignedShare,
  type OverviewTile,
} from "../overview";
import type { OverviewMetric } from "../overviewMetrics";

interface Props {
  tile: OverviewTile;
  total: number;
  metric: OverviewMetric;
  source: "hover" | "selected" | "focus";
  isCurrent: boolean; // the layer stack is on this node
  action?: ReactNode; // the explore button, under the name
  hasCharacteristics: boolean;
  normalized: boolean;
}

// Shifts past this are drawn at full length; the number beside the bar still
// says how far they go.
const SD_MAX = 3;

const KICKER = { hover: "Highlighted tile", selected: "Selected tile", focus: "Map focus" } as const;

export function OverviewDetails(props: Props) {
  const { tile, total, metric, source, isCurrent, action, hasCharacteristics, normalized } = props;
  const parent = tile.owner;
  const split = unassignedShare(tile);
  const value = metric.value?.(tile);
  const rows: [string, string][] = [
    ["Points", fmtCount(tile.size)],
    ["Share of all data", total > 0 ? fmtPct(tile.size / total) : "—"],
  ];
  if (parent) rows.push(["Share of parent", fmtPct(tile.size / parent.n_points)]);
  rows.push(["Node type", kindLabel(tile.kind)]);
  if (split != null) rows.push(["Unassigned when split", fmtPct(split)]);
  if (metric.value && metric.key !== "unassigned")
    rows.push([metric.label, value != null ? metric.format(value) : "—"]);

  const assigned = tile.children.filter((c) => c.kind !== "unassigned");
  const shifts = tile.shifts.slice(0, 3);

  return (
    <div className="overview__details">
      <div className="kicker">
        {KICKER[source]}
        {isCurrent ? " · open below" : ""}
      </div>
      <div className="overview__name">{tile.label}</div>
      <div className="overview__path">{tile.pathLabel}</div>
      {action}

      <dl className="overview__facts">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>

      <div className="overview__block">
        <div className="overview__caption">Sub-clusters vs. unassigned</div>
        {split != null ? (
          <>
            <div className="overview__comp" aria-hidden="true">
              {1 - split > 0 && <span className="overview__comp-in" style={{ width: `${100 * (1 - split)}%` }} />}
              {split > 0 && <span className="overview__comp-out" style={{ width: `${100 * split}%` }} />}
            </div>
            <p className="hint">
              {assigned.length} sub-cluster{assigned.length === 1 ? "" : "s"} · {fmtPct(split)} unassigned
            </p>
          </>
        ) : (
          <p className="hint">
            {tile.kind === "unassigned"
              ? "HDBSCAN noise left over when its parent was split."
              : "A leaf — not split further."}
          </p>
        )}
      </div>

      <div className="overview__block">
        <div className="overview__caption">Largest mean shifts vs. parent (SDs of the parent)</div>
        {!parent ? (
          <p className="hint">The root has no parent to compare with.</p>
        ) : !shifts.length && !normalized && tile.depth === 1 ? (
          <p className="hint">
            Built without normalisation, so the root’s own means are not on a common scale and nothing
            can be compared against it. Shifts appear from the second level down.
          </p>
        ) : !shifts.length ? (
          <p className="hint">
            {hasCharacteristics ? "Not available for this tile." : "This run carries no characteristics."}
          </p>
        ) : (
          <div className="overview__shifts">
            {shifts.map((s) => {
              const w = (Math.min(Math.abs(s.sd), SD_MAX) / SD_MAX) * 50;
              return (
                <div key={s.feature} className="overview__shift">
                  <span className="overview__shift-name" title={s.feature}>
                    {s.feature}
                  </span>
                  <span className="overview__shift-bar" aria-hidden="true">
                    <i
                      style={{
                        left: `${s.sd < 0 ? 50 - w : 50}%`,
                        width: `${w}%`,
                        background: s.sd < 0 ? theme.divNeg : theme.divPos,
                      }}
                    />
                  </span>
                  <span className="overview__shift-val">
                    {s.sd > 0 ? "+" : s.sd < 0 ? "−" : ""}
                    {Math.abs(s.sd).toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// One treemap tile's paint: fill, the unassigned hatch and its labels. The
// interactive wrapper (role, tab stop, handlers) stays in TreemapOverview; this
// is shared with the tiles that slide out during a zoom, which have none.
import { isFrame } from "../overviewFocus";
import { textOn } from "../overviewMetrics";
import type { OverviewTile } from "../overview";
import { theme } from "./theme";
import { LINE, headerText, tileLines, type Cell } from "./treemapLayout";
import type { Box } from "./useTileTween";

interface Props {
  cell: Cell; // final geometry: what the labels are sized for
  box: Box; // where it is drawn right now
  fill: (t: OverviewTile) => string;
  hatchId: string;
  labels: boolean; // off while the map moves: text sized for the end rect would spill
}

export function TreemapTileShape({ cell, box, fill, hatchId, labels }: Props) {
  const t = cell.data;
  const w = Math.max(0, box.x1 - box.x0);
  const h = Math.max(0, box.y1 - box.y0);
  const bg = fill(t);
  const ink = textOn(bg);
  const header = labels && isFrame(t) ? headerText(cell) : "";
  const lines = labels && !isFrame(t) ? tileLines(cell) : [];
  return (
    <>
      <rect x={box.x0} y={box.y0} width={w} height={h} fill={bg} stroke={theme.borderStrong} strokeWidth={0.75} />
      {t.kind === "unassigned" && (
        <rect
          x={box.x0}
          y={box.y0}
          width={w}
          height={h}
          fill={`url(#${ink === theme.textPrimary ? hatchId : `${hatchId}-light`})`}
          pointerEvents="none"
        />
      )}
      {header && (
        <text x={box.x0 + 5} y={box.y0 + 12} fontSize={11} fill={ink} pointerEvents="none">
          <tspan fontWeight={600}>{header.split(" · ")[0]}</tspan>
          {header.includes(" · ") && <tspan>{header.slice(header.indexOf(" · "))}</tspan>}
        </text>
      )}
      {lines.map((s, i) => (
        <text
          key={i}
          x={box.x0 + 4}
          y={box.y0 + 12 + i * LINE}
          fontSize={11}
          fontWeight={i === 0 ? 600 : 400}
          fill={ink}
          // A halo in the tile's own fill keeps the label clear of the hatch.
          stroke={t.kind === "unassigned" ? bg : "none"}
          strokeWidth={3}
          paintOrder="stroke"
          pointerEvents="none"
        >
          {s}
        </text>
      ))}
    </>
  );
}

// Two rects, white under the colour, so a mark holds on the dark ramp steps.
export function Outline({ box, width, color = theme.textPrimary, dashed = false, inset = 0 }: {
  box: Box;
  width: number;
  color?: string;
  dashed?: boolean;
  inset?: number;
}) {
  const w = box.x1 - box.x0;
  const h = box.y1 - box.y0;
  const i = w > 4 * inset && h > 4 * inset ? inset : -Math.abs(inset);
  const r = { x: box.x0 + i, y: box.y0 + i, width: Math.max(0, w - 2 * i), height: Math.max(0, h - 2 * i), fill: "none" };
  return (
    <>
      <rect {...r} stroke="#ffffff" strokeWidth={width + 2} />
      <rect {...r} stroke={color} strokeWidth={width} strokeDasharray={dashed ? "4 2" : undefined} />
    </>
  );
}

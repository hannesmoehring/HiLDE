// The map's own breadcrumb: how far the treemap is zoomed in. Unlike the path
// crumbs beside it, these never move the layer stack, only the map.
import type { OverviewTile } from "../overview";

interface Props {
  chain: OverviewTile[]; // root → focused frame
  onZoom: (t: OverviewTile) => void;
}

export function MapCrumbs({ chain, onZoom }: Props) {
  const last = chain.length - 1;
  return (
    <nav className="overview__crumbs overview__crumbs--map" aria-label="Map zoom">
      <span className="overview__crumbs-label">Map</span>
      {chain.map((t, i) => (
        <span key={t.key}>
          {i > 0 && <span aria-hidden="true"> → </span>}
          <button
            aria-current={i === last ? "location" : undefined}
            onClick={() => onZoom(t)}
            title={i === last ? `The map shows ${t.pathLabel}` : `Zoom the map out to ${t.pathLabel}`}
          >
            {t.label}
          </button>
        </span>
      ))}
    </nav>
  );
}

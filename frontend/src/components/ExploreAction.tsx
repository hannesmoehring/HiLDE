// "Explore here": the one control in the details column that moves the layer
// stack. It opens the pinned tile (else the map's focused frame) exactly as
// activating a tile used to, and says what that opens, stack cut-off included.
import { useId } from "react";
import { pathLabelOf, type OverviewTile } from "../overview";
import type { TileTarget } from "../overviewTarget";

interface Props {
  tile: OverviewTile;
  target: TileTarget;
  isOpen: boolean; // the stack is already on the target
  onExplore: () => void;
}

const sentence = (s: string) => `${s.charAt(0).toUpperCase()}${s.slice(1)}.`;

export function ExploreAction({ tile, target, isOpen, onExplore }: Props) {
  const id = useId();
  return (
    <div className="overview__explore">
      <button
        className="primary"
        onClick={onExplore}
        aria-describedby={id}
        title={`Explore ${tile.pathLabel}: ${target.what}`}
      >
        Explore {pathLabelOf(target.path)}
      </button>
      <p id={id} className="hint">
        {tile.kind === "root" ? "" : `${tile.pathLabel} `}
        {tile.kind === "root" ? sentence(target.what) : `${target.what}.`}
        {isOpen ? " Already open below." : ""}
      </p>
    </div>
  );
}

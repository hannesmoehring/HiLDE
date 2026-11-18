// A polite live region for the overview map: zooming (by click, key or the Map
// crumbs) and selecting change the map without moving keyboard focus, so a
// screen reader would otherwise hear nothing. Silent on first render.
import { useEffect, useRef, useState } from "react";
import type { OverviewTile } from "../overview";

interface Props {
  focus: OverviewTile;
  selected: OverviewTile | null;
}

export function MapStatus({ focus, selected }: Props) {
  const [text, setText] = useState("");
  const last = useRef({ focus: focus.key, selected: selected?.key ?? null });

  useEffect(() => {
    const prev = last.current;
    const sel = selected?.key ?? null;
    last.current = { focus: focus.key, selected: sel };
    if (prev.focus !== focus.key) setText(`Map zoomed to ${focus.pathLabel}`);
    else if (prev.selected !== sel) setText(selected ? `Selected ${selected.pathLabel}` : "Selection cleared");
  }, [focus, selected]);

  return (
    <p className="visually-hidden" role="status" aria-live="polite">
      {text}
    </p>
  );
}

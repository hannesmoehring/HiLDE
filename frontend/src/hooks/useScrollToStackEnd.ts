import { useEffect, useRef, useState } from "react";

// After a jump from the overview, bring the panel it opened into view. That is
// always the LAST panel of the layer stack: a navigation to an internal node ends
// on the layer that waits for a click into it, one to a leaf on the exploration
// panel. Looking it up after render avoids repeating the stack's own rules here.
export function useScrollToStackEnd<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [request, setRequest] = useState(0);

  useEffect(() => {
    if (!request || !ref.current) return;
    const panels = ref.current.querySelectorAll<HTMLElement>(":scope > .panel:not(.overview)");
    const target = panels[panels.length - 1];
    if (!target) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, [request]);

  return { ref, scroll: () => setRequest((n) => n + 1) };
}

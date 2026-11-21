// Tour state: whether it is running, which step is on screen, and whether this
// browser has seen it (so a first visit offers it once, on its own).
import { useCallback, useEffect, useRef, useState } from "react";
import { TOUR_STEPS, type TourContext } from "./steps";

const SEEN_KEY = "hilde.tour.seen";

function readSeen(): boolean {
  try {
    return window.localStorage.getItem(SEEN_KEY) === "1";
  } catch {
    return true; // storage blocked: never nag on every load
  }
}

function markSeen() {
  try {
    window.localStorage.setItem(SEEN_KEY, "1");
  } catch {
    // Not remembered, which is fine.
  }
}

export interface TourController {
  active: boolean;
  index: number;
  start: () => void;
  next: () => void;
  back: () => void;
  exit: () => void;
}

/** `ready`: the app can take a tour now (false under maintenance or before the
 *  mode is known). `ctx` is read fresh on every step change. */
export function useTour(ctx: TourContext, ready: boolean): TourController {
  const [active, setActive] = useState(false);
  const [index, setIndex] = useState(0);
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;

  // Offer the tour once, on a first visit: the welcome card asks before
  // anything on screen changes.
  useEffect(() => {
    if (ready && !readSeen()) setActive(true);
  }, [ready]);

  const go = useCallback((from: number, dir: 1 | -1) => {
    let i = from + dir;
    while (i > 0 && i < TOUR_STEPS.length && TOUR_STEPS[i].skip?.(ctxRef.current)) i += dir;
    if (i >= TOUR_STEPS.length) {
      markSeen();
      setActive(false);
      return;
    }
    i = Math.max(0, i);
    TOUR_STEPS[i].enter?.(ctxRef.current);
    setIndex(i);
  }, []);

  const start = useCallback(() => {
    setIndex(0);
    setActive(true);
  }, []);
  const exit = useCallback(() => {
    markSeen();
    setActive(false);
  }, []);
  const next = useCallback(() => go(index, 1), [go, index]);
  const back = useCallback(() => go(index, -1), [go, index]);

  return { active, index, start, next, back, exit };
}

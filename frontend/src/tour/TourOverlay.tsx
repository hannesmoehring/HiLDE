// The tour on screen: a spotlight outline around the current step's target that
// dims everything else, and a card with the step's text and Back / Next / Exit.
// The dimming never takes clicks, so the user can try the spotlighted control
// (lasso some points, click a tile) while the card stays up.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { placeCard, visibleBox, type Box } from "./placement";
import { TOUR_STEPS } from "./steps";
import type { TourController } from "./useTour";

function findTarget(id: string | undefined): HTMLElement | null {
  return id ? document.querySelector<HTMLElement>(`[data-tour="${id}"]`) : null;
}

export function TourOverlay({ tour }: { tour: TourController }) {
  const step = TOUR_STEPS[tour.index];
  const [box, setBox] = useState<Box | null>(null);
  const [found, setFound] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const [cardPos, setCardPos] = useState({ top: 0, left: 0 });

  // Follow the target every frame: it may appear late (a build), move (layout
  // reflows as panels load) or scroll. Bring it into view once, when found.
  useEffect(() => {
    let raf = 0;
    let scrolled = false;
    setFound(false);
    setBox(null);
    const tick = () => {
      const el = findTarget(step.target);
      if (el && el.getClientRects().length > 0) {
        if (!scrolled) {
          scrolled = true;
          setFound(true);
          step.shown?.();
          const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
          el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
        }
        const b = visibleBox(el.getBoundingClientRect());
        setBox((prev) =>
          prev &&
          Math.abs(prev.top - b.top) < 0.5 &&
          Math.abs(prev.left - b.left) < 0.5 &&
          Math.abs(prev.width - b.width) < 0.5 &&
          Math.abs(prev.height - b.height) < 0.5
            ? prev
            : b,
        );
      } else if (scrolled) {
        setBox(null); // the target went away (the user navigated elsewhere)
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step]);

  useLayoutEffect(() => {
    const c = cardRef.current;
    if (!c) return;
    setCardPos(placeCard(box, { width: c.offsetWidth, height: c.offsetHeight }));
  }, [box, step, found]);

  // A step whose target is not on screen yet — the overview while the run is still
  // building — holds the tour there: stepping on would only land on more steps
  // about a run that does not exist. Exit and Back stay available.
  const waiting = step.target !== undefined && !box;
  const waitingRef = useRef(waiting);
  waitingRef.current = waiting;

  // Keyboard: Esc exits from anywhere but a field; arrows step only from inside the
  // card, since the treemap and the plots use them for their own navigation.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.key === "Escape") {
        tour.exit();
        return;
      }
      if (!t || !cardRef.current?.contains(t)) return;
      if (e.key === "ArrowRight" && !waitingRef.current) tour.next();
      else if (e.key === "ArrowLeft" && tour.index > 0) tour.back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tour]);

  // Move keyboard focus to the card on every step, so a screen reader reads it.
  useEffect(() => {
    cardRef.current?.focus({ preventScroll: true });
  }, [tour.index]);

  const first = tour.index === 0;
  const last = tour.index === TOUR_STEPS.length - 1;

  return (
    <div className="tour" aria-live="polite">
      {box ? (
        <div
          className="tour__spot"
          style={{ top: box.top, left: box.left, width: box.width, height: box.height }}
        />
      ) : (
        <div className="tour__dim" />
      )}
      <div
        ref={cardRef}
        className="tour__card"
        role="dialog"
        aria-modal="false"
        aria-labelledby="tour-title"
        tabIndex={-1}
        style={cardPos}
      >
        <div className="tour__count">
          {first ? "Guided tour" : `Step ${tour.index} of ${TOUR_STEPS.length - 1}`}
        </div>
        <h2 id="tour-title" className="tour__title">
          {step.title}
        </h2>
        <p className="tour__body">
          {waiting ? (step.waiting ?? "Waiting for this part of the page to appear…") : step.body}
        </p>
        <div className="tour__actions">
          <button onClick={tour.exit}>{first ? "Not now" : "Exit"}</button>
          <span className="tour__spacer" />
          {!first && <button onClick={tour.back}>Back</button>}
          <button
            className="primary"
            onClick={tour.next}
            disabled={waiting}
            title={waiting ? "Available once this part of the page has loaded" : undefined}
          >
            {first ? "Start tour" : last ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}

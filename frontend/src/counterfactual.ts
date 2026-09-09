// Counterfactual session state — the stack of applied edits and the one key
// every request in the app is sent against.
//
// Apply is a two-step transition: `applying` (the movement is written into a
// server-side copy of the dataset and a new dataset key comes back), then
// `rebuilding` (the whole analysis is rebuilt on that key while the CURRENT run
// stays on screen and interactive). Only when the rebuild has an answer is the
// switch made — stack pushed, active key swapped, displayed run replaced,
// navigation reset to root — and it is made in ONE state update, so no render
// ever pairs the new key with the old tree or the other way round. Any failure
// leaves the stack and the displayed run exactly as they were.
//
// Undo and Reset need no endpoint: keys are immutable snapshots, so they are a
// rebuild on the parent key (or the base), which the server's tree cache
// normally answers inline.
import { useCallback, useEffect, useRef, useState } from "react";
import { applyCounterfactual } from "./api";
import type {
  AnalysisResponse,
  CounterfactualApplyRequest,
  CounterfactualEdit,
  CounterfactualPhase,
} from "./types";

export interface CounterfactualStackEntry {
  datasetKey: string;
  cfId: string;
  edits: CounterfactualEdit[]; // the whole chain, base first (from the server)
}

export interface UseCounterfactualArgs {
  /** The plain dataset the session is built on. A change resets the stack. */
  baseDataset: string;
  /** Rebuild the analysis on a key — the caller supplies the run's own feature
   *  columns and built config, never the rail's unsaved knobs. */
  rebuild: (datasetKey: string) => Promise<AnalysisResponse>;
  /** Commit a switch: show `analysis` (built on `datasetKey`) and go to root.
   *  Called inside the same update that changes the stack. */
  onSwitched: (analysis: AnalysisResponse, datasetKey: string) => void;
}

export interface CounterfactualController {
  stack: CounterfactualStackEntry[];
  /** What every request sends as `dataset`: the stack head, or the base. */
  activeDatasetKey: string;
  phase: CounterfactualPhase;
  error: string | null;
  /** Write a movement into data and rebuild on the new key. No-op unless idle. */
  apply: (req: CounterfactualApplyRequest) => void;
  /** Back to the previous key (or the base). No-op unless idle and stacked. */
  undo: () => void;
  /** Back to the base. No-op unless idle and stacked. */
  reset: () => void;
}

export function useCounterfactual(args: UseCounterfactualArgs): CounterfactualController {
  const { baseDataset, rebuild, onSwitched } = args;
  const [stack, setStack] = useState<CounterfactualStackEntry[]>([]);
  const [phase, setPhase] = useState<CounterfactualPhase>("idle");
  const [error, setError] = useState<string | null>(null);

  const activeDatasetKey = stack.length ? stack[stack.length - 1].datasetKey : baseDataset;

  // Mirrors so the transitions below stay identity-stable while reading the
  // newest values, and so a settling response can be checked against the state
  // as it stands THEN, never as it was captured when the request went out.
  const stackRef = useRef(stack);
  stackRef.current = stack;
  const baseRef = useRef(baseDataset);
  baseRef.current = baseDataset;
  const activeRef = useRef(activeDatasetKey);
  activeRef.current = activeDatasetKey;
  const rebuildRef = useRef(rebuild);
  rebuildRef.current = rebuild;
  const switchedRef = useRef(onSwitched);
  switchedRef.current = onSwitched;
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  // Every transition bumps the generation; a response is dropped if the
  // generation, the base dataset or the stack head changed while it was out.
  const genRef = useRef(0);
  const stale = (gen: number, base: string, head: string) =>
    gen !== genRef.current || base !== baseRef.current || head !== activeRef.current;

  // A new base dataset is a new session: nothing on the stack refers to it.
  useEffect(() => {
    genRef.current += 1;
    setStack([]);
    setPhase("idle");
    setError(null);
  }, [baseDataset]);

  /** Rebuild on `key`, then commit `next` as the new stack in one update. */
  const switchTo = useCallback(
    (key: string, next: (prev: CounterfactualStackEntry[]) => CounterfactualStackEntry[]) => {
      const gen = ++genRef.current;
      const base = baseRef.current;
      const head = activeRef.current;
      setPhase("rebuilding");
      setError(null);
      rebuildRef.current(key).then(
        (analysis) => {
          if (stale(gen, base, head)) return;
          // React batches these into one render: the stack, the phase and the
          // displayed run change together, and the path resets with them.
          setStack(next);
          setPhase("idle");
          switchedRef.current(analysis, key);
        },
        (err: unknown) => {
          if (stale(gen, base, head)) return;
          setPhase("error");
          setError(`Rebuild on the counterfactual data failed: ${String(err)}`);
        },
      );
    },
    [],
  );

  const apply = useCallback(
    (req: CounterfactualApplyRequest) => {
      // A second click while one is out is a no-op: no double submit.
      if (phaseRef.current === "applying" || phaseRef.current === "rebuilding") return;
      const gen = ++genRef.current;
      const base = baseRef.current;
      const head = activeRef.current;
      setPhase("applying");
      setError(null);
      applyCounterfactual({ ...req, dataset: head }).then(
        (res) => {
          if (stale(gen, base, head)) return;
          switchTo(res.dataset_key, (prev) => [
            ...prev,
            { datasetKey: res.dataset_key, cfId: res.cf_id, edits: res.edits },
          ]);
        },
        (err: unknown) => {
          if (stale(gen, base, head)) return;
          setPhase("error");
          setError(`Apply failed: ${String(err)}`);
        },
      );
    },
    [switchTo],
  );

  const undo = useCallback(() => {
    const s = stackRef.current;
    if (s.length === 0 || phaseRef.current === "applying" || phaseRef.current === "rebuilding")
      return;
    const target = s.length >= 2 ? s[s.length - 2].datasetKey : baseRef.current;
    switchTo(target, (prev) => prev.slice(0, -1));
  }, [switchTo]);

  const reset = useCallback(() => {
    const s = stackRef.current;
    if (s.length === 0 || phaseRef.current === "applying" || phaseRef.current === "rebuilding")
      return;
    switchTo(baseRef.current, () => []);
  }, [switchTo]);

  return { stack, activeDatasetKey, phase, error, apply, undo, reset };
}

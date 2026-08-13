// Side column of a hierarchical layer: DR quality tiles, then one of two accounts
// of the selected cluster — its characteristics, or the predicate separating it
// from the rest of the space it was selected out of. The view is per layer, so
// different depths can show different accounts at the same time.
import { useEffect, useMemo, useState } from "react";
import { runPredicate } from "../api";
import { CharacteristicsBar } from "../charts/CharacteristicsBar";
import { PredicateBands } from "../charts/PredicateBands";
import { ScoreTiles } from "../charts/ScoreTiles";
import type { AnalysisConfig, PredicateResponse, TreeNode } from "../types";

type View = "characteristics" | "predicate";

interface Props {
  parent: TreeNode; // the space the cluster was selected out of
  child: TreeNode; // the selected cluster
  childIndex: number;
  dataset: string;
  featureCols: string[];
  config: AnalysisConfig;
  charNonFeatureOnly: boolean;
}

export function LayerSide({
  parent,
  child,
  childIndex,
  dataset,
  featureCols,
  config,
  charNonFeatureOnly,
}: Props) {
  const [view, setView] = useState<View>("characteristics");
  const [predicate, setPredicate] = useState<PredicateResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);

  const label = `C${childIndex}`;

  // /api/predicate wants the cluster as positions into the parent's rows, but the
  // tree stores both as dataframe ids. A child is always a subset of its parent,
  // so every id resolves; HDBSCAN noise stays behind in the parent, which is
  // exactly right — it is part of what the cluster is being contrasted against.
  const selectedLocal = useMemo(() => {
    const at = new Map(parent.row_indices.map((id, i) => [id, i]));
    return child.row_indices.flatMap((id) => {
      const i = at.get(id);
      return i === undefined ? [] : [i];
    });
  }, [parent.row_indices, child.row_indices]);

  // Fetched only while the predicate view is showing: the default view is served
  // entirely from the tree, and stays as instant as it is today.
  useEffect(() => {
    if (view !== "predicate") return;
    let cancelled = false;
    setPredicate(null);
    setPending(true);
    setFailed(false);
    runPredicate({
      dataset,
      feature_cols: featureCols,
      config,
      row_indices: parent.row_indices,
      selected_local_indices: selectedLocal,
      scope: "local", // background = the rest of this layer's space
    })
      .then((p) => {
        if (cancelled) return;
        setPredicate(p);
        setPending(false);
      })
      .catch(() => {
        if (cancelled) return;
        setFailed(true);
        setPending(false);
      });
    return () => {
      cancelled = true;
    };
  }, [view, dataset, featureCols, config, parent.row_indices, selectedLocal]);

  // Only the dimensions that carry the explanation: the clauses the greedy search
  // kept. `trimmed` is a separate RCM 0.9 run whose own flags can differ, so the
  // core bands are matched by feature name against the clauses shown.
  const clauses = useMemo(() => {
    if (!predicate) return null;
    const full = predicate.full.filter((r) => r.in_predicate);
    const keep = new Set(full.map((r) => r.feature));
    return { full, trimmed: predicate.trimmed.filter((r) => keep.has(r.feature)) };
  }, [predicate]);

  return (
    <>
      <ScoreTiles scores={child.scores} title={`${label} — DR quality`} />

      <div className="tabs" role="tablist" aria-label={`${label} explanation`}>
        <button
          role="tab"
          aria-selected={view === "characteristics"}
          className={view === "characteristics" ? "is-active" : undefined}
          onClick={() => setView("characteristics")}
        >
          Characteristics
        </button>
        <button
          role="tab"
          aria-selected={view === "predicate"}
          className={view === "predicate" ? "is-active" : undefined}
          onClick={() => setView("predicate")}
        >
          Predicate
        </button>
      </div>

      {/* Both views share one fixed-height viewport, so switching tabs never
          reflows the layer row around them — whichever is taller scrolls. */}
      <div className="layer__view" role="tabpanel">
        {view === "characteristics" ? (
          <CharacteristicsBar
            data={child.rel_characteristics}
            title={`${label} characteristics`}
            nonFeatureOnly={charNonFeatureOnly}
          />
        ) : (
          <div className="layer__predicate">
            <div className="layer__predicate-title">{label} predicate</div>
            <p className="hint">
              Feature ranges separating {label} from the rest of this layer. Each track spans
              the feature&apos;s range within this layer, not the whole dataset.
            </p>
            {pending && <p className="hint">Inducing predicate…</p>}
            {failed && <p className="hint">Could not induce a predicate for {label}.</p>}
            {clauses &&
              (clauses.full.length === 0 ? (
                <p className="hint">
                  No feature range separates {label} from the rest of this layer.
                </p>
              ) : (
                <>
                  {predicate?.summary && (
                    <div className="predicate-summary">
                      <span>F1 {predicate.summary.predicate_f1.toFixed(2)}</span>
                      <span>
                        {predicate.summary.n_features_used} / {predicate.summary.n_features_total}{" "}
                        features
                      </span>
                      <span>{predicate.summary.n_selected} points</span>
                    </div>
                  )}
                  <PredicateBands full={clauses.full} trimmed={clauses.trimmed} />
                </>
              ))}
          </div>
        )}
      </div>
    </>
  );
}

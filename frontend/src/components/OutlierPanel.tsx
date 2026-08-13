// GLOSH outlier scores for one internal (clustered) layer.
//
// Folded away in a <details> so it stays out of the way: the closed summary
// carries just the headline numbers. Open it for the score distribution and a
// ranked table of the most outlying points; clicking a row reveals that point's
// actual column values and rings it in the projection above.
import { useEffect, useState } from "react";
import { fetchRows } from "../api";
import { OutlierHistogram } from "../charts/OutlierHistogram";
import type { RowsResponse, TreeNode } from "../types";

interface Props {
  node: TreeNode; // internal node; node.outlier_scores is parallel to node.row_indices
  dataset: string;
  selectedRow: number | null; // row id echoed back from the parent (drives the scatter ring)
  onSelectRow: (rowId: number | null) => void;
}

const TOP_N = 100;

interface Ranked {
  rowId: number;
  score: number;
}

export function OutlierPanel({
  node,
  dataset,
  selectedRow,
  onSelectRow,
}: Props) {
  const [detail, setDetail] = useState<RowsResponse | null>(null);

  const scores = node.outlier_scores;

  // Pair each score with its row id, dropping non-finite entries (serialized as null).
  const ranked: Ranked[] = [];
  if (scores) {
    scores.forEach((s, i) => {
      if (s != null) ranked.push({ rowId: node.row_indices[i], score: s });
    });
  }
  ranked.sort((a, b) => b.score - a.score);

  // Fetch every column for the clicked point — "what is this point?" includes the
  // non-feature columns (labels/targets), not just the ones used for clustering.
  useEffect(() => {
    if (selectedRow == null) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    fetchRows(dataset, [selectedRow])
      .then((r) => !cancelled && setDetail(r))
      .catch(() => !cancelled && setDetail(null));
    return () => {
      cancelled = true;
    };
  }, [selectedRow, dataset]);

  if (ranked.length === 0) return null;

  const max = ranked[0].score;
  const mean = ranked.reduce((s, r) => s + r.score, 0) / ranked.length;
  const shown = ranked.slice(0, TOP_N);

  return (
    <details className="outliers">
      <summary>
        <span className="outliers__title">GLOSH outlier scores</span>
        <span className="outliers__teaser">
          max {max.toFixed(3)} · mean {mean.toFixed(3)} · {ranked.length} points
        </span>
      </summary>

      <div className="outliers__body">
        <p className="hint outliers__note">
          HDBSCAN's Global-Local Outlier Score from Hierarchies, per point of
          this layer: 0 = sits firmly inside its cluster, 1 = strongly outlying.
        </p>

        <OutlierHistogram scores={ranked.map((r) => r.score)} />

        <p className="outliers__caption">
          {ranked.length <= TOP_N
            ? `All ${ranked.length} points, most outlying first. Click a row to inspect it.`
            : `Top ${TOP_N} of ${ranked.length} points by score. Click a row to inspect it.`}
        </p>

        <div className="outliers__table-scroll">
          <table className="outliers__table">
            <thead>
              <tr>
                <th className="num">#</th>
                <th className="num">Row</th>
                <th>Score</th>
                <th className="num">Value</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r, i) => {
                const active = selectedRow === r.rowId;
                return (
                  <tr
                    key={r.rowId}
                    className={active ? "is-active" : undefined}
                    onClick={() => onSelectRow(active ? null : r.rowId)}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelectRow(active ? null : r.rowId);
                      }
                    }}
                    aria-expanded={active}
                  >
                    <td className="num outliers__rank">{i + 1}</td>
                    <td className="num">{r.rowId}</td>
                    <td>
                      <span className="outliers__track">
                        <span
                          className="outliers__fill"
                          style={{ width: `${r.score * 100}%` }}
                        />
                      </span>
                    </td>
                    <td className="num">{r.score.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {selectedRow != null && (
          <div className="outliers__detail">
            <div className="outliers__detail-head">
              <strong>Row {selectedRow}</strong>
              <span className="hint">ringed in the projection above</span>
              <button type="button" onClick={() => onSelectRow(null)}>
                Close
              </button>
            </div>
            {detail && detail.rows.length > 0 ? (
              <dl className="outliers__kv">
                {detail.columns.map((c) => (
                  <div key={c}>
                    <dt>{c}</dt>
                    <dd>{formatValue(detail.rows[0][c])}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="hint">Loading row values…</p>
            )}
          </div>
        )}
      </div>
    </details>
  );
}

function formatValue(v: unknown): string {
  if (v == null) return "—";
  // Round to 4 decimals but drop the trailing zeros: 7.8000 -> 7.8, 1.0390 -> 1.039.
  if (typeof v === "number") return String(Number(v.toFixed(4)));
  return String(v);
}

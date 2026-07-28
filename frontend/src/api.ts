// Typed client for the FastAPI backend. In dev, Vite proxies /api -> uvicorn.
import type {
  AnalysisConfig,
  AnalysisResponse,
  DatasetColumns,
  DatasetInfo,
  PredicateResponse,
  PredicateScope,
  RowsResponse,
} from "./types";

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return get("/api/datasets");
}

export function datasetColumns(key: string): Promise<DatasetColumns> {
  return get(`/api/datasets/${encodeURIComponent(key)}/columns`);
}

export function runAnalysis(
  dataset: string,
  feature_cols: string[],
  config: Partial<AnalysisConfig>,
): Promise<AnalysisResponse> {
  return post("/api/analysis", { dataset, feature_cols, config });
}

export function runPredicate(args: {
  dataset: string;
  feature_cols: string[];
  config: Partial<AnalysisConfig>;
  row_indices: number[];
  selected_local_indices: number[];
  scope: PredicateScope;
}): Promise<PredicateResponse> {
  return post("/api/predicate", args);
}

export function fetchRows(
  dataset: string,
  ids: number[],
  columns?: string[],
): Promise<RowsResponse> {
  return post("/api/rows", { dataset, ids, columns });
}

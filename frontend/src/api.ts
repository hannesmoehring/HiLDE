// Typed client for the FastAPI backend. In dev, Vite proxies /api -> uvicorn.
import type {
  AnalysisConfig,
  AnalysisJob,
  AnalysisResponse,
  DatasetColumns,
  DatasetInfo,
  ModeInfo,
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

export function getMode(): Promise<ModeInfo> {
  return get("/api/mode");
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return get("/api/datasets");
}

export function datasetColumns(key: string): Promise<DatasetColumns> {
  return get(`/api/datasets/${encodeURIComponent(key)}/columns`);
}

const POLL_INTERVAL_MS = 2000;
const POLL_RETRIES = 3; // a hiccup on one poll must not throw away a run in flight

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Starts a build and waits for it, polling the job until it finishes. */
export async function runAnalysis(
  dataset: string,
  feature_cols: string[],
  config: Partial<AnalysisConfig>,
  use_cache = true,
): Promise<AnalysisResponse> {
  let job = await post<AnalysisJob>("/api/analysis", { dataset, feature_cols, config, use_cache });
  let failures = 0;
  while (job.status === "running") {
    await sleep(POLL_INTERVAL_MS);
    try {
      job = await get<AnalysisJob>(`/api/analysis/jobs/${job.job_id}`);
      failures = 0;
    } catch (e) {
      if (++failures > POLL_RETRIES) throw e;
    }
  }
  if (job.status === "error") throw new Error(job.detail);
  return job;
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

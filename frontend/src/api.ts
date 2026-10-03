// Typed client for the FastAPI backend. In dev, Vite proxies /api -> uvicorn.
import type {
  AnalysisConfig,
  AnalysisJob,
  AnalysisResponse,
  CachedRun,
  CharacteristicsResponse,
  CounterfactualApplyRequest,
  CounterfactualApplyResponse,
  CounterfactualChain,
  CounterfactualJob,
  DatasetColumns,
  DatasetInfo,
  ImagePixels,
  ModeInfo,
  MovementJob,
  MovementRequest,
  MovementResponse,
  PredicateResponse,
  PredicateScope,
  RowsResponse,
  TargetsResponse,
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

/** The stored runs a cache-only server can show. */
export function listCachedRuns(): Promise<CachedRun[]> {
  return get("/api/cached-runs");
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return get("/api/datasets");
}

export function datasetColumns(key: string): Promise<DatasetColumns> {
  return get(`/api/datasets/${encodeURIComponent(key)}/columns`);
}

/** Pixels of one row — only for datasets whose `image` spec is non-null. */
export function fetchPointImage(key: string, rowId: number): Promise<ImagePixels> {
  return get(`/api/datasets/${encodeURIComponent(key)}/image/${rowId}`);
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

/** Characteristics of a selection, on the same z-score baseline as the tree. */
export function fetchSelectionCharacteristics(args: {
  dataset: string;
  feature_cols: string[];
  config: Partial<AnalysisConfig>; // only `normalize` is read, as on /api/predicate
  row_indices: number[];
  selected_local_indices: number[];
}): Promise<CharacteristicsResponse> {
  return post("/api/characteristics", args);
}

export function fetchTargets(args: {
  dataset: string;
  target_cols: string[];
  row_indices: number[];
  selected_local_indices: number[];
}): Promise<TargetsResponse> {
  return post("/api/targets", args);
}

export function fetchRows(
  dataset: string,
  ids: number[],
  columns?: string[],
): Promise<RowsResponse> {
  return post("/api/rows", { dataset, ids, columns });
}

/** Requests a cluster-movement preview, polling the job when the server needs a
 *  reducer refit (UMAP). PCA and cache hits answer inline, so the loop usually
 *  runs zero times. Retry behaviour matches `runAnalysis`: a hiccup on one poll
 *  must not throw away work already in flight. */
export async function runMovement(req: MovementRequest): Promise<MovementResponse> {
  let job = await post<MovementJob>("/api/movement", req);
  let failures = 0;
  while (job.status === "running") {
    await sleep(POLL_INTERVAL_MS);
    try {
      job = await get<MovementJob>(`/api/movement/jobs/${job.job_id}`);
      failures = 0;
    } catch (e) {
      if (++failures > POLL_RETRIES) throw e;
    }
  }
  if (job.status === "error") throw new Error(job.detail);
  return job;
}

/** Writes a previewed movement into a counterfactual copy of the dataset and
 *  returns the new dataset key. Same job rule as `runMovement`: only an Apply
 *  that needs a UMAP refit polls. The analysis itself is NOT rebuilt here — the
 *  caller runs `runAnalysis` on the returned key. */
export async function applyCounterfactual(
  req: CounterfactualApplyRequest,
): Promise<CounterfactualApplyResponse> {
  let job = await post<CounterfactualJob>("/api/counterfactual/apply", req);
  let failures = 0;
  while (job.status === "running") {
    await sleep(POLL_INTERVAL_MS);
    try {
      job = await get<CounterfactualJob>(`/api/counterfactual/jobs/${job.job_id}`);
      failures = 0;
    } catch (e) {
      if (++failures > POLL_RETRIES) throw e;
    }
  }
  if (job.status === "error") throw new Error(job.detail);
  return job;
}

/** The edit chain behind a counterfactual key, base first. */
export function fetchCounterfactualChain(cfId: string): Promise<CounterfactualChain> {
  return get(`/api/counterfactual/${encodeURIComponent(cfId)}`);
}

/** Download URL of the changed rows (row_id, original and counterfactual values). */
export function counterfactualRowsUrl(cfId: string): string {
  return `/api/counterfactual/${encodeURIComponent(cfId)}/rows.csv`;
}

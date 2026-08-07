/**
 * api.ts — the typed surface of the PlanGen backend.
 *
 * Every shape here was read off api/server.py rather than assumed. The flow
 * the UI walks is:
 *
 *   createSession()            -> session_id
 *   parseText(brief)           -> is_valid | needs more info
 *   nextQuestion()/answer()    -> the gathering loop, until is_valid
 *   runPipeline()              -> run_id
 *   pipelineStatus(run_id)     -> {step 1..5, logs[]}   <- drives the render page
 *   runFiles(run_id)           -> svg/dxf/json artefacts
 */

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(
      "Cannot reach the PlanGen server. Is it running on port 8000?",
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body; statusText stands */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

/* ── types ──────────────────────────────────────────────────────────── */

export interface DiagnosticCheck {
  name: string;
  status: "ok" | "degraded" | "missing";
  detail: string;
  impact?: string;
  remedy?: string;
}

export interface Diagnostics {
  healthy: boolean;
  overall: "ok" | "degraded" | "missing";
  counts: { ok: number; degraded: number; missing: number };
  checks: DiagnosticCheck[];
}

export interface Health {
  status: string;
  engine: string;
  engine_ready: boolean;
  version: string;
  diagnostics: Diagnostics;
}

export interface ParseResult {
  status?: string;
  is_valid?: boolean;
  clarification_prompt?: string;
  missing_tier1?: string[];
  missing_tier2?: string[];
  data?: Record<string, unknown>;
  message?: string;
  [k: string]: unknown;
}

export interface PipelineStatus {
  status: "running" | "complete" | "error" | string;
  step: number;
  run_id: string;
  logs: string[];
  error?: string;
  result?: Record<string, unknown>;
}

export interface RunFile {
  name: string;
  size_kb: number;
  type: string;
}

export interface RunFiles {
  run_id: string;
  files: RunFile[];
}

/* ── calls ──────────────────────────────────────────────────────────── */

export const api = {
  health: () => req<Health>("/health"),
  diagnostics: () => req<Diagnostics>("/diagnostics"),

  createSession: () =>
    req<{ session_id: string }>("/sessions", { method: "POST", body: "{}" }),

  deleteSession: (sessionId: string) =>
    req<{ status: string }>(`/sessions/${sessionId}`, { method: "DELETE" }),

  parseText: (sessionId: string, text: string) =>
    req<ParseResult>("/parse/text", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, text }),
    }),

  nextQuestion: (sessionId: string) =>
    req<ParseResult>(
      `/parse/next-question?session_id=${encodeURIComponent(sessionId)}`,
    ),

  answer: (sessionId: string, answer: string) =>
    req<ParseResult>("/parse/answer", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, answer }),
    }),

  runPipeline: (sessionId: string, options: Record<string, unknown> = {}) =>
    req<{ run_id: string; status: string }>("/pipeline/run", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, options }),
    }),

  pipelineStatus: (runId: string) =>
    req<PipelineStatus>(`/pipeline/status/${runId}`),

  runFiles: (runId: string) => req<RunFiles>(`/runs/${runId}/files`),

  runJson: (runId: string, artifact: string) =>
    req<Record<string, unknown>>(`/runs/${runId}/json/${artifact}`),

  /** Direct URL — used as an <img>/<object> src, not fetched as JSON. */
  svgUrl: (runId: string, filename: string) =>
    `${BASE}/runs/${runId}/svg/${filename}`,
};

/** The five pipeline stages, in the order the backend reports them. */
export const STAGES = [
  { step: 1, code: "PARSE", label: "Reading the brief" },
  { step: 2, code: "MATCH", label: "Matching reference plans" },
  { step: 3, code: "ENRICH", label: "Sizing and zoning rooms" },
  { step: 4, code: "GENERATE", label: "Carving the layout" },
  { step: 5, code: "RENDER", label: "Drawing the sheet" },
] as const;

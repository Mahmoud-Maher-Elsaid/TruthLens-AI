import { DEMO_TEXT, localDemoReport } from "./demo";
import { documentResponseSchema, reportSchema, type AuditReport, type DocumentResponse, type ProgressEvent } from "./types";

const apiBase = () => process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function uploadDocument(file: File): Promise<DocumentResponse> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${apiBase()}/api/v1/documents`, {
    method: "POST", body, signal: AbortSignal.timeout(120000)
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message = payload?.error?.message || `Document upload failed (HTTP ${response.status}).`;
    throw new Error(message);
  }
  return documentResponseSchema.parse(await response.json());
}

export async function streamAnalysis(text: string, mode: string, onEvent: (event: ProgressEvent) => void): Promise<AuditReport> {
  if (mode === "demo") {
    if (text !== DEMO_TEXT) throw new Error("Load the labeled demo example before running Demo mode.");
    const report = localDemoReport(text);
    onEvent({ event: "analysis_complete", message: "Loaded labeled precomputed demo", progress: 100, data: report });
    return report;
  }
    const response = await fetch(`${apiBase()}/api/v1/analyze/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode }), signal: AbortSignal.timeout(120000)
    });
    if (!response.ok || !response.body) throw new Error(`Analysis API returned HTTP ${response.status}.`);
    const reader = response.body.getReader(); const decoder = new TextDecoder();
    let buffer = ""; let report: AuditReport | null = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop() || "";
      for (const line of lines) if (line.trim()) {
        const event = JSON.parse(line) as ProgressEvent;
        onEvent(event);
        if (event.event === "analysis_failed") throw new Error(event.message);
        if (event.event === "analysis_complete") report = reportSchema.parse(event.data);
      }
      if (done) break;
    }
    if (!report) throw new Error("Analysis stream ended without a report.");
    return report;
}

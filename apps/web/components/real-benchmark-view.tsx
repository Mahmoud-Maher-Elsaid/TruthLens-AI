"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, DatabaseZap, Gauge, RefreshCw } from "lucide-react";

type Artifact = { name: string; data: Record<string, unknown> };
type Metric = { label: string; value: string };

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const asNumber = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const percent = (value: unknown) => { const number = asNumber(value); return number === null ? "Unavailable" : `${(number * 100).toFixed(1)}%`; };
const decimal = (value: unknown) => { const number = asNumber(value); return number === null ? "Unavailable" : number.toFixed(4); };
const milliseconds = (value: unknown) => { const number = asNumber(value); return number === null ? "Unavailable" : `${number.toFixed(1)} ms`; };
const mebibytes = (value: unknown) => { const number = asNumber(value); return number === null ? "Unavailable" : `${(number / 1024 / 1024).toFixed(0)} MiB`; };

function ResultCard({ title, artifact, metrics }: { title: string; artifact?: Artifact; metrics: Metric[] }) {
  const completed = artifact?.data.status === "completed" || artifact?.data.status === "passed";
  return <article>
    <span className="tiny-label">{completed ? "MEASURED" : "UNAVAILABLE"}</span>
    <h2>{title}</h2>
    {!artifact ? <p>No verified artifact was generated for this experiment.</p> : !completed ? <p>The experiment did not complete successfully. Inspect the artifact for its recorded failure.</p> : <dl className="metric-list">{metrics.map(metric => <div key={metric.label}><dt>{metric.label}</dt><dd>{metric.value}</dd></div>)}</dl>}
    {artifact && <details><summary>Machine-readable details</summary><pre>{JSON.stringify(artifact.data, null, 2)}</pre></details>}
  </article>;
}

export function RealBenchmarkView() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 7000);
    fetch("/api/benchmarks", { signal: controller.signal, cache: "no-store" })
      .then(response => response.ok ? response.json() : Promise.reject(new Error("unavailable")))
      .then((data: { artifacts?: Artifact[]; message?: string | null }) => { setArtifacts(data.artifacts || []); setMessage(data.message || null); })
      .catch(() => { setArtifacts([]); setMessage("The benchmark API is unavailable. No measurements are shown."); })
      .finally(() => { window.clearTimeout(timeout); setLoading(false); });
    return () => { controller.abort(); window.clearTimeout(timeout); };
  }, [attempt]);
  const byName = useMemo(() => new Map(artifacts.map(artifact => [artifact.name, artifact])), [artifacts]);
  if (loading) return <div className="benchmark-empty"><div className="skeleton wide"/><div className="skeleton cards"/></div>;
  if (!artifacts.length) return <div className="benchmark-empty"><div className="empty-orbit"><BarChart3/></div><span className="tiny-label">NO VERIFIED ARTIFACTS</span><h2>Benchmarks are unavailable.</h2><p>{message || "The evaluation scripts have not generated measurements."}</p><button className="button ghost" onClick={() => { setLoading(true); setAttempt(value => value + 1); }}><RefreshCw size={15}/> Check again</button><div className="planned-metrics"><span><Gauge/>Accuracy · macro F1 · latency</span><span><DatabaseZap/>Recall@K · MRR · evidence rate</span></div></div>;

  const baselineComparison = byName.get("experiment2_verifier_comparison.json");
  const comparisonBaseline = asRecord(asRecord(baselineComparison?.data.models).baseline);
  const legacyBaseline = byName.get("baseline_verifier.json");
  const baseline = Object.keys(comparisonBaseline).length ? baselineComparison : legacyBaseline;
  const baselineMetrics = Object.keys(comparisonBaseline).length ? asRecord(comparisonBaseline.metrics) : asRecord(legacyBaseline?.data.metrics);
  const fineTuned = byName.get("finetuned_verifier.json");
  const fineMetrics = asRecord(fineTuned?.data.metrics);
  const fineLatency = asRecord(fineTuned?.data.latency);
  const retrieval = byName.get("rag_validation.json");
  const retrievalMetrics = asRecord(retrieval?.data.metrics);
  const retrievalTiming = asRecord(retrieval?.data.timing);
  const quantization = byName.get("quantization_results.json");
  const higher = asRecord(quantization?.data.higher_precision);
  const quantized = asRecord(quantization?.data.quantized);
  const quantizedComparison = asRecord(quantization?.data.comparison);
  const systemLatency = byName.get("system_latency.json");
  const latency = asRecord(systemLatency?.data.latency);

  return <div className="artifact-grid">
    <ResultCard title="Baseline Verifier" artifact={baseline} metrics={[
      { label: "Samples", value: String(asNumber(baselineMetrics.samples) ?? "Unavailable") },
      { label: "Accuracy", value: percent(baselineMetrics.accuracy) },
      { label: "Macro F1", value: decimal(baselineMetrics.macro_f1) },
    ]}/>
    <ResultCard title="Fine-Tuned Verifier" artifact={fineTuned} metrics={[
      { label: "Samples", value: String(asNumber(fineMetrics.samples) ?? "Unavailable") },
      { label: "Accuracy", value: percent(fineMetrics.accuracy) },
      { label: "Macro F1", value: decimal(fineMetrics.macro_f1) },
      { label: "Median latency", value: milliseconds(fineLatency.p50_ms) },
    ]}/>
    <ResultCard title="Retrieval Metrics" artifact={retrieval} metrics={[
      { label: "Queries", value: String(asNumber(retrievalMetrics.queries) ?? "Unavailable") },
      { label: "Recall@K", value: decimal(retrievalMetrics["recall@3"]) },
      { label: "MRR", value: decimal(retrievalMetrics.mrr) },
      { label: "Mean retrieval", value: milliseconds(retrievalTiming.mean_retrieval_ms) },
    ]}/>
    <ResultCard title="Quantization" artifact={quantization} metrics={[
      { label: "BF16 peak VRAM", value: mebibytes(higher.peak_vram_bytes) },
      { label: "4-bit peak VRAM", value: mebibytes(quantized.peak_vram_bytes) },
      { label: "VRAM reduction", value: percent(quantizedComparison.peak_vram_reduction_fraction) },
      { label: "Label consistency", value: percent(quantizedComparison.label_output_consistency) },
    ]}/>
    <ResultCard title="Latency" artifact={systemLatency} metrics={[
      { label: "Runs", value: String(asNumber(latency.runs) ?? "Unavailable") },
      { label: "Mean", value: milliseconds(latency.mean_ms) },
      { label: "p50", value: milliseconds(latency.p50_ms) },
      { label: "p95", value: milliseconds(latency.p95_ms) },
    ]}/>
  </div>;
}

"use client";

import { useEffect, useState } from "react";
import { BarChart3, DatabaseZap, Gauge, RefreshCw } from "lucide-react";

type Artifact = { name: string; data: Record<string, unknown> };

export function BenchmarkView() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]); const [loading, setLoading] = useState(true); const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 4000);
    const base = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";
    fetch(`${base}/api/v1/benchmarks`, { signal: controller.signal })
      .then(response => response.ok ? response.json() : Promise.reject(new Error("unavailable")))
      .then((data: { artifacts?: Artifact[] }) => setArtifacts(data.artifacts || []))
      .catch(() => setArtifacts([]))
      .finally(() => { window.clearTimeout(timeout); setLoading(false); });
    return () => { controller.abort(); window.clearTimeout(timeout); };
  }, [attempt]);
  if (loading) return <div className="benchmark-empty"><div className="skeleton wide"/><div className="skeleton cards"/></div>;
  if (!artifacts.length) return <div className="benchmark-empty"><div className="empty-orbit"><BarChart3/></div><span className="tiny-label">NO VERIFIED ARTIFACTS</span><h2>Benchmarks have not been generated yet.</h2><p>This page only renders machine-readable results produced by the evaluation scripts. It never substitutes example numbers.</p><button className="button ghost" onClick={() => { setLoading(true); setAttempt(value => value + 1); }}><RefreshCw size={15}/> Check again</button><div className="planned-metrics"><span><Gauge/>Accuracy · macro F1 · latency</span><span><DatabaseZap/>Recall@K · MRR · evidence rate</span></div></div>;
  return <div className="artifact-grid">{artifacts.map(artifact => <article key={artifact.name}><span className="tiny-label">MEASURED ARTIFACT</span><h2>{artifact.name}</h2><pre>{JSON.stringify(artifact.data, null, 2)}</pre></article>)}</div>;
}

import type { Metadata } from "next";
import { RealBenchmarkView } from "@/components/real-benchmark-view";
export const metadata: Metadata = { title: "Benchmarks" };
export default function BenchmarksPage() { return <main id="main" className="shell page"><div className="page-hero"><span className="eyebrow">MEASURED, NEVER INVENTED</span><h1>Benchmark evidence.</h1><p>Evaluation, retrieval, latency, and quantization results appear only after the corresponding scripts have produced real JSON artifacts.</p></div><RealBenchmarkView/></main>; }

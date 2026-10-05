import type { Metadata } from "next";
import { AnalyzeWorkspace } from "@/components/analyze-workspace";
export const metadata: Metadata = { title: "Analyze", description: "Audit an AI-generated answer claim by claim." };
export default function AnalyzePage() { return <main id="main" className="shell page"><div className="page-intro"><span className="eyebrow">AUDIT WORKSPACE</span><p>Every verdict stays attached to the evidence used to reach it.</p></div><AnalyzeWorkspace/></main>; }

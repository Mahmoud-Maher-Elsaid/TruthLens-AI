"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { Check, Database, Layers3, Play, Sparkles } from "lucide-react";
import { DEMO_TEXT, localDemoReport } from "@/lib/demo";

export function CompareLab() {
  const [ran, setRan] = useState(false); const report = localDemoReport();
  const cards = [
    { title: "Base LLM", icon: Sparkles, tone: "muted", features: ["No retrieval", "No citations", "No verifier"], body: "Not run: no remote base model is configured. TruthLens does not invent a baseline output." },
    { title: "RAG-enhanced", icon: Database, tone: "cyan", features: ["Illustrative context", "No independent run", "No measured score"], body: "This card illustrates retrieved context using the labeled fixture. A separate RAG system was not run." },
    { title: "Full TruthLens", icon: Layers3, tone: "green", features: ["Precomputed fixture", "Claim-level verdicts", "Evidence graph"], body: `Labeled demo: ${report.summary.supported} supported, ${report.summary.contradicted} contradicted, ${report.summary.insufficient_evidence} insufficient.` }
  ];
  return <div><section className="compare-input"><span className="tiny-label">CONTROLLED INPUT</span><p>{DEMO_TEXT}</p><button className="button primary" onClick={() => setRan(true)} disabled={ran}><Play size={16}/>{ran ? "Demo comparison complete" : "Run demo comparison"}</button><small>Uses the same labeled precomputed fixture as Demo Mode. No remote model calls.</small></section>{ran && <motion.div className="compare-grid" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>{cards.map(card => <article className={`compare-card ${card.tone}`} key={card.title}><card.icon/><span className="tiny-label">PIPELINE VARIANT</span><h2>{card.title}</h2><p>{card.body}</p><ul>{card.features.map(feature => <li key={feature}><Check size={14}/>{feature}</li>)}</ul></article>)}</motion.div>}</div>;
}

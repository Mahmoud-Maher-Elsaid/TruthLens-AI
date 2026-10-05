import type { AuditReport, ClaimResult } from "./types";

const evidence = {
  id: "evidence-demo", text: "The Eiffel Tower is in Paris, France. It was constructed from 1887 to 1889 and is 330 metres tall with its current antenna.",
  metadata: { document_name: "Eiffel Tower reference", document_id: null, section: "Overview", chunk_id: "demo-kb-001" },
  retrieval_score: 0.82, rerank_score: 0.91, relationship: "supports" as const
};

function claim(id: string, text: string, verdict: ClaimResult["verdict"], explanation: string, correction?: string): ClaimResult {
  return { id, claim: text, normalized_claim: text.replace(/\.$/, ""), verdict, evidence_strength: verdict === "INSUFFICIENT_EVIDENCE" ? "LOW" : "HIGH", evidence: [{ ...evidence, relationship: verdict === "CONTRADICTED" ? "contradicts" : verdict === "SUPPORTED" ? "supports" : "related" }], explanation, corrected_statement: correction ?? null };
}

export const DEMO_TEXT = "The Eiffel Tower is in Paris. It was completed in 1920. It is 330 metres tall. Gustave Eiffel personally invented radio broadcasting from the tower.";

export function localDemoReport(text = DEMO_TEXT): AuditReport {
  const claims = [
    claim("claim-1", "The Eiffel Tower is in Paris.", "SUPPORTED", "The bundled reference directly confirms the location."),
    claim("claim-2", "The Eiffel Tower was completed in 1920.", "CONTRADICTED", "The reference dates construction to 1887–1889.", "The Eiffel Tower was completed in 1889 in Paris, France."),
    claim("claim-3", "The Eiffel Tower is 330 metres tall.", "SUPPORTED", "The bundled reference directly confirms the current height."),
    claim("claim-4", "Gustave Eiffel personally invented radio broadcasting from the tower.", "INSUFFICIENT_EVIDENCE", "The available reference does not establish or refute this claim.")
  ];
  return { analysis_id: "precomputed-demo", mode: "DEMO", provider: "frontend-precomputed-demo", is_precomputed_demo: true, original_text: text, summary: { claims_analyzed: 4, supported: 2, contradicted: 1, insufficient_evidence: 1, evidence_coverage_score: 0.75, score_definition: "Share of extracted claims with evidence sufficient to support or contradict them." }, claims, limitations: ["This is a precomputed demo fixture, not fresh model inference.", "Evidence strength labels are categorical, not calibrated probabilities."] };
}

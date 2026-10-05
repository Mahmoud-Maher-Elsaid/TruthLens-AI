import { z } from "zod";

export const verdictSchema = z.enum(["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]);
export const evidenceSchema = z.object({
  id: z.string(), text: z.string(),
  metadata: z.object({ document_name: z.string(), document_id: z.string().nullable().optional(), source_url: z.string().nullable().optional(), section: z.string().nullable().optional(), page: z.number().nullable().optional(), chunk_id: z.string() }),
  retrieval_score: z.number().nullable().optional(), rerank_score: z.number().nullable().optional(),
  relationship: z.enum(["supports", "contradicts", "related"])
});
export const claimSchema = z.object({
  id: z.string(), claim: z.string(), normalized_claim: z.string(), verdict: verdictSchema,
  evidence_strength: z.enum(["HIGH", "MEDIUM", "LOW"]), evidence: z.array(evidenceSchema),
  explanation: z.string(), corrected_statement: z.string().nullable().optional(), conflicting_evidence: z.boolean().optional()
});
export const reportSchema = z.object({
  analysis_id: z.string(), mode: z.enum(["DEMO", "LOCAL", "PRODUCTION"]), provider: z.string(),
  is_precomputed_demo: z.boolean(), original_text: z.string(),
  summary: z.object({ claims_analyzed: z.number(), supported: z.number(), contradicted: z.number(), insufficient_evidence: z.number(), evidence_coverage_score: z.number(), score_definition: z.string() }),
  claims: z.array(claimSchema), limitations: z.array(z.string())
});
export type AuditReport = z.infer<typeof reportSchema>;
export type ClaimResult = z.infer<typeof claimSchema>;
export type ProgressEvent = { event: string; message: string; progress: number; claim_index?: number; total_claims?: number; data?: unknown };

export const documentResponseSchema = z.object({
  document_id: z.string().uuid(), filename: z.string(), pages: z.number().nullable(),
  characters_indexed: z.number().nonnegative(), chunks_indexed: z.number().int().positive(),
  vector_backend: z.string()
});
export type DocumentResponse = z.infer<typeof documentResponseSchema>;

"use client";

import { useMemo, useState } from "react";
import { Background, Controls, Edge, MarkerType, MiniMap, Node, ReactFlow } from "@xyflow/react";
import type { AuditReport } from "@/lib/types";

const verdictColor = { SUPPORTED: "#39e58c", CONTRADICTED: "#ff5b68", INSUFFICIENT_EVIDENCE: "#f4b942" } as const;

export function EvidenceGraph({ report }: { report: AuditReport }) {
  const [selected, setSelected] = useState<string | null>(null);
  const { nodes, edges } = useMemo(() => {
    const graphNodes: Node[] = [{ id: "answer", position: { x: 0, y: 150 }, data: { label: "Original answer" }, className: "node-answer" }];
    const graphEdges: Edge[] = [];
    report.claims.forEach((claim, index) => {
      const claimId = claim.id; const evidenceId = `evidence-${claimId}-${claim.evidence[0]?.id || "empty"}`; const verdictId = `verdict-${claimId}`; const y = index * 145;
      graphNodes.push({ id: claimId, position: { x: 250, y }, data: { label: claim.normalized_claim }, className: "node-claim" });
      graphNodes.push({ id: evidenceId, position: { x: 590, y }, data: { label: claim.evidence[0]?.metadata.document_name || "No sufficient evidence" }, className: `node-evidence ${claim.verdict.toLowerCase()}` });
      graphNodes.push({ id: verdictId, position: { x: 885, y }, data: { label: claim.verdict.replace("_", " ") }, className: `node-verdict ${claim.verdict.toLowerCase()}` });
      const color = verdictColor[claim.verdict]; const dim = selected && !["answer", claimId, evidenceId, verdictId].includes(selected) ? 0.16 : 0.9;
      graphEdges.push({ id: `a-${claimId}`, source: "answer", target: claimId, type: "smoothstep", animated: false, style: { stroke: "#50707a", opacity: dim }, markerEnd: { type: MarkerType.ArrowClosed, color: "#50707a" } });
      graphEdges.push({ id: `e-${claimId}`, source: claimId, target: evidenceId, label: claim.evidence[0]?.relationship || "related", type: "smoothstep", animated: claim.verdict !== "INSUFFICIENT_EVIDENCE", style: { stroke: color, opacity: dim }, labelStyle: { fill: color, fontSize: 10 }, markerEnd: { type: MarkerType.ArrowClosed, color } });
      graphEdges.push({ id: `v-${claimId}`, source: evidenceId, target: verdictId, type: "smoothstep", style: { stroke: color, opacity: dim }, markerEnd: { type: MarkerType.ArrowClosed, color } });
    });
    return { nodes: graphNodes, edges: graphEdges };
  }, [report, selected]);
  return <div className="graph-shell"><div className="graph-header"><div><span className="tiny-label">INTERACTIVE RELATIONSHIP MAP</span><h3>Evidence graph</h3></div><p>Click a node to isolate its connection.</p></div><div className="graph-canvas" role="img" aria-label="Interactive graph connecting the original answer to claims, evidence, and verdicts"><ReactFlow nodes={nodes} edges={edges} onNodeClick={(_, node) => setSelected(current => current === node.id ? null : node.id)} fitView minZoom={0.35} maxZoom={1.5} proOptions={{ hideAttribution: true }}><Background color="#18313a" gap={24} size={1}/><Controls showInteractive={false}/><MiniMap nodeColor={node => node.className?.includes("supported") ? "#39e58c" : node.className?.includes("contradicted") ? "#ff5b68" : "#2de2e6"} maskColor="rgba(2,7,9,.72)"/></ReactFlow></div></div>;
}

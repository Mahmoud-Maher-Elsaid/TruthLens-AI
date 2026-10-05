"use client";

import { useMemo } from "react";
import { Background, Controls, Edge, MarkerType, Node, ReactFlow } from "@xyflow/react";

export function ArchitectureMap() {
  const nodes: Node[] = useMemo(() => [
    { id: "browser", position: { x: 0, y: 110 }, data: { label: "Browser / Next.js" }, className: "node-answer" },
    { id: "api", position: { x: 250, y: 110 }, data: { label: "FastAPI / NDJSON" }, className: "node-claim" },
    { id: "graph", position: { x: 500, y: 10 }, data: { label: "Audit orchestrator" }, className: "node-evidence" },
    { id: "models", position: { x: 500, y: 210 }, data: { label: "HF models / adapters" }, className: "node-evidence" },
    { id: "vector", position: { x: 800, y: 10 }, data: { label: "FAISS / Qdrant" }, className: "node-verdict supported" },
    { id: "report", position: { x: 800, y: 210 }, data: { label: "Cited audit report" }, className: "node-verdict supported" }
  ], []);
  const edges: Edge[] = useMemo(() => [["browser","api"],["api","graph"],["api","models"],["graph","vector"],["graph","report"],["models","report"]].map(([source,target], i) => ({ id: String(i), source, target, animated: i === 0 || i === 3, type: "smoothstep", style: { stroke: "#2de2e6" }, markerEnd: { type: MarkerType.ArrowClosed, color: "#2de2e6" } })), []);
  return <div className="architecture-canvas" role="img" aria-label="Interactive system architecture map"><ReactFlow nodes={nodes} edges={edges} fitView minZoom={.5} maxZoom={1.4} proOptions={{ hideAttribution: true }}><Background color="#18313a" gap={26}/><Controls showInteractive={false}/></ReactFlow></div>;
}

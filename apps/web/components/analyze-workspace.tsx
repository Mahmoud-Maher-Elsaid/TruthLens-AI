"use client";

import { ChangeEvent, FormEvent, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertCircle, ArrowRight, Check, ChevronDown, CircleDot, FileText, LoaderCircle, Network, RotateCcw, ScanLine, ShieldAlert, ShieldCheck, Upload, X } from "lucide-react";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import { toast } from "sonner";
import { streamAnalysis, uploadDocument } from "@/lib/api";
import { DEMO_TEXT } from "@/lib/demo";
import type { AuditReport, ClaimResult, ProgressEvent } from "@/lib/types";
import { EvidenceGraph } from "./evidence-graph";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const allowed = [".pdf", ".txt", ".md", ".markdown"];
const verdictMeta = {
  SUPPORTED: { label: "Supported", icon: ShieldCheck, color: "green" },
  CONTRADICTED: { label: "Contradicted", icon: ShieldAlert, color: "red" },
  INSUFFICIENT_EVIDENCE: { label: "Insufficient evidence", icon: AlertCircle, color: "amber" }
} as const;

function ClaimCard({ claim, index }: { claim: ClaimResult; index: number }) {
  const [open, setOpen] = useState(index === 0); const meta = verdictMeta[claim.verdict]; const Icon = meta.icon;
  return <motion.article layout className={`claim-card ${meta.color}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(index * 0.07, .35) }}>
    <button className="claim-main" onClick={() => setOpen(!open)} aria-expanded={open}><span className="claim-number">{String(index + 1).padStart(2, "0")}</span><span className="claim-copy"><b>{claim.claim}</b><small>{claim.normalized_claim}</small></span><span className={`verdict ${meta.color}`}><Icon size={15}/>{meta.label}</span><ChevronDown className={open ? "rotate" : ""} size={18}/></button>
    <AnimatePresence initial={false}>{open && <motion.div className="claim-detail" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}><div className="explanation"><span className="tiny-label">REASONING</span><p>{claim.explanation}</p>{claim.corrected_statement && <div className="correction"><Check size={15}/><span><b>Evidence-backed correction</b>{claim.corrected_statement}</span></div>}</div><div className="evidence-list"><span className="tiny-label">EVIDENCE USED · {claim.evidence_strength} STRENGTH</span>{claim.evidence.map(item => <blockquote key={item.id}><p>“{item.text}”</p><footer><FileText size={13}/>{item.metadata.document_name}{item.metadata.page ? ` · p.${item.metadata.page}` : ""}<span>retrieval {item.retrieval_score?.toFixed(2) ?? "n/a"}</span></footer></blockquote>)}</div></motion.div>}</AnimatePresence>
  </motion.article>;
}

export function AnalyzeWorkspace() {
  const [text, setText] = useState(""); const [mode, setMode] = useState("standard"); const [file, setFile] = useState<File | null>(null); const [uploadStatus, setUploadStatus] = useState<string | null>(null); const [events, setEvents] = useState<ProgressEvent[]>([]); const [report, setReport] = useState<AuditReport | null>(null); const [running, setRunning] = useState(false); const [error, setError] = useState<string | null>(null);
  const loadDemo = () => { setText(DEMO_TEXT); setMode("demo"); setFile(null); setUploadStatus(null); setReport(null); setEvents([]); toast.info("Loaded a labeled precomputed demo example"); };
  const onFile = (event: ChangeEvent<HTMLInputElement>) => { const candidate = event.target.files?.[0]; if (!candidate) return; const ext = candidate.name.slice(candidate.name.lastIndexOf(".")).toLowerCase(); if (!allowed.includes(ext)) { toast.error("Use a PDF, TXT, or Markdown document."); event.target.value = ""; return; } if (candidate.size > MAX_FILE_SIZE || candidate.size === 0) { toast.error(candidate.size === 0 ? "The document is empty." : "The maximum document size is 10 MB."); event.target.value = ""; return; } setFile(candidate); setUploadStatus(null); };
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (text.trim().length < 8) { setError("Enter at least one complete factual statement."); return; }
    if (file && mode === "demo") { setError("Attached evidence requires Standard or Deep mode and a live API."); return; }
    setError(null); setUploadStatus(null); setRunning(true); setReport(null); setEvents([]);
    try {
      if (file) {
        const uploaded = await uploadDocument(file);
        setUploadStatus(`Indexed ${uploaded.chunks_indexed} chunk${uploaded.chunks_indexed === 1 ? "" : "s"} from ${uploaded.filename}.`);
      }
      const result = await streamAnalysis(text.trim(), mode, item => setEvents(previous => [...previous, item]));
      setReport(result);
      if (result.is_precomputed_demo) toast.info("Showing the labeled precomputed demo example.");
      else toast.success("Audit complete");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The analysis service is unavailable.");
    } finally { setRunning(false); }
  }
  const progress = events.at(-1)?.progress ?? 0;
  return <div className="workspace">
    <form className="input-panel" onSubmit={submit}><div className="panel-heading"><div><span className="tiny-label">NEW AUDIT</span><h1>Inspect an AI response</h1></div><span className={`mode-indicator ${mode}`}>{mode === "demo" ? "PRECOMPUTED DEMO" : `${mode.toUpperCase()} AUDIT`}</span></div><label className="field-label" htmlFor="audit-text">Response or factual claim</label><div className="textarea-wrap"><textarea id="audit-text" value={text} onChange={event => setText(event.target.value)} maxLength={50000} placeholder="Paste an AI-generated response, factual claim, or analysis..." aria-describedby="text-help"/><span id="text-help">{text.length.toLocaleString()} / 50,000 characters</span></div><div className="mode-row" role="radiogroup" aria-label="Audit mode">{[{ id: "standard", title: "Standard Audit", text: "Current claim and evidence pass" }, { id: "deep", title: "Deep Audit (preview)", text: "Currently uses the same audit path" }, { id: "demo", title: "Demo Example", text: "Deterministic fixture" }].map(item => <label className={mode === item.id ? "selected" : ""} key={item.id}><input type="radio" name="mode" value={item.id} checked={mode === item.id} onChange={() => item.id === "demo" ? loadDemo() : setMode(item.id)}/><span><b>{item.title}</b><small>{item.text}</small></span></label>)}</div><div className="upload-row"><label className="upload"><Upload size={17}/><span>{file ? file.name : "Attach evidence document"}</span><input type="file" accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain,text/markdown" onChange={onFile} disabled={running}/></label>{file && <button type="button" className="icon-button" aria-label="Remove attached document" onClick={() => { setFile(null); setUploadStatus(null); }} disabled={running}><X size={16}/></button>}<small>PDF, TXT, MD · max 10 MB</small></div>{uploadStatus && <p role="status">{uploadStatus}</p>}{error && <div className="form-error" role="alert"><AlertCircle size={16}/>{error}</div>}<div className="form-actions"><button type="button" className="button ghost" onClick={loadDemo}>Load demo</button><button className="button primary" disabled={running}>{running ? <><LoaderCircle className="spin" size={18}/> Auditing…</> : <><ScanLine size={18}/> Run evidence audit <ArrowRight size={17}/></>}</button></div></form>
    <AnimatePresence>{running && <motion.section className="progress-panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} aria-live="polite"><div className="progress-head"><span><CircleDot size={15}/> PIPELINE ACTIVE</span><b>{progress}%</b></div><div className="progress-track"><motion.i animate={{ width: `${progress}%` }}/></div><div className="event-log">{events.slice(-5).map((item, index) => <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} key={`${item.event}-${index}`}><Check size={13}/>{item.message}</motion.p>)}</div></motion.section>}</AnimatePresence>
    {report && <motion.section className="report" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><div className="report-bar"><div><span className="tiny-label">AUDIT REPORT / {report.analysis_id}</span><h2>Evidence coverage summary</h2></div><span className="demo-disclosure">{report.is_precomputed_demo ? "Precomputed demo — not fresh inference" : `${report.mode} analysis`}</span></div><div className="summary-grid"><div className="coverage-card"><div className="chart"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={[{ value: report.summary.evidence_coverage_score }, { value: 1 - report.summary.evidence_coverage_score }]} dataKey="value" innerRadius={48} outerRadius={60} startAngle={90} endAngle={-270} stroke="none"><Cell fill="#2de2e6"/><Cell fill="#172329"/></Pie></PieChart></ResponsiveContainer><strong>{Math.round(report.summary.evidence_coverage_score * 100)}<small>%</small></strong></div><div><span className="tiny-label">EVIDENCE COVERAGE</span><p>{report.summary.score_definition}</p></div></div><div className="stat"><span>Claims</span><strong>{report.summary.claims_analyzed}</strong></div><div className="stat green"><span>Supported</span><strong>{report.summary.supported}</strong></div><div className="stat red"><span>Contradicted</span><strong>{report.summary.contradicted}</strong></div><div className="stat amber"><span>Insufficient</span><strong>{report.summary.insufficient_evidence}</strong></div></div><div className="report-layout"><div className="claims-column"><div className="column-title"><h3>Claim ledger</h3><span>{report.claims.length} atomic claims</span></div>{report.claims.map((claim, index) => <ClaimCard key={claim.id} claim={claim} index={index}/>)}</div><aside className="method-card"><Network size={20}/><span className="tiny-label">ACTIVE METHOD</span><h3>{report.provider}</h3><p>Claims are normalized, matched to evidence, reranked, and classified with bounded orchestration.</p><dl><div><dt>Mode</dt><dd>{report.mode}</dd></div><div><dt>Confidence</dt><dd>Categorical</dd></div><div><dt>Citations</dt><dd>Used evidence only</dd></div></dl><button className="button ghost" onClick={() => { setReport(null); setEvents([]); }}><RotateCcw size={15}/> New audit</button></aside></div><EvidenceGraph report={report}/><div className="limitations"><b>Read this report carefully.</b>{report.limitations.map(item => <p key={item}>{item}</p>)}</div></motion.section>}
  </div>;
}

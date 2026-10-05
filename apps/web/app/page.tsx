import Link from "next/link";
import { ArrowRight, CheckCircle2, CircleAlert, FileSearch, Network, ScanLine, ShieldCheck, Sparkles } from "lucide-react";

const stages = ["Extract", "Normalize", "Retrieve", "Rerank", "Verify", "Cite"];

export default function Home() {
  return <main id="main">
    <section className="hero shell">
      <div className="hero-glow" aria-hidden />
      <div className="eyebrow reveal"><span className="live-dot" /> AI RESPONSE FORENSICS / EVIDENCE ENGINE</div>
      <h1 className="display reveal delay-1">Don&apos;t trust an answer.<br/><span>Verify it.</span></h1>
      <p className="hero-copy reveal delay-2">TruthLens AI disassembles fluent AI responses into atomic claims, traces each one to evidence, and shows you exactly what holds up.</p>
      <div className="hero-actions reveal delay-3"><Link className="button primary" href="/analyze"><ScanLine size={18}/> Analyze a response <ArrowRight size={17}/></Link><Link className="button ghost" href="/architecture">Explore the system</Link></div>
      <div className="hero-proof reveal delay-3"><span><ShieldCheck size={15}/> Cited evidence</span><span><Network size={15}/> Claim graph</span><span><Sparkles size={15}/> Clear corrections</span></div>
      <div className="audit-visual reveal delay-2" aria-label="Example claim audit visualization">
        <div className="scan-beam" aria-hidden />
        <div className="visual-head"><span>LIVE AUDIT / TL-2026-042</span><span className="mode-chip">DEMO MODE</span></div>
        <div className="visual-grid">
          <div className="source-answer"><span className="tiny-label">ORIGINAL ANSWER</span><p>The Eiffel Tower is in Paris. It was completed in <mark>1920</mark> and stands 330 metres tall.</p><div className="token-line"><i/><i/><i/><i/><i/></div></div>
          <div className="claims-stack">
            <div className="mini-claim supported"><CheckCircle2/><span><b>Claim 01</b>Located in Paris</span><em>SUPPORTED</em></div>
            <div className="mini-claim contradicted"><CircleAlert/><span><b>Claim 02</b>Completed in 1920</span><em>CONTRADICTED</em></div>
            <div className="mini-claim supported"><CheckCircle2/><span><b>Claim 03</b>330 metres tall</span><em>SUPPORTED</em></div>
          </div>
          <div className="evidence-rail"><div className="ring"><strong>75</strong><span>% COVERAGE</span></div><p><FileSearch size={14}/> 1 cited source</p><p><Network size={14}/> 3 claim links</p></div>
        </div>
      </div>
    </section>
    <section className="section shell">
      <div className="section-kicker">THE PROBLEM</div><div className="split-heading"><h2>Fluency is not evidence.</h2><p>Language models are optimized to continue text—not to guarantee that every statement is supported. TruthLens makes the missing verification layer visible.</p></div>
      <div className="feature-grid"><article><span>01</span><FileSearch/><h3>Atomic claim extraction</h3><p>Separates checkable facts from opinion, rhetoric, and vague language.</p></article><article><span>02</span><Network/><h3>Evidence relationships</h3><p>Retrieves, reranks, and maps cited passages to the claims they actually address.</p></article><article><span>03</span><ShieldCheck/><h3>Careful verdicts</h3><p>Distinguishes contradiction from missing evidence—because unknown is not false.</p></article></div>
    </section>
    <section className="pipeline-section"><div className="shell"><div className="section-kicker">VERIFICATION PIPELINE</div><h2 className="pipeline-title">From answer to audit trail.</h2><div className="pipeline-row">{stages.map((stage, index) => <div className="stage" key={stage}><span>{String(index + 1).padStart(2, "0")}</span><b>{stage}</b>{index < stages.length - 1 && <i aria-hidden/>}</div>)}</div><div className="cta-panel"><div><span className="tiny-label">READY TO INSPECT</span><h2>Put an answer under the lens.</h2></div><Link href="/analyze" className="button primary">Launch TruthLens <ArrowRight size={17}/></Link></div></div></section>
  </main>;
}

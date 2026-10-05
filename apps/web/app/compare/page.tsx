import type { Metadata } from "next";
import { CompareLab } from "@/components/compare-lab";
export const metadata: Metadata = { title: "Compare" };
export default function ComparePage() { return <main id="main" className="shell page"><div className="page-hero"><span className="eyebrow">CONTROLLED COMPARISON</span><h1>See what each layer adds.</h1><p>Compare an input-only baseline, evidence retrieval, and the full audited pipeline without passing demo output off as a live model benchmark.</p></div><CompareLab/></main>; }

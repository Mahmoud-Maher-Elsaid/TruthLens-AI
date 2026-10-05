"use client";
import { AlertTriangle, RotateCcw } from "lucide-react";
export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) { return <main id="main" className="shell error-state"><AlertTriangle/><p className="eyebrow">SYSTEM EXCEPTION</p><h1>The lens lost focus.</h1><p>The interface hit an unexpected error. No audit data was submitted again.</p><button className="button primary" onClick={reset}><RotateCcw size={16}/> Retry safely</button></main>; }

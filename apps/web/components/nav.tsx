"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { useState } from "react";
import { Logo } from "./logo";

const links = [{ href: "/analyze", label: "Analyze" }, { href: "/compare", label: "Compare" }, { href: "/benchmarks", label: "Benchmarks" }, { href: "/architecture", label: "Architecture" }, { href: "/about", label: "About" }];

export function Nav() {
  const path = usePathname(); const [open, setOpen] = useState(false);
  return <header className="site-header"><nav className="nav-wrap" aria-label="Primary navigation"><Link href="/" className="logo-link"><Logo /></Link><button className="menu-button" aria-label={open ? "Close navigation" : "Open navigation"} aria-expanded={open} onClick={() => setOpen(!open)}>{open ? <X /> : <Menu />}</button><div className={`nav-links ${open ? "is-open" : ""}`}>{links.map(link => <Link key={link.href} href={link.href} onClick={() => setOpen(false)} className={path === link.href ? "active" : ""}>{link.label}</Link>)}<Link href="/analyze" className="nav-cta">Open auditor <span aria-hidden>↗</span></Link></div></nav></header>;
}

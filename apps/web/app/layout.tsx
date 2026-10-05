import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import { Nav } from "@/components/nav";
import "@xyflow/react/dist/style.css";
import "./globals.css";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://truthlens-ai.vercel.app"),
  title: { default: "TruthLens AI — Verify every claim", template: "%s | TruthLens AI" },
  description: "An AI response auditor and evidence verification engine. Don't trust an answer. Verify it.",
  openGraph: { title: "TruthLens AI", description: "Claim-level evidence verification for AI-generated answers.", type: "website" },
  twitter: { card: "summary_large_image", title: "TruthLens AI", description: "Don't trust an answer. Verify it." },
  icons: { icon: "/favicon.svg" }
};
export const viewport: Viewport = { themeColor: "#070a0d", colorScheme: "dark" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${sans.variable} ${mono.variable}`}><a className="skip-link" href="#main">Skip to content</a><Nav />{children}<Toaster theme="dark" richColors /></body></html>;
}

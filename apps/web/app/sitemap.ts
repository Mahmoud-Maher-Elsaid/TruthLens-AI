import type { MetadataRoute } from "next";
export default function sitemap(): MetadataRoute.Sitemap { return ["", "/analyze", "/compare", "/benchmarks", "/architecture", "/about"].map(path => ({ url: `https://truthlens-ai.vercel.app${path}`, lastModified: new Date(), changeFrequency: "monthly" as const, priority: path ? 0.7 : 1 })); }

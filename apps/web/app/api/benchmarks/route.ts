import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const base = (process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
  try {
    const response = await fetch(`${base}/api/v1/benchmarks`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Benchmark API returned ${response.status}`);
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({
      status: "unavailable",
      artifacts: [],
      message: "The benchmark API is unavailable. No measurements are shown.",
    });
  }
}

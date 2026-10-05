import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { RealBenchmarkView } from "@/components/real-benchmark-view";

afterEach(() => vi.unstubAllGlobals());

describe("RealBenchmarkView", () => {
  it("renders measured API artifact values when available", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: "available",
      artifacts: [{ name: "experiment2_verifier_comparison.json", data: { status: "completed", models: { baseline: { metrics: { samples: 180, accuracy: 0.6944444444, macro_f1: 0.6758189800742992 } } } } }],
    }))));
    render(<RealBenchmarkView />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Baseline Verifier" })).toBeInTheDocument());
    expect(screen.getByText("69.4%")).toBeInTheDocument();
    expect(screen.getByText("0.6758")).toBeInTheDocument();
  });

  it("keeps an explicit unavailable state when the proxy cannot respond", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    render(<RealBenchmarkView />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Benchmarks are unavailable." })).toBeInTheDocument());
    expect(screen.getByText("The benchmark API is unavailable. No measurements are shown.")).toBeInTheDocument();
  });
});

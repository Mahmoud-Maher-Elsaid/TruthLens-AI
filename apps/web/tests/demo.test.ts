import { describe, expect, it } from "vitest";
import { localDemoReport } from "@/lib/demo";
import { reportSchema } from "@/lib/types";

describe("demo audit fixture", () => {
  it("is schema-valid and clearly labeled", () => {
    const report = localDemoReport();
    expect(reportSchema.parse(report)).toEqual(report);
    expect(report.is_precomputed_demo).toBe(true);
    expect(report.provider).toContain("demo");
  });

  it("keeps insufficient evidence distinct from contradiction", () => {
    const report = localDemoReport();
    expect(report.summary.contradicted).toBe(1);
    expect(report.summary.insufficient_evidence).toBe(1);
    expect(report.claims.some(claim => claim.verdict === "INSUFFICIENT_EVIDENCE")).toBe(true);
  });
});

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

test("landing page loads", async ({ page }) => { await page.goto("/"); await expect(page.getByRole("heading", { name: /don't trust an answer/i })).toBeVisible(); });
test("analyze validates and demo completes", async ({ page }) => { await page.goto("/analyze"); await page.getByLabel(/response or factual claim/i).fill("tiny"); await page.getByRole("button", { name: /run evidence audit/i }).click(); await expect(page.getByText(/enter at least one complete factual statement/i)).toBeVisible(); await page.getByRole("button", { name: /load demo/i }).click(); await page.getByRole("button", { name: /run evidence audit/i }).click(); await expect(page.getByRole("heading", { name: /evidence coverage summary/i })).toBeVisible({ timeout: 10000 }); await expect(page.getByText(/evidence graph/i).first()).toBeVisible(); });
test("compare page runs fixture", async ({ page }) => { await page.goto("/compare"); await page.getByRole("button", { name: /run demo comparison/i }).click(); await expect(page.getByRole("heading", { name: "Full TruthLens" })).toBeVisible(); });
test("benchmark page renders measured data returned by the proxy", async ({ page }) => {
  const comparison = JSON.parse(readFileSync(resolve(process.cwd(), "../../ml/benchmarks/experiment2_verifier_comparison.json"), "utf8"));
  await page.route("**/api/benchmarks", route => route.fulfill({ contentType: "application/json", body: JSON.stringify({ status: "available", artifacts: [{ name: "experiment2_verifier_comparison.json", data: comparison }] }) }));
  await page.goto("/benchmarks");
  const baselineCard = page.getByRole("heading", { name: "Baseline Verifier" }).locator("..");
  await expect(baselineCard).toBeVisible();
  await expect(baselineCard.getByText("69.4%", { exact: true })).toBeVisible();
  await expect(baselineCard.getByText("0.6758", { exact: true })).toBeVisible();
});
test("architecture graph loads", async ({ page }) => { await page.goto("/architecture"); await expect(page.getByRole("img", { name: /interactive system architecture map/i })).toBeVisible(); });

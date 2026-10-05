import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalyzeWorkspace } from "@/components/analyze-workspace";
import * as api from "@/lib/api";
import { localDemoReport } from "@/lib/demo";

afterEach(() => vi.restoreAllMocks());

describe("AnalyzeWorkspace", () => {
  it("shows input validation", () => {
    render(<AnalyzeWorkspace />);
    fireEvent.change(screen.getByLabelText(/response or factual claim/i), { target: { value: "tiny" } });
    fireEvent.click(screen.getByRole("button", { name: /run evidence audit/i }));
    expect(screen.getByRole("alert")).toHaveTextContent(/complete factual statement/i);
  });

  it("loads the labeled demo", () => {
    render(<AnalyzeWorkspace />);
    fireEvent.click(screen.getByRole("button", { name: /load demo/i }));
    expect((screen.getByLabelText(/response or factual claim/i) as HTMLTextAreaElement).value).toContain("Eiffel Tower");
    expect(screen.getByText("PRECOMPUTED DEMO")).toBeInTheDocument();
  });

  it("indexes an attached document before starting analysis", async () => {
    const upload = vi.spyOn(api, "uploadDocument").mockResolvedValue({
      document_id: "c2b1f6a7-0a87-4e55-a250-782737841d68", filename: "notes.txt",
      pages: null, characters_indexed: 20, chunks_indexed: 1, vector_backend: "faiss"
    });
    const analyze = vi.spyOn(api, "streamAnalysis").mockResolvedValue(localDemoReport());
    render(<AnalyzeWorkspace />);
    fireEvent.change(screen.getByLabelText(/response or factual claim/i), { target: { value: "The Lantern Museum opened in 2005." } });
    const file = new File(["The Lantern Museum opened in 2005."], "notes.txt", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText(/attach evidence document/i), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /run evidence audit/i }));
    await waitFor(() => expect(analyze).toHaveBeenCalledOnce());
    expect(upload).toHaveBeenCalledWith(file);
    expect(upload.mock.invocationCallOrder[0]).toBeLessThan(analyze.mock.invocationCallOrder[0]);
    expect(screen.getByRole("status")).toHaveTextContent("Indexed 1 chunk from notes.txt.");
  });
});

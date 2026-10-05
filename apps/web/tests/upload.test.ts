import { afterEach, describe, expect, it, vi } from "vitest";
import { uploadDocument } from "@/lib/api";

afterEach(() => vi.unstubAllGlobals());

describe("document upload API", () => {
  const file = new File(["Evidence lives here."], "notes.txt", { type: "text/plain" });

  it("posts multipart data and validates a successful ingestion", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      document_id: "c2b1f6a7-0a87-4e55-a250-782737841d68", filename: "notes.txt",
      pages: null, characters_indexed: 20, chunks_indexed: 1, vector_backend: "faiss"
    }), { status: 201 }));
    vi.stubGlobal("fetch", fetchMock);
    const result = await uploadDocument(file);
    expect(result.chunks_indexed).toBe(1);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/v1\/documents$/);
    expect(fetchMock.mock.calls[0][1].body).toBeInstanceOf(FormData);
    expect(fetchMock.mock.calls[0][1].body.get("file")).toBe(file);
  });

  it("shows the backend error instead of claiming ingestion", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code: "VALIDATION_ERROR", message: "Document is empty.", request_id: "req-1" }
    }), { status: 422 })));
    await expect(uploadDocument(file)).rejects.toThrow("Document is empty.");
  });

  it("rejects a malformed success response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      document_id: "c2b1f6a7-0a87-4e55-a250-782737841d68", filename: "notes.txt",
      pages: null, characters_indexed: 20, chunks_indexed: 0, vector_backend: "faiss"
    }), { status: 201 })));
    await expect(uploadDocument(file)).rejects.toThrow();
  });
});

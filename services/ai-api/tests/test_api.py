import io

DEMO_TEXT = "The Eiffel Tower is in Paris. It was completed in 1920. It is 330 metres tall."


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["backend_mode"] == "DEMO"
    assert response.json()["model_readiness"] == "ready (deterministic demo)"
    assert response.headers["X-Request-ID"]


def test_analyze_structured_verdicts(client):
    response = client.post("/api/v1/analyze", json={"text": DEMO_TEXT, "mode": "demo"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_precomputed_demo"] is True
    assert {claim["verdict"] for claim in data["claims"]} >= {"SUPPORTED", "CONTRADICTED"}
    assert data["summary"]["claims_analyzed"] == 3


def test_stream_finishes_with_report(client):
    response = client.post("/api/v1/analyze/stream", json={"text": DEMO_TEXT, "mode": "demo"})
    assert response.status_code == 200
    lines = response.text.strip().splitlines()
    assert '"event":"validation_started"' in lines[0]
    assert '"event":"analysis_complete"' in lines[-1]
    assert not any('"event":"evidence_found"' in line for line in lines)
    assert not any('"event":"reranking_complete"' in line for line in lines)


def test_rejects_blank_text(client):
    response = client.post("/api/v1/analyze", json={"text": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_rejects_unsupported_file(client):
    response = client.post("/api/v1/documents", files={"file": ("unsafe.exe", io.BytesIO(b"x"), "application/octet-stream")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "HTTP_ERROR"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_indexes_text_document(client):
    response = client.post("/api/v1/documents", files={"file": ("notes.txt", io.BytesIO(b"Evidence lives here."), "text/plain")})
    assert response.status_code == 201
    assert response.json()["chunks_indexed"] == 1
    assert response.json()["vector_backend"] == "lexical-demo"
    assert client.app.state.store._chunks[-1].metadata["document_id"] == response.json()["document_id"]


def test_rejects_malformed_pdf_with_structured_error(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", io.BytesIO(b"not a pdf"), "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "HTTP_ERROR"


def test_benchmarks_reports_available_artifacts(client):
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "available"
    assert isinstance(payload["artifacts"], list)
    assert payload["artifacts"]
    assert all("name" in artifact and "data" in artifact for artifact in payload["artifacts"])

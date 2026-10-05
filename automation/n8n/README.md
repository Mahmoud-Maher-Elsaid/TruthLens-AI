# TruthLens n8n automation

n8n provides the external operational layer around TruthLens. The FastAPI orchestrator owns claim extraction, retrieval, reranking, NLI assessment, evidence aggregation, and report construction. n8n owns request correlation, review routing, scheduled health checks, and integrations that are outside the model pipeline.

## Workflows

- `analysis_review_workflow.json`: receives an external JSON request, validates and correlates it, calls `POST /api/v1/analyze` with a bounded retry, classifies review risk, and returns either an approved structured result or a human-review payload. Contradictions, low-strength insufficient evidence, conflicting evidence, low evidence strength, malformed responses, timeouts, and fallback providers are routed for review. An insufficient-evidence claim is escalated only when its strength is below HIGH; absence of evidence is not treated as false automatically.
- `quality_monitor_workflow.json`: runs every 15 minutes, checks `/api/v1/health` and `/api/v1/benchmarks`, validates the response shape, checks the configured readiness signals and selected artifact presence, and emits a structured report. It does not invent live latency or quality values.

## Environment and import

For the preferred local setup, FastAPI runs on the Windows host and n8n runs in Docker. Set these n8n container environment variables:

```text
TRUTHLENS_API_BASE_URL=http://host.docker.internal:8000
TRUTHLENS_MIN_BASELINE_MACRO_F1=0.6758189800742992
N8N_BLOCK_ENV_ACCESS_IN_NODE=false
TZ=Africa/Cairo
GENERIC_TIMEZONE=Africa/Cairo
```

`N8N_BLOCK_ENV_ACCESS_IN_NODE=false` is required because the workflow reads `TRUTHLENS_API_BASE_URL` through `$env`. The monitor uses the same full measured baseline macro F1 (`0.6758189800742992`) when the optional floor is absent. It raises `QUALITY_REGRESSION_MACRO_F1` only when `observed < reference - 1e-9`, so floating-point representation does not create a false regression. No API key, Slack credential, or email credential is embedded. Add credentials through n8n's credential store only when an external integration is intentionally added.

Import each JSON file from **Workflows → Import from File**. Both workflows are inactive exports and use generic webhook/output nodes; activate them only after setting the environment variable and reviewing the response route.

## Test payloads and behavior

Send `examples/analysis_request.json` to the webhook URL. A valid low-risk response returns `{ok:true, route:"approved_output"}`. A contradiction, conflicting evidence flag, low evidence strength, provider fallback, timeout, malformed response, or qualifying insufficient-evidence claim returns HTTP 202 with a `human_review` payload. The payload maps only real API fields; `confidence` is `null` because the API exposes categorical evidence strength rather than calibrated probability.

The monitor's output is in `examples/quality_alert.json` shape. Healthy runs have `ok:true` and an empty `alerts` list. Missing or malformed health/benchmark responses become explicit alerts. The current API does not expose a reliable live latency signal, so `latency_signal` remains `null`. Health exposes readiness text rather than the exact configured verifier identity; verify the production setting from the service configuration when that identity matters.

## Preferred local Docker setup

Start FastAPI on the Windows host with `--host 0.0.0.0`. n8n reaches that host through `host.docker.internal`; it must not use `localhost` from inside its container. The persistent n8n Docker volume is `truthlens_n8n_data`.

```powershell
docker run --rm --name truthlens-n8n -p 5678:5678 -e TRUTHLENS_API_BASE_URL=http://host.docker.internal:8000 -e TRUTHLENS_MIN_BASELINE_MACRO_F1=0.6758189800742992 -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false -e TZ=Africa/Cairo -e GENERIC_TIMEZONE=Africa/Cairo -v truthlens_n8n_data:/home/node/.n8n n8nio/n8n
```

No ngrok or WSL command is required for normal local development.

## Local runtime smoke test

1. Start the FastAPI backend in `LOCAL` mode with the baseline verifier and bind it to `0.0.0.0`, then confirm `GET http://127.0.0.1:8000/api/v1/health` returns a JSON health response. Keep that PowerShell window running.
2. Start local n8n in Docker with the environment variables above. Its API base URL must be `http://host.docker.internal:8000`.
3. Import `analysis_review_workflow.json`, then `quality_monitor_workflow.json`. Keep both inactive until their test settings are reviewed.
4. In the analysis workflow, use n8n's test webhook URL: `POST /webhook-test/truthlens-analysis-review`. After activation, the production webhook path is `POST /webhook/truthlens-analysis-review`.
5. Send the contents of `examples/analysis_request.json`. The HTTP Request node sends exactly two JSON body fields: `text` from `{{$json.original_input}}` and `mode` from `{{$json.mode}}`. With the verified LOCAL reference case, the result should be an HTTP 202 `human_review` route because the claim is contradicted. The response must contain a real FastAPI audit report and a `CONTRADICTED` escalation reason.
6. Execute `quality_monitor_workflow.json` once manually from n8n. It calls only `/api/v1/health` and `/api/v1/benchmarks`; inspect the emitted JSON. A healthy local service can still report an explicit vector-store or artifact alert when the corresponding API signal is degraded or absent.

## Secrets and operational limits

Keep the API base URL and any credentials in n8n environment variables or managed credentials. Do not place tokens in workflow JSON or example payloads. The analysis route has a 120-second request timeout and two HTTP attempts; n8n cannot undo synchronous model work after a timeout. The monitor checks artifacts returned by the API and cannot replace the project's fixed held-out evaluation protocol.

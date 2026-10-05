from __future__ import annotations
import argparse, json, statistics, time, urllib.request
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--url", default="http://localhost:8000/api/v1/analyze"); p.add_argument("--runs", type=int, default=20); p.add_argument("--output", type=Path, default=Path("ml/benchmarks/system_latency.json")); a = p.parse_args()
    payload = json.dumps({"text": "The Eiffel Tower is in Paris.", "mode": "demo"}).encode(); samples = []
    for _ in range(a.runs):
        start = time.perf_counter(); urllib.request.urlopen(urllib.request.Request(a.url, data=payload, headers={"Content-Type": "application/json"}), timeout=120).read(); samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples); result = {"runs": len(samples), "p50_ms": statistics.median(samples), "p95_ms": ordered[min(len(ordered)-1, round(.95*(len(ordered)-1)))] if len(samples) >= 20 else None, "p95_note": None if len(samples) >= 20 else "Not measured: at least 20 samples required."}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

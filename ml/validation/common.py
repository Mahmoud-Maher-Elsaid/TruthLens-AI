from __future__ import annotations

import gc
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
FEVER_LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment() -> dict[str, Any]:
    import torch

    gpu: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        left = torch.tensor([1.0, 2.0], device="cuda")
        right = torch.tensor([3.0, 4.0], device="cuda")
        probe = float(torch.dot(left, right).item())
        torch.cuda.synchronize()
        gpu.update(
            {
                "model": properties.name,
                "vram_bytes": properties.total_memory,
                "vram_mib": round(properties.total_memory / 1024**2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "tensor_probe": {"operation": "dot([1,2], [3,4])", "result": probe},
            }
        )
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu["driver_version"] = query.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        gpu["driver_version"] = None
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": gpu,
    }


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "samples": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=LABELS).tolist(),
        "per_class": {
            label: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }


def latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    result: dict[str, Any] = {
        "runs": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms) if samples_ms else None,
        "p50_ms": statistics.median(samples_ms) if samples_ms else None,
        "p95_ms": None,
    }
    if len(ordered) >= 20:
        result["p95_ms"] = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    else:
        result["p95_note"] = "Unavailable: at least 20 observations are required."
    return result


def parse_generated_label(text: str) -> str | None:
    normalized = text.upper().strip()
    matches = re.findall(r"\b(?:SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE)\b", normalized)
    return matches[0] if matches else None


def release_cuda(*objects: object) -> None:
    import torch

    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_demo_kb_chunks(kb_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            continue
        section = "Document"
        paragraphs: list[str] = []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            clean = " ".join(line.strip() for line in block.splitlines() if line.strip())
            if not clean:
                continue
            if clean.startswith("#"):
                section = clean.lstrip("# ")
                continue
            paragraphs.append(clean)
        for index, text in enumerate(paragraphs):
            texts.append(text)
            metadata.append(
                {
                    "document_name": path.name,
                    "section": section,
                    "chunk_index": index,
                    "source_path": str(path.as_posix()),
                    "expected_chunk_id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                }
            )
    if not texts:
        raise RuntimeError(f"No supported knowledge-base files found under {kb_dir}.")
    return texts, metadata

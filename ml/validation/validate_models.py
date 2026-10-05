from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common import environment, parse_generated_label, release_cuda, utc_now, write_json

MODELS = [
    ("embedding", "sentence-transformers/all-MiniLM-L6-v2"),
    ("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("baseline_verifier", "cross-encoder/nli-deberta-v3-small"),
    ("finetuning_base", "Qwen/Qwen2.5-1.5B-Instruct"),
]


def metadata(model_id: str) -> dict[str, Any]:
    from huggingface_hub import model_info

    info = model_info(model_id)
    card = info.card_data
    license_name = getattr(card, "license", None) if card else None
    return {
        "requested_id": model_id,
        "resolved_id": info.id,
        "revision": info.sha,
        "private": info.private,
        "gated": info.gated,
        "library": info.library_name,
        "pipeline_tag": info.pipeline_tag,
        "license": license_name,
        "usage_assessment": (
            "Apache-2.0 is a permissive license suitable for this research/demo use; "
            "downstream users remain responsible for data and deployment obligations."
            if license_name == "apache-2.0"
            else "Review the model card and license before use."
        ),
    }


def validate_embedding(model_id: str) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(model_id)
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    vectors = model.encode(
        ["The Eiffel Tower is in Paris.", "Paris is the capital of France."],
        normalize_embeddings=True,
    )
    inference = time.perf_counter() - started
    result = {
        "tokenizer_loaded": getattr(model, "tokenizer", None) is not None,
        "model_loaded": True,
        "load_seconds": loaded,
        "inference_seconds": inference,
        "inference": {
            "shape": list(vectors.shape),
            "finite": bool(__import__("numpy").isfinite(vectors).all()),
            "cosine_similarity": float(vectors[0] @ vectors[1]),
        },
    }
    release_cuda(model)
    return result


def validate_cross_encoder(model_id: str, verifier: bool) -> dict[str, Any]:
    import numpy as np
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    model = CrossEncoder(model_id)
    loaded = time.perf_counter() - started
    pairs = (
        [
            ("Paris is the capital of France.", "Paris is in France."),
            ("Paris is the capital of France.", "Paris is in Italy."),
        ]
        if verifier
        else [
            ("Where is the Eiffel Tower?", "The Eiffel Tower is in Paris, France."),
            ("Where is the Eiffel Tower?", "Water freezes at zero degrees Celsius."),
        ]
    )
    started = time.perf_counter()
    scores = np.asarray(model.predict(pairs))
    inference = time.perf_counter() - started
    labels = getattr(model.model.config, "id2label", {})
    result = {
        "tokenizer_loaded": model.tokenizer is not None,
        "model_loaded": True,
        "load_seconds": loaded,
        "inference_seconds": inference,
        "inference": {
            "shape": list(scores.shape),
            "finite": bool(np.isfinite(scores).all()),
            "scores": scores.tolist(),
            "id2label": {str(key): value for key, value in labels.items()},
        },
    }
    release_cuda(model)
    return result


def validate_qwen(model_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    quantization = None
    if torch.cuda.is_available():
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if quantization else None,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    load_seconds = time.perf_counter() - started
    prompt = (
        "Return exactly SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.\n"
        "Claim: Paris is in France.\nEvidence: Paris is the capital of France.\nLabel:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    result = {
        "tokenizer_loaded": True,
        "model_loaded": True,
        "tokenizer_load_seconds": tokenizer_seconds,
        "load_seconds": load_seconds,
        "load_precision": "4-bit NF4" if quantization else "float32",
        "inference_seconds": inference_seconds,
        "inference": {"generated_text": generated, "parsed_label": parse_generated_label(generated)},
    }
    release_cuda(model, tokenizer)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("ml/benchmarks/model_validation.json")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve passed model records in an existing artifact and run only unfinished roles.",
    )
    args = parser.parse_args()
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        payload["status"] = "running"
        payload["resumed_at"] = utc_now()
        payload.setdefault("models", {})
    else:
        payload = {
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "environment": environment(),
            "models": {},
        }
    for role, model_id in MODELS:
        if args.resume and payload["models"].get(role, {}).get("status") == "passed":
            continue
        record: dict[str, Any] = {"status": "failed"}
        try:
            record.update(metadata(model_id))
            if role == "embedding":
                record.update(validate_embedding(model_id))
            elif role == "reranker":
                record.update(validate_cross_encoder(model_id, verifier=False))
            elif role == "baseline_verifier":
                record.update(validate_cross_encoder(model_id, verifier=True))
            else:
                record.update(validate_qwen(model_id))
            record["status"] = "passed"
        except Exception as exc:  # artifact must preserve the real failure
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
        payload["models"][role] = record
        write_json(args.output, payload)
    payload["completed_at"] = utc_now()
    payload["status"] = (
        "passed"
        if all(item["status"] == "passed" for item in payload["models"].values())
        else "partial_failure"
    )
    write_json(args.output, payload)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

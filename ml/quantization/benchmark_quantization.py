"""Measure real load, generation latency, and CUDA allocation for baseline vs 4-bit."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path


def run(model_id: str, four_bit: bool) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if four_bit else None
    started = time.perf_counter(); tokenizer = AutoTokenizer.from_pretrained(model_id); model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant, torch_dtype="auto"); load_s = time.perf_counter() - started
    inputs = tokenizer("Claim: Paris is in France. Evidence: Paris is the capital of France. Label:", return_tensors="pt").to(model.device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    started = time.perf_counter(); model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return {"mode": "4bit" if four_bit else "higher_precision", "model": model_id, "load_seconds": load_s, "inference_seconds": time.perf_counter()-started, "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None, "vram_note": None if torch.cuda.is_available() else "Not measured: CUDA unavailable."}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); p.add_argument("--output", type=Path, default=Path("ml/benchmarks/quantization.json")); a=p.parse_args()
    result = {"baseline": run(a.model, False), "quantized": run(a.model, True), "quality_delta": "Not measured yet. Run verifier evaluation for each model."}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

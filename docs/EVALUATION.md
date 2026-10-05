# Evaluation

Evaluation artifacts are data, not marketing copy. The verifier evaluator accepts JSONL rows containing `label` and `prediction`, writes accuracy, macro precision/recall/F1, and a fixed-order confusion matrix. Retrieval evaluation consumes relevant and retrieved chunk IDs and writes Recall@K and MRR. The latency runner measures an actual API, emitting p95 only with at least 20 observations. Quantization benchmarking loads both variants and records observed timing and CUDA allocation.

Commands:

```powershell
& .\services\ai-api\.venv\Scripts\python.exe ml/training/evaluate_verifier.py --predictions path/to/predictions.jsonl
& .\services\ai-api\.venv\Scripts\python.exe ml/evaluation/evaluate_retrieval.py path/to/retrieval-results.json
& .\services\ai-api\.venv\Scripts\python.exe ml/evaluation/benchmark_latency.py --runs 20
& .\services\ai-api\.venv\Scripts\python.exe ml/quantization/benchmark_quantization.py
```

Selected measured JSON summaries in `ml/benchmarks` are eligible for Git, including the 180-sample verifier comparison and six-case real end-to-end summary. Other generated JSON remains ignored pending review. The 90-sample baseline/fine-tuned artifacts are historical and must not be compared as if they were the 180-sample experiment. Before publishing a new result, review dataset licensing and document hardware, revisions, seed, split, and sample count.

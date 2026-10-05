# ML pipeline

The input contract is `Claim + Evidence`; the output labels are `SUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT_EVIDENCE`. Dataset preparation writes normalized JSONL outside Git. The default training configuration selects the verified public identifier `Qwen/Qwen2.5-1.5B-Instruct`, sized for practical LoRA/QLoRA experiments; actual feasibility still depends on sequence length, batch settings, CUDA/toolkit compatibility, and available VRAM.

Prepare a dependency-free smoke fixture:

`& .\services\ai-api\.venv\Scripts\python.exe ml/datasets/prepare_dataset.py --smoke`

Run a single training step after installing `requirements-gpu.txt`:

`& .\services\ai-api\.venv\Scripts\python.exe ml/training/train_qlora.py --smoke`

Validate the complete configuration and four-row smoke subset without downloading weights:

`& .\services\ai-api\.venv\Scripts\python.exe ml/training/train_qlora.py --smoke --dry-run`

The standard local workflow uses PowerShell and `services/ai-api/.venv/Scripts/python.exe`. Training outputs and checkpoints remain local. The completed adapter experiments are preserved as unsuccessful comparisons; the pretrained NLI baseline remains the production default. Any new training must use train/validation separation and evaluate the untouched held-out test set only after checkpoint selection.

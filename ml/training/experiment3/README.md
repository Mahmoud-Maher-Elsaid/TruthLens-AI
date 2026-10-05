# Experiment 3 — Multi-Evidence Verifier

## Purpose

This experiment tests whether standard BF16 PEFT LoRA can improve the boundary between `CONTRADICTED` and `INSUFFICIENT_EVIDENCE` when the verifier receives a claim plus multiple evidence passages.

It is distinct from the earlier pair-level experiments. Each training and validation example contains one source-audited decisive passage and two deterministically selected passages. Related-neutral and high-overlap contradiction examples are tagged from their audited source rows. The primary held-out protocol remains the frozen 180-row pair-level test, rendered as a single `EVIDENCE 1` block under the same frozen prompt.

## Data contract

`prepare.py` writes these local generated files under `ml/datasets/processed/experiment3_multi_evidence/`:

- `train.jsonl` — 1,200 rows, 400 per label.
- `validation.jsonl` — 180 rows, 60 per label.
- `conflict_challenge.jsonl` — 50 source-audited support/contradiction pairs for a secondary challenge only.

Each train/validation row preserves the decisive source label and provenance:

```text
CLAIM:
...

EVIDENCE 1:
...

EVIDENCE 2:
...

EVIDENCE 3:
...
```

Evidence ordering is deterministic by source ID. Distractors come from a different claim group and satisfy a fixed low-overlap rule. The conflict challenge has relationship labels only; it is excluded from training, validation, early stopping, checkpoint selection, and promotion.

The prepared manifest records the excluded duplicate source, deterministic replacement, source hashes, split counts, duplicate audit, and cross-split leakage audit. The held-out test is never rewritten or loaded by `train.py`.

## Training design

- Base model/tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`
- Strategy: standard PEFT LoRA, BF16, no QLoRA/NF4
- LoRA: rank 8, alpha 16, dropout 0.05; targets `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Base model: frozen; gradient checkpointing enabled
- Batch size: 1; gradient accumulation: 8
- Maximum sequence length: 1024
- Optimizer: `adamw_torch`; cosine schedule; 10% warmup; weight decay 0.01
- CUDA and BF16 are required. The runner refuses CPU execution.

`train.py` evaluates generation labels on the validation split at every evaluation cycle, saves accuracy, macro precision/recall/F1, per-class metrics, and confusion matrices to `validation_history.jsonl`, and selects the best checkpoint using validation macro F1. It never reads held-out test rows.

## Evaluation and promotion

`evaluate.py` accepts a completed adapter and verifies that the frozen 180-row test hash matches `manifest.json` before inference. It reports the full metrics set, mean/p50/p95 latency, peak VRAM, and adapter identity.

Promotion compares that result against the baseline macro F1 of `0.6758189801`. Training or validation metrics cannot promote a model.

## Completed result

Experiment 3 completed three epochs and 450 optimizer steps. `checkpoint-400` was selected using validation macro F1 only (`0.6718931026492623`). On the frozen 180-row held-out protocol (SHA-256 `390ca93b325f60a910c1c4a973901915574c570011690c3a7301a708458db94c`), it produced accuracy `0.6444444444` and macro F1 `0.6232936923`. This missed the frozen production baseline (accuracy `0.6944444444`, macro F1 `0.6758189801`) and did not meet promotion criteria. The baseline remains production; this adapter is retained as a negative experiment. No test-set tuning occurred.

## Commands

Run `train.py --validate-only` for a CPU-only structural check. Run smoke training before a full run. The smoke output is explicitly marked `smoke_completed_not_scientific` and cannot support a quality claim.

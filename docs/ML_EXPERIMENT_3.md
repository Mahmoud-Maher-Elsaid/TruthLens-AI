# Experiment 3 — completed negative result

Experiment 3 tested a distinct multi-evidence contradiction-boundary hypothesis. Training and frozen held-out evaluation completed successfully, but the result did not meet the pre-registered promotion criteria. The production verifier remains `cross-encoder/nli-deberta-v3-small`.

The defined real end-to-end suite separately achieved **6/6 expected-case matches on the defined real end-to-end validation suite** after pipeline fixes. That validates this bounded suite; it does not establish general verifier accuracy or a model-promotion result.

## Hypothesis

The hypothesis is to improve semantic separation between `CONTRADICTED` and `INSUFFICIENT_EVIDENCE` using claim-plus-multiple-evidence examples with explicit contradiction, numeric/date mismatch, related-neutral, and support/contradiction contrasts. The implementation is under `ml/training/experiment3/`: preparation creates three-passage train/validation rows, training uses a dedicated LoRA runner, and evaluation preserves the frozen 180-row held-out protocol.

This is different from Experiments 1 and 2 because the input unit is an evidence set with passage metadata, rather than one claim and one passage. It tests verifier behavior after retrieval, not retrieval quality itself.

## Data and safeguards

The natural source is the existing `pietrolesci/nli_fever` projection. Controlled contrast examples are allowed for training and validation only. The fixed held-out test remains `ml/datasets/processed/experiment2/test.jsonl`, with its current 180 balanced rows and source/group exclusion rules. A preparation run must record exact class counts, source revision, exact and near-duplicate checks, source/claim-group overlap, and SHA-256 hashes before any training starts. The test split is never used for checkpoint or hyperparameter decisions.

The tracked preparation manifest at `ml/training/experiment3/manifest.json` records 1,200 train rows, 180 validation rows, and 180 frozen test rows, balanced at 400/60/60 per class. It reports zero exact and near duplicates and zero source or claim-group overlap across splits. The generated multi-evidence train and validation rows remain local under `ml/datasets/processed/experiment3_multi_evidence/`.

The pair is claim group `160785`: source `160785:a463fee3-d463-4b10-a461-9f16d1c20cbb` is excluded, while `160785:7c095d99-5d1d-418d-8c40-59ef87d7d5f6` is retained. Both state the same Stratford-west-London claim and have the `CONTRADICTED` label; the longer excluded evidence adds only an unrelated retail/cultural sentence. The deterministic audited natural replacement is `130096:8b4afc54-21ec-4bc0-97a2-9ef07024e6c4`, from claim group `130096`.

## Training design

The configuration uses Qwen/Qwen2.5-1.5B-Instruct with standard BF16 PEFT LoRA. `bitsandbytes` is not installed in the verified Windows environment, so QLoRA/NF4 is not a requirement. The base model remains frozen; gradient checkpointing is enabled; batch size is 1 with gradient accumulation 8; maximum sequence length is 1024; learning rate is 5e-5; scheduler is cosine; warmup is 10%; weight decay is 0.01; and seed is 113. LoRA targets `q_proj`, `k_proj`, `v_proj`, and `o_proj` with rank 8, alpha 16, and dropout 0.05.

The trainer evaluates generated validation labels at each evaluation cycle, records accuracy, macro precision/recall/F1, per-class metrics, and confusion matrices, and selects checkpoints using validation macro F1. The held-out test is read only by the separate evaluator after training and cannot influence checkpoint selection or early stopping.

## Measured outcome and disposition

Training used 1,200 train rows and 180 validation rows for three epochs and 450 optimizer steps. `checkpoint-400` was selected by **validation macro F1 only** (`0.6718931026492623`), before the held-out test was evaluated. Peak training VRAM was `4,763,118,080` bytes and training wall time was about `1204.6` seconds.

The frozen 180-row held-out protocol has SHA-256 `390ca93b325f60a910c1c4a973901915574c570011690c3a7301a708458db94c`. Checkpoint 400's adapter SHA-256 is `2f59b54e3d26aed4ae5395b57b645d2089bf43dce033f32cd0d533e8b7937df5`. It produced accuracy `0.6444444444`, macro precision `0.6637115839`, macro recall `0.6444444444`, and macro F1 `0.6232936923`.

This is below the frozen production baseline (accuracy `0.6944444444`, macro F1 `0.6758189801`) and worsened `INSUFFICIENT_EVIDENCE` F1 to `0.4444444444`. It did not satisfy promotion criteria. **Keep the baseline; do not promote Experiment 3.** No held-out labels, thresholds, prompts, or datasets were changed in response to these results. The configuration, manifest, preparation code, trainer, evaluator, checkpoint identity, and held-out evaluation artifact are preserved as an honest negative experiment.

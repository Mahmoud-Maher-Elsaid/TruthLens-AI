from __future__ import annotations

from pathlib import Path

from .evaluate import evaluation_sources, resolve_adapter_base_model
from .format import user_content, validate_row
from .metrics import (
    callback_classification_metrics,
    classification_report,
    decode_label,
    deterministic_generation_config,
    generation_inputs,
)


def _row() -> dict[str, object]:
    return {
        "claim": "The archive opened in 2001.",
        "label": "SUPPORTED",
        "source_id": "source-1",
        "claim_group_id": "group-1",
        "evidence_set": ["The archive opened in 2001.", "A museum opened in 1990.", "The river is long."],
        "evidence_metadata": [{}, {}, {}],
    }


def test_three_passage_contract_is_explicit_and_ordered():
    row = _row()
    validate_row(row)
    text = user_content(row)
    assert text.index("EVIDENCE 1:") < text.index("EVIDENCE 2:") < text.index("EVIDENCE 3:")


def test_metrics_use_the_fixed_three_label_order():
    expected = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    report = classification_report(expected, expected)
    assert report["macro_f1"] == 1.0
    assert report["confusion_matrix_labels"] == expected
    assert decode_label("CONTRADICTED") == "CONTRADICTED"


class _Tensor:
    def to(self, device: object) -> "_Tensor":
        self.device = device
        return self


class _Tokenizer:
    pad_token_id = 7

    def apply_chat_template(self, *args: object, **kwargs: object) -> dict[str, _Tensor]:
        self.kwargs = kwargs
        return {"input_ids": _Tensor(), "attention_mask": _Tensor()}


class _GenerationConfig:
    do_sample = True
    temperature = 0.7
    top_p = 0.9
    top_k = 20
    max_new_tokens = 32
    pad_token_id = None


class _Model:
    generation_config = _GenerationConfig()


def test_callback_metric_contract_exposes_eval_macro_f1_before_callback_return():
    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    report = classification_report(labels, labels)
    metrics = callback_classification_metrics(report)
    assert metrics["eval_macro_f1"] == 1.0
    source = (Path(__file__).with_name("train.py")).read_text(encoding="utf-8")
    assert "def evaluation_loop(" in source
    assert source.index("loop_output.metrics.update(callback_classification_metrics") < source.index("return loop_output")


def test_generation_contract_passes_attention_mask_and_disables_sampling_flags():
    tokenizer = _Tokenizer()
    encoded = generation_inputs(tokenizer, _row(), 1024, "cuda")
    assert set(encoded) == {"input_ids", "attention_mask"}
    assert tokenizer.kwargs["return_dict"] is True
    config = deterministic_generation_config(_Model(), tokenizer)
    assert config.do_sample is False
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.pad_token_id == tokenizer.pad_token_id


def test_huggingface_early_stopping_receives_eval_macro_f1(tmp_path: Path):
    """Exercise Trainer's callback lifecycle without loading a language model."""
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    from transformers.trainer_utils import EvalLoopOutput

    class _TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids: object = None, labels: object = None) -> dict[str, object]:
            return {"loss": self.weight * 0}

    class _CallbackProbe(EarlyStoppingCallback):
        seen_metric: float | None = None

        def on_evaluate(self, args: object, state: object, control: object, metrics: dict[str, float], **kwargs: object) -> object:
            self.seen_metric = metrics.get("eval_macro_f1")
            return super().on_evaluate(args, state, control, metrics, **kwargs)

    class _LifecycleTrainer(Trainer):
        def evaluation_loop(self, dataloader: object, description: str, prediction_loss_only: object = None, ignore_keys: object = None, metric_key_prefix: str = "eval") -> EvalLoopOutput:
            report = classification_report(list(("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")), list(("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")))
            return EvalLoopOutput(predictions=None, label_ids=None, metrics=callback_classification_metrics(report, metric_key_prefix), num_samples=3)

    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        use_cpu=True,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1,
        save_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )
    probe = _CallbackProbe(early_stopping_patience=2)
    trainer = _LifecycleTrainer(model=_TinyModel(), args=arguments, eval_dataset=[{"input_ids": [1]}], callbacks=[probe])
    metrics = trainer.evaluate()
    assert metrics["eval_macro_f1"] == 1.0
    assert probe.seen_metric == 1.0


def test_adapter_metadata_resolves_base_model_and_tokenizer_source_without_model_load():
    root = Path(__file__).resolve().parents[3]
    adapter = root / "ml/training/outputs/experiment3-qwen2.5-1.5b-lora-bf16/checkpoint-400"
    expected = "Qwen/Qwen2.5-1.5B-Instruct"
    assert resolve_adapter_base_model(adapter) == expected
    assert evaluation_sources(adapter) == {
        "base_model_name_or_path": expected,
        "tokenizer_source": expected,
    }

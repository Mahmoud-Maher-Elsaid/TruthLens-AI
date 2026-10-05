"""Deterministic label decoding and metric reporting shared by Experiment 3."""

from __future__ import annotations

import copy
from typing import Any

try:
    from .format import LABELS, messages
except ImportError:  # Supports direct PowerShell execution of package scripts.
    from format import LABELS, messages


def decode_label(text: str) -> str:
    upper = text.upper()
    return next((label for label in LABELS if label in upper), "INSUFFICIENT_EVIDENCE")


def classification_report(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=list(LABELS), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=list(LABELS), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_metrics": {
            label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(LABELS)).tolist(),
    }


def callback_classification_metrics(report: dict[str, Any], metric_key_prefix: str = "eval") -> dict[str, float]:
    """Return the generated-label metrics under the names Trainer callbacks consume."""
    return {
        f"{metric_key_prefix}_{key}": float(report[key])
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }


def deterministic_generation_config(model: Any, tokenizer: Any) -> Any:
    """Copy and sanitize generation defaults for reproducible label classification."""
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.max_new_tokens = 8
    config.pad_token_id = tokenizer.pad_token_id
    return config


def generation_inputs(tokenizer: Any, row: dict[str, Any], max_length: int, device: Any) -> dict[str, Any]:
    """Tokenize a classification prompt and require an explicit attention mask."""
    encoded = tokenizer.apply_chat_template(
        messages(row),
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_dict=True,
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise RuntimeError("Tokenizer did not return input_ids and attention_mask for generation.")
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def generate_labels(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> list[str]:
    import torch

    device = next(model.parameters()).device
    predictions: list[str] = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            encoded = generation_inputs(tokenizer, row, max_length, device)
            output = model.generate(
                **encoded,
                generation_config=deterministic_generation_config(model, tokenizer),
            )
            predictions.append(decode_label(tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)))
    return predictions

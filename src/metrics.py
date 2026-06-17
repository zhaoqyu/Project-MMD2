from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .utils import slugify
from scipy.special import softmax as scipy_softmax

def softmax(logits: np.ndarray) -> np.ndarray:
    return scipy_softmax(np.asarray(logits), axis=-1)

# def softmax(logits: np.ndarray) -> np.ndarray:
#     logits = np.asarray(logits)
#     shifted = logits - np.max(logits, axis=-1, keepdims=True)
#     exp = np.exp(shifted)
#     return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_trainer_metrics(label_names: list[str]):
    def _compute(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return basic_metrics(labels, preds, label_names, include_per_class=True)

    return _compute


def compute_multitask_trainer_metrics(task1_labels: list[str], task2_labels: list[str]):
    def _compute(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred
        logits_task1, logits_task2 = predictions[:2]
        labels_task1, labels_task2 = labels
        pred_task1 = np.argmax(logits_task1, axis=-1)
        pred_task2 = np.argmax(logits_task2, axis=-1)
        metrics = {}
        for prefix, gold, pred, names in [
            ("task1", labels_task1, pred_task1, task1_labels),
            ("task2", labels_task2, pred_task2, task2_labels),
        ]:
            for key, value in basic_metrics(gold, pred, names, include_per_class=False).items():
                metrics[f"{prefix}_{key}"] = value
        return metrics

    return _compute


def basic_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    label_names: list[str],
    include_per_class: bool = True,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(label_names)))
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if include_per_class:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        for idx, name in enumerate(label_names):
            key = slugify(name)
            metrics[f"{key}_precision"] = float(precision[idx])
            metrics[f"{key}_recall"] = float(recall[idx])
            metrics[f"{key}_f1"] = float(f1[idx])
            metrics[f"{key}_support"] = float(support[idx])
    return metrics


def report_dict(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_dict(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": label_names, "matrix": matrix.tolist()}


def task1_ambivalent_nonreply_analysis(gold: list[str], pred: list[str]) -> dict[str, Any]:
    ambivalent_labels = {"Ambivalent", "Ambiguous", "Ambivalent Reply"}
    nonreply = "Clear Non-Reply"
    ambiv_to_nonreply = sum(g in ambivalent_labels and p == nonreply for g, p in zip(gold, pred))
    nonreply_to_ambiv = sum(g == nonreply and p in ambivalent_labels for g, p in zip(gold, pred))
    ambiv_gold = sum(g in ambivalent_labels for g in gold)
    nonreply_gold = sum(g == nonreply for g in gold)
    return {
        "ambivalent_gold_count": ambiv_gold,
        "clear_non_reply_gold_count": nonreply_gold,
        "ambivalent_predicted_clear_non_reply": ambiv_to_nonreply,
        "clear_non_reply_predicted_ambivalent": nonreply_to_ambiv,
        "ambivalent_to_clear_non_reply_rate": _safe_rate(ambiv_to_nonreply, ambiv_gold),
        "clear_non_reply_to_ambivalent_rate": _safe_rate(nonreply_to_ambiv, nonreply_gold),
    }


def task2_test_metrics(
    pred_labels: list[str],
    annotator_labels: list[list[str | None]],
    label_names: list[str],
) -> dict[str, Any]:
    majority_gold = []
    majority_pred = []
    unanimous_gold = []
    unanimous_pred = []
    unresolved = 0
    any_match_count = 0
    any_match_denominator = 0
    row_details = []

    label2id = {label: idx for idx, label in enumerate(label_names)}

    for idx, (pred, annotations) in enumerate(zip(pred_labels, annotator_labels)):
        clean = [_clean_label(label) for label in annotations]
        clean = [label for label in clean if label]
        majority = majority_vote(clean)
        unanimous = unanimous_vote(clean)
        has_any = bool(clean)

        if has_any:
            any_match_denominator += 1
            any_match_count += int(pred in clean)

        if majority is None:
            unresolved += 1
        elif majority in label2id and pred in label2id:
            majority_gold.append(label2id[majority])
            majority_pred.append(label2id[pred])

        if unanimous is not None and unanimous in label2id and pred in label2id:
            unanimous_gold.append(label2id[unanimous])
            unanimous_pred.append(label2id[pred])

        row_details.append(
            {
                "row_idx": idx,
                "prediction": pred,
                "annotations": clean,
                "majority_label": majority,
                "unanimous_label": unanimous,
                "any_annotator_match": pred in clean if clean else None,
                "unresolved_disagreement": majority is None,
            }
        )

    metrics: dict[str, Any] = {
        "majority_consensus_count": len(majority_gold),
        "unanimous_count": len(unanimous_gold),
        "unresolved_disagreement_count": unresolved,
        "any_annotator_match_rate": _safe_rate(any_match_count, any_match_denominator),
        "any_annotator_match_count": any_match_count,
        "any_annotator_match_denominator": any_match_denominator,
        "row_details": row_details,
    }
    if majority_gold:
        metrics["majority_macro_f1"] = float(
            f1_score(majority_gold, majority_pred, labels=list(range(len(label_names))), average="macro", zero_division=0)
        )
        metrics["majority_weighted_f1"] = float(
            f1_score(majority_gold, majority_pred, labels=list(range(len(label_names))), average="weighted", zero_division=0)
        )
        metrics["majority_accuracy"] = float(accuracy_score(majority_gold, majority_pred))
        metrics["majority_classification_report"] = report_dict(majority_gold, majority_pred, label_names)
        metrics["majority_confusion_matrix"] = confusion_matrix_dict(majority_gold, majority_pred, label_names)
    else:
        metrics["majority_macro_f1"] = None

    if unanimous_gold:
        metrics["unanimous_macro_f1"] = float(
            f1_score(unanimous_gold, unanimous_pred, labels=list(range(len(label_names))), average="macro", zero_division=0)
        )
        metrics["unanimous_weighted_f1"] = float(
            f1_score(unanimous_gold, unanimous_pred, labels=list(range(len(label_names))), average="weighted", zero_division=0)
        )
        metrics["unanimous_accuracy"] = float(accuracy_score(unanimous_gold, unanimous_pred))
        metrics["unanimous_classification_report"] = report_dict(unanimous_gold, unanimous_pred, label_names)
        metrics["unanimous_confusion_matrix"] = confusion_matrix_dict(unanimous_gold, unanimous_pred, label_names)
    else:
        metrics["unanimous_macro_f1"] = None
    return metrics


def majority_vote(labels: list[str]) -> str | None:
    counts = Counter(labels)
    for label, count in counts.most_common():
        if count >= 2:
            return label
    return None


def unanimous_vote(labels: list[str]) -> str | None:
    if len(labels) == 3 and len(set(labels)) == 1:
        return labels[0]
    return None


def _clean_label(label: Any) -> str | None:
    if label is None:
        return None
    label = str(label).strip()
    return label or None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


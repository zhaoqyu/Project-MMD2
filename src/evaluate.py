from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import normalize_label
from .metrics import (
    basic_metrics,
    confusion_matrix_dict,
    report_dict,
    softmax,
    task1_ambivalent_nonreply_analysis,
    task2_test_metrics,
)
from .utils import ensure_dir, save_json, slugify


def save_supervised_artifacts(
    out_dir: str | Path,
    split_name: str,
    raw_dataset,
    logits: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    data_cfg: DataConfig,
    task: str,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits)
    gold_labels = [label_names[int(i)] for i in label_ids]
    pred_labels = [label_names[int(i)] for i in predictions]

    df = prediction_frame(raw_dataset, logits, probs, pred_labels, data_cfg, label_names)
    df["gold_label"] = gold_labels
    df["correct"] = df["gold_label"] == df["predicted_label"]
    df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    np.save(out_dir / f"{split_name}_logits.npy", logits)
    np.save(out_dir / f"{split_name}_probabilities.npy", probs)
    np.save(out_dir / f"{split_name}_gold_labels.npy", label_ids)

    metrics = basic_metrics(label_ids, predictions, label_names, include_per_class=True)
    metrics["classification_report"] = report_dict(label_ids.tolist(), predictions.tolist(), label_names)
    metrics["confusion_matrix"] = confusion_matrix_dict(label_ids.tolist(), predictions.tolist(), label_names)
    if task == "task1":
        metrics["ambivalent_vs_clear_non_reply"] = task1_ambivalent_nonreply_analysis(gold_labels, pred_labels)
    save_json(metrics, out_dir / f"{split_name}_metrics.json")
    save_json(metrics["classification_report"], out_dir / f"{split_name}_classification_report.json")
    save_json(metrics["confusion_matrix"], out_dir / f"{split_name}_confusion_matrix.json")
    df.loc[~df["correct"]].to_csv(out_dir / f"{split_name}_misclassified_examples.csv", index=False)
    return metrics


def save_task2_test_artifacts(
    out_dir: str | Path,
    split_name: str,
    raw_dataset,
    logits: np.ndarray,
    label_names: list[str],
    data_cfg: DataConfig,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    probs = softmax(logits)
    predictions = np.argmax(logits, axis=-1)
    pred_labels = [label_names[int(i)] for i in predictions]
    annotations = [
        [row.get(column) for column in data_cfg.annotator_columns]
        for row in raw_dataset
    ]
    metrics = task2_test_metrics(pred_labels, annotations, label_names)

    df = prediction_frame(raw_dataset, logits, probs, pred_labels, data_cfg, label_names)
    details = pd.DataFrame(metrics.pop("row_details"))
    df = pd.concat([df.reset_index(drop=True), details.drop(columns=["row_idx"])], axis=1)
    df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    df.loc[df["unresolved_disagreement"] == True].to_csv(  # noqa: E712
        out_dir / f"{split_name}_unresolved_disagreements.csv",
        index=False,
    )
    if "majority_label" in df.columns:
        majority_errors = df[
            (df["majority_label"].notna()) & (df["majority_label"] != df["predicted_label"])
        ]
        majority_errors.to_csv(out_dir / f"{split_name}_misclassified_examples.csv", index=False)

    np.save(out_dir / f"{split_name}_logits.npy", logits)
    np.save(out_dir / f"{split_name}_probabilities.npy", probs)
    save_json(metrics, out_dir / f"{split_name}_metrics.json")
    if metrics.get("majority_classification_report"):
        save_json(metrics["majority_classification_report"], out_dir / f"{split_name}_classification_report.json")
    if metrics.get("majority_confusion_matrix"):
        save_json(metrics["majority_confusion_matrix"], out_dir / f"{split_name}_confusion_matrix.json")
    return metrics


def prediction_frame(
    raw_dataset,
    logits,
    probs,
    pred_labels: list[str],
    data_cfg: DataConfig,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(raw_dataset):
        item = {
            "row_idx": idx,
            "example_id": row.get(data_cfg.id_column, idx),
            "question": row.get(data_cfg.question_column),
            "answer": row.get(data_cfg.answer_column),
            "predicted_label": pred_labels[idx],
        }
        for column in ["title", "date", "president", "url", "question_order"]:
            if column in row:
                item[column] = row.get(column)
        for column in data_cfg.annotator_columns:
            if column in row:
                item[column] = row.get(column)
        for label_idx in range(logits.shape[1]):
            suffix = slugify(label_names[label_idx]) if label_names else slugify(str(label_idx))
            item[f"logit_{suffix}"] = float(logits[idx, label_idx])
            item[f"prob_{suffix}"] = float(probs[idx, label_idx])
        rows.append(item)
    return pd.DataFrame(rows)


def labels_from_raw(raw_dataset, data_cfg: DataConfig, task: str, label2id: dict[str, int]) -> np.ndarray:
    label_column = data_cfg.task1_label_column if task == "task1" else data_cfg.task2_label_column
    labels = []
    for row in raw_dataset:
        label = normalize_label(row.get(label_column), task)
        labels.append(label2id[label])
    return np.asarray(labels, dtype=np.int64)

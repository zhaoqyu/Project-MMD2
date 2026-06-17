#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DataConfig
from src.metrics import basic_metrics, confusion_matrix_dict, report_dict, task2_test_metrics
from src.utils import ensure_dir, load_json, save_json


REPLY_STRATEGIES = {"Explicit", "Implicit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple hierarchical Task 1 -> Task 2 evaluation.")
    parser.add_argument("--task1-run-dir", required=True)
    parser.add_argument("--task2-run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task1_run = Path(args.task1_run_dir)
    task2_run = Path(args.task2_run_dir)
    out_dir = ensure_dir(args.output_dir)

    task1_df = pd.read_csv(task1_run / "artifacts" / f"{args.split}_predictions.csv")
    task2_df = pd.read_csv(task2_run / "artifacts" / f"{args.split}_predictions.csv")
    task2_probs = np.load(task2_run / "artifacts" / f"{args.split}_probabilities.npy")
    task2_labels = load_json(task2_run / "label_mapping.json")["label_names"]
    reply_indices = [idx for idx, label in enumerate(task2_labels) if label in REPLY_STRATEGIES]

    direct_pred = [task2_labels[int(idx)] for idx in np.argmax(task2_probs, axis=-1)]
    hierarchical_pred = []
    for row_idx, clarity_pred in enumerate(task1_df["predicted_label"].tolist()):
        if clarity_pred == "Clear Reply" and reply_indices:
            local = reply_indices[int(np.argmax(task2_probs[row_idx, reply_indices]))]
            hierarchical_pred.append(task2_labels[local])
        else:
            hierarchical_pred.append(direct_pred[row_idx])

    result = task2_df.copy()
    result["direct_task2_prediction"] = direct_pred
    result["hierarchical_task2_prediction"] = hierarchical_pred
    result["task1_prediction"] = task1_df["predicted_label"].tolist()
    result.to_csv(out_dir / f"{args.split}_hierarchical_predictions.csv", index=False)

    if args.split == "dev" and "gold_label" in result.columns:
        label2id = {label: idx for idx, label in enumerate(task2_labels)}
        y_true = np.asarray([label2id[label] for label in result["gold_label"].tolist()])
        y_pred = np.asarray([label2id[label] for label in hierarchical_pred])
        metrics = basic_metrics(y_true, y_pred, task2_labels, include_per_class=True)
        metrics["classification_report"] = report_dict(y_true.tolist(), y_pred.tolist(), task2_labels)
        metrics["confusion_matrix"] = confusion_matrix_dict(y_true.tolist(), y_pred.tolist(), task2_labels)
    else:
        data_cfg = DataConfig()
        annotations = result[data_cfg.annotator_columns].values.tolist()
        metrics = task2_test_metrics(hierarchical_pred, annotations, task2_labels)

    save_json(metrics, out_dir / f"{args.split}_hierarchical_metrics.json")
    print(f"Wrote hierarchical evaluation to {out_dir}")


if __name__ == "__main__":
    main()


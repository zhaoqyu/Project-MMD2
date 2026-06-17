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
from src.metrics import basic_metrics, confusion_matrix_dict, report_dict, task1_ambivalent_nonreply_analysis, task2_test_metrics
from src.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average probabilities from multiple seed runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to ensemble.")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--task", required=True, choices=["task1", "task2"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    probs = []
    for run_dir in run_dirs:
        prob_path = run_dir / "artifacts" / f"{args.split}_probabilities.npy"
        if not prob_path.exists():
            raise FileNotFoundError(prob_path)
        probs.append(np.load(prob_path))
    mean_probs = np.mean(np.stack(probs, axis=0), axis=0)

    label_info = load_json(run_dirs[0] / "label_mapping.json")
    label_names = label_info["label_names"]
    pred_ids = np.argmax(mean_probs, axis=-1)
    pred_labels = [label_names[int(i)] for i in pred_ids]

    base_predictions = pd.read_csv(run_dirs[0] / "artifacts" / f"{args.split}_predictions.csv")
    base_predictions["ensemble_predicted_label"] = pred_labels
    for idx in range(mean_probs.shape[1]):
        base_predictions[f"ensemble_prob_{idx}"] = mean_probs[:, idx]

    out_dir = ensure_dir(args.output_dir)
    base_predictions.to_csv(out_dir / f"{args.split}_ensemble_predictions.csv", index=False)
    np.save(out_dir / f"{args.split}_ensemble_probabilities.npy", mean_probs)

    if args.task == "task1" or (args.split == "dev" and "gold_label" in base_predictions.columns):
        gold_labels = base_predictions["gold_label"].tolist()
        label2id = {label: idx for idx, label in enumerate(label_names)}
        y_true = np.asarray([label2id[label] for label in gold_labels])
        metrics = basic_metrics(y_true, pred_ids, label_names, include_per_class=True)
        metrics["classification_report"] = report_dict(y_true.tolist(), pred_ids.tolist(), label_names)
        metrics["confusion_matrix"] = confusion_matrix_dict(y_true.tolist(), pred_ids.tolist(), label_names)
        if args.task == "task1":
            metrics["ambivalent_vs_clear_non_reply"] = task1_ambivalent_nonreply_analysis(gold_labels, pred_labels)
    else:
        data_cfg = DataConfig()
        annotations = base_predictions[data_cfg.annotator_columns].values.tolist()
        metrics = task2_test_metrics(pred_labels, annotations, label_names)
    save_json(metrics, out_dir / f"{args.split}_ensemble_metrics.json")
    print(f"Wrote ensemble results to {out_dir}")


if __name__ == "__main__":
    main()


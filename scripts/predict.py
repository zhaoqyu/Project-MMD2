#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_label_mapping, load_qevasion_dataset, validate_expected_columns
from src.evaluate import prediction_frame
from src.metrics import softmax
from src.preprocess import tokenize_dataset
from src.utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a saved single-task QEvasion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model or a checkpoint directory.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.task == "multitask":
        raise ValueError("scripts/predict.py currently supports single-task checkpoints.")

    raw = load_qevasion_dataset(cfg.data)
    validate_expected_columns(raw, cfg.data)
    label_names, _, _ = build_label_mapping(raw["train"], cfg.task, cfg.data)
    dataset = raw[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    tokenized = tokenize_dataset(dataset, tokenizer, cfg.data, remove_columns=dataset.column_names)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    pred = trainer.predict(tokenized)
    logits = np.asarray(pred.predictions)
    probs = softmax(logits)
    pred_labels = [label_names[int(i)] for i in np.argmax(logits, axis=-1)]
    df = prediction_frame(dataset, logits, probs, pred_labels, cfg.data, label_names)
    output = Path(args.output)
    ensure_dir(output.parent)
    df.to_csv(output, index=False)
    np.save(output.with_suffix(".logits.npy"), logits)
    np.save(output.with_suffix(".probabilities.npy"), probs)
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()

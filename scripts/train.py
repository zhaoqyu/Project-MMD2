#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, save_config
from src.data import (
    add_multitask_labels,
    add_single_task_labels,
    build_label_mapping,
    class_weights_from_labels,
    get_or_create_split,
    load_qevasion_dataset,
    select_split,
    task_label_column,
    validate_expected_columns,
)
from src.evaluate import (
    labels_from_raw,
    save_supervised_artifacts,
    save_task2_test_artifacts,
)
from src.metrics import compute_multitask_trainer_metrics, compute_trainer_metrics
from src.models import build_multitask_classifier, build_sequence_classifier
from src.preprocess import tokenize_dataset
from src.trainer_utils import (
    MultiTaskTrainer,
    WeightedLossTrainer,
    build_training_arguments,
    early_stopping_callbacks,
    focal_gamma_from_config,
)
from src.utils import ensure_dir, now_timestamp, save_json, set_global_seed, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QEvasion models.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seeds with one seed.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and prepare run dirs without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else list(cfg.seeds)
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seeds = [seed]
        run_one_seed(run_cfg, config_path=Path(args.config), dry_run=args.dry_run)


def run_one_seed(cfg, config_path: Path, dry_run: bool = False) -> Path:
    set_global_seed(cfg.seeds[0])
    run_dir = make_run_dir(cfg)
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    save_config(cfg, run_dir / "config.yaml")
    save_json({"source_config": str(config_path)}, run_dir / "run_metadata.json")

    raw = load_qevasion_dataset(cfg.data)
    schema = validate_expected_columns(raw, cfg.data)
    save_json(schema, run_dir / "dataset_schema.json")

    split = get_or_create_split(
        raw["train"],
        cfg.data,
        cfg.task,
        cfg.seeds[0],
        cfg.output_dir,
    )
    save_json(split, run_dir / "split.json")
    raw_train, raw_eval = select_split(raw["train"], split)
    raw_test = raw["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        use_fast=cfg.model.tokenizer_use_fast,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")

    if dry_run:
        print(f"Dry run prepared {run_dir}")
        return run_dir

    if cfg.task == "multitask":
        summary = run_multitask(cfg, run_dir, artifacts_dir, tokenizer, raw_train, raw_eval, raw_test)
    else:
        summary = run_single_task(cfg, run_dir, artifacts_dir, tokenizer, raw_train, raw_eval, raw_test)

    save_json(summary, run_dir / "results_summary.json")
    print(f"Finished run: {run_dir}")
    return run_dir


def run_single_task(cfg, run_dir: Path, artifacts_dir: Path, tokenizer, raw_train, raw_eval, raw_test) -> dict:
    label_names, label2id, id2label = build_label_mapping(raw_train, cfg.task, cfg.data)
    save_json({"label_names": label_names, "label2id": label2id, "id2label": id2label}, run_dir / "label_mapping.json")

    label_column = task_label_column(cfg.task, cfg.data)
    train_labeled = add_single_task_labels(raw_train, label_column, label2id, cfg.task)
    eval_labeled = add_single_task_labels(raw_eval, label_column, label2id, cfg.task)

    train_dataset = tokenize_keep_labels(train_labeled, tokenizer, cfg.data, ["labels"])
    eval_dataset = tokenize_keep_labels(eval_labeled, tokenizer, cfg.data, ["labels"])
    test_dataset = make_single_task_test_dataset(cfg, raw_test, tokenizer, label2id)

    model = build_sequence_classifier(
        cfg.model.model_name,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=cfg.model.trust_remote_code,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )

    train_label_ids = [int(label) for label in train_labeled["labels"]]
    class_weights = None
    if cfg.loss.use_class_weights or cfg.loss.name == "weighted_cross_entropy":
        class_weights = class_weights_from_labels(train_label_ids, len(label_names))
        save_json({"class_weights": class_weights, "label_names": label_names}, run_dir / "class_weights.json")

    trainer = WeightedLossTrainer(
        model=model,
        args=build_training_arguments(cfg, run_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_trainer_metrics(label_names),
        callbacks=early_stopping_callbacks(cfg),
        class_weights=class_weights,
        focal_gamma=focal_gamma_from_config(cfg),
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(run_dir / "best_model")
    tokenizer.save_pretrained(run_dir / "best_model")
    save_json(trainer.state.log_history, artifacts_dir / "trainer_log_history.json")

    eval_predictions = trainer.predict(eval_dataset)
    eval_metrics = save_supervised_artifacts(
        artifacts_dir,
        "dev",
        raw_eval,
        np.asarray(eval_predictions.predictions),
        np.asarray(eval_predictions.label_ids),
        label_names,
        cfg.data,
        cfg.task,
    )

    test_predictions = trainer.predict(test_dataset)
    if cfg.task == "task1":
        test_label_ids = labels_from_raw(raw_test, cfg.data, "task1", label2id)
        test_metrics = save_supervised_artifacts(
            artifacts_dir,
            "test",
            raw_test,
            np.asarray(test_predictions.predictions),
            test_label_ids,
            label_names,
            cfg.data,
            cfg.task,
        )
    else:
        test_metrics = save_task2_test_artifacts(
            artifacts_dir,
            "test",
            raw_test,
            np.asarray(test_predictions.predictions),
            label_names,
            cfg.data,
        )

    return {
        "run_dir": str(run_dir),
        "experiment_name": cfg.experiment_name,
        "task": cfg.task,
        "model_name": cfg.model.model_name,
        "seed": cfg.seeds[0],
        "dev_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }


def run_multitask(cfg, run_dir: Path, artifacts_dir: Path, tokenizer, raw_train, raw_eval, raw_test) -> dict:
    task1_labels, task1_label2id, task1_id2label = build_label_mapping(raw_train, "task1", cfg.data)
    task2_labels, task2_label2id, task2_id2label = build_label_mapping(raw_train, "task2", cfg.data)
    save_json(
        {
            "task1": {"label_names": task1_labels, "label2id": task1_label2id, "id2label": task1_id2label},
            "task2": {"label_names": task2_labels, "label2id": task2_label2id, "id2label": task2_id2label},
        },
        run_dir / "label_mapping.json",
    )

    train_labeled = add_multitask_labels(raw_train, task1_label2id, task2_label2id, cfg.data)
    eval_labeled = add_multitask_labels(raw_eval, task1_label2id, task2_label2id, cfg.data)
    train_dataset = tokenize_keep_labels(train_labeled, tokenizer, cfg.data, ["labels_task1", "labels_task2"])
    eval_dataset = tokenize_keep_labels(eval_labeled, tokenizer, cfg.data, ["labels_task1", "labels_task2"])
    test_dataset = tokenize_dataset(raw_test, tokenizer, cfg.data, remove_columns=raw_test.column_names)

    task1_weights = task2_weights = None
    if cfg.loss.use_class_weights or cfg.loss.name == "weighted_cross_entropy":
        task1_weights = class_weights_from_labels([int(x) for x in train_labeled["labels_task1"]], len(task1_labels))
        task2_weights = class_weights_from_labels([int(x) for x in train_labeled["labels_task2"]], len(task2_labels))
        save_json(
            {"task1_class_weights": task1_weights, "task2_class_weights": task2_weights},
            run_dir / "class_weights.json",
        )

    model = build_multitask_classifier(
        cfg.model.model_name,
        num_task1_labels=len(task1_labels),
        num_task2_labels=len(task2_labels),
        lambda_task2=cfg.multitask.lambda_task2,
        trust_remote_code=cfg.model.trust_remote_code,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )
    trainer = MultiTaskTrainer(
        model=model,
        args=build_training_arguments(cfg, run_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_multitask_trainer_metrics(task1_labels, task2_labels),
        callbacks=early_stopping_callbacks(cfg),
        lambda_task2=cfg.multitask.lambda_task2,
        task1_class_weights=task1_weights,
        task2_class_weights=task2_weights,
        focal_gamma=focal_gamma_from_config(cfg),
    )
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(run_dir / "best_model")
    tokenizer.save_pretrained(run_dir / "best_model")
    save_json(trainer.state.log_history, artifacts_dir / "trainer_log_history.json")

    dev_pred = trainer.predict(eval_dataset)
    dev_task1_logits, dev_task2_logits = dev_pred.predictions[:2]
    dev_task1_metrics = save_supervised_artifacts(
        artifacts_dir / "task1",
        "dev",
        raw_eval,
        np.asarray(dev_task1_logits),
        np.asarray(dev_pred.label_ids[0]),
        task1_labels,
        cfg.data,
        "task1",
    )
    dev_task2_metrics = save_supervised_artifacts(
        artifacts_dir / "task2",
        "dev",
        raw_eval,
        np.asarray(dev_task2_logits),
        np.asarray(dev_pred.label_ids[1]),
        task2_labels,
        cfg.data,
        "task2",
    )

    test_pred = trainer.predict(test_dataset)
    test_task1_logits, test_task2_logits = test_pred.predictions[:2]
    test_task1_metrics = save_supervised_artifacts(
        artifacts_dir / "task1",
        "test",
        raw_test,
        np.asarray(test_task1_logits),
        labels_from_raw(raw_test, cfg.data, "task1", task1_label2id),
        task1_labels,
        cfg.data,
        "task1",
    )
    test_task2_metrics = save_task2_test_artifacts(
        artifacts_dir / "task2",
        "test",
        raw_test,
        np.asarray(test_task2_logits),
        task2_labels,
        cfg.data,
    )
    return {
        "run_dir": str(run_dir),
        "experiment_name": cfg.experiment_name,
        "task": "multitask",
        "model_name": cfg.model.model_name,
        "seed": cfg.seeds[0],
        "dev_metrics": {"task1": dev_task1_metrics, "task2": dev_task2_metrics},
        "test_metrics": {"task1": test_task1_metrics, "task2": test_task2_metrics},
    }


def tokenize_keep_labels(dataset, tokenizer, data_cfg, label_columns: list[str]):
    remove_columns = [column for column in dataset.column_names if column not in label_columns]
    return tokenize_dataset(dataset, tokenizer, data_cfg, remove_columns=remove_columns)


def make_single_task_test_dataset(cfg, raw_test, tokenizer, label2id):
    if cfg.task == "task1":
        test_labeled = add_single_task_labels(raw_test, cfg.data.task1_label_column, label2id, "task1")
        return tokenize_keep_labels(test_labeled, tokenizer, cfg.data, ["labels"])
    return tokenize_dataset(raw_test, tokenizer, cfg.data, remove_columns=raw_test.column_names)


def make_run_dir(cfg) -> Path:
    model_slug = slugify(cfg.model.model_name.replace("/", "_"))
    run_name = f"{cfg.experiment_name}_{model_slug}_{cfg.task}_seed{cfg.seeds[0]}_{now_timestamp()}"
    return ensure_dir(Path(cfg.output_dir) / "runs" / cfg.task / run_name)


if __name__ == "__main__":
    main()

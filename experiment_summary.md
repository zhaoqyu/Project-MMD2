# CLARITY / QEvasion Experiment Summary

This file summarizes the experiments conducted so far in the CLARITY SemEval 2026 project repository.

## 1. Experiment Overview

The project has conducted **15 saved training experiments**:

- 7 single-task Task 1 runs
- 7 single-task Task 2 runs
- 1 multi-task run that produces both Task 1 and Task 2 outputs

All saved experiments use:

- Dataset: `ailsntua/QEvasion`
- Official train split split internally into:
  - train: `2758`
  - dev: `690`
- Official test split:
  - Task 1: `308` examples
  - Task 2: `308` examples, but majority-consensus evaluation uses `275` examples because `33` have no majority agreement
- Seeds: only seed `42`
- Folds: no cross-validation
- Main metric: macro-F1

The experiments were designed in stages:

1. **BERT-base baseline**
   - Fast, cheap, standard baseline.
2. **DeBERTa-v3-base**
   - Main strong encoder baseline.
3. **Class-weighted DeBERTa-v3-base**
   - Tests whether label imbalance handling helps.
4. **Alternative input wording**
   - Tests whether `Politician's answer:` helps compared with simple `Answer:`.
5. **RoBERTa-large and DeBERTa-v3-large**
   - Stronger model comparison.
6. **DeBERTa-xlarge**
   - Larger model, memory-safe settings.
7. **Multi-task DeBERTa-v3-base**
   - Tests whether Task 1 and Task 2 help each other.

No saved ensemble, focal-loss, or hierarchical-result directories were found.

## 2. Experiment Table

| Experiment | Task | Model | Seeds/Folds | Important Settings | Metric | Result | Output Path | Conclusion |
|---|---|---:|---|---|---|---:|---|---|
| BERT baseline | Task 1 | `bert-base-uncased` | seed 42 / no folds | max_len 256, bs 16, lr 2e-5, 5 epochs | test macro-F1 | 0.570 | `outputs/runs/task1/bert_base...` | Useful baseline, later improved. |
| BERT baseline | Task 2 | `bert-base-uncased` | seed 42 / no folds | max_len 256, bs 16, lr 2e-5, 5 epochs | majority test macro-F1 | 0.239 | `outputs/runs/task2/bert_base...` | Weak baseline for fine-grained labels. |
| DeBERTa-v3-base | Task 1 | `microsoft/deberta-v3-base` | seed 42 / no folds | max_len 256, bs 8, accum 2, lr 1.5e-5, fp16 | test macro-F1 | 0.605 | `outputs/runs/task1/deberta_v3_base...` | Strong improvement over BERT. |
| DeBERTa-v3-base | Task 2 | `microsoft/deberta-v3-base` | seed 42 / no folds | max_len 256, bs 8, accum 2, lr 1.5e-5, fp16 | majority test macro-F1 | 0.301 | `outputs/runs/task2/deberta_v3_base...` | Better than BERT, but still difficult. |
| Weighted DeBERTa-v3-base | Task 1 | `microsoft/deberta-v3-base` | seed 42 / no folds | weighted CE | test macro-F1 | 0.546 | `outputs/runs/task1/deberta_v3_base_task1_weighted...` | Class weighting hurt Task 1. |
| Weighted DeBERTa-v3-base | Task 2 | `microsoft/deberta-v3-base` | seed 42 / no folds | weighted CE | majority test macro-F1 | 0.318 | `outputs/runs/task2/deberta_v3_base_task2_weighted...` | Helps Task 2 macro-F1, hurts accuracy. |
| Alt input wording | Task 1 | `microsoft/deberta-v3-base` | seed 42 / no folds | `Politician's answer:` input | test macro-F1 | 0.588 | `outputs/runs/task1/deberta_v3_base_task1_politician_answer...` | Did not beat simple `Answer:`. |
| Alt input wording | Task 2 | `microsoft/deberta-v3-base` | seed 42 / no folds | `Politician's answer:` input | majority test macro-F1 | 0.238 | `outputs/runs/task2/deberta_v3_base_task2_politician_answer...` | Clearly worse. |
| RoBERTa-large | Task 1 | `roberta-large` | seed 42 / no folds | bs 2, accum 8, fp16, gradient checkpointing | test macro-F1 | 0.596 | `outputs/runs/task1/roberta_large...` | Competitive but not best Task 1. |
| RoBERTa-large | Task 2 | `roberta-large` | seed 42 / no folds | bs 2, accum 8, fp16, gradient checkpointing | majority test macro-F1 | 0.342 | `outputs/runs/task2/roberta_large...` | Best Task 2 system. |
| DeBERTa-v3-large | Task 1 | `microsoft/deberta-v3-large` | seed 42 / no folds | bs 2, accum 8, fp16 | test macro-F1 | 0.595 | `outputs/runs/task1/deberta_v3_large...` | Best dev score, not best test score. |
| DeBERTa-v3-large | Task 2 | `microsoft/deberta-v3-large` | seed 42 / no folds | bs 2, accum 8, fp16 | majority test macro-F1 | 0.325 | `outputs/runs/task2/deberta_v3_large...` | Second-best Task 2. |
| DeBERTa-xlarge | Task 1 | `microsoft/deberta-xlarge` | seed 42 / no folds | max_len 192, bs 1, accum 16, grad checkpointing | test macro-F1 | 0.632 | `outputs/runs/task1/deberta_xlarge...` | Best Task 1 system. |
| DeBERTa-xlarge | Task 2 | `microsoft/deberta-xlarge` | seed 42 / no folds | max_len 192, bs 1, accum 16, grad checkpointing | majority test macro-F1 | 0.267 | `outputs/runs/task2/deberta_xlarge...` | Larger model did not help Task 2. |
| Multi-task DeBERTa-v3-base | Both | `microsoft/deberta-v3-base` | seed 42 / no folds | shared encoder, two heads, lambda_task2=1.0 | Task 1 / Task 2 test macro-F1 | 0.586 / 0.262 | `outputs/runs/multitask/...` | Did not improve over single-task. |

## 3. Baseline Experiments

The baseline is BERT-base:

```bash
python scripts/train.py --config configs/bert_base_task1.yaml
python scripts/train.py --config configs/bert_base_task2.yaml
```

Why it counts as baseline:

- It is a standard small transformer encoder.
- It is fast and cheap compared with RoBERTa-large or DeBERTa-large.
- It gives a reference point before stronger models.

Results:

- Task 1 BERT-base test macro-F1: `0.570`
- Task 2 BERT-base majority test macro-F1: `0.239`

Was it improved later?

Yes.

- Task 1 improved from `0.570` to `0.632` with DeBERTa-xlarge.
- Task 2 improved from `0.239` to `0.342` with RoBERTa-large.

## 4. Model Comparison Experiments

Models tested:

- `bert-base-uncased`
- `microsoft/deberta-v3-base`
- `roberta-large`
- `microsoft/deberta-v3-large`
- `microsoft/deberta-xlarge`

Why each was tried:

- **BERT-base**: quick baseline.
- **DeBERTa-v3-base**: strong modern baseline with manageable GPU cost.
- **RoBERTa-large**: strong large encoder, useful comparison against DeBERTa.
- **DeBERTa-v3-large**: stronger DeBERTa experiment.
- **DeBERTa-xlarge**: largest experiment, mainly to see whether model scale helps.

What worked best:

- Task 1: `microsoft/deberta-xlarge`, test macro-F1 `0.632`.
- Task 2: `roberta-large`, majority-consensus test macro-F1 `0.342`.

No saved evidence shows that any of these final saved model runs failed. The configs suggest memory-aware settings were used for large models:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_length: 192
gradient_checkpointing: true
```

That is in the DeBERTa-xlarge configs.

## 5. Seed / Fold / Ensemble Experiments

What I found:

- Multiple seeds: code supports them, but saved runs use only seed `42`.
- Cross-validation: not implemented.
- Out-of-fold predictions: not found.
- Ensemble code: implemented in `scripts/ensemble.py`.
- Saved ensemble results: not found.
- Ensemble method: probability averaging.

The ensemble script does this:

```python
mean_probs = np.mean(np.stack(probs, axis=0), axis=0)
```

So if you later train 3 seeds, the ensemble will average probabilities, not majority vote.

Conclusion: ensembling is prepared but not yet conducted.

## 6. Task 1 Experiments

Task 1 models trained:

| Model | Test macro-F1 |
|---|---:|
| DeBERTa-xlarge | 0.632 |
| DeBERTa-v3-base | 0.605 |
| RoBERTa-large | 0.596 |
| DeBERTa-v3-large | 0.595 |
| DeBERTa-v3-base + alt input | 0.588 |
| DeBERTa-v3-base multi-task | 0.586 |
| BERT-base | 0.570 |
| DeBERTa-v3-base + weighted | 0.546 |

Best Task 1 experiment:

```bash
python scripts/train.py --config configs/deberta_xlarge_task1.yaml
```

Why it worked:

- Larger model capacity helped the coarse clarity classification task.
- Even though `max_length` was reduced to 192, it still achieved the best test macro-F1.

Error patterns from `paper_assets/tables/top_confusions_task1.csv`:

- `Ambivalent -> Clear Reply`: 47 examples
- `Clear Reply -> Ambivalent`: 26 examples
- `Clear Non-Reply -> Ambivalent`: 10 examples

Interpretation:

Task 1 mainly struggles with the boundary between a clear answer and a partially/ambiguously responsive answer.

What still needs testing:

- Multiple seeds for DeBERTa-xlarge.
- Whether max_length 256 helps DeBERTa-xlarge if memory allows.
- Calibration or threshold tuning.
- Ensemble of best Task 1 models.

## 7. Task 2 Experiments

Task 2 models trained:

| Model | Majority test macro-F1 |
|---|---:|
| RoBERTa-large | 0.342 |
| DeBERTa-v3-large | 0.325 |
| DeBERTa-v3-base + weighted | 0.318 |
| DeBERTa-v3-base | 0.301 |
| DeBERTa-xlarge | 0.267 |
| DeBERTa-v3-base multi-task | 0.262 |
| BERT-base | 0.239 |
| DeBERTa-v3-base + alt input | 0.238 |

Best Task 2 experiment:

```bash
python scripts/train.py --config configs/roberta_large_task2.yaml
```

Important Task 2 evaluation:

- Majority-consensus examples: `275`
- Unanimous examples: `125`
- No-majority examples: `33`

Best Task 2 disagreement metrics:

- RoBERTa-large majority macro-F1: `0.342`
- RoBERTa-large unanimous macro-F1: `0.402`
- RoBERTa-large any-annotator match rate: `0.526`

Error patterns from `top_confusions_task2.csv`:

- `Implicit -> Explicit`: 26
- `Implicit -> Dodging`: 22
- `General -> Dodging`: 20
- `General -> Explicit`: 17

Interpretation:

Task 2 is much harder than Task 1. The model often detects that an answer is broadly responsive/evasive, but confuses the exact evasion strategy.

## 8. Ablation Experiments

### Class Weighting

Task 1:

- DeBERTa-v3-base: `0.605`
- Weighted DeBERTa-v3-base: `0.546`

Conclusion: class weighting hurt Task 1.

Task 2:

- DeBERTa-v3-base: `0.301`
- Weighted DeBERTa-v3-base: `0.318`

Conclusion: class weighting helped Task 2 macro-F1, likely because Task 2 has stronger class imbalance. But accuracy dropped from `0.400` to `0.280`, so it trades overall accuracy for minority-class sensitivity.

### Input Wording

Default:

```text
Question: {question} Answer: {answer}
```

Alternative:

```text
Question: {question} Politician's answer: {answer}
```

Results:

- Task 1 default DeBERTa-v3-base: `0.605`
- Task 1 alt input: `0.588`
- Task 2 default DeBERTa-v3-base: `0.301`
- Task 2 alt input: `0.238`

Conclusion: the simpler `Answer:` format is better.

### Model Scale

Task 1 improves with DeBERTa-xlarge.

Task 2 does not improve with DeBERTa-xlarge. RoBERTa-large is best.

### Multi-task Learning

Multi-task DeBERTa-v3-base:

- Task 1 test macro-F1: `0.586`
- Task 2 majority test macro-F1: `0.262`

Compared with single-task DeBERTa-v3-base:

- Task 1: `0.605`
- Task 2: `0.301`

Conclusion: the simple multi-task setup did not help.

Not found as completed ablations:

- focal loss
- max-length sweep
- learning-rate sweep
- multi-seed vs single-seed comparison
- ensemble comparison
- hierarchical result comparison

## 9. Failed Or Unfinished Experiments

I searched for error traces like `Traceback`, `RuntimeError`, `out of memory`, `protobuf`, `FileNotFoundError`, and `failed`.

What I found in the repo:

- No saved failed training logs.
- No `outputs/slurm_logs/` in the git worktree.
- No saved CUDA OOM log.
- No saved protobuf failure log.
- No saved checkpoint-not-found failure.

Unfinished or incomplete experiments:

1. **Focal loss**
   - Implemented in `src/trainer_utils.py`.
   - No focal-loss config/results found.
   - Should be retried only after multi-seed baselines.

2. **Seed ensembling**
   - Implemented in `scripts/ensemble.py`.
   - No `outputs/ensembles/` found.
   - Should be run after training 3 seeds.

3. **Hierarchical Task 2**
   - Implemented in `scripts/hierarchical.py`.
   - No `outputs/hierarchical/` found.
   - Should be tested with best Task 1 and best Task 2 checkpoints.

4. **Submission generation**
   - Inference exists.
   - Official Codabench/SemEval submission writer not found.
   - This is important to add.

5. **Slurm file uncertainty**
   - README mentions `slurm/run_all.sh`.
   - Git tree lists `slurm/run_all_jobs.sh` and `slurm/run_large_models_jobs.sh`.
   - In this sparse checkout, Slurm files were not materialized; the outer non-git folder has `run_all.sh` and `run_large_models.sh`.
   - Verify names on Marvin before running.

## 10. Result Interpretation

What worked best:

- Task 1: DeBERTa-xlarge.
- Task 2: RoBERTa-large.
- Class weighting: helpful for Task 2 macro-F1, harmful for Task 1.
- Input wording ablation: did not help.
- Multi-task learning: did not help.

What did not work:

- DeBERTa-v3-base weighted loss for Task 1.
- Alternative `Politician's answer:` input.
- Simple multi-task learning.
- DeBERTa-xlarge for Task 2.

Is the improvement meaningful?

Likely yes for model comparison, but not fully reliable because all saved results are **single seed only**. Example:

- Task 1 BERT `0.570` -> DeBERTa-xlarge `0.632`
- Task 2 BERT `0.239` -> RoBERTa-large `0.342`

These are sizable improvements, but without multiple seeds you cannot claim stability.

Evidence:

- Main result CSVs:
  - `paper_assets/tables/main_results_task1.csv`
  - `paper_assets/tables/main_results_task2.csv`
- Prediction files under `outputs/runs/.../artifacts/`
- Confusion/error files under `paper_assets/tables/` and `paper_assets/examples/`

## 11. Reproducibility Commands

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is loaded automatically from Hugging Face, so no manual data placement is needed.

BERT baselines:

```bash
python scripts/train.py --config configs/bert_base_task1.yaml
python scripts/train.py --config configs/bert_base_task2.yaml
```

Main DeBERTa-v3-base:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1.yaml
python scripts/train.py --config configs/deberta_v3_base_task2.yaml
```

Class-weighted ablations:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1_weighted.yaml
python scripts/train.py --config configs/deberta_v3_base_task2_weighted.yaml
```

Input wording ablations:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1_politician_answer.yaml
python scripts/train.py --config configs/deberta_v3_base_task2_politician_answer.yaml
```

Large models:

```bash
python scripts/train.py --config configs/roberta_large_task1.yaml
python scripts/train.py --config configs/roberta_large_task2.yaml
python scripts/train.py --config configs/deberta_v3_large_task1.yaml
python scripts/train.py --config configs/deberta_v3_large_task2.yaml
python scripts/train.py --config configs/deberta_xlarge_task1.yaml
python scripts/train.py --config configs/deberta_xlarge_task2.yaml
```

Multi-task:

```bash
python scripts/train.py --config configs/deberta_v3_base_multitask.yaml
```

Evaluate saved metrics:

```bash
python scripts/evaluate.py --run-dir outputs/runs/task1/<run-name>
python scripts/evaluate.py --run-dir outputs/runs/task2/<run-name>
```

Inference:

```bash
python scripts/predict.py \
  --config configs/deberta_v3_base_task1.yaml \
  --checkpoint outputs/runs/task1/<run-name>/best_model \
  --split test \
  --output outputs/predictions/task1_test.csv
```

Expected outputs per run:

- `config.yaml`
- `split.json`
- `label_mapping.json`
- `results_summary.json`
- `artifacts/dev_predictions.csv`
- `artifacts/test_predictions.csv`
- `artifacts/*_metrics.json`
- `artifacts/*_classification_report.json`
- `artifacts/*_confusion_matrix.json`
- `artifacts/*_misclassified_examples.csv`
- `best_model/`

GPU expectations:

- BERT: relatively light.
- DeBERTa-v3-base: moderate GPU, fp16.
- RoBERTa-large / DeBERTa-v3-large: small batch, gradient accumulation.
- DeBERTa-xlarge: memory-safe config, batch size 1, accumulation 16, max length 192, gradient checkpointing.

## 12. What You Can Write In Report/Poster

### Experimental setup paragraph

We fine-tuned transformer encoders on the Hugging Face `ailsntua/QEvasion` dataset for both CLARITY subtasks. The official training split was divided into an 80/20 internal train-development split using stratification where possible. We evaluated Task 1 with standard macro-F1 on the official test labels and Task 2 with majority-consensus macro-F1, unanimous-only macro-F1, and any-annotator match rate because Task 2 test examples have three annotator labels.

### Model comparison paragraph

We compared BERT-base, DeBERTa-v3-base, RoBERTa-large, DeBERTa-v3-large, and DeBERTa-xlarge. BERT-base served as a quick baseline. DeBERTa-v3-base was the main strong baseline, while RoBERTa-large and larger DeBERTa models tested whether scaling improved performance. Larger encoders helped Task 1 most clearly, while Task 2 was best handled by RoBERTa-large.

### Best result paragraph

The strongest Task 1 system was DeBERTa-xlarge with test macro-F1 `0.632`. The strongest Task 2 system was RoBERTa-large with majority-consensus test macro-F1 `0.342`. Task 2 remained much harder, especially because fine-grained labels are imbalanced and annotators sometimes disagree.

### Limitations/future work paragraph

All saved experiments use a single seed, so results should be treated as preliminary. No cross-validation or seed ensemble results are available yet. Focal loss, hierarchical evaluation, and ensemble scripts are implemented but not yet run. Future work should add multi-seed experiments, probability averaging ensembles, a submission-generation script, better hierarchical modeling, and methods that explicitly handle annotator disagreement.

## 13. Code Explanation After Experiments

Main training:

- `scripts/train.py`
- Loads config, data, split, tokenizer, model, trains, saves predictions and metrics.

Main evaluation:

- `scripts/evaluate.py`
- Reads saved `*_metrics.json` files and prints compact metrics.

Main inference:

- `scripts/predict.py`
- Loads a single-task checkpoint and writes prediction CSV plus logits/probabilities.

Important utilities:

- `src/config.py`: YAML config dataclasses.
- `src/data.py`: dataset loading, labels, splits, class weights.
- `src/preprocess.py`: input formatting and tokenization.
- `src/models.py`: single-task and multi-task model builders.
- `src/trainer_utils.py`: weighted/focal loss and training arguments.
- `src/metrics.py`: macro-F1, reports, Task 2 annotator metrics.
- `src/evaluate.py`: saves predictions, logits, reports, confusion matrices.

Results are stored in:

- `outputs/runs/`
- `paper_assets/tables/`
- `paper_assets/figures/`
- `paper_assets/examples/`

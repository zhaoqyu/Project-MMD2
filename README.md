# QEvasion / CLARITY SemEval 2026 Baselines

Clean Hugging Face Transformers code for the SemEval 2026 CLARITY / QEvasion dataset:
`ailsntua/QEvasion`.

The code supports:

- Task 1: clarity / evasion detection from `question` + `interview_answer`
- Task 2: fine-grained evasion strategy detection
- single-task `Trainer` runs
- an optional shared-encoder multi-task model
- class-weighted cross entropy and focal loss
- input wording ablations
- seed ensembling
- Task 2 test metrics for annotator disagreement

## Dataset Schema

The Hugging Face dataset has one config, `default`, with `train` and `test`.

Important columns:

- question: `question`
- answer: `interview_answer`
- Task 1 label: `clarity_label`
- Task 2 train label: `evasion_label`
- Task 2 test annotator labels: `annotator1`, `annotator2`, `annotator3`
- stable row id: `index`

The dataset currently uses `Ambivalent` for the middle Task 1 class. Some task descriptions call this class `Ambiguous`; the code preserves the dataset label and accepts common aliases when normalizing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For DeBERTa-v3 models, keep `sentencepiece` installed.

## Training

Every training run:

- loads `ailsntua/QEvasion`
- creates or reuses a reproducible 80/20 internal train-dev split
- saves split indices and dataset IDs under `outputs/splits/`
- saves a run snapshot under `outputs/runs/<task>/<run-name>/`
- saves metrics, predictions, probabilities, logits, reports, confusion matrices, and misclassified examples

Baseline:

```bash
python scripts/train.py --config configs/bert_base_task1.yaml
python scripts/train.py --config configs/bert_base_task2.yaml
```

Main strong model:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1.yaml
python scripts/train.py --config configs/deberta_v3_base_task2.yaml
```

Class-weighted experiments:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1_weighted.yaml
python scripts/train.py --config configs/deberta_v3_base_task2_weighted.yaml
```

Input wording experiments:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1_politician_answer.yaml
python scripts/train.py --config configs/deberta_v3_base_task2_politician_answer.yaml
```

Large-model runs:

```bash
python scripts/train.py --config configs/roberta_large_task1.yaml
python scripts/train.py --config configs/roberta_large_task2.yaml
python scripts/train.py --config configs/deberta_v3_large_task1.yaml
python scripts/train.py --config configs/deberta_v3_large_task2.yaml
python scripts/train.py --config configs/deberta_xlarge_task1.yaml
python scripts/train.py --config configs/deberta_xlarge_task2.yaml
```

Multi-task model:

```bash
python scripts/train.py --config configs/deberta_v3_base_multitask.yaml
```

Multiple seeds can be placed in a config, for example:

```yaml
seeds: [13, 42, 87]
```

or run a single override:

```bash
python scripts/train.py --config configs/deberta_v3_base_task1.yaml --seed 13
```

## Evaluation

Training already evaluates dev and test. To print compact saved metrics:

```bash
python scripts/evaluate.py --run-dir outputs/runs/task1/<run-name>
```

Task 1 test reports standard macro-F1, weighted-F1, accuracy, per-class metrics, a confusion matrix, and Ambivalent vs Clear Non-Reply confusion.

Task 2 test reports:

- majority-consensus macro-F1 where at least two annotators agree
- unanimous-only macro-F1 where all three annotators agree
- any-annotator match rate
- unresolved-disagreement count for rows with no majority

Rows with no majority are saved separately and are not counted as ordinary errors for majority-consensus F1.

## Prediction

```bash
python scripts/predict.py \
  --config configs/deberta_v3_base_task1.yaml \
  --checkpoint outputs/runs/task1/<run-name>/best_model \
  --split test \
  --output outputs/predictions/task1_test.csv
```

## Seed Ensembling

Run the same config with multiple seeds, then average probabilities:

```bash
python scripts/ensemble.py \
  --task task1 \
  --split test \
  --run-dirs outputs/runs/task1/<seed-run-1> outputs/runs/task1/<seed-run-2> outputs/runs/task1/<seed-run-3> \
  --output-dir outputs/ensembles/deberta_v3_base_task1
```

Use `--task task2` for Task 2.

## Hierarchical Task 2 Evaluation

This optional comparison uses Task 1 predictions to constrain clear-reply examples to reply strategies (`Explicit`, `Implicit`) and leaves non-clear-reply examples to the Task 2 classifier.

```bash
python scripts/hierarchical.py \
  --task1-run-dir outputs/runs/task1/<task1-run> \
  --task2-run-dir outputs/runs/task2/<task2-run> \
  --split test \
  --output-dir outputs/hierarchical/deberta_v3_base
```

## Slurm

From the project root:

```bash
sbatch slurm/run_all.sh
sbatch slurm/run_large_models.sh
```

Edit these variables at the top of each Slurm script:

- `#SBATCH --partition`
- `#SBATCH --gres`
- `#SBATCH --mem`
- `#SBATCH --time`
- `VENV_PATH`
- `CONFIGS`

The scripts print host, CUDA device, `nvidia-smi`, Python path, PyTorch version, Transformers version, and CUDA availability. Per-config logs are saved in `outputs/slurm_logs/`.

## Analysis Notebook

Open:

```bash
jupyter lab notebooks/analysis.ipynb
```

The notebook does not train models. It loads saved `results_summary.json`, metrics JSON files, prediction CSV files, and training logs from `outputs/runs/`.


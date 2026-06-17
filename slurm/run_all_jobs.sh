#!/bin/bash
#SBATCH --job-name=qevasion
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8%2
#SBATCH --output=outputs/slurm_logs/%x_%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x_%A_%a.err

set -euo pipefail

# Edit these for Marvin.
VENV_PATH="${VENV_PATH:-$PWD/.venv}"

CONFIGS=(
  configs/bert_base_task1.yaml
  configs/bert_base_task2.yaml
  configs/deberta_v3_base_task1.yaml
  configs/deberta_v3_base_task2.yaml
  configs/deberta_v3_base_task1_weighted.yaml
  configs/deberta_v3_base_task2_weighted.yaml
  configs/deberta_v3_base_task1_politician_answer.yaml
  configs/deberta_v3_base_task2_politician_answer.yaml
  configs/deberta_v3_base_multitask.yaml
)

mkdir -p outputs/slurm_logs

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(basename "$CONFIG" .yaml)"

LOG="outputs/slurm_logs/${NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "=========================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: $CONFIG"
echo "Name: $NAME"
echo "Log: $LOG"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

nvidia-smi || true

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Virtual environment not found at $VENV_PATH"
  exit 1
fi

echo "python: $(which python)"

python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo "Running $CONFIG"

python scripts/train.py --config "$CONFIG" 2>&1 | tee "$LOG"

echo "Finished $CONFIG"
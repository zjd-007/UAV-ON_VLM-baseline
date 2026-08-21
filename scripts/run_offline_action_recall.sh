#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/data/zhujd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-octmem_openvla_nomemory}"
RUN_ID="${RUN_ID:-offline_action_recall_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/${RUN_ID}.log}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -u "${ROOT}/scripts/offline_action_recall.py" \
  --samples_per_class "${SAMPLES_PER_CLASS}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

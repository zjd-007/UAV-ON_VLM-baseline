#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}/eval:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MODEL_PATH="${MODEL_PATH:-${ROOT}/outputs/phi35_uavon_lora_r256_merged}"
if [[ ! -e "${MODEL_PATH}" ]]; then
  MODEL_PATH="${MODEL_PATH_FALLBACK:-${ROOT}/outputs/phi35_uavon_lora_r256}"
fi

python "${ROOT}/eval/eval_phi35_uavon.py" \
  --model_path "${MODEL_PATH}" \
  --eval_dataset "${EVAL_DATASET:-${REPO_ROOT}/UAV-ON_dataset/splits/uavon_raw_json/test.json}" \
  --output_foler "${OUTPUT_FOLDER:-${ROOT}/results/phi35_uavon}" \
  --eval_max_steps "${EVAL_MAX_STEPS:-100}" \
  --airsim_default_port "${AIRSIM_DEFAULT_PORT:-30000}" \
  "$@"

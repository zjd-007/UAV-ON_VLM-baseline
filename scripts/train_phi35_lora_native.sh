#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_phi35_lora_native.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,6,7}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICES[@]}}"
MASTER_PORT="${MASTER_PORT:-29541}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-port "${MASTER_PORT}" \
    "${ROOT}/scripts/train_phi35_lora_native.py" "${CONFIG}" "$@"
else
  python "${ROOT}/scripts/train_phi35_lora_native.py" "${CONFIG}" "$@"
fi

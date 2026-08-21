#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_phi35_lora.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICES[@]}}"
MASTER_PORT="${MASTER_PORT:-29531}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --master-port "${MASTER_PORT}" \
    "${ROOT}/scripts/llamafactory_phi3v.py" train "${CONFIG}" "$@"
else
  python "${ROOT}/scripts/llamafactory_phi3v.py" train "${CONFIG}" "$@"
fi

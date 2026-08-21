#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_qwen25_vl_lora_language_only_depth_grid_stopbank_v1.yaml}"
NVIDIA_COMPAT_LIB="${NVIDIA_COMPAT_LIB:-/usr/lib/x86_64-linux-gnu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,4,5}"
export LD_LIBRARY_PATH="${NVIDIA_COMPAT_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICES[@]}}"
export MASTER_PORT="${MASTER_PORT:-29616}"

python -m llamafactory.cli train "${CONFIG}" "$@"

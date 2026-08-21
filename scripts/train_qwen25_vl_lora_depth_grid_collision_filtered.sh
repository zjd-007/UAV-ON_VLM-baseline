#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_qwen25_vl_lora_depth_grid_collision_filtered.yaml}"
NVIDIA_COMPAT_LIB="${NVIDIA_COMPAT_LIB:-/data/zhujd/Aerial-ObjectNav/.nvidia-compat/580.159.03/lib}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LD_LIBRARY_PATH="${NVIDIA_COMPAT_LIB}:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICES[@]}}"
export MASTER_PORT="${MASTER_PORT:-29547}"

python -m llamafactory.cli train "${CONFIG}" "$@"

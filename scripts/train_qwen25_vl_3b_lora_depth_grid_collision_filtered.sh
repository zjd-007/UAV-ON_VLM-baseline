#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_qwen25_vl_3b_lora_depth_grid_collision_filtered.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"

# The host driver is usable directly after the server reboot. Keep the
# compatibility library opt-in so an older cached driver cannot shadow it.
if [[ -n "${NVIDIA_COMPAT_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_COMPAT_LIB}:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICES[@]}}"
export MASTER_PORT="${MASTER_PORT:-29653}"

python -m llamafactory.cli train "${CONFIG}" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/train_phi35_lora_smoke.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/scripts/llamafactory_phi3v.py" train "${CONFIG}" "$@"

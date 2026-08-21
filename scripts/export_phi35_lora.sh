#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/export_phi35_lora.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/scripts/llamafactory_phi3v.py" export "${CONFIG}" "$@"

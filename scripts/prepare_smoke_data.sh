#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/scripts/prepare_data.py" \
  --limit "${LIMIT:-100}" \
  --output "${ROOT}/data/uavon_phi35_sft_smoke.jsonl" \
  --manifest "${ROOT}/data/uavon_phi35_sft_smoke_manifest.json" \
  "$@"

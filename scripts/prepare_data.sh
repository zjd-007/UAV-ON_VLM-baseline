#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/scripts/prepare_data.py" "$@"

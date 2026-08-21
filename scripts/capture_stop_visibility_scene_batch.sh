#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 GPU BASE_PORT SCENE_LIST TRAJECTORY_KEYS OUTPUT_DIR LOG_DIR PYTHON_BIN" >&2
  exit 2
}

[[ $# -eq 7 ]] || usage

GPU="$1"
BASE_PORT="$2"
SCENE_LIST="$3"
TRAJECTORY_KEYS="$4"
OUTPUT_DIR="$5"
LOG_DIR="$6"
PYTHON_BIN="$7"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

IFS=',' read -r -a SCENES <<< "$SCENE_LIST"
for SCENE in "${SCENES[@]}"; do
  [[ -n "$SCENE" ]] || continue
  LOG_PATH="$LOG_DIR/${SCENE}.log"
  echo "[$(date '+%F %T')] starting scene=$SCENE gpu=$GPU" | tee -a "$LOG_PATH"
  PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/capture_stop_visibility_cache.py" \
    --gpu "$GPU" \
    --base-port "$BASE_PORT" \
    --scene-list "$SCENE" \
    --trajectory-keys "$TRAJECTORY_KEYS" \
    --output-dir "$OUTPUT_DIR" \
    --save-debug \
    2>&1 | tee -a "$LOG_PATH"
  echo "[$(date '+%F %T')] finished scene=$SCENE gpu=$GPU" | tee -a "$LOG_PATH"
done

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_DIR LANE GPU BASE_PORT" >&2
  exit 2
fi

RUN_DIR="$1"
LANE="$2"
GPU="$3"
BASE_PORT="$4"
ROOT="/data/zhujd/Aerial-ObjectNav"
PYTHON_BIN="/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python"
PROJECT="$ROOT/VLM-baseline"

mkdir -p "$RUN_DIR/visibility/$LANE" "$RUN_DIR/collision/$LANE"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src"

"$PYTHON_BIN" "$PROJECT/scripts/capture_stop_visibility_cache.py" \
  --aligned-root "$RUN_DIR/aligned_xy_shifted" \
  --metadata "$RUN_DIR/train_xy_shifted.json" \
  --output-dir "$RUN_DIR/visibility/$LANE" \
  --trajectory-keys "$RUN_DIR/${LANE}_trajectory_keys.txt" \
  --scene-list Neighborhood \
  --gpu "$GPU" \
  --base-port "$BASE_PORT" \
  --distance-threshold 20 \
  --settle-frames 2 \
  --segmentation-settle-frames 4 \
  --save-debug

"$PYTHON_BIN" "$PROJECT/scripts/check_train_label_action_collision.py" \
  --source "$RUN_DIR/${LANE}_collision_source.jsonl" \
  --aligned-root "$RUN_DIR/aligned_xy_shifted" \
  --output-dir "$RUN_DIR/collision/$LANE" \
  --scene-list Neighborhood \
  --gpu "$GPU" \
  --base-port "$((BASE_PORT + 100))" \
  --settle-frames 1 \
  --action-velocity 2.0 \
  --action-move-timeout 5.0 \
  --action-rotate-timeout 3.0 \
  --fix-vertical-actions \
  --fix-yaw-actions

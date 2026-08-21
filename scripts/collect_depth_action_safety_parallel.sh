#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zhujd/Aerial-ObjectNav/VLM-baseline"
DATASET_ROOT="/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset"
CONDA_SH="${CONDA_SH:-/data/zhujd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-octmem_openvla_nomemory}"
RUN_ID="${RUN_ID:-depth_action_safety_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_ROOT}/processed/depth_action_safety/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_ID}}"
BASE_PORT="${BASE_PORT:-47000}"
MAX_ROWS_PER_SCENE="${MAX_ROWS_PER_SCENE:-0}"
LIMIT="${LIMIT:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
DEPTH_OUTPUT_SIZE="${DEPTH_OUTPUT_SIZE:-128}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-60}"
WATCHDOG_STALE_SECONDS="${WATCHDOG_STALE_SECONDS:-300}"
ORIGINAL_COLLISION_DIR="${DATASET_ROOT}/processed/label_action_collision_check/label_action_collision_full_20260715_175048"
REPAIR_COLLISION_DIR="${DATASET_ROOT}/processed/label_action_collision_check/label_action_collision_full_20260715_175048_repair_lane3"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Output directory is not empty: ${OUTPUT_DIR}" >&2
  echo "Use a new RUN_ID or an explicit empty OUTPUT_DIR." >&2
  exit 1
fi

cat > "${OUTPUT_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "launch_unix": $(date +%s),
  "script": "${ROOT}/scripts/collect_depth_action_safety_data.py",
  "output_dir": "${OUTPUT_DIR}",
  "log_dir": "${LOG_DIR}",
  "source": "${DATASET_ROOT}/processed/nomemory_baseline/train_frames.jsonl",
  "aligned_root": "${DATASET_ROOT}/generated/record_output_transition_aligned",
  "original_collision_dir": "${ORIGINAL_COLLISION_DIR}",
  "repair_collision_dir": "${REPAIR_COLLISION_DIR}",
  "collision_filter_manifest": "${ROOT}/data/uavon_phi35_sft_depth_grid_collision_filtered_manifest.json",
  "gpus": [2, 3, 4, 5, 6],
  "base_port": ${BASE_PORT},
  "max_rows_per_scene": ${MAX_ROWS_PER_SCENE},
  "limit_per_lane": ${LIMIT},
  "depth_camera": "uav_on_0",
  "depth_output_size": ${DEPTH_OUTPUT_SIZE},
  "save_full_resolution_depth": true,
  "full_resolution_depth_shape": [512, 512],
  "depth_encoding": "uint16_png",
  "depth_scale_meters": 0.01,
  "progress_interval": ${PROGRESS_INTERVAL},
  "watchdog_interval_seconds": ${WATCHDOG_INTERVAL},
  "watchdog_stale_seconds": ${WATCHDOG_STALE_SECONDS},
  "label_order": ["stop", "forward 3m", "turn left 30 degree", "turn right 30 degree", "ascend 3m", "descend 3m"],
  "stop_is_fixed_safe": true,
  "reuse_collision_audited_expert_action": true,
  "action_execution_mode": "apex_join",
  "action_velocity": 2.0,
  "fix_vertical_actions": true,
  "fix_yaw_actions": true,
  "client_reset_per_action": true,
  "client_reset_settle_seconds": 0.05,
  "initial_pose_retries": 3
}
EOF

cat > "${OUTPUT_DIR}/lanes.tsv" <<'EOF'
2	BrushifyUrban,CabinLake
3	CityPark,DownTown
4	Neighborhood,Slum
5	UrbanJapan,Venice
6	WesternTown,WinterTown
EOF

: > "${LOG_DIR}/sessions.tsv"
while IFS=$'\t' read -r GPU SCENES; do
  SESSION="${RUN_ID}_gpu${GPU}"
  LOG="${LOG_DIR}/gpu${GPU}.log"
  CMD=$(cat <<EOF
source '${CONDA_SH}'
conda activate '${CONDA_ENV}'
export PYTHONPATH='${ROOT}/src:${ROOT}/eval:${ROOT}/scripts':\${PYTHONPATH:-}
cd '${ROOT}'
python '${ROOT}/scripts/collect_depth_action_safety_data.py' \
  --output-dir '${OUTPUT_DIR}' \
  --scene-list '${SCENES}' \
  --gpu '${GPU}' \
  --base-port '${BASE_PORT}' \
  --max-rows-per-scene '${MAX_ROWS_PER_SCENE}' \
  --limit '${LIMIT}' \
  --progress-interval '${PROGRESS_INTERVAL}' \
  --depth-output-size '${DEPTH_OUTPUT_SIZE}' \
  --expected-full-depth-size '512' \
  --save-full-resolution-depth \
  --reuse-collision-audited-expert-action \
  --retry-incomplete \
  --summary-name 'summary_gpu${GPU}.json' \
  2>&1 | tee '${LOG}'
EOF
)
  screen -dmS "${SESSION}" bash -lc "${CMD}"
  printf "%s\tgpu=%s\tport=%s\tscenes=%s\tlog=%s\n" \
    "${SESSION}" "${GPU}" "$((BASE_PORT + GPU))" "${SCENES}" "${LOG}" \
    | tee -a "${LOG_DIR}/sessions.tsv"
done < "${OUTPUT_DIR}/lanes.tsv"

WATCHDOG_SESSION="${RUN_ID}_watchdog"
WATCHDOG_CMD=$(cat <<EOF
source '${CONDA_SH}'
conda activate '${CONDA_ENV}'
export PYTHONPATH='${ROOT}/src:${ROOT}/eval:${ROOT}/scripts':\${PYTHONPATH:-}
cd '${ROOT}'
python -u '${ROOT}/scripts/depth_action_safety_watchdog.py' \
  --run-dir '${OUTPUT_DIR}' \
  --interval-seconds '${WATCHDOG_INTERVAL}' \
  --stale-seconds '${WATCHDOG_STALE_SECONDS}' \
  2>&1 | tee -a '${LOG_DIR}/watchdog.log'
EOF
)
screen -dmS "${WATCHDOG_SESSION}" bash -lc "${WATCHDOG_CMD}"
printf "%s\twatchdog\tlog=%s\n" "${WATCHDOG_SESSION}" "${LOG_DIR}/watchdog.log" \
  | tee -a "${LOG_DIR}/sessions.tsv"

echo "run_id=${RUN_ID}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_dir=${LOG_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/data/zhujd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-octmem_openvla_nomemory}"
NVIDIA_COMPAT_LIB="${NVIDIA_COMPAT_LIB-/data/zhujd/Aerial-ObjectNav/.nvidia-compat/580.159.03/lib}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-VL-7B-Instruct}"
EVAL_DATASET="${EVAL_DATASET:-${REPO_ROOT}/UAV-ON_dataset/splits/uavon_raw_json/test.json}"
EVAL_SAMPLES_PER_ENV="${EVAL_SAMPLES_PER_ENV:-}"
RUN_ID="${RUN_ID:-qwen25vl7b_zero_shot_cfmem_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_ID}}"
LANE_GPUS="${LANE_GPUS:-0,2,3}"
SIMULATOR_GPUS="${SIMULATOR_GPUS:-${LANE_GPUS}}"
BASE_PORT="${BASE_PORT:-48100}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
SAVE_STEP_IMAGES="${SAVE_STEP_IMAGES:-1}"
IMAGE_SAVE_STRIDE="${IMAGE_SAVE_STRIDE:-1}"
IMAGE_QUALITY="${IMAGE_QUALITY:-85}"
DEPTH_AVOIDANCE="${DEPTH_AVOIDANCE:-uavon_single_view_prompt}"
MEMORY_CONTEXT="${MEMORY_CONTEXT:-uavon_pose_history}"
MEMORY_HISTORY_SIZE="${MEMORY_HISTORY_SIZE:-5}"
MEMORY_SEARCH_RADIUS="${MEMORY_SEARCH_RADIUS:-50.0}"
MEMORY_INCLUDE_SEARCH_BOUNDS="${MEMORY_INCLUDE_SEARCH_BOUNDS:-0}"
MEMORY_POSE_YAW_UNIT="${MEMORY_POSE_YAW_UNIT:-radians}"
ACTION_REDIRECT="${ACTION_REDIRECT:-none}"
START_WATCHDOG="${START_WATCHDOG:-1}"
WATCHDOG_INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-60}"
WATCHDOG_STALE_SECONDS="${WATCHDOG_STALE_SECONDS:-300}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
QWEN_MERGE_ADAPTER_FOR_INFERENCE="${QWEN_MERGE_ADAPTER_FOR_INFERENCE:-0}"

IFS=',' read -r -a LANE_GPU_ARRAY <<< "${LANE_GPUS}"
IFS=',' read -r -a SIMULATOR_GPU_ARRAY <<< "${SIMULATOR_GPUS}"
LANE_COUNT="${#LANE_GPU_ARRAY[@]}"
if [[ "${LANE_COUNT}" != "3" && "${LANE_COUNT}" != "4" ]]; then
  echo "LANE_GPUS must contain three or four comma-separated GPU ids, got: ${LANE_GPUS}" >&2
  exit 1
fi
if [[ "${#SIMULATOR_GPU_ARRAY[@]}" != "${LANE_COUNT}" ]]; then
  echo "SIMULATOR_GPUS must contain ${LANE_COUNT} comma-separated GPU ids, got: ${SIMULATOR_GPUS}" >&2
  exit 1
fi
if [[ -e "${RUN_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "Refusing to overwrite an existing run: ${RUN_ID}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [[ "${LANE_COUNT}" == "3" ]]; then
  printf "lane0\t%s\tNYC_test,WinterTown_test,UrbanJapan_test,WesternTown_test\n" "${LANE_GPU_ARRAY[0]}" > "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tSlum_test,Barnyard_test,BrushifyUrban_test,CabinLake_test,DownTown_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane2\t%s\tCityStreet_test,BrushifyRoad_test,CityPark_test,ModularNeighborhood_test,Venice_test\n" "${LANE_GPU_ARRAY[2]}" >> "${RUN_DIR}/lanes.tsv"
else
  printf "lane0\t%s\tNYC_test,UrbanJapan_test,DownTown_test\n" "${LANE_GPU_ARRAY[0]}" > "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tSlum_test,BrushifyUrban_test,CabinLake_test,Venice_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane2\t%s\tCityStreet_test,BrushifyRoad_test,CityPark_test,WesternTown_test\n" "${LANE_GPU_ARRAY[2]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane3\t%s\tBarnyard_test,WinterTown_test,ModularNeighborhood_test\n" "${LANE_GPU_ARRAY[3]}" >> "${RUN_DIR}/lanes.tsv"
fi

ZERO_SHOT=true
if [[ -f "${MODEL_PATH}/adapter_config.json" ]]; then
  ZERO_SHOT=false
fi

SIMULATOR_GPU_BY_LANE_JSON="{"
for ((i = 0; i < LANE_COUNT; i++)); do
  if ((i > 0)); then
    SIMULATOR_GPU_BY_LANE_JSON+=", "
  fi
  SIMULATOR_GPU_BY_LANE_JSON+="\"lane${i}\": ${SIMULATOR_GPU_ARRAY[$i]}"
done
SIMULATOR_GPU_BY_LANE_JSON+="}"

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "lane_count": ${LANE_COUNT},
  "evaluator_script": "eval/eval_qwen25_vl_uavon.py",
  "model_family": "qwen2.5-vl",
  "zero_shot": ${ZERO_SHOT},
  "nvidia_compat_lib": "${NVIDIA_COMPAT_LIB}",
  "model_path": "${MODEL_PATH}",
  "eval_dataset": "${EVAL_DATASET}",
  "eval_samples_per_env": ${EVAL_SAMPLES_PER_ENV:-null},
  "eval_max_steps": ${EVAL_MAX_STEPS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "base_port": ${BASE_PORT},
  "simulator_gpu_by_lane": ${SIMULATOR_GPU_BY_LANE_JSON},
  "save_step_images": ${SAVE_STEP_IMAGES},
  "image_save_stride": ${IMAGE_SAVE_STRIDE},
  "image_quality": ${IMAGE_QUALITY},
  "inference_mode": "generate",
  "pytorch_cuda_alloc_conf": "${PYTORCH_CUDA_ALLOC_CONF}",
  "qwen_merge_adapter_for_inference": ${QWEN_MERGE_ADAPTER_FOR_INFERENCE},
  "depth_avoidance": "${DEPTH_AVOIDANCE}",
  "depth_grid_size": 3,
  "depth_max_meters": 100.0,
  "depth_forward_threshold": 4.0,
  "depth_turn_threshold": 1.5,
  "depth_descend_threshold": 6.0,
  "depth_ascend_top_threshold": 8.0,
  "action_redirect": "${ACTION_REDIRECT}",
  "action_redirect_search_radius": 50.0,
  "action_redirect_near_obstacle_threshold": 2.0,
  "memory_context": "${MEMORY_CONTEXT}",
  "memory_history_size": ${MEMORY_HISTORY_SIZE},
  "memory_search_radius": ${MEMORY_SEARCH_RADIUS},
  "memory_include_search_bounds": ${MEMORY_INCLUDE_SEARCH_BOUNDS},
  "memory_pose_yaw_unit": "${MEMORY_POSE_YAW_UNIT}",
  "fix_vertical_actions": 1,
  "fix_yaw_actions": 1,
  "pose_wait_timeout": 1.0,
  "pose_wait_position_tol": 0.2,
  "pose_wait_yaw_tol": 0.05,
  "pose_wait_poll_interval": 0.05,
  "render_settle_seconds": 0.0,
  "action_execution_mode": "apex_join",
  "action_sim_frames": 150,
  "action_velocity": 2.0,
  "action_move_timeout": 5.0,
  "action_rotate_timeout": 3.0,
  "level_after_action": 0,
  "level_settle_frames": 1,
  "initial_pose_retries": 3,
  "initial_pose_settle_frames": 1,
  "zero_kinematics_reset": 1,
  "client_reset_per_episode": 1,
  "kill_env_process": 0
}
EOF

while IFS="$(printf '\t')" read -r LANE GPU SCENES; do
  LANE_INDEX="${LANE#lane}"
  SIMULATOR_GPU="${SIMULATOR_GPU_ARRAY[$LANE_INDEX]}"
  PORT=$((BASE_PORT + GPU))
  SESSION="${RUN_ID}_${LANE}"
  OUT="${RUN_DIR}/${LANE}"
  LOG="${LOG_DIR}/${LANE}.log"
  mkdir -p "${OUT}"

  SAMPLE_ARGS=""
  if [[ -n "${EVAL_SAMPLES_PER_ENV}" ]]; then
    SAMPLE_ARGS="--eval_samples_per_env ${EVAL_SAMPLES_PER_ENV}"
  fi

  COMPAT_SETUP=""
  if [[ -n "${NVIDIA_COMPAT_LIB}" ]]; then
    COMPAT_SETUP="export LD_LIBRARY_PATH='${NVIDIA_COMPAT_LIB}':\${LD_LIBRARY_PATH:-} &&"
  fi

  CMD=$(cat <<EOF
cd '${ROOT}' &&
source '${CONDA_SH}' &&
conda activate '${CONDA_ENV}' &&
${COMPAT_SETUP}
export CUDA_VISIBLE_DEVICES='${GPU}' &&
export PYTHONUNBUFFERED=1 &&
export TOKENIZERS_PARALLELISM=false &&
export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}' &&
export QWEN_MERGE_ADAPTER_FOR_INFERENCE='${QWEN_MERGE_ADAPTER_FOR_INFERENCE}' &&
export PYTHONPATH='${ROOT}/src:${ROOT}:${ROOT}/eval':\${PYTHONPATH:-} &&
python -u eval/eval_qwen25_vl_uavon.py \
  --model_path '${MODEL_PATH}' \
  --eval_dataset '${EVAL_DATASET}' \
  ${SAMPLE_ARGS} \
  --output_foler '${OUT}' \
  --eval_max_steps '${EVAL_MAX_STEPS}' \
  --airsim_default_port '${PORT}' \
  --simulator_gpu '${SIMULATOR_GPU}' \
  --device cuda:0 \
  --max_new_tokens '${MAX_NEW_TOKENS}' \
  --inference_mode generate \
  --depth_avoidance '${DEPTH_AVOIDANCE}' \
  --depth_grid_size 3 \
  --depth_max_meters 100.0 \
  --depth_forward_threshold 4.0 \
  --depth_turn_threshold 1.5 \
  --depth_descend_threshold 6.0 \
  --depth_ascend_top_threshold 8.0 \
  --action_redirect '${ACTION_REDIRECT}' \
  --action_redirect_search_radius 50.0 \
  --action_redirect_near_obstacle_threshold 2.0 \
  --memory_context '${MEMORY_CONTEXT}' \
  --memory_history_size '${MEMORY_HISTORY_SIZE}' \
  --memory_search_radius '${MEMORY_SEARCH_RADIUS}' \
  --memory_include_search_bounds '${MEMORY_INCLUDE_SEARCH_BOUNDS}' \
  --memory_pose_yaw_unit '${MEMORY_POSE_YAW_UNIT}' \
  --scene_list '${SCENES}' \
  --skip_kill_env_process \
  --fix_vertical_actions \
  --fix_yaw_actions \
  --pose_wait_timeout 1.0 \
  --pose_wait_position_tol 0.2 \
  --pose_wait_yaw_tol 0.05 \
  --pose_wait_poll_interval 0.05 \
  --render_settle_seconds 0.0 \
  --action_execution_mode apex_join \
  --action_sim_frames 150 \
  --action_velocity 2.0 \
  --action_move_timeout 5.0 \
  --action_rotate_timeout 3.0 \
  --level_settle_frames 1 \
  --initial_pose_retries 3 \
  --initial_pose_settle_frames 1 \
  --zero_kinematics_reset \
  --client_reset_per_episode \
  --save_step_images \
  --image_save_stride '${IMAGE_SAVE_STRIDE}' \
  --image_format jpg \
  --image_quality '${IMAGE_QUALITY}' \
  2>&1 | tee '${LOG}'
EOF
)

  screen -dmS "${SESSION}" bash -lc "${CMD}"
  printf "%s\tgpu=%s\tsimulator_gpu=%s\tport=%s\tout=%s\tlog=%s\tscenes=%s\n" \
    "${SESSION}" "${GPU}" "${SIMULATOR_GPU}" "${PORT}" "${OUT}" "${LOG}" "${SCENES}" | tee -a "${RUN_DIR}/sessions.tsv"
done < "${RUN_DIR}/lanes.tsv"

if [[ "${START_WATCHDOG}" == "1" ]]; then
  WATCHDOG_SESSION="${RUN_ID}_watchdog"
  WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
  WATCHDOG_CMD="cd '${ROOT}' && export PYTHONUNBUFFERED=1 && python3 -u scripts/eval_lane_watchdog.py --run_dir '${RUN_DIR}' --log_dir '${LOG_DIR}' --interval_seconds '${WATCHDOG_INTERVAL_SECONDS}' --stale_seconds '${WATCHDOG_STALE_SECONDS}' --conda_env '${CONDA_ENV}' 2>&1 | tee '${WATCHDOG_LOG}'"
  screen -dmS "${WATCHDOG_SESSION}" bash -lc "${WATCHDOG_CMD}"
fi

printf "%s\n" "${RUN_ID}" > "${ROOT}/logs/current_qwen_eval_run.txt"
printf "run_id=%s\nrun_dir=%s\nlog_dir=%s\n" "${RUN_ID}" "${RUN_DIR}" "${LOG_DIR}"
screen -ls || true

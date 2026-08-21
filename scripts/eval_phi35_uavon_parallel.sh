#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/data/zhujd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-octmem_openvla_nomemory}"
NVIDIA_COMPAT_LIB="${NVIDIA_COMPAT_LIB:-}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/outputs/phi35_uavon_lora_r256}"
EVAL_DATASET="${EVAL_DATASET:-${REPO_ROOT}/UAV-ON_dataset/splits/uavon_raw_json/test.json}"
EVAL_SAMPLES_PER_ENV="${EVAL_SAMPLES_PER_ENV:-}"
RUN_ID="${RUN_ID:-phi35_uavon_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/results/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${RUN_ID}}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
BASE_PORT="${BASE_PORT:-31700}"
SAVE_STEP_IMAGES="${SAVE_STEP_IMAGES:-1}"
IMAGE_SAVE_STRIDE="${IMAGE_SAVE_STRIDE:-1}"
IMAGE_QUALITY="${IMAGE_QUALITY:-85}"
INFERENCE_MODE="${INFERENCE_MODE:-generate}"
DEPTH_AVOIDANCE="${DEPTH_AVOIDANCE:-uavon_single_view_prompt}"
DEPTH_GRID_SIZE="${DEPTH_GRID_SIZE:-3}"
DEPTH_MAX_METERS="${DEPTH_MAX_METERS:-100.0}"
DEPTH_FORWARD_THRESHOLD="${DEPTH_FORWARD_THRESHOLD:-4.0}"
DEPTH_TURN_THRESHOLD="${DEPTH_TURN_THRESHOLD:-1.5}"
DEPTH_DESCEND_THRESHOLD="${DEPTH_DESCEND_THRESHOLD:-6.0}"
DEPTH_ASCEND_TOP_THRESHOLD="${DEPTH_ASCEND_TOP_THRESHOLD:-8.0}"
ACTION_REDIRECT="${ACTION_REDIRECT:-none}"
ACTION_REDIRECT_SEARCH_RADIUS="${ACTION_REDIRECT_SEARCH_RADIUS:-50.0}"
ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD="${ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD:-2.0}"
MEMORY_CONTEXT="${MEMORY_CONTEXT:-uavon_pose_history}"
MEMORY_HISTORY_SIZE="${MEMORY_HISTORY_SIZE:-5}"
MEMORY_SEARCH_RADIUS="${MEMORY_SEARCH_RADIUS:-50.0}"
MEMORY_INCLUDE_SEARCH_BOUNDS="${MEMORY_INCLUDE_SEARCH_BOUNDS:-0}"
MEMORY_POSE_YAW_UNIT="${MEMORY_POSE_YAW_UNIT:-radians}"
FIX_VERTICAL_ACTIONS="${FIX_VERTICAL_ACTIONS:-1}"
FIX_YAW_ACTIONS="${FIX_YAW_ACTIONS:-1}"
POSE_WAIT_TIMEOUT="${POSE_WAIT_TIMEOUT:-1.0}"
POSE_WAIT_POSITION_TOL="${POSE_WAIT_POSITION_TOL:-0.2}"
POSE_WAIT_YAW_TOL="${POSE_WAIT_YAW_TOL:-0.05}"
POSE_WAIT_POLL_INTERVAL="${POSE_WAIT_POLL_INTERVAL:-0.05}"
RENDER_SETTLE_SECONDS="${RENDER_SETTLE_SECONDS:-0.0}"
ACTION_EXECUTION_MODE="${ACTION_EXECUTION_MODE:-apex_join}"
ACTION_SIM_FRAMES="${ACTION_SIM_FRAMES:-150}"
ACTION_VELOCITY="${ACTION_VELOCITY:-2.0}"
ACTION_MOVE_TIMEOUT="${ACTION_MOVE_TIMEOUT:-5.0}"
ACTION_ROTATE_TIMEOUT="${ACTION_ROTATE_TIMEOUT:-3.0}"
LEVEL_AFTER_ACTION="${LEVEL_AFTER_ACTION:-0}"
LEVEL_SETTLE_FRAMES="${LEVEL_SETTLE_FRAMES:-1}"
INITIAL_POSE_RETRIES="${INITIAL_POSE_RETRIES:-3}"
INITIAL_POSE_SETTLE_FRAMES="${INITIAL_POSE_SETTLE_FRAMES:-1}"
ZERO_KINEMATICS_RESET="${ZERO_KINEMATICS_RESET:-1}"
CLIENT_RESET_PER_EPISODE="${CLIENT_RESET_PER_EPISODE:-1}"
KILL_ENV_PROCESS="${KILL_ENV_PROCESS:-0}"
LANE_GPUS="${LANE_GPUS:-3,4,6,7}"

IFS=',' read -r -a LANE_GPU_ARRAY <<< "${LANE_GPUS}"
LANE_COUNT="${#LANE_GPU_ARRAY[@]}"
if [[ "${LANE_COUNT}" != "2" && "${LANE_COUNT}" != "3" && "${LANE_COUNT}" != "4" && "${LANE_COUNT}" != "5" ]]; then
  echo "LANE_GPUS must contain two, three, four, or five comma-separated GPU ids, got: ${LANE_GPUS}" >&2
  exit 1
fi
for GPU in "${LANE_GPU_ARRAY[@]}"; do
  if [[ -z "${GPU}" ]]; then
    echo "LANE_GPUS contains an empty GPU id: ${LANE_GPUS}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cat > "${RUN_DIR}/lanes.tsv" <<'EOF'
EOF
if [[ "${LANE_COUNT}" == "2" ]]; then
  printf "lane0\t%s\tNYC_test,WinterTown_test,BrushifyRoad_test,UrbanJapan_test,CityPark_test,Venice_test,WesternTown_test\n" "${LANE_GPU_ARRAY[0]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tSlum_test,CityStreet_test,Barnyard_test,BrushifyUrban_test,CabinLake_test,ModularNeighborhood_test,DownTown_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
elif [[ "${LANE_COUNT}" == "3" ]]; then
  printf "lane0\t%s\tNYC_test,WinterTown_test,UrbanJapan_test,WesternTown_test\n" "${LANE_GPU_ARRAY[0]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tSlum_test,Barnyard_test,BrushifyUrban_test,CabinLake_test,DownTown_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane2\t%s\tCityStreet_test,BrushifyRoad_test,CityPark_test,ModularNeighborhood_test,Venice_test\n" "${LANE_GPU_ARRAY[2]}" >> "${RUN_DIR}/lanes.tsv"
elif [[ "${LANE_COUNT}" == "4" ]]; then
  printf "lane0\t%s\tNYC_test,UrbanJapan_test,DownTown_test\n" "${LANE_GPU_ARRAY[0]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tSlum_test,BrushifyUrban_test,CabinLake_test,Venice_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane2\t%s\tCityStreet_test,BrushifyRoad_test,CityPark_test,WesternTown_test\n" "${LANE_GPU_ARRAY[2]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane3\t%s\tBarnyard_test,WinterTown_test,ModularNeighborhood_test\n" "${LANE_GPU_ARRAY[3]}" >> "${RUN_DIR}/lanes.tsv"
else
  # Balanced by measured step workload from the current 1,000-task reference run.
  printf "lane0\t%s\tNYC_test,CabinLake_test\n" "${LANE_GPU_ARRAY[0]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane1\t%s\tBarnyard_test,UrbanJapan_test,DownTown_test\n" "${LANE_GPU_ARRAY[1]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane2\t%s\tBrushifyRoad_test,CityStreet_test,WesternTown_test\n" "${LANE_GPU_ARRAY[2]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane3\t%s\tSlum_test,BrushifyUrban_test,ModularNeighborhood_test\n" "${LANE_GPU_ARRAY[3]}" >> "${RUN_DIR}/lanes.tsv"
  printf "lane4\t%s\tCityPark_test,WinterTown_test,Venice_test\n" "${LANE_GPU_ARRAY[4]}" >> "${RUN_DIR}/lanes.tsv"
fi

cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "lane_count": ${LANE_COUNT},
  "nvidia_compat_lib": "${NVIDIA_COMPAT_LIB}",
  "model_path": "${MODEL_PATH}",
  "eval_dataset": "${EVAL_DATASET}",
  "eval_samples_per_env": ${EVAL_SAMPLES_PER_ENV:-null},
  "eval_max_steps": ${EVAL_MAX_STEPS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "base_port": ${BASE_PORT},
  "save_step_images": ${SAVE_STEP_IMAGES},
  "image_save_stride": ${IMAGE_SAVE_STRIDE},
  "image_quality": ${IMAGE_QUALITY},
  "inference_mode": "${INFERENCE_MODE}",
  "depth_avoidance": "${DEPTH_AVOIDANCE}",
  "depth_grid_size": ${DEPTH_GRID_SIZE},
  "depth_max_meters": ${DEPTH_MAX_METERS},
  "depth_forward_threshold": ${DEPTH_FORWARD_THRESHOLD},
  "depth_turn_threshold": ${DEPTH_TURN_THRESHOLD},
  "depth_descend_threshold": ${DEPTH_DESCEND_THRESHOLD},
  "depth_ascend_top_threshold": ${DEPTH_ASCEND_TOP_THRESHOLD},
  "action_redirect": "${ACTION_REDIRECT}",
  "action_redirect_search_radius": ${ACTION_REDIRECT_SEARCH_RADIUS},
  "action_redirect_near_obstacle_threshold": ${ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD},
  "memory_context": "${MEMORY_CONTEXT}",
  "memory_history_size": ${MEMORY_HISTORY_SIZE},
  "memory_search_radius": ${MEMORY_SEARCH_RADIUS},
  "memory_include_search_bounds": ${MEMORY_INCLUDE_SEARCH_BOUNDS},
  "memory_pose_yaw_unit": "${MEMORY_POSE_YAW_UNIT}",
  "fix_vertical_actions": ${FIX_VERTICAL_ACTIONS},
  "fix_yaw_actions": ${FIX_YAW_ACTIONS},
  "pose_wait_timeout": ${POSE_WAIT_TIMEOUT},
  "pose_wait_position_tol": ${POSE_WAIT_POSITION_TOL},
  "pose_wait_yaw_tol": ${POSE_WAIT_YAW_TOL},
  "pose_wait_poll_interval": ${POSE_WAIT_POLL_INTERVAL},
  "render_settle_seconds": ${RENDER_SETTLE_SECONDS},
  "action_execution_mode": "${ACTION_EXECUTION_MODE}",
  "action_sim_frames": ${ACTION_SIM_FRAMES},
  "action_velocity": ${ACTION_VELOCITY},
  "action_move_timeout": ${ACTION_MOVE_TIMEOUT},
  "action_rotate_timeout": ${ACTION_ROTATE_TIMEOUT},
  "level_after_action": ${LEVEL_AFTER_ACTION},
  "level_settle_frames": ${LEVEL_SETTLE_FRAMES},
  "initial_pose_retries": ${INITIAL_POSE_RETRIES},
  "initial_pose_settle_frames": ${INITIAL_POSE_SETTLE_FRAMES},
  "zero_kinematics_reset": ${ZERO_KINEMATICS_RESET},
  "client_reset_per_episode": ${CLIENT_RESET_PER_EPISODE},
  "kill_env_process": ${KILL_ENV_PROCESS}
}
EOF

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/eval:${PYTHONPATH:-}"
if [[ "${KILL_ENV_PROCESS}" == "1" ]]; then
python - <<'PY'
from eval_utils import kill_all_env_process
kill_all_env_process()
PY
fi

while IFS="$(printf '\t')" read -r LANE GPU SCENES; do
  PORT=$((BASE_PORT + GPU))
  SESSION="${RUN_ID}_${LANE}"
  OUT="${RUN_DIR}/${LANE}"
  LOG="${LOG_DIR}/${LANE}.log"
  mkdir -p "${OUT}"

  IMAGE_ARGS=""
  if [[ "${SAVE_STEP_IMAGES}" == "1" ]]; then
    IMAGE_ARGS="--save_step_images --image_save_stride ${IMAGE_SAVE_STRIDE} --image_format jpg --image_quality ${IMAGE_QUALITY}"
  fi
  FIX_ARGS=""
  if [[ "${FIX_VERTICAL_ACTIONS}" == "1" ]]; then
    FIX_ARGS="--fix_vertical_actions"
  fi
	  if [[ "${FIX_YAW_ACTIONS}" == "1" ]]; then
	    FIX_ARGS="${FIX_ARGS} --fix_yaw_actions"
	  fi
	  LEVEL_ARGS=""
	  if [[ "${LEVEL_AFTER_ACTION}" == "1" ]]; then
	    LEVEL_ARGS="--level_after_action"
	  fi
	  ZERO_KINEMATICS_ARGS="--zero_kinematics_reset"
	  if [[ "${ZERO_KINEMATICS_RESET}" == "0" ]]; then
	    ZERO_KINEMATICS_ARGS="--no-zero_kinematics_reset"
	  fi
  CLIENT_RESET_ARGS=""
	  if [[ "${CLIENT_RESET_PER_EPISODE}" == "1" ]]; then
	    CLIENT_RESET_ARGS="--client_reset_per_episode"
	  fi
  SAMPLE_ARGS=""
  if [[ -n "${EVAL_SAMPLES_PER_ENV}" ]]; then
    SAMPLE_ARGS="--eval_samples_per_env ${EVAL_SAMPLES_PER_ENV}"
  fi
  COMPAT_EXPORT=""
  if [[ -n "${NVIDIA_COMPAT_LIB}" ]]; then
    COMPAT_EXPORT="export LD_LIBRARY_PATH='${NVIDIA_COMPAT_LIB}':\${LD_LIBRARY_PATH:-} &&"
  fi

	  CMD=$(cat <<EOF
	cd '${ROOT}' &&
	source '${CONDA_SH}' &&
	conda activate '${CONDA_ENV}' &&
	${COMPAT_EXPORT}
	export CUDA_VISIBLE_DEVICES='${GPU}' &&
export PYTHONUNBUFFERED=1 &&
export TOKENIZERS_PARALLELISM=false &&
export PYTHONPATH='${ROOT}/src:${ROOT}:${ROOT}/eval':\${PYTHONPATH:-} &&
python -u eval/eval_phi35_uavon.py \
  --model_path '${MODEL_PATH}' \
  --eval_dataset '${EVAL_DATASET}' \
  ${SAMPLE_ARGS} \
  --output_foler '${OUT}' \
  --eval_max_steps '${EVAL_MAX_STEPS}' \
  --airsim_default_port '${PORT}' \
  --simulator_gpu '${GPU}' \
  --device cuda:0 \
  --max_new_tokens '${MAX_NEW_TOKENS}' \
  --inference_mode '${INFERENCE_MODE}' \
  --depth_avoidance '${DEPTH_AVOIDANCE}' \
  --depth_grid_size '${DEPTH_GRID_SIZE}' \
  --depth_max_meters '${DEPTH_MAX_METERS}' \
  --depth_forward_threshold '${DEPTH_FORWARD_THRESHOLD}' \
  --depth_turn_threshold '${DEPTH_TURN_THRESHOLD}' \
  --depth_descend_threshold '${DEPTH_DESCEND_THRESHOLD}' \
  --depth_ascend_top_threshold '${DEPTH_ASCEND_TOP_THRESHOLD}' \
  --action_redirect '${ACTION_REDIRECT}' \
  --action_redirect_search_radius '${ACTION_REDIRECT_SEARCH_RADIUS}' \
  --action_redirect_near_obstacle_threshold '${ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD}' \
  --memory_context '${MEMORY_CONTEXT}' \
  --memory_history_size '${MEMORY_HISTORY_SIZE}' \
  --memory_search_radius '${MEMORY_SEARCH_RADIUS}' \
  --memory_include_search_bounds '${MEMORY_INCLUDE_SEARCH_BOUNDS}' \
  --memory_pose_yaw_unit '${MEMORY_POSE_YAW_UNIT}' \
  --scene_list '${SCENES}' \
  --skip_kill_env_process \
  ${FIX_ARGS} \
  --pose_wait_timeout '${POSE_WAIT_TIMEOUT}' \
  --pose_wait_position_tol '${POSE_WAIT_POSITION_TOL}' \
  --pose_wait_yaw_tol '${POSE_WAIT_YAW_TOL}' \
	  --pose_wait_poll_interval '${POSE_WAIT_POLL_INTERVAL}' \
	  --render_settle_seconds '${RENDER_SETTLE_SECONDS}' \
	  --action_execution_mode '${ACTION_EXECUTION_MODE}' \
	  --action_sim_frames '${ACTION_SIM_FRAMES}' \
	  --action_velocity '${ACTION_VELOCITY}' \
	  --action_move_timeout '${ACTION_MOVE_TIMEOUT}' \
	  --action_rotate_timeout '${ACTION_ROTATE_TIMEOUT}' \
	  ${LEVEL_ARGS} \
	  --level_settle_frames '${LEVEL_SETTLE_FRAMES}' \
	  --initial_pose_retries '${INITIAL_POSE_RETRIES}' \
  --initial_pose_settle_frames '${INITIAL_POSE_SETTLE_FRAMES}' \
  ${ZERO_KINEMATICS_ARGS} \
  ${CLIENT_RESET_ARGS} \
  ${IMAGE_ARGS} \
  2>&1 | tee '${LOG}'
EOF
)

  screen -dmS "${SESSION}" bash -lc "${CMD}"
  printf "%s\tgpu=%s\tport=%s\tout=%s\tlog=%s\tscenes=%s\n" "${SESSION}" "${GPU}" "${PORT}" "${OUT}" "${LOG}" "${SCENES}" | tee -a "${RUN_DIR}/sessions.tsv"
done < "${RUN_DIR}/lanes.tsv"

printf "%s\n" "${RUN_ID}" > "${ROOT}/logs/current_eval_run.txt"
printf "run_id=%s\nrun_dir=%s\nlog_dir=%s\n" "${RUN_ID}" "${RUN_DIR}" "${LOG_DIR}"
screen -ls || true

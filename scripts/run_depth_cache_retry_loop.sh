#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/data/zhujd/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-octmem_openvla_nomemory}"
SOURCE="${SOURCE:-${REPO_ROOT}/UAV-ON_dataset/processed/nomemory_baseline/train_frames.jsonl}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/UAV-ON_dataset/processed/depth_grid_cache/train}"
ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-1200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"

usage() {
  echo "Usage: $0 --scene SCENE --gpu GPU --log LOG_PATH" >&2
}

SCENE=""
GPU=""
LOG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      SCENE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --log)
      LOG_PATH="$2"
      shift 2
      ;;
    --attempt-timeout)
      ATTEMPT_TIMEOUT="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${SCENE}" || -z "${GPU}" || -z "${LOG_PATH}" ]]; then
  usage
  exit 2
fi

mkdir -p "$(dirname "${LOG_PATH}")"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROOT}/eval:${PYTHONPATH:-}"

missing_count() {
  SCENE="${SCENE}" SOURCE="${SOURCE}" CACHE_DIR="${CACHE_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

scene = os.environ["SCENE"]
source = Path(os.environ["SOURCE"])
cache = Path(os.environ["CACHE_DIR"]) / f"{scene}.jsonl"

expected = set()
with source.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row["scene_id"] != scene:
            continue
        expected.add(f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}")

valid = set()
if cache.exists():
    with cache.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("depth_grid") is not None and row.get("key"):
                valid.add(str(row["key"]))

print(len(expected - valid))
PY
}

attempt=0
while true; do
  missing="$(missing_count)"
  echo "[$(date '+%F %T')] scene=${SCENE} gpu=${GPU} missing=${missing}" | tee -a "${LOG_PATH}"
  if [[ "${missing}" == "0" ]]; then
    echo "[$(date '+%F %T')] scene=${SCENE} complete" | tee -a "${LOG_PATH}"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] scene=${SCENE} attempt=${attempt} timeout=${ATTEMPT_TIMEOUT}s start" | tee -a "${LOG_PATH}"
  set +e
  timeout --signal=TERM --kill-after=30s "${ATTEMPT_TIMEOUT}s" \
    python "${ROOT}/scripts/capture_depth_grid_cache.py" --gpu "${GPU}" --scene-list "${SCENE}" \
    >> "${LOG_PATH}" 2>&1
  status=$?
  set -e
  echo "[$(date '+%F %T')] scene=${SCENE} attempt=${attempt} exit_status=${status}" | tee -a "${LOG_PATH}"
  sleep "${SLEEP_SECONDS}"
done

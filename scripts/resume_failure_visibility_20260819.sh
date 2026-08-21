#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zhujd/Aerial-ObjectNav/VLM-baseline"
PYTHON="/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python"
SOURCE="$ROOT/results/phi35_stopbank_cfmem_v2_ckpt19764_full_20260814_162448/all_episodes.jsonl"
OUTPUT="$ROOT/results/phi35_stopbank_cfmem_v2_ckpt19764_full_20260814_162448/failure_target_visibility_full_20260819_121809"
LOG_ROOT="$ROOT/logs"

screen_running() {
  screen -list 2>/dev/null | grep -q "\.$1[[:space:]]"
}

run_gpu1_phase2() {
  while screen_running "failure_visibility_gpu1_20260819_121809"; do
    printf '[%s] GPU1 waiting for phase 1\n' "$(date '+%F %T')"
    sleep 60
  done
  printf '[%s] GPU1 starting resumable phase 2\n' "$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
    "$ROOT/scripts/audit_inference_failure_visibility.py" \
    --all-episodes "$SOURCE" \
    --output-dir "$OUTPUT" \
    --scene-list NYC_test,Barnyard_test,CityPark_test,CityStreet_test,UrbanJapan_test,CabinLake_test,WesternTown_test \
    --gpu 1 \
    --resume \
    --retry-errors \
    > "$LOG_ROOT/failure_target_visibility_gpu1_phase2_20260819_121809.log" 2>&1
}

run_gpu3_phase2() {
  while screen_running "failure_visibility_gpu3_20260819_121809"; do
    printf '[%s] GPU3 waiting for phase 1\n' "$(date '+%F %T')"
    sleep 60
  done
  printf '[%s] GPU3 starting resumable phase 2\n' "$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" \
    "$ROOT/scripts/audit_inference_failure_visibility.py" \
    --all-episodes "$SOURCE" \
    --output-dir "$OUTPUT" \
    --scene-list Slum_test,BrushifyRoad_test,WinterTown_test,BrushifyUrban_test,ModularNeighborhood_test,DownTown_test,Venice_test \
    --gpu 3 \
    --resume \
    --retry-errors \
    > "$LOG_ROOT/failure_target_visibility_gpu3_phase2_20260819_121809.log" 2>&1
}

run_gpu1_phase2 &
pid1=$!
run_gpu3_phase2 &
pid3=$!

status=0
wait "$pid1" || status=$?
wait "$pid3" || status=$?

"$PYTHON" "$ROOT/scripts/audit_inference_failure_visibility.py" \
  --all-episodes "$SOURCE" \
  --output-dir "$OUTPUT" \
  --summarize-only \
  > "$LOG_ROOT/failure_target_visibility_summary_20260819_121809.log" 2>&1

printf '[%s] phase 2 finished with status=%s\n' "$(date '+%F %T')" "$status"
exit "$status"

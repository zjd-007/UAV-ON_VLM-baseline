#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zhujd/Aerial-ObjectNav"
PROJECT="$ROOT/VLM-baseline"
DATASET="$ROOT/UAV-ON_dataset"
REPAIR_ROOT="$DATASET/processed/neighborhood_coordinate_repair_v1_20260812_194807"
SOURCE="$REPAIR_ROOT/final_dataset_per_frame_safe_v1"
STRICT="$REPAIR_ROOT/final_dataset"
OUTPUT="$REPAIR_ROOT/final_dataset_per_frame_safe_stopbank_v1"
STATE="$OUTPUT/.state"
LOG_ROOT="$PROJECT/logs/actor_stop_bank_v1_20260813"
PY="/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python"

mkdir -p "$OUTPUT" "$STATE" "$LOG_ROOT"
exec 9>"$STATE/pipeline.lock"
if ! flock -n 9; then
  echo "Another actor-stop-bank pipeline instance is running." >&2
  exit 3
fi

done_stage() { [[ -f "$STATE/$1.done" ]]; }
mark_done() { printf '%(%F %T)T\n' -1 >"$STATE/$1.done"; }
log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_ROOT/pipeline.log"; }

if ! done_stage build; then
  log "building diverse actor Stop banks and appending matched Stop rows"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/append_actor_stop_bank.py" \
    --source "$SOURCE/train_frames.jsonl" \
    --source-quarantine "$SOURCE/quarantine_trajectories.jsonl" \
    --depth-cache "$SOURCE/depth_grid_cache" \
    --output "$OUTPUT/train_frames.jsonl" \
    --quarantine-output "$OUTPUT/quarantine_trajectories.jsonl" \
    --bank-output "$OUTPUT/actor_stop_bank.jsonl" \
    --assignments-output "$OUTPUT/actor_stop_bank_assignments.jsonl" \
    --unresolved-output "$OUTPUT/unresolved_navigation_only_actors.jsonl" \
    --manifest "$OUTPUT/manifest.json" \
    --bank-size 5 \
    >>"$LOG_ROOT/build.log" 2>&1
  ln -s "$SOURCE/allowed_uncovered_actors.jsonl" \
    "$OUTPUT/allowed_uncovered_actors.jsonl"
  mark_done build
fi

if ! done_stage validate; then
  log "validating Stop, image, inline depth, and episode invariants"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/validate_stop_visible_v4_production.py" \
    --frames "$OUTPUT/train_frames.jsonl" \
    --quarantine "$OUTPUT/quarantine_trajectories.jsonl" \
    --allowed-uncovered-actors "$OUTPUT/allowed_uncovered_actors.jsonl" \
    --allowed-image-root "$REPAIR_ROOT/record_output/images" \
    --report "$OUTPUT/validation_report.json" \
    >>"$LOG_ROOT/validate.log" 2>&1
  mark_done validate
fi

if ! done_stage sft; then
  log "building depth-grid SFT JSONL"
  mkdir -p "$OUTPUT/depth_grid_cache"
  for path in "$SOURCE/depth_grid_cache"/*.jsonl; do
    name=$(basename "$path")
    [[ -e "$OUTPUT/depth_grid_cache/$name" ]] || \
      ln -s "$path" "$OUTPUT/depth_grid_cache/$name"
  done
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/prepare_collision_filtered_depth_sft_data.py" \
    --source "$OUTPUT/train_frames.jsonl" \
    --output "$OUTPUT/uavon_phi35_sft_depth_grid_stop_visible_v4_per_frame_safe_stopbank_v1.jsonl" \
    --manifest "$OUTPUT/sft_manifest.json" \
    --aligned-root "$REPAIR_ROOT/record_output" \
    --depth-cache "$OUTPUT/depth_grid_cache" \
    --original-collision-dir "$REPAIR_ROOT/collision_merged" \
    --repair-collision-dir "$REPAIR_ROOT/empty_collision_repair" \
    --missing-depth-policy error \
    --overwrite \
    >>"$LOG_ROOT/build_sft.log" 2>&1
  mark_done sft
fi

if ! done_stage pair_audit; then
  log "archiving Stop/previous pairs"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/archive_stop_visible_pairs.py" \
    --frames "$OUTPUT/train_frames.jsonl" \
    --output-dir "$OUTPUT/stop_pair_audit" \
    --page-size 50 \
    >>"$LOG_ROOT/archive_stop_pairs.log" 2>&1
  mark_done pair_audit
fi

if ! done_stage mask_audit; then
  log "rendering target mask boxes"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/annotate_stop_pair_archive.py" \
    --source-archive "$OUTPUT/stop_pair_audit" \
    --expert-mask-audit-dir "$STRICT/combined_visibility_masks" \
    --standoff-capture-dir "$DATASET/processed/stop_visible_v4_production_20260812/standoff_capture_repairable" \
    --standoff-capture-dir "$DATASET/processed/stop_visible_v4_production_20260812/standoff_capture_rescue" \
    --standoff-capture-dir "$DATASET/processed/stop_visible_v4_production_20260812/standoff_capture_rescue_mug_ultraclose" \
    --standoff-capture-dir "$REPAIR_ROOT/standoff_capture" \
    --output-dir "$OUTPUT/stop_pair_audit_maskboxed" \
    --page-size 50 \
    >>"$LOG_ROOT/annotate_stop_pairs.log" 2>&1
  mark_done mask_audit
fi

log "actor Stop bank v1 pipeline complete: $OUTPUT"

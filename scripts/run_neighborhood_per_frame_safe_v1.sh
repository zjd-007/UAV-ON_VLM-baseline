#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zhujd/Aerial-ObjectNav"
PROJECT="$ROOT/VLM-baseline"
DATASET="$ROOT/UAV-ON_dataset"
REPAIR_ROOT="$DATASET/processed/neighborhood_coordinate_repair_v1_20260812_194807"
STRICT="$REPAIR_ROOT/final_dataset"
OUTPUT="$REPAIR_ROOT/final_dataset_per_frame_safe_v1"
STATE="$OUTPUT/.state"
LOG_ROOT="$PROJECT/logs/neighborhood_per_frame_safe_v1_20260813"
PY="/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python"

mkdir -p "$OUTPUT" "$STATE" "$LOG_ROOT"
exec 9>"$STATE/pipeline.lock"
if ! flock -n 9; then
  echo "Another per-frame-safe pipeline instance is running." >&2
  exit 3
fi

done_stage() {
  [[ -f "$STATE/$1.done" ]]
}
mark_done() {
  printf '%(%F %T)T\n' -1 >"$STATE/$1.done"
}
log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_ROOT/pipeline.log"
}

if ! done_stage build; then
  log "building the independent per-frame-safe v1 dataset"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/build_per_frame_safe_salvage.py" \
    --strict-frames "$STRICT/train_frames.jsonl" \
    --strict-quarantine "$STRICT/quarantine_trajectories.jsonl" \
    --source-scene-frames "$REPAIR_ROOT/frames/train_frames.jsonl" \
    --collision-audit "$REPAIR_ROOT/collision_merged" \
    --visibility-audit "$REPAIR_ROOT/visibility_audit" \
    --semantic-scores "$REPAIR_ROOT/clip_scores/Neighborhood.jsonl" \
    --policy "$PROJECT/configs/stop_visible_v4_production_policy.json" \
    --stop-bank "$REPAIR_ROOT/standoff_stop_bank/train_frames_standoff_stop_bank.jsonl" \
    --queue-manifest "$REPAIR_ROOT/standoff_queue/standoff_queue.jsonl" \
    --scene Neighborhood \
    --output "$OUTPUT/train_frames.jsonl" \
    --quarantine-output "$OUTPUT/quarantine_trajectories.jsonl" \
    --decisions-output "$OUTPUT/salvage_decisions.jsonl" \
    --manifest "$OUTPUT/manifest.json" \
    >>"$LOG_ROOT/build.log" 2>&1
  ln -s "$STRICT/allowed_uncovered_actors.jsonl" \
    "$OUTPUT/allowed_uncovered_actors.jsonl"
  mark_done build
fi

if ! done_stage validate; then
  log "validating frame, Stop, image, and quarantine invariants"
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
  log "building the depth-grid single-frame SFT JSONL"
  mkdir -p "$OUTPUT/depth_grid_cache"
  for path in "$STRICT/depth_grid_cache"/*.jsonl; do
    name=$(basename "$path")
    [[ -e "$OUTPUT/depth_grid_cache/$name" ]] || \
      ln -s "$path" "$OUTPUT/depth_grid_cache/$name"
  done
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/prepare_collision_filtered_depth_sft_data.py" \
    --source "$OUTPUT/train_frames.jsonl" \
    --output "$OUTPUT/uavon_phi35_sft_depth_grid_stop_visible_v4_per_frame_safe_v1.jsonl" \
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
  log "archiving every Stop and its previous retained sample"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY" \
    "$PROJECT/scripts/archive_stop_visible_pairs.py" \
    --frames "$OUTPUT/train_frames.jsonl" \
    --output-dir "$OUTPUT/stop_pair_audit" \
    --page-size 50 \
    >>"$LOG_ROOT/archive_stop_pairs.log" 2>&1
  mark_done pair_audit
fi

if ! done_stage mask_audit; then
  log "rendering target-mask boxes for the Stop pair audit"
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

log "per-frame-safe v1 pipeline complete: $OUTPUT"

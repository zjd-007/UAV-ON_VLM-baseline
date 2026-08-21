#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/zhujd/Aerial-ObjectNav"
PROJECT="$ROOT/VLM-baseline"
DATASET="$ROOT/UAV-ON_dataset"
RUN_ID="neighborhood_coordinate_repair_v1_20260812_194807"
REPAIR_ROOT="$DATASET/processed/$RUN_ID"
LOG_ROOT="$PROJECT/logs/$RUN_ID"
STATE_DIR="$REPAIR_ROOT/.pipeline_state"
GPU=4

PY_OCT="/data/zhujd/miniconda3/envs/octmem_uavon/bin/python"
PY_AIR="/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python"
PY_CLIP="/data/zhujd/miniconda3/envs/ovdet/bin/python"

PLAN_ROOT="$REPAIR_ROOT/plans"
METADATA="$REPAIR_ROOT/metadata/train_xy_shifted.json"
RECORD_ROOT="$REPAIR_ROOT/record_output"
FRAME_DIR="$REPAIR_ROOT/frames"
SOURCE_FRAMES="$FRAME_DIR/train_frames.jsonl"
SHARD_DIR="$REPAIR_ROOT/frame_shards"
COLLISION_SHARDS="$REPAIR_ROOT/collision_shards"
COLLISION_PENDING="$REPAIR_ROOT/collision_pending_10way"
COLLISION_PENDING_SOURCE="$COLLISION_PENDING/pending_frames.jsonl"
COLLISION_SHARDS_10="$REPAIR_ROOT/collision_shards_10way"
COLLISION_MERGED="$REPAIR_ROOT/collision_merged"
AUDIT_DIR="$REPAIR_ROOT/visibility_audit"
ALIGNMENT_DIR="$AUDIT_DIR/actor_pose_alignment"
SEMANTIC_SCORES="$REPAIR_ROOT/clip_scores/Neighborhood.jsonl"
PREPARED_DIR="$REPAIR_ROOT/stop_visible_prepared_base"
QUEUE_DIR="$REPAIR_ROOT/standoff_queue"
STANDOFF_CAPTURE="$REPAIR_ROOT/standoff_capture"
STANDOFF_SCORES="$REPAIR_ROOT/clip_scores_standoff/Neighborhood.jsonl"
STOP_BANK="$REPAIR_ROOT/standoff_stop_bank"
NEIGHBORHOOD_ASSEMBLED="$REPAIR_ROOT/assembled_neighborhood"
FINAL_DIR="$REPAIR_ROOT/final_dataset"
COMBINED_CACHE="$FINAL_DIR/depth_grid_cache"

OLD_V4="$DATASET/processed/stop_visible_v4_production_20260812"
OLD_FRAMES="$OLD_V4/assembled_v4_appended/train_frames.jsonl"
OLD_QUARANTINE="$OLD_V4/prepared_base_v2/quarantine_trajectories.jsonl"
OLD_AUDIT="$DATASET/processed/stop_visible_full_audit/full_canonical_geometry_v1_20260812_153000"
OLD_DEPTH_CACHE="$DATASET/processed/depth_grid_cache/train"
OLD_UNCOVERED="$OLD_V4/unresolved_stop_coverage_actors.jsonl"
POLICY="$PROJECT/configs/stop_visible_v4_production_policy.json"

mkdir -p "$LOG_ROOT" "$STATE_DIR"
exec 9>"$STATE_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "Another coordinate-repair pipeline instance is already running." >&2
  exit 3
fi

PIPELINE_LOG="$LOG_ROOT/pipeline.log"
log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$PIPELINE_LOG"
}
done_stage() {
  [[ -f "$STATE_DIR/$1.done" ]]
}
mark_done() {
  printf '%(%F %T)T\n' -1 >"$STATE_DIR/$1.done"
}
archive_partial_dir() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mv "$path" "${path}.partial_$(date +%Y%m%d_%H%M%S)"
  fi
}
cleanup_port() {
  local port="$1"
  pkill -TERM -f "settings_512_${port}.json" 2>/dev/null || true
  sleep 1
  pkill -KILL -f "settings_512_${port}.json" 2>/dev/null || true
}
wait_for_gpu_idle() {
  local gpu_uuid
  gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}')
  [[ -n "$gpu_uuid" ]]
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q "$gpu_uuid"; do
    log "GPU$GPU is occupied by another compute process; waiting"
    sleep 60
  done
}
wait_for_gpu_ids_idle() {
  local gpu gpu_uuid
  for gpu in "$@"; do
    gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$gpu" '$1 == gpu {print $2}')
    [[ -n "$gpu_uuid" ]]
    while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -q "$gpu_uuid"; do
      log "GPU$gpu is occupied by another compute process; waiting"
      sleep 60
    done
  done
}

trap 'status=$?; log "pipeline stopped at line $LINENO with status=$status"; exit $status' ERR

if ! done_stage astar; then
  log "waiting for the isolated 850-target A* planning screen"
  while screen -ls 2>/dev/null | grep -q '[.]neighborhood_repair_astar_20260812_194807'; do
    progress=$(grep '"event": "plan_progress"' "$LOG_ROOT/astar.log" 2>/dev/null | tail -n 1 || true)
    [[ -n "$progress" ]] && log "A* $progress"
    sleep 60
  done
  [[ -f "$PLAN_ROOT/manifest.json" ]]
  "$PY_OCT" - "$PLAN_ROOT/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest["errors"] == 0, manifest
assert manifest["successful_plans"] > 0, manifest
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
  plan_count=$(find "$PLAN_ROOT/Neighborhood" -mindepth 2 -maxdepth 2 \
    -type f -name '*.json' | wc -l)
  if (( plan_count < 848 )); then
    recovery_output="$PLAN_ROOT/recovery_manifest.json"
    recovery_log="$LOG_ROOT/astar_recovery.log"
    recovery_radius=9
    if [[ -e "$recovery_output" ]]; then
      recovery_output="$PLAN_ROOT/recovery_manifest_v2.json"
      recovery_log="$LOG_ROOT/astar_recovery_v2.log"
      recovery_radius=30
      log "9m start recovery left $((850 - plan_count)) tasks; expanding to 30m"
    fi
    rm -f "$recovery_output"
    OCTMEN_ASTAR_TIMEOUT_SECONDS=300 "$PY_OCT" \
      "$PROJECT/scripts/recover_neighborhood_astar_no_path.py" \
      --metadata "$METADATA" \
      --plan-root "$PLAN_ROOT" \
      --output "$recovery_output" \
      --max-horizontal-m "$recovery_radius" \
      --minimum-plans 848 \
      >>"$recovery_log" 2>&1
  fi
  "$PY_OCT" - "$PLAN_ROOT" "$METADATA" <<'PY'
import json, pathlib, sys
root, metadata = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected = set()
for row in json.load(open(metadata)):
    scene = str(row.get("scene_key") or row.get("map_name", ""))
    if scene.replace("_TrainSets", "") not in {"Neighborhood", "NeighborhoodTrain", "ModularNeighborhood"}:
        continue
    for pose_idx, _ in enumerate(row.get("pose") or []):
        expected.add(f"Neighborhood::{row['episode_id']}::{pose_idx}")
present = {
    f"Neighborhood::{path.parent.name}::{path.stem}"
    for path in (root / "Neighborhood").glob("*/*.json")
}
legacy = {"Neighborhood::373::3", "Neighborhood::374::3"}
missing = expected - present
assert len(present) >= 848, len(present)
assert missing <= legacy, sorted(missing)
print({"validated_plans": len(present), "missing_legacy": sorted(missing)})
PY
  mark_done astar
  log "A* planning validated"
fi

if ! done_stage recording; then
  for port in 30204 30205 30206 30207; do cleanup_port "$port"; done
  wait_for_gpu_idle
  log "GPU$GPU is free; starting RGB+Depth recording"
  mkdir -p "$RECORD_ROOT"
  recording_ok=0
  for attempt in 1 2 3 4 5; do
    log "RGB+Depth recording attempt $attempt/5 on GPU$GPU"
    rm -f "$REPAIR_ROOT/record_validation_attempt${attempt}.json"
    set +e
    (
      cd "$ROOT/octmem_nomemory_repro/octmen-agent/tools"
      OCTMEN_GPU_IDS="$GPU" \
      OCTMEN_AIRSIM_READY_TIMEOUT=300 \
      OCTMEN_AIRSIM_STABLE_SECONDS=15 \
      OCTMEN_RECORD_RETRIES=5 \
      timeout --signal=TERM --kill-after=30s 5h \
        "$PY_OCT" traj_gen/record_traj.py \
          --env Neighborhood \
          --base_folder "$PLAN_ROOT" \
          --output_folder "$RECORD_ROOT" \
          --capture_depth
    ) >>"$LOG_ROOT/record_attempt${attempt}.log" 2>&1
    record_status=$?
    set -e
    for port in 30204 30205 30206 30207; do cleanup_port "$port"; done

    set +e
    PYTHONPATH="$PROJECT/src" "$PY_AIR" \
      "$PROJECT/scripts/validate_coordinate_repair_recording.py" \
      --plan-root "$PLAN_ROOT" \
      --record-root "$RECORD_ROOT" \
      --scene Neighborhood \
      --require-depth \
      --clean-invalid \
      --report "$REPAIR_ROOT/record_validation_attempt${attempt}.json" \
      >>"$LOG_ROOT/record_validation_attempt${attempt}.log" 2>&1
    validation_status=$?
    set -e
    if [[ $validation_status -eq 0 ]]; then
      recording_ok=1
      log "recording validation passed on attempt $attempt (record command status=$record_status)"
      break
    fi
    log "recording attempt $attempt incomplete; invalid outputs were removed for resume"
  done
  [[ $recording_ok -eq 1 ]]
  mark_done recording
fi

if ! done_stage frames; then
  if [[ -f "$FRAME_DIR/manifest.json" ]]; then
    mark_done frames
  elif [[ -d "$FRAME_DIR" ]]; then
    archive_partial_dir "$FRAME_DIR"
  fi
  if ! done_stage frames; then
    log "building aligned frame labels and 3x3 DepthGrid from synchronized captures"
    PYTHONPATH="$PROJECT/src" "$PY_AIR" \
      "$PROJECT/scripts/build_neighborhood_coordinate_repair_frames.py" \
      --record-root "$RECORD_ROOT" \
      --metadata "$METADATA" \
      --output-dir "$FRAME_DIR" \
      >>"$LOG_ROOT/build_frames.log" 2>&1
    mark_done frames
  fi
fi

if ! done_stage collision_smoke; then
  cleanup_port 44004
  wait_for_gpu_idle
  archive_partial_dir "$REPAIR_ROOT/collision_smoke"
  mkdir -p "$REPAIR_ROOT/collision_smoke"
  log "running a 20-frame exact action-collision smoke on GPU$GPU"
  PYTHONPATH="$PROJECT/src" timeout --signal=TERM --kill-after=30s 30m \
    "$PY_AIR" "$PROJECT/scripts/check_train_label_action_collision.py" \
    --source "$SOURCE_FRAMES" \
    --aligned-root "$RECORD_ROOT" \
    --output-dir "$REPAIR_ROOT/collision_smoke" \
    --scene-list Neighborhood \
    --gpu "$GPU" \
    --base-port 44000 \
    --limit 20 \
    --settle-frames 1 \
    --action-velocity 2.0 \
    --action-move-timeout 5.0 \
    --action-rotate-timeout 3.0 \
    --fix-vertical-actions \
    --fix-yaw-actions \
    >>"$LOG_ROOT/collision_smoke.log" 2>&1
  cleanup_port 44004
  "$PY_AIR" - "$REPAIR_ROOT/collision_smoke/Neighborhood.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert len(rows) == 20, len(rows)
assert not [row for row in rows if row.get("error")], rows
print({"rows": len(rows), "new_collisions": sum(bool(r.get("new_collision_after_action")) for r in rows)})
PY
  mark_done collision_smoke
fi

if ! done_stage shards; then
  if [[ -f "$SHARD_DIR/manifest.json" ]]; then
    mark_done shards
  elif [[ -d "$SHARD_DIR" ]]; then
    archive_partial_dir "$SHARD_DIR"
  fi
  if ! done_stage shards; then
    "$PY_AIR" "$PROJECT/scripts/split_frame_jsonl_by_episode.py" \
      --source "$SOURCE_FRAMES" \
      --output-dir "$SHARD_DIR" \
      --shards 4 \
      >>"$LOG_ROOT/split_frames.log" 2>&1
    mark_done shards
  fi
fi

if ! done_stage collision; then
  collision_gpus=(0 0 1 1 2 2 3 3 4 4)
  if ! done_stage collision_10way_inputs; then
    archive_partial_dir "$COLLISION_PENDING"
    archive_partial_dir "$COLLISION_SHARDS_10"
    mkdir -p "$COLLISION_PENDING" "$COLLISION_SHARDS_10"
    "$PY_AIR" "$PROJECT/scripts/build_pending_collision_source.py" \
      --source "$SOURCE_FRAMES" \
      --audit-root "$COLLISION_SHARDS" \
      --output "$COLLISION_PENDING_SOURCE" \
      --manifest "$COLLISION_PENDING/manifest.json" \
      >>"$LOG_ROOT/collision_pending_10way.log" 2>&1
    "$PY_AIR" "$PROJECT/scripts/split_frame_jsonl_by_episode.py" \
      --source "$COLLISION_PENDING_SOURCE" \
      --output-dir "$COLLISION_PENDING/shards" \
      --shards 10 \
      >>"$LOG_ROOT/collision_split_10way.log" 2>&1
    mark_done collision_10way_inputs
  fi

  for lane in $(seq 0 9); do
    gpu=${collision_gpus[$lane]}
    cleanup_port "$((46200 + lane * 100 + gpu))"
  done
  wait_for_gpu_ids_idle 0 1 2 3 4
  mkdir -p "$COLLISION_SHARDS_10"
  collision_ok=0
  for attempt in 1 2 3 4 5; do
    log "full exact collision audit attempt $attempt/5: 10 lanes use GPUs 0-4, two lanes per GPU"
    pids=()
    ports=()
    for lane in $(seq 0 9); do
      gpu=${collision_gpus[$lane]}
      base_port=$((46200 + lane * 100))
      port=$((base_port + gpu))
      ports+=("$port")
      mkdir -p "$COLLISION_SHARDS_10/lane${lane}"
      setsid env PYTHONPATH="$PROJECT/src" \
        timeout --signal=TERM --kill-after=30s 10h \
        "$PY_AIR" "$PROJECT/scripts/check_train_label_action_collision.py" \
        --source "$COLLISION_PENDING/shards/lane${lane}.jsonl" \
        --aligned-root "$RECORD_ROOT" \
        --output-dir "$COLLISION_SHARDS_10/lane${lane}" \
        --scene-list Neighborhood \
        --gpu "$gpu" \
        --base-port "$base_port" \
        --retry-errors \
        --settle-frames 1 \
        --action-velocity 2.0 \
        --action-move-timeout 5.0 \
        --action-rotate-timeout 3.0 \
        --fix-vertical-actions \
        --fix-yaw-actions \
        >>"$LOG_ROOT/collision_lane${lane}_attempt${attempt}.log" 2>&1 &
      pids+=("$!")
      sleep 10
    done

    last_counts=()
    idle_ticks=()
    for lane in $(seq 0 9); do
      last_counts+=("-1")
      idle_ticks+=("0")
    done
    stalled=0
    while true; do
      running=0
      counts=()
      for lane in $(seq 0 9); do
        file="$COLLISION_SHARDS_10/lane${lane}/Neighborhood.jsonl"
        count=$(wc -l <"$file" 2>/dev/null || echo 0)
        counts+=("lane${lane}=$count")
        if kill -0 "${pids[$lane]}" 2>/dev/null; then
          running=1
          if (( count > last_counts[lane] )); then
            idle_ticks[$lane]=0
          else
            idle_ticks[$lane]=$((idle_ticks[lane] + 1))
          fi
          if (( idle_ticks[lane] >= 5 )); then
            stalled=1
          fi
        fi
        last_counts[$lane]=$count
      done
      log "collision progress: ${counts[*]}"
      [[ $running -eq 0 ]] && break
      if [[ $stalled -eq 1 ]]; then
        log "collision attempt $attempt stalled for 5 minutes; restarting unresolved rows"
        for pid in "${pids[@]}"; do
          kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 10
        for pid in "${pids[@]}"; do
          kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        done
        break
      fi
      sleep 60
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done
    for port in "${ports[@]}"; do cleanup_port "$port"; done

    if [[ -d "$COLLISION_MERGED" ]]; then
      archive_partial_dir "$COLLISION_MERGED"
    fi
    mkdir -p "$COLLISION_MERGED"
    set +e
    "$PY_AIR" "$PROJECT/scripts/merge_collision_audit_shards.py" \
      --source "$SOURCE_FRAMES" \
      --shard-root "$COLLISION_SHARDS" \
      --shard-root "$COLLISION_SHARDS_10" \
      --output-dir "$COLLISION_MERGED" \
      --scene Neighborhood \
      >>"$LOG_ROOT/collision_merge_attempt${attempt}.log" 2>&1
    merge_status=$?
    set -e
    if [[ $merge_status -eq 0 ]]; then
      collision_ok=1
      break
    fi
    log "collision audit still has missing/error rows; retrying only unresolved keys"
  done
  [[ $collision_ok -eq 1 ]]
  mark_done collision
  log "full collision audit completed"
fi

if ! done_stage visibility; then
  cleanup_port 45504
  wait_for_gpu_idle
  visibility_ok=0
  for attempt in 1 2 3; do
    mkdir -p "$AUDIT_DIR"
    log "full synchronized target-visibility audit attempt $attempt/3 on GPU$GPU"
    set +e
    PYTHONPATH="$PROJECT/src" timeout --signal=TERM --kill-after=30s 6h \
      "$PY_AIR" "$PROJECT/scripts/audit_full_training_target_visibility.py" \
      --aligned-root "$RECORD_ROOT" \
      --metadata "$METADATA" \
      --output-dir "$AUDIT_DIR" \
      --scene-list Neighborhood \
      --gpu "$GPU" \
      --base-port 45500 \
      --distance-threshold 20 \
      --settle-frames 2 \
      --segmentation-settle-frames 4 \
      --save-replay-evidence \
      --resume \
      >>"$LOG_ROOT/visibility_attempt${attempt}.log" 2>&1
    visibility_status=$?
    set -e
    cleanup_port 45504
    if [[ $visibility_status -eq 0 ]]; then
      set +e
      "$PY_AIR" - "$PLAN_ROOT" "$AUDIT_DIR" <<'PY'
import json, pathlib, sys
plans = list((pathlib.Path(sys.argv[1]) / "Neighborhood").glob("*/*.json"))
rows = [json.loads(line) for line in open(pathlib.Path(sys.argv[2]) / "Neighborhood.jsonl") if line.strip()]
actors = [json.loads(line) for line in open(pathlib.Path(sys.argv[2]) / "Neighborhood_actors.jsonl") if line.strip()]
assert len(rows) == len(plans), (len(rows), len(plans))
assert not [row for row in rows if row.get("status") == "error"]
assert actors and not [row for row in actors if row.get("status") == "error"]
print({"trajectories": len(rows), "actors": len(actors)})
PY
      check_status=$?
      set -e
      if [[ $check_status -eq 0 ]]; then
        visibility_ok=1
        break
      fi
    fi
    mkdir -p "$AUDIT_DIR/failed_attempts/attempt${attempt}"
    for path in "$AUDIT_DIR/Neighborhood.jsonl" "$AUDIT_DIR/Neighborhood_actors.jsonl"; do
      [[ -f "$path" ]] && mv "$path" "$AUDIT_DIR/failed_attempts/attempt${attempt}/"
    done
  done
  [[ $visibility_ok -eq 1 ]]
  mark_done visibility
fi

if ! done_stage actor_alignment; then
  cleanup_port 45604
  wait_for_gpu_idle
  archive_partial_dir "$ALIGNMENT_DIR"
  mkdir -p "$ALIGNMENT_DIR"
  PYTHONPATH="$PROJECT/src" "$PY_AIR" \
    "$PROJECT/scripts/audit_target_actor_pose_alignment.py" \
    --input-dir "$AUDIT_DIR" \
    --output-dir "$ALIGNMENT_DIR" \
    --scene-list Neighborhood \
    --gpu "$GPU" \
    --base-port 45600 \
    >>"$LOG_ROOT/actor_alignment.log" 2>&1
  cleanup_port 45604
  "$PY_AIR" - "$ALIGNMENT_DIR/Neighborhood.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert rows and not [row for row in rows if row.get("status") != "ok"]
assert max(float(row["actor_to_target_error_xy_m"]) for row in rows) < 0.1
print({"actors": len(rows), "max_xy_error_m": max(float(row["actor_to_target_error_xy_m"]) for row in rows)})
PY
  mark_done actor_alignment
fi

if ! done_stage visibility_summary; then
  archive_partial_dir "$AUDIT_DIR/summary_collision_filtered"
  "$PY_AIR" "$PROJECT/scripts/summarize_full_training_target_visibility.py" \
    --input-dir "$AUDIT_DIR" \
    --output-dir "$AUDIT_DIR/summary_collision_filtered" \
    --actor-pose-alignment-dir "$ALIGNMENT_DIR" \
    >>"$LOG_ROOT/visibility_summary.log" 2>&1
  mark_done visibility_summary
fi

if ! done_stage semantic_scores; then
  wait_for_gpu_idle
  mkdir -p "$(dirname "$SEMANTIC_SCORES")"
  log "scoring synchronized visible crops with CLIP on GPU$GPU"
  PYTHONPATH="$PROJECT/src" "$PY_CLIP" \
    "$PROJECT/scripts/score_stop_visibility_clip.py" \
    --visibility-cache "$AUDIT_DIR" \
    --metadata "$METADATA" \
    --output "$SEMANTIC_SCORES" \
    --gpu "$GPU" \
    --batch-size 128 \
    --torch-num-threads 8 \
    --resume \
    >>"$LOG_ROOT/clip_expert.log" 2>&1
  mark_done semantic_scores
fi

if ! done_stage prepare_base; then
  if [[ -d "$PREPARED_DIR" && ! -f "$PREPARED_DIR/manifest_base.json" ]]; then
    archive_partial_dir "$PREPARED_DIR"
  fi
  mkdir -p "$REPAIR_ROOT/empty_collision_repair"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/prepare_stop_visible_v4_production_frames.py" \
    --source "$SOURCE_FRAMES" \
    --audit-dir "$AUDIT_DIR" \
    --semantic-scores "$SEMANTIC_SCORES" \
    --policy "$POLICY" \
    --aligned-root "$RECORD_ROOT" \
    --original-collision-dir "$COLLISION_MERGED" \
    --repair-collision-dir "$REPAIR_ROOT/empty_collision_repair" \
    --reject-initial-collisions \
    --output-dir "$PREPARED_DIR" \
    >>"$LOG_ROOT/prepare_stop_visible_base.log" 2>&1
  mark_done prepare_base
fi

if ! done_stage standoff_queue; then
  archive_partial_dir "$QUEUE_DIR"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/build_stop_visible_v4_standoff_queue.py" \
    --audit-dir "$AUDIT_DIR" \
    --selections "$PREPARED_DIR/selections.jsonl" \
    --quarantine "$PREPARED_DIR/quarantine_trajectories.jsonl" \
    --output-dir "$QUEUE_DIR" \
    >>"$LOG_ROOT/standoff_queue.log" 2>&1
  sort -u \
    "$QUEUE_DIR/repairable_actor_keys.txt" \
    "$QUEUE_DIR/below_threshold_actor_keys.txt" \
    >"$QUEUE_DIR/all_actor_keys.txt"
  mark_done standoff_queue
fi

if ! done_stage standoff_capture; then
  cleanup_port 45704
  wait_for_gpu_idle
  mkdir -p "$STANDOFF_CAPTURE"
  standoff_count=$(grep -cve '^[[:space:]]*$' "$QUEUE_DIR/all_actor_keys.txt" || true)
  if [[ $standoff_count -gt 0 ]]; then
    log "capturing target-facing standoff candidates for $standoff_count actors"
    PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
      "$PROJECT/scripts/capture_target_standoff_candidates.py" \
      --input-cache "$AUDIT_DIR" \
      --trajectory-keys "$QUEUE_DIR/all_actor_keys.txt" \
      --output-dir "$STANDOFF_CAPTURE" \
      --scene-list Neighborhood \
      --gpu "$GPU" \
      --base-port 45700 \
      --settle-frames 2 \
      --resume \
      >>"$LOG_ROOT/standoff_capture.log" 2>&1
    cleanup_port 45704
  else
    log "no Neighborhood actor requires a standoff recapture"
  fi
  mark_done standoff_capture
fi

if ! done_stage standoff_semantic; then
  wait_for_gpu_idle
  mkdir -p "$(dirname "$STANDOFF_SCORES")"
  PYTHONPATH="$PROJECT/src" "$PY_CLIP" \
    "$PROJECT/scripts/score_stop_visibility_clip.py" \
    --visibility-cache "$STANDOFF_CAPTURE" \
    --metadata "$METADATA" \
    --output "$STANDOFF_SCORES" \
    --gpu "$GPU" \
    --batch-size 128 \
    --torch-num-threads 8 \
    --resume \
    >>"$LOG_ROOT/clip_standoff.log" 2>&1
  mark_done standoff_semantic
fi

if ! done_stage stop_bank; then
  archive_partial_dir "$STOP_BANK"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/build_stop_visible_v4_stop_bank.py" \
    --capture-cache "$STANDOFF_CAPTURE" \
    --semantic-scores "$STANDOFF_SCORES" \
    --queue-manifest "$QUEUE_DIR/standoff_queue.jsonl" \
    --policy "$POLICY" \
    --output-dir "$STOP_BANK" \
    >>"$LOG_ROOT/build_stop_bank.log" 2>&1
  "$PY_AIR" - \
    "$STOP_BANK/standoff_rejected.jsonl" \
    "$QUEUE_DIR/standoff_queue.jsonl" \
    "$REPAIR_ROOT/unresolved_stop_coverage_actors.jsonl" <<'PY'
import json, sys
rejected_path, queue_path, output_path = sys.argv[1:]
rejected = {
    row["trajectory_key"]: row
    for row in (json.loads(line) for line in open(rejected_path) if line.strip())
}
queue = {
    row["trajectory_key"]: row
    for row in (json.loads(line) for line in open(queue_path) if line.strip())
}
with open(output_path, "w") as output:
    for key, row in sorted(rejected.items()):
        queued = queue[key]
        result = {
            "actor_key": f"{queued['scene_id']}::{queued['object_name']}",
            "scene_id": queued["scene_id"],
            "object_name": queued["object_name"],
            "true_name": queued.get("true_name"),
            "decision": "retain_safe_navigation_without_stop",
            "represented_trajectory_count": queued["represented_trajectory_count"],
            "reason": "No eligible Stop after synchronized expert-path audit and standard target-facing standoff capture.",
            "selection": row,
        }
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
print({"unresolved_actors": len(rejected)})
PY
  mark_done stop_bank
fi

if ! done_stage neighborhood_assembled; then
  archive_partial_dir "$NEIGHBORHOOD_ASSEMBLED"
  mkdir -p "$NEIGHBORHOOD_ASSEMBLED"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/assemble_stop_visible_v4_appended_frames.py" \
    --base-frames "$PREPARED_DIR/train_frames_base.jsonl" \
    --stop-bank "$STOP_BANK/train_frames_standoff_stop_bank.jsonl" \
    --queue-manifest "$QUEUE_DIR/standoff_queue.jsonl" \
    --source-frames "$FRAME_DIR/train_frames.jsonl" \
    --base-quarantine "$PREPARED_DIR/quarantine_trajectories.jsonl" \
    --quarantine-output "$NEIGHBORHOOD_ASSEMBLED/quarantine_trajectories.jsonl" \
    --quarantine-missing-base-episodes \
    --output "$NEIGHBORHOOD_ASSEMBLED/train_frames.jsonl" \
    --manifest "$NEIGHBORHOOD_ASSEMBLED/manifest.json" \
    >>"$LOG_ROOT/assemble_neighborhood.log" 2>&1
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/validate_stop_visible_v4_production.py" \
    --frames "$NEIGHBORHOOD_ASSEMBLED/train_frames.jsonl" \
    --quarantine "$NEIGHBORHOOD_ASSEMBLED/quarantine_trajectories.jsonl" \
    --allowed-uncovered-actors "$REPAIR_ROOT/unresolved_stop_coverage_actors.jsonl" \
    --allowed-image-root "$RECORD_ROOT/images" \
    --report "$NEIGHBORHOOD_ASSEMBLED/validation_report.json" \
    >>"$LOG_ROOT/validate_neighborhood.log" 2>&1
  mark_done neighborhood_assembled
fi

if ! done_stage final_merge; then
  archive_partial_dir "$FINAL_DIR"
  mkdir -p "$FINAL_DIR"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/merge_repaired_scene_frames.py" \
    --base "$OLD_FRAMES" \
    --repair "$NEIGHBORHOOD_ASSEMBLED/train_frames.jsonl" \
    --scene Neighborhood \
    --base-quarantine "$OLD_QUARANTINE" \
    --repair-quarantine "$NEIGHBORHOOD_ASSEMBLED/quarantine_trajectories.jsonl" \
    --quarantine-output "$FINAL_DIR/quarantine_trajectories.jsonl" \
    --output "$FINAL_DIR/train_frames.jsonl" \
    --manifest "$FINAL_DIR/manifest.json" \
    >>"$LOG_ROOT/merge_final.log" 2>&1
  "$PY_AIR" - \
    "$OLD_UNCOVERED" \
    "$REPAIR_ROOT/unresolved_stop_coverage_actors.jsonl" \
    "$FINAL_DIR/allowed_uncovered_actors.jsonl" <<'PY'
import json, sys
rows = {}
for path in sys.argv[1:3]:
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            rows[row["actor_key"]] = row
with open(sys.argv[3], "w") as output:
    for key in sorted(rows):
        output.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
print({"allowed_uncovered_actors": len(rows)})
PY
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/validate_stop_visible_v4_production.py" \
    --frames "$FINAL_DIR/train_frames.jsonl" \
    --quarantine "$FINAL_DIR/quarantine_trajectories.jsonl" \
    --allowed-uncovered-actors "$FINAL_DIR/allowed_uncovered_actors.jsonl" \
    --allowed-image-root "$RECORD_ROOT/images" \
    --report "$FINAL_DIR/validation_report.json" \
    >>"$LOG_ROOT/validate_final.log" 2>&1
  mark_done final_merge
fi

if ! done_stage sft; then
  mkdir -p "$COMBINED_CACHE"
  for path in "$OLD_DEPTH_CACHE"/*.jsonl; do
    name=$(basename "$path")
    [[ "$name" == "Neighborhood.jsonl" ]] && continue
    [[ -e "$COMBINED_CACHE/$name" ]] || ln -s "$path" "$COMBINED_CACHE/$name"
  done
  [[ -e "$COMBINED_CACHE/Neighborhood.jsonl" ]] || \
    ln -s "$FRAME_DIR/depth_grid_cache/Neighborhood.jsonl" \
      "$COMBINED_CACHE/Neighborhood.jsonl"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/prepare_collision_filtered_depth_sft_data.py" \
    --source "$FINAL_DIR/train_frames.jsonl" \
    --output "$FINAL_DIR/uavon_phi35_sft_depth_grid_stop_visible_v4_neighborhood_repaired.jsonl" \
    --manifest "$FINAL_DIR/sft_manifest.json" \
    --aligned-root "$RECORD_ROOT" \
    --depth-cache "$COMBINED_CACHE" \
    --original-collision-dir "$COLLISION_MERGED" \
    --repair-collision-dir "$REPAIR_ROOT/empty_collision_repair" \
    --missing-depth-policy error \
    --overwrite \
    >>"$LOG_ROOT/build_sft.log" 2>&1
  mark_done sft
fi

if ! done_stage audit_archive; then
  archive_partial_dir "$FINAL_DIR/stop_pair_audit"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/archive_stop_visible_pairs.py" \
    --frames "$FINAL_DIR/train_frames.jsonl" \
    --output-dir "$FINAL_DIR/stop_pair_audit" \
    --page-size 50 \
    >>"$LOG_ROOT/archive_stop_pairs.log" 2>&1

  combined_masks="$FINAL_DIR/combined_visibility_masks"
  mkdir -p "$combined_masks"
  for path in "$OLD_AUDIT"/*.jsonl; do
    name=$(basename "$path")
    [[ "$name" == Neighborhood.jsonl || "$name" == Neighborhood_actors.jsonl ]] && continue
    [[ -e "$combined_masks/$name" ]] || ln -s "$path" "$combined_masks/$name"
  done
  ln -s "$AUDIT_DIR/Neighborhood.jsonl" "$combined_masks/Neighborhood.jsonl"
  ln -s "$AUDIT_DIR/Neighborhood_actors.jsonl" "$combined_masks/Neighborhood_actors.jsonl"

  archive_partial_dir "$FINAL_DIR/stop_pair_audit_maskboxed"
  PYTHONPATH="$PROJECT/src:$PROJECT/scripts" "$PY_AIR" \
    "$PROJECT/scripts/annotate_stop_pair_archive.py" \
    --source-archive "$FINAL_DIR/stop_pair_audit" \
    --expert-mask-audit-dir "$combined_masks" \
    --standoff-capture-dir "$OLD_V4/standoff_capture_repairable" \
    --standoff-capture-dir "$OLD_V4/standoff_capture_rescue" \
    --standoff-capture-dir "$OLD_V4/standoff_capture_rescue_mug_ultraclose" \
    --standoff-capture-dir "$STANDOFF_CAPTURE" \
    --output-dir "$FINAL_DIR/stop_pair_audit_maskboxed" \
    --page-size 50 \
    >>"$LOG_ROOT/annotate_stop_pairs.log" 2>&1
  mark_done audit_archive
fi

log "pipeline complete: $FINAL_DIR"

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from vlm_baseline.stop_visibility import VisibilityPolicy, select_first_clear_frame


STOP_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TARGET_QUARANTINE_REASON = "no_retained_collision_safe_prefix"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(path)
    for source_path in paths:
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def sample_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def load_policy(path: Path) -> VisibilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VisibilityPolicy(**payload.get("policy", payload))


def load_target_quarantine(
    path: Path,
    scene: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows = list(iter_jsonl(path))
    targets = {
        str(row["trajectory_key"])
        for row in rows
        if str(row.get("scene_id")) == scene
        and str(row.get("reason")) == TARGET_QUARANTINE_REASON
    }
    if not targets:
        raise ValueError(
            f"no {scene} trajectories with reason={TARGET_QUARANTINE_REASON}"
        )
    return rows, targets


def load_source_groups(
    path: Path,
    targets: set[str],
) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in iter_jsonl(path):
        episode_key = str(row["episode_key"])
        if episode_key in targets:
            groups.setdefault(episode_key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["frame_idx"]))
    missing = sorted(targets - set(groups))
    if missing:
        raise KeyError(f"source frames missing {len(missing)} targets: {missing[:10]}")
    return groups


def load_collision_status(
    path: Path,
    targets: set[str],
) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for row in iter_jsonl(path):
        key = str(row.get("key") or "")
        parts = key.split("::")
        if len(parts) != 4 or "::".join(parts[:3]) not in targets:
            continue
        result[key] = {
            "error": bool(row.get("error")),
            "initial_collided": bool(row.get("initial_collided")),
            "new_collision_after_action": bool(
                row.get("new_collision_after_action")
            ),
        }
    return result


def load_visibility(
    path: Path,
    targets: set[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in iter_jsonl(path):
        key = str(row.get("trajectory_key") or "")
        if key in targets and "status" in row:
            result[key] = row
    missing = sorted(targets - set(result))
    if missing:
        raise KeyError(f"visibility audit missing {len(missing)} targets: {missing[:10]}")
    return result


def attach_semantic_scores(
    visibility: dict[str, dict[str, Any]],
    path: Path,
) -> int:
    attached = 0
    for row in iter_jsonl(path):
        key = str(row.get("trajectory_key") or "")
        trajectory = visibility.get(key)
        if trajectory is None:
            continue
        frame_idx = int(row["frame_idx"])
        frame = next(
            (
                item
                for item in trajectory.get("frames") or []
                if int(item["frame_idx"]) == frame_idx
            ),
            None,
        )
        if frame is None:
            continue
        frame.update(
            {
                name: value
                for name, value in row.items()
                if name not in {"trajectory_key", "frame_idx"}
            }
        )
        attached += 1
    return attached


def load_stop_specs(
    stop_bank: Path,
    queue_manifests: list[Path],
    targets: set[str],
) -> dict[str, dict[str, Any]]:
    queue_by_capture_key = {}
    for path in queue_manifests:
        for row in iter_jsonl(path):
            key = str(row["trajectory_key"])
            if key in queue_by_capture_key:
                raise ValueError(f"duplicate queue capture key: {key}")
            queue_by_capture_key[key] = row

    specs = {}
    for stop_row in iter_jsonl(stop_bank):
        visibility = stop_row.get("stop_visibility") or {}
        capture_key = str(visibility.get("source_trajectory_key") or "")
        queue_row = queue_by_capture_key.get(capture_key)
        if queue_row is None:
            raise KeyError(f"Stop bank row has no queue entry: {capture_key}")
        for episode_key in map(str, queue_row["represented_trajectory_keys"]):
            if episode_key not in targets:
                continue
            if episode_key in specs:
                raise ValueError(f"episode covered by multiple Stop rows: {episode_key}")
            specs[episode_key] = {
                "stop_row": stop_row,
                "queue_row": queue_row,
            }
    missing = sorted(targets - set(specs))
    if missing:
        raise KeyError(f"Stop bank missing {len(missing)} targets: {missing[:10]}")
    return specs


def attach_authoritative_metadata(
    row: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    for field in ("target_description", "true_name", "object_name", "size"):
        value = audit.get(field)
        if value is not None:
            row[field] = value


def audit_frame_lookup(audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame["frame_idx"]): frame for frame in audit.get("frames") or []
    }


def current_pose_safe(status: dict[str, bool] | None) -> bool:
    return bool(
        status is not None
        and not status["error"]
        and not status["initial_collided"]
    )


def motion_label_safe(
    row: dict[str, Any],
    status: dict[str, bool] | None,
) -> bool:
    return bool(
        current_pose_safe(status)
        and not status["new_collision_after_action"]
        and str(row.get("action_name") or "").strip().lower() != "stop"
    )


def select_stop(
    audit: dict[str, Any],
    collision_status: dict[str, dict[str, bool]],
    source_by_idx: dict[int, dict[str, Any]],
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    candidates = []
    for source_frame in audit.get("frames") or []:
        frame_idx = int(source_frame["frame_idx"])
        source_row = source_by_idx.get(frame_idx)
        if source_row is None:
            continue
        status = collision_status.get(sample_key(source_row))
        if not current_pose_safe(status):
            continue
        frame = copy.deepcopy(source_frame)
        frame["collision_info"] = {
            "has_collided": False,
            "initial_collided": False,
            "collision_check_error": False,
            "source": "exact_action_replay_per_frame_safe",
        }
        candidates.append(frame)
    return select_first_clear_frame(candidates, audit.get("size"), policy)


def prepare_safe_motion_rows(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    collision_status: dict[str, dict[str, bool]],
) -> tuple[list[dict[str, Any]], dict[int, int], Counter[str]]:
    frame_lookup = audit_frame_lookup(audit)
    safe_source_rows = [
        row
        for row in rows
        if motion_label_safe(row, collision_status.get(sample_key(row)))
    ]
    segment_by_idx = {}
    segment_id = -1
    previous_idx = None
    for row in safe_source_rows:
        frame_idx = int(row["frame_idx"])
        if previous_idx is None or frame_idx != previous_idx + 1:
            segment_id += 1
        segment_by_idx[frame_idx] = segment_id
        previous_idx = frame_idx

    stats: Counter[str] = Counter()
    prepared = []
    for source_row in safe_source_rows:
        frame_idx = int(source_row["frame_idx"])
        row = copy.deepcopy(source_row)
        attach_authoritative_metadata(row, audit)
        audit_frame = frame_lookup.get(frame_idx) or {}
        if audit_frame.get("replay_image_path"):
            row["image_path"] = str(audit_frame["replay_image_path"])
        status = collision_status[sample_key(source_row)]
        row["trajectory_repair"] = {
            "version": "per_frame_safe_v1",
            "mode": "individually_collision_safe_motion",
            "original_frame_idx": frame_idx,
            "safe_segment_id": segment_by_idx[frame_idx],
            "per_frame_training_only": True,
            "cross_row_continuity_required": False,
        }
        row["collision_safety"] = {
            "source": "exact_action_replay",
            "current_pose_collision_free": True,
            "label_action_collision_free": True,
            "initial_collided": status["initial_collided"],
            "new_collision_after_action": status["new_collision_after_action"],
        }
        prepared.append(row)
        stats["safe_motion_rows"] += 1
    stats["safe_segments"] = segment_id + 1
    return prepared, segment_by_idx, stats


def build_original_stop(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    selection: dict[str, Any],
    output_frame_idx: int,
) -> dict[str, Any]:
    selected_idx = int(selection["selected_frame_idx"])
    source_by_idx = {int(row["frame_idx"]): row for row in rows}
    source_row = source_by_idx[selected_idx]
    frame = audit_frame_lookup(audit)[selected_idx]
    row = copy.deepcopy(source_row)
    attach_authoritative_metadata(row, audit)
    row["frame_idx"] = output_frame_idx
    row["step_id"] = output_frame_idx
    row["image_path"] = str(frame.get("replay_image_path") or row["image_path"])
    row["original_frame_idx"] = selected_idx
    row["original_action_name"] = row.get("action_name")
    row["original_action_vector"] = row.get("action_vector")
    row["original_action_id"] = row.get("action_id")
    row["action_name"] = "Stop"
    row["action_id"] = 0
    row["uavon_action"] = "stop"
    row["action_vector"] = list(STOP_VECTOR)
    if row.get("pose") is not None:
        row["next_pose"] = copy.deepcopy(row["pose"])
    row["trajectory_repair"] = {
        "version": "per_frame_safe_v1",
        "mode": "clear_expert_frame_appended_for_single_frame_sft",
        "source_original_frame_idx": selected_idx,
        "per_frame_training_only": True,
        "cross_row_continuity_required": False,
    }
    row["collision_safety"] = {
        "source": "exact_action_replay",
        "current_pose_collision_free": True,
        "label_action_replaced_by_zero_motion_stop": True,
    }
    row["stop_visibility"] = {
        "selected": True,
        "version": "per_frame_safe_v1",
        "selection_mode": selection.get("selection_mode"),
        "quality_score": selection.get("selected_quality_score"),
        "size_bucket": selection.get("size_bucket"),
        "peak_pixels": selection.get("peak_pixels"),
        "source_type": "expert_path",
        "source_original_frame_idx": selected_idx,
        "image_source": (
            "synchronized_replay"
            if frame.get("replay_image_path")
            else "original_recording"
        ),
        "mask_path": frame.get("mask_path"),
        "appended_to_original_trajectory": True,
        "recaptured_pose_is_not_physical_next_pose": True,
        "per_frame_training_only": True,
    }
    return row


def build_standoff_stop(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    spec: dict[str, Any],
    output_frame_idx: int,
) -> dict[str, Any]:
    source = rows[0]
    row = copy.deepcopy(spec["stop_row"])
    queue_row = spec["queue_row"]
    source_visibility = row.get("stop_visibility") or {}
    row.update(
        {
            "episode_key": str(source["episode_key"]),
            "scene_id": str(source["scene_id"]),
            "episode_id": str(source["episode_id"]),
            "pose_idx": str(source["pose_idx"]),
            "frame_idx": output_frame_idx,
            "step_id": output_frame_idx,
            "action_name": "Stop",
            "action_id": 0,
            "uavon_action": "stop",
            "action_vector": list(STOP_VECTOR),
            "coordinate_repair": source.get("coordinate_repair"),
            "coordinate_repair_start_recovery": source.get(
                "coordinate_repair_start_recovery"
            ),
        }
    )
    attach_authoritative_metadata(row, audit)
    if row.get("pose") is not None:
        row["next_pose"] = copy.deepcopy(row["pose"])
    row["trajectory_repair"] = {
        "version": "per_frame_safe_v1",
        "mode": "individually_safe_motion_with_appended_standoff_stop",
        "per_frame_training_only": True,
        "cross_row_continuity_required": False,
    }
    row["collision_safety"] = {
        "source": "verified_standoff_capture",
        "current_pose_collision_free": True,
        "label_action_replaced_by_zero_motion_stop": True,
    }
    row["stop_visibility"] = {
        **source_visibility,
        "version": "per_frame_safe_v1",
        "source_type": "target_facing_standoff_appended_to_original_episode",
        "capture_group": str(queue_row["capture_group"]),
        "source_capture_trajectory_key": str(queue_row["trajectory_key"]),
        "source_stop_bank_episode_key": str(spec["stop_row"]["episode_key"]),
        "appended_to_original_trajectory": True,
        "not_appended_to_original_trajectory": False,
        "recaptured_pose_is_not_physical_next_pose": True,
        "per_frame_training_only": True,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add individually collision-safe rows from strict-prefix quarantined "
            "episodes to an existing single-frame training dataset."
        )
    )
    parser.add_argument("--strict-frames", type=Path, required=True)
    parser.add_argument("--strict-quarantine", type=Path, required=True)
    parser.add_argument("--source-scene-frames", type=Path, required=True)
    parser.add_argument("--collision-audit", type=Path, required=True)
    parser.add_argument("--visibility-audit", type=Path, required=True)
    parser.add_argument("--semantic-scores", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--stop-bank", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, action="append", required=True)
    parser.add_argument("--scene", default="Neighborhood")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.output,
        args.quarantine_output,
        args.decisions_output,
        args.manifest,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    quarantine_rows, targets = load_target_quarantine(
        args.strict_quarantine,
        args.scene,
    )
    groups = load_source_groups(args.source_scene_frames, targets)
    collision_status = load_collision_status(args.collision_audit, targets)
    visibility = load_visibility(args.visibility_audit, targets)
    semantic_attached = attach_semantic_scores(visibility, args.semantic_scores)
    stop_specs = load_stop_specs(args.stop_bank, args.queue_manifest, targets)
    policy = replace(load_policy(args.policy), reject_collided=True)

    strict_sample_keys: set[str] = set()
    strict_episode_keys: set[str] = set()
    output_sample_keys: set[str] = set()
    output_episode_keys: set[str] = set()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    decisions = []
    salvaged_keys: set[str] = set()

    def write_row(output, row: dict[str, Any], source: str) -> None:
        key = sample_key(row)
        if key in output_sample_keys:
            raise ValueError(f"duplicate output sample key: {key}")
        output_sample_keys.add(key)
        output_episode_keys.add(str(row["episode_key"]))
        action_counts[str(row["action_name"])] += 1
        scene_counts[str(row["scene_id"])] += 1
        stats["rows"] += 1
        stats[f"rows_{source}"] += 1
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.output.open("x", encoding="utf-8") as output:
        for row in iter_jsonl(args.strict_frames):
            key = sample_key(row)
            strict_sample_keys.add(key)
            strict_episode_keys.add(str(row["episode_key"]))
            write_row(output, row, "strict_base")

        overlap = sorted(targets & strict_episode_keys)
        if overlap:
            raise ValueError(
                f"target quarantined episodes already occur in strict frames: {overlap[:10]}"
            )

        for episode_key, source_rows in groups.items():
            audit = visibility[episode_key]
            source_by_idx = {
                int(row["frame_idx"]): row for row in source_rows
            }
            missing_collision = [
                sample_key(row)
                for row in source_rows
                if sample_key(row) not in collision_status
            ]
            if missing_collision:
                raise KeyError(
                    f"collision audit missing {len(missing_collision)} rows for "
                    f"{episode_key}: {missing_collision[:5]}"
                )
            safe_rows, segment_by_idx, safe_stats = prepare_safe_motion_rows(
                source_rows,
                audit,
                collision_status,
            )
            stats.update(safe_stats)
            decision = {
                "trajectory_key": episode_key,
                "scene_id": audit.get("scene_id"),
                "episode_id": audit.get("episode_id"),
                "pose_idx": audit.get("pose_idx"),
                "object_name": audit.get("object_name"),
                "true_name": audit.get("true_name"),
                "source_rows": len(source_rows),
                "safe_motion_rows": len(safe_rows),
                "safe_segments": len(set(segment_by_idx.values())),
            }
            if not safe_rows:
                stats["episodes_still_quarantined_no_safe_motion"] += 1
                decision.update(
                    {
                        "decision": "quarantine",
                        "reason": "no_individually_collision_safe_motion_rows",
                    }
                )
                decisions.append(decision)
                continue

            selection = select_stop(
                audit,
                collision_status,
                source_by_idx,
                policy,
            )
            output_frame_idx = max(source_by_idx) + 1
            selected_idx = selection.get("selected_frame_idx")
            if selected_idx is not None:
                selected_idx = int(selected_idx)
                before = len(safe_rows)
                safe_rows = [
                    row
                    for row in safe_rows
                    if int((row.get("trajectory_repair") or {}).get("original_frame_idx"))
                    != selected_idx
                ]
                if len(safe_rows) < before:
                    stats["motion_rows_replaced_by_stop"] += 1
                stop_row = build_original_stop(
                    source_rows,
                    audit,
                    selection,
                    output_frame_idx,
                )
                stop_source = "clear_original_frame"
                stats["episodes_salvaged_with_original_stop"] += 1
            else:
                stop_row = build_standoff_stop(
                    source_rows,
                    audit,
                    stop_specs[episode_key],
                    output_frame_idx,
                )
                stop_source = "standoff_capture"
                stats["episodes_salvaged_with_standoff_stop"] += 1

            for row in safe_rows:
                write_row(output, row, "per_frame_safe_motion")
            write_row(output, stop_row, "per_frame_safe_stop")
            salvaged_keys.add(episode_key)
            stats["episodes_salvaged"] += 1
            decision.update(
                {
                    "decision": "salvage",
                    "stop_source": stop_source,
                    "selected_original_stop_frame_idx": selected_idx,
                    "output_stop_frame_idx": output_frame_idx,
                    "rows_written": len(safe_rows) + 1,
                }
            )
            decisions.append(decision)

    with args.quarantine_output.open("x", encoding="utf-8") as output:
        for source_row in quarantine_rows:
            key = str(source_row["trajectory_key"])
            if key in salvaged_keys:
                stats["quarantine_rows_removed_after_salvage"] += 1
                continue
            row = copy.deepcopy(source_row)
            if key in targets:
                row["reason"] = "no_individually_collision_safe_motion_rows"
                row["repair_decision"] = "retain_in_quarantine_per_frame_safe_v1"
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["quarantine_rows_written"] += 1

    with args.decisions_output.open("x", encoding="utf-8") as output:
        for row in decisions:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(decisions) != len(targets):
        raise AssertionError(f"decision count mismatch: {len(decisions)} != {len(targets)}")
    manifest = {
        "format": "uavon_per_frame_safe_salvage_v1",
        "training_mode": "single_frame_sft",
        "strict_frames": str(args.strict_frames.resolve()),
        "strict_quarantine": str(args.strict_quarantine.resolve()),
        "source_scene_frames": str(args.source_scene_frames.resolve()),
        "collision_audit": str(args.collision_audit.resolve()),
        "visibility_audit": str(args.visibility_audit.resolve()),
        "semantic_scores": str(args.semantic_scores.resolve()),
        "policy": str(args.policy.resolve()),
        "output": str(args.output.resolve()),
        "quarantine_output": str(args.quarantine_output.resolve()),
        "decisions_output": str(args.decisions_output.resolve()),
        "target_quarantined_episodes": len(targets),
        "strict_rows": len(strict_sample_keys),
        "strict_episodes": len(strict_episode_keys),
        "output_rows": len(output_sample_keys),
        "output_episodes": len(output_episode_keys),
        "semantic_scores_attached": semantic_attached,
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "safety_policy": {
            "motion_row": (
                "collision replay succeeded, current pose was collision-free, and "
                "the labeled action caused no new collision"
            ),
            "original_stop": (
                "current pose was collision-free and the RGB target passed the frozen "
                "v4 geometry and semantic visibility policy within the 20m audit range"
            ),
            "standoff_stop": "verified clear target-facing zero-motion Stop capture",
            "sequence_policy": (
                "Rows are independent single-frame SFT samples; safe segments are "
                "recorded but cross-row continuity is not required."
            ),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

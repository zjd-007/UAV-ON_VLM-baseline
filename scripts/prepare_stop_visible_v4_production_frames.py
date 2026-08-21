#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_AUDIT_DIR = (
    DATASET_ROOT
    / "processed"
    / "stop_visible_full_audit"
    / "full_canonical_geometry_v1_20260812_153000"
)
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "stop_visible_v4_production_policy.json"
DEFAULT_ORIGINAL_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048"
)
DEFAULT_REPAIR_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048_repair_lane3"
)
STOP_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_collision_filtered_depth_sft_data import (  # noqa: E402
    collision_key,
    is_error_row,
    iter_collision_rows,
    load_collision_filter,
)
from prepare_stop_visible_frames import (  # noqa: E402
    attach_semantic_scores,
    load_semantic_scores,
    load_visibility_cache,
)
from vlm_baseline.stop_visibility import VisibilityPolicy, select_first_clear_frame  # noqa: E402


RETAIN_NAVIGATION_CAUSES = {
    "trajectory_viewpoint_distance_or_occlusion",
    "target_facing_standoff_repairable",
    "target_pixels_exist_but_below_clear_threshold_after_standoff",
}
QUARANTINE_CAUSES = {
    "dataset_simulator_xy_coordinate_mismatch",
    "target_actor_at_expected_xy_but_no_pixels_in_path_or_standoff",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_source_groups(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            groups.setdefault(str(row["episode_key"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["frame_idx"]))
    return groups


def load_problem_causes(audit_dir: Path) -> dict[str, str]:
    path = audit_dir / "summary_collision_filtered" / "problem_trajectories.jsonl"
    return {
        str(row["trajectory_key"]): str(row.get("refined_problem_cause") or "unknown")
        for row in read_jsonl(path)
    }


def load_policy(path: Path) -> VisibilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VisibilityPolicy(**payload.get("policy", payload))


def load_collision_replay_status(
    original_dir: Path,
    repair_dir: Path | None,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    decisions: dict[str, dict[str, bool]] = {}
    stats: Counter[str] = Counter()
    for source_name, directory in (
        ("original", original_dir),
        ("repair", repair_dir),
    ):
        if directory is None or not directory.exists():
            continue
        for _, row in iter_collision_rows(directory):
            stats[f"{source_name}_rows"] += 1
            if is_error_row(row):
                stats[f"{source_name}_errors"] += 1
                decisions.setdefault(
                    collision_key(row),
                    {
                        "error": True,
                        "initial_collided": False,
                        "new_collision_after_action": False,
                    },
                )
                continue
            key = collision_key(row)
            decisions[key] = {
                "error": False,
                "initial_collided": bool(row.get("initial_collided")),
                "new_collision_after_action": bool(
                    row.get("new_collision_after_action")
                ),
            }
            stats[f"{source_name}_checked"] += 1
    return decisions, {
        "keys": len(decisions),
        "initial_collision_keys": sum(
            status["initial_collided"] for status in decisions.values()
        ),
        "new_collision_keys": sum(
            status["new_collision_after_action"] for status in decisions.values()
        ),
        "error_keys": sum(status["error"] for status in decisions.values()),
        "repair_overrides_original": bool(repair_dir and repair_dir.exists()),
        **dict(stats),
    }


def attach_collision_reachability(
    groups: OrderedDict[str, list[dict[str, Any]]],
    audit: dict[str, dict[str, Any]],
    collision_status: dict[str, dict[str, bool]],
) -> tuple[set[str], set[str], dict[str, int]]:
    initial_collision_keys: set[str] = set()
    reachable_motion_keys: set[str] = set()
    stats: Counter[str] = Counter()
    for trajectory_key, rows in groups.items():
        frame_lookup = {
            int(frame["frame_idx"]): frame
            for frame in (audit.get(trajectory_key) or {}).get("frames") or []
        }
        prefix_reachable = True
        barrier_frame_idx = None
        for row in rows:
            frame_idx = int(row["frame_idx"])
            key = sample_key(row)
            status = collision_status.get(key)
            if status is None:
                status = {
                    "error": True,
                    "initial_collided": False,
                    "new_collision_after_action": False,
                }
                stats["missing_collision_status"] += 1
            initial_collided = bool(status["initial_collided"])
            failed = bool(status["error"])
            current_reachable = bool(
                prefix_reachable and not initial_collided and not failed
            )
            if initial_collided:
                initial_collision_keys.add(key)
            frame = frame_lookup.get(frame_idx)
            if frame is not None:
                frame["collision_info"] = {
                    "has_collided": not current_reachable,
                    "initial_collided": initial_collided,
                    "collision_check_error": failed,
                    "source": "exact_action_replay_contiguous_safe_prefix",
                }
                frame["trajectory_reachability"] = {
                    "reachable_from_episode_start": current_reachable,
                    "barrier_frame_idx": barrier_frame_idx,
                    "original_action_new_collision": bool(
                        status["new_collision_after_action"]
                    ),
                }
                stats["audit_frames_attached"] += 1
                if not current_reachable:
                    stats["audit_frames_rejected_as_unreachable"] += 1
            action_is_stop = (
                str(row.get("action_name") or "").strip().lower() == "stop"
            )
            if (
                current_reachable
                and not action_is_stop
                and not status["new_collision_after_action"]
            ):
                reachable_motion_keys.add(key)
            if current_reachable and status["new_collision_after_action"]:
                stats["reachable_frames_with_unsafe_original_action"] += 1
            if (
                not current_reachable
                or status["new_collision_after_action"]
            ):
                if prefix_reachable:
                    barrier_frame_idx = frame_idx
                    stats["trajectories_with_reachability_barrier"] += 1
                prefix_reachable = False
        if prefix_reachable:
            stats["fully_reachable_trajectories"] += 1
    return initial_collision_keys, reachable_motion_keys, dict(stats)


def sample_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def aligned_image_path(path_text: str, aligned_root: Path) -> str:
    normalized = str(path_text).replace("\\", "/")
    if "record_output_transition_aligned/images/" in normalized:
        path = Path(normalized)
    elif "record_output/images/" in normalized:
        suffix = normalized.split("record_output/images/", 1)[1]
        path = aligned_root / "images" / suffix
    else:
        path = Path(normalized)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.absolute())


def has_authoritative_metadata(audit: dict[str, Any]) -> bool:
    return bool(
        str(audit.get("true_name") or "").strip()
        and str(audit.get("target_description") or "").strip()
        and str(audit.get("size") or "").strip()
    )


def collision_safe_rows(
    rows: list[dict[str, Any]], excluded_keys: set[str]
) -> list[dict[str, Any]]:
    return [row for row in rows if sample_key(row) not in excluded_keys]


def attach_authoritative_metadata(
    row: dict[str, Any], audit: dict[str, Any] | None
) -> None:
    for field in ("target_description", "true_name", "object_name", "size"):
        value = (audit or {}).get(field)
        if value is not None:
            row[field] = value


def prepare_selected_trajectory(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
    selection: dict[str, Any],
    excluded_keys: set[str],
    aligned_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    selected_idx = int(selection["selected_frame_idx"])
    audit_frames = {
        int(frame["frame_idx"]): frame for frame in audit.get("frames") or []
    }
    eligible_motion_frames = {
        int(assessment["frame_idx"]): assessment
        for assessment in selection.get("assessments") or []
        if assessment.get("clear")
        and int(assessment["frame_idx"]) < selected_idx
    }
    selected_frame = next(
        frame
        for frame in audit.get("frames") or []
        if int(frame["frame_idx"]) == selected_idx
    )
    output = []
    removed_collision = 0
    for source_row in rows:
        frame_idx = int(source_row["frame_idx"])
        if frame_idx > selected_idx:
            break
        key = sample_key(source_row)
        is_selected = frame_idx == selected_idx
        if key in excluded_keys and not is_selected:
            removed_collision += 1
            continue
        row = dict(source_row)
        attach_authoritative_metadata(row, audit)
        audit_frame = audit_frames.get(frame_idx) or {}
        row["image_path"] = aligned_image_path(
            str(audit_frame.get("replay_image_path") or row["image_path"]),
            aligned_root,
        )
        row["trajectory_repair"] = {
            "version": "stop_visible_v4_production",
            "mode": "preserve_all_real_motion_before_selected_stop",
            "selected_stop_frame_idx": selected_idx,
            "contiguous_safe_prefix_enforced": bool(
                selected_frame.get("trajectory_reachability")
            ),
        }
        if not is_selected and frame_idx in eligible_motion_frames:
            assessment = eligible_motion_frames[frame_idx]
            row["trajectory_repair"].update(
                {
                    "earlier_stop_eligible_motion_preserved": True,
                    "stop_eligible_quality_score": assessment.get("quality_score"),
                }
            )
        if is_selected:
            row["image_path"] = aligned_image_path(
                str(
                    selected_frame.get("replay_image_path")
                    or selected_frame["image_path"]
                ),
                aligned_root,
            )
            row["original_action_name"] = row.get("action_name")
            row["original_action_vector"] = row.get("action_vector")
            row["action_name"] = "Stop"
            row["action_vector"] = list(STOP_VECTOR)
            row["stop_visibility"] = {
                "selected": True,
                "version": "v4_production",
                "selection_mode": selection.get("selection_mode"),
                "quality_score": selection.get("selected_quality_score"),
                "size_bucket": selection.get("size_bucket"),
                "peak_pixels": selection.get("peak_pixels"),
                "source_type": "expert_path",
                "image_source": (
                    "synchronized_replay"
                    if selected_frame.get("replay_image_path")
                    else "original_recording"
                ),
                "mask_path": selected_frame.get("mask_path"),
                "trajectory_reachability": selected_frame.get(
                    "trajectory_reachability"
                ),
                "preserved_earlier_motion_labels": True,
                "collision_filter_override_reason": (
                    "original movement label was replaced by zero-motion Stop"
                    if key in excluded_keys
                    else None
                ),
            }
        output.append(row)
    if not output or int(output[-1]["frame_idx"]) != selected_idx:
        raise RuntimeError(
            f"selected frame {selected_idx} missing after rewrite: {audit['trajectory_key']}"
        )
    return output, removed_collision


def prepare_navigation_only_trajectory(
    rows: list[dict[str, Any]],
    cause: str,
    excluded_keys: set[str],
    aligned_root: Path,
    audit: dict[str, Any] | None = None,
    reachable_motion_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    output = []
    audit_frames = {
        int(frame["frame_idx"]): frame for frame in (audit or {}).get("frames") or []
    }
    removed_collision = 0
    removed_stop = 0
    for source_row in rows:
        if str(source_row.get("action_name") or "").strip().lower() == "stop":
            removed_stop += 1
            continue
        key = sample_key(source_row)
        if (
            key in excluded_keys
            or (
                reachable_motion_keys is not None
                and key not in reachable_motion_keys
            )
        ):
            removed_collision += 1
            continue
        row = dict(source_row)
        attach_authoritative_metadata(row, audit)
        audit_frame = audit_frames.get(int(source_row["frame_idx"])) or {}
        row["image_path"] = aligned_image_path(
            str(audit_frame.get("replay_image_path") or row["image_path"]),
            aligned_root,
        )
        row["trajectory_repair"] = {
            "version": "stop_visible_v4_production",
            "mode": "navigation_only_pending_independent_stop_bank",
            "removed_invalid_stop": True,
            "problem_cause": cause,
            "contiguous_safe_prefix_enforced": reachable_motion_keys is not None,
        }
        output.append(row)
    return output, removed_collision, removed_stop


def write_jsonl_rows(output, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build v4 Stop-visible production frames while preserving every real motion "
            "sample before the selected Stop."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--semantic-scores", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--original-collision-dir", type=Path, default=DEFAULT_ORIGINAL_COLLISION_DIR)
    parser.add_argument("--repair-collision-dir", type=Path, default=DEFAULT_REPAIR_COLLISION_DIR)
    parser.add_argument(
        "--reject-initial-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude motion rows whose replay pose is already collided and reject "
            "those poses during Stop selection."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_frames": args.output_dir / "train_frames_base.jsonl",
        "selections": args.output_dir / "selections.jsonl",
        "quarantine": args.output_dir / "quarantine_trajectories.jsonl",
        "manifest": args.output_dir / "manifest_base.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    groups = load_source_groups(args.source)
    audit = load_visibility_cache(args.audit_dir)
    semantic_scores = load_semantic_scores(args.semantic_scores)
    semantic_attached, visible_without_semantic = attach_semantic_scores(
        audit, semantic_scores
    )
    causes = load_problem_causes(args.audit_dir)
    excluded_keys, collision_manifest = load_collision_filter(
        args.original_collision_dir,
        args.repair_collision_dir,
    )
    policy = load_policy(args.policy)
    initial_collision_keys: set[str] = set()
    reachable_motion_keys: set[str] | None = None
    initial_collision_manifest: dict[str, Any] | None = None
    reachability_manifest: dict[str, int] | None = None
    if args.reject_initial_collisions:
        collision_status, initial_collision_manifest = load_collision_replay_status(
            args.original_collision_dir,
            args.repair_collision_dir,
        )
        (
            initial_collision_keys,
            reachable_motion_keys,
            reachability_manifest,
        ) = attach_collision_reachability(groups, audit, collision_status)
        excluded_keys.update(initial_collision_keys)
        policy = replace(policy, reject_collided=True)
    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    cause_rows: dict[str, Counter[str]] = {}

    with (
        paths["train_frames"].open("x", encoding="utf-8") as output,
        paths["selections"].open("x", encoding="utf-8") as selections,
        paths["quarantine"].open("x", encoding="utf-8") as quarantine,
    ):
        for key, rows in groups.items():
            cached = audit.get(key)
            if cached is None:
                raise KeyError(f"full visibility audit missing trajectory: {key}")
            stats["source_trajectories"] += 1
            stats["source_rows"] += len(rows)
            if not has_authoritative_metadata(cached):
                reason = "missing_authoritative_target_metadata"
                stats["quarantined_trajectories"] += 1
                stats["quarantined_rows"] += len(rows)
                stats[f"quarantine_{reason}"] += 1
                quarantine.write(
                    json.dumps(
                        {
                            "trajectory_key": key,
                            "reason": reason,
                            "row_count": len(rows),
                            "scene_id": cached.get("scene_id"),
                            "object_name": cached.get("object_name"),
                            "true_name": cached.get("true_name"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            selection = select_first_clear_frame(
                cached.get("frames") or [], cached.get("size"), policy
            )
            decision = {
                "trajectory_key": key,
                "scene_id": cached.get("scene_id"),
                "episode_id": cached.get("episode_id"),
                "pose_idx": cached.get("pose_idx"),
                "true_name": cached.get("true_name"),
                "object_name": cached.get("object_name"),
                "size": cached.get("size"),
                "problem_cause": causes.get(key),
                **selection,
            }
            selections.write(json.dumps(decision, ensure_ascii=False) + "\n")
            if selection["selected_frame_idx"] is not None:
                prepared, removed_collision = prepare_selected_trajectory(
                    rows,
                    cached,
                    selection,
                    excluded_keys,
                    args.aligned_root,
                )
                stats["selected_stop_trajectories"] += 1
                stats["selected_stop_rows"] += len(prepared)
                stats["collision_rows_removed"] += removed_collision
                stats["stop_rows_written"] += 1
                stats["earlier_stop_eligible_motion_rows_preserved"] += sum(
                    bool(
                        (row.get("trajectory_repair") or {}).get(
                            "earlier_stop_eligible_motion_preserved"
                        )
                    )
                    for row in prepared
                )
                if int(selection["selected_frame_idx"]) < int(cached["original_stop_frame_idx"]):
                    stats["stop_moved_earlier"] += 1
                else:
                    stats["stop_unchanged"] += 1
            else:
                cause = causes.get(key) or "semantic_gate_no_eligible_stop"
                if cause in QUARANTINE_CAUSES:
                    stats["quarantined_trajectories"] += 1
                    stats["quarantined_rows"] += len(rows)
                    stats[f"quarantine_{cause}"] += 1
                    quarantine.write(
                        json.dumps(
                            {
                                "trajectory_key": key,
                                "reason": cause,
                                "row_count": len(rows),
                                "scene_id": cached.get("scene_id"),
                                "object_name": cached.get("object_name"),
                                "true_name": cached.get("true_name"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                if cause not in RETAIN_NAVIGATION_CAUSES and cause != "semantic_gate_no_eligible_stop":
                    raise ValueError(f"unhandled no-Stop cause for {key}: {cause}")
                prepared, removed_collision, removed_stop = prepare_navigation_only_trajectory(
                    rows,
                    cause,
                    excluded_keys,
                    args.aligned_root,
                    cached,
                    reachable_motion_keys,
                )
                stats["navigation_only_trajectories"] += 1
                stats["navigation_only_rows"] += len(prepared)
                stats["collision_rows_removed"] += removed_collision
                stats["invalid_stop_rows_removed"] += removed_stop
                cause_rows.setdefault(cause, Counter())
                cause_rows[cause]["trajectories"] += 1
                cause_rows[cause]["rows"] += len(prepared)

            write_jsonl_rows(output, prepared)
            for row in prepared:
                stats["rows_written"] += 1
                action_counts[str(row["action_name"])] += 1
                scene_counts[str(row["scene_id"])] += 1

    manifest = {
        "format": "uavon_stop_visible_v4_production_frames_base",
        "source": str(args.source.resolve()),
        "audit_dir": str(args.audit_dir.resolve()),
        "semantic_scores": str(args.semantic_scores.resolve()),
        "semantic_scores_attached": semantic_attached,
        "visible_frames_without_semantic_score": visible_without_semantic,
        "policy": str(args.policy.resolve()),
        "preserve_all_real_motion_before_selected_stop": True,
        "standoff_stop_samples_are_appended_as_independent_episodes": True,
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "stats": dict(stats),
        "navigation_only_by_cause": {
            cause: dict(counts) for cause, counts in sorted(cause_rows.items())
        },
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "collision_filter": collision_manifest,
        "initial_collision_filter": {
            "enabled": args.reject_initial_collisions,
            "stats": initial_collision_manifest,
            "contiguous_safe_prefix": reachability_manifest,
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

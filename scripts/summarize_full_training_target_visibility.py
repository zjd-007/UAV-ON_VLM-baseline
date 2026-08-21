#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_key_from_image(image_path: str) -> tuple[str, int] | None:
    path = Path(image_path)
    try:
        marker = path.parts.index("record_output_transition_aligned")
        scene = path.parts[marker + 2]
        episode_id = path.parts[marker + 3]
        pose_idx = path.parts[marker + 4]
        frame_idx = int(path.stem)
    except (ValueError, IndexError):
        return None
    return f"{scene}::{episode_id}::{pose_idx}", frame_idx


def count_sft_impact(
    rows_by_key: dict[str, dict[str, Any]],
    sft_path: Path | None,
    refined_causes: dict[str, str | None],
) -> dict[str, Any] | None:
    if sft_path is None:
        return None
    counts = Counter()
    affected_trajectories = set()
    rows_by_problem_cause = Counter()
    stop_rows_by_problem_cause = Counter()
    with sft_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            counts["sft_rows"] += 1
            row = json.loads(line)
            images = row.get("images") or []
            if not images:
                counts["unmapped_rows"] += 1
                continue
            key_frame = sample_key_from_image(str(images[0]))
            if key_frame is None:
                counts["unmapped_rows"] += 1
                continue
            key, frame_idx = key_frame
            audit = rows_by_key.get(key)
            if audit is None:
                counts["unmapped_rows"] += 1
                continue
            if audit.get("clear_geometry_frame_count", 0) == 0:
                counts["rows_from_no_clear_path"] += 1
                affected_trajectories.add(key)
                cause = refined_causes.get(key) or "unknown"
                rows_by_problem_cause[cause] += 1
            if frame_idx == int(audit["original_stop_frame_idx"]):
                counts["stop_rows"] += 1
                if not audit.get("original_stop_clear_geometry"):
                    counts["invalid_original_stop_rows"] += 1
                if audit.get("clear_geometry_frame_count", 0) == 0:
                    counts["stop_rows_from_no_clear_path"] += 1
                    cause = refined_causes.get(key) or "unknown"
                    stop_rows_by_problem_cause[cause] += 1
    counts["no_clear_trajectories_present"] = len(affected_trajectories)
    return {
        **dict(counts),
        "rows_by_problem_cause": dict(rows_by_problem_cause),
        "stop_rows_by_problem_cause": dict(stop_rows_by_problem_cause),
    }


def load_actor_pose_alignment(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    rows = {}
    for source in sorted(path.glob("*.jsonl")):
        for row in read_jsonl(source):
            rows[(str(row.get("scene_id")), str(row.get("object_name")))] = row
    return rows


def refine_problem_cause(
    row: dict[str, Any],
    actor_by_key: dict[tuple[str, str], dict[str, Any]],
    alignment_by_actor: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    if row.get("path_status") == "clear_geometry_available":
        return None
    original = str(row.get("no_target_cause") or "")
    if original != "target_mapping_coordinate_or_scene_asset_unresolved":
        return original or "unknown"
    actor_key = (str(row.get("scene_id")), str(row.get("object_name")))
    alignment = alignment_by_actor.get(actor_key) or {}
    xy_error = alignment.get("actor_to_target_error_xy_m")
    if xy_error is not None and float(xy_error) > 20.0:
        return "dataset_simulator_xy_coordinate_mismatch"
    actor = actor_by_key.get(actor_key) or {}
    best_pixels = int(((actor.get("standoff") or {}).get("best_pixel_count") or 0))
    if best_pixels <= 0:
        return "target_actor_at_expected_xy_but_no_pixels_in_path_or_standoff"
    return "target_pixels_exist_but_below_clear_threshold_after_standoff"


def nearest_target_angles(
    pose: list[float],
    target_positions: list[list[float]],
) -> tuple[float, float, float]:
    x, y, z, yaw = (float(value) for value in pose)
    candidates = []
    for target in target_positions:
        dx = float(target[0]) - x
        dy = float(target[1]) - y
        dz = float(target[2]) - z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        horizontal = math.sqrt(dx * dx + dy * dy)
        yaw_error = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
        pitch_error = math.atan2(dz, horizontal)
        candidates.append(
            (distance, abs(math.degrees(yaw_error)), abs(math.degrees(pitch_error)))
        )
    return min(candidates) if candidates else (math.inf, math.inf, math.inf)


def geometry_summary(row: dict[str, Any]) -> dict[str, Any]:
    frames = row.get("frames") or []
    targets = row.get("target_positions") or []
    in_fov = []
    stop_frame = None
    for frame in frames:
        _, yaw_error, pitch_error = nearest_target_angles(frame["pose"], targets)
        center_in_fov = yaw_error <= 45.0 and pitch_error <= 45.0
        in_fov.append(center_in_fov)
        if int(frame["frame_idx"]) == int(row.get("original_stop_frame_idx", -1)):
            stop_frame = frame
            stop_frame = {
                **stop_frame,
                "target_center_in_fov": center_in_fov,
                "target_yaw_error_deg": yaw_error,
                "target_pitch_error_deg": pitch_error,
            }
    return {
        "path_has_target_center_in_fov": any(in_fov),
        "path_target_center_in_fov_frame_count": sum(in_fov),
        "stop_frame": stop_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a completed full training target visibility audit."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sft-jsonl", type=Path)
    parser.add_argument("--actor-pose-alignment-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows = []
    actor_rows = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        if path.name.endswith("_actors.jsonl"):
            actor_rows.extend(read_jsonl(path))
        else:
            trajectory_rows.extend(read_jsonl(path))
    rows_by_key = {str(row["trajectory_key"]): row for row in trajectory_rows}
    actor_by_key = {
        (str(row.get("scene_id")), str(row.get("object_name"))): row
        for row in actor_rows
    }
    alignment_by_actor = load_actor_pose_alignment(args.actor_pose_alignment_dir)
    refined_causes = {
        str(row["trajectory_key"]): refine_problem_cause(
            row,
            actor_by_key,
            alignment_by_actor,
        )
        for row in trajectory_rows
    }
    path_status = Counter(str(row.get("path_status")) for row in trajectory_rows)
    cause = Counter(
        refined_causes[str(row["trajectory_key"])]
        for row in trajectory_rows
        if refined_causes[str(row["trajectory_key"])]
    )
    actor_status = Counter(str(row.get("actor_audit_status")) for row in actor_rows)
    by_scene: dict[str, Counter] = defaultdict(Counter)
    by_size: dict[str, Counter] = defaultdict(Counter)
    by_object: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    stop_diagnostics = Counter()
    path_problem_diagnostics = Counter()
    for row in trajectory_rows:
        scene = str(row.get("scene_id"))
        size = str(row.get("size", "unknown")).split("(", 1)[0].strip() or "unknown"
        status = str(row.get("path_status"))
        by_scene[scene]["trajectories"] += 1
        by_scene[scene][status] += 1
        if not row.get("original_stop_clear_geometry"):
            by_scene[scene]["invalid_original_stop"] += 1
        geometry = geometry_summary(row)
        stop_frame = geometry["stop_frame"]
        if row.get("status") == "ok" and not row.get("original_stop_clear_geometry"):
            if int(row.get("clear_geometry_frame_count", 0)) > 0:
                stop_diagnostics["earlier_clear_frame_but_original_stop_invalid"] += 1
            else:
                stop_diagnostics["entire_path_has_no_clear_frame"] += 1
            stop_pixels = int(((stop_frame or {}).get("mask") or {}).get("pixel_count", 0))
            if stop_pixels == 0:
                stop_diagnostics["original_stop_zero_target_pixels"] += 1
            else:
                stop_diagnostics["original_stop_pixels_but_not_clear"] += 1
            if stop_frame and not stop_frame["target_center_in_fov"]:
                stop_diagnostics["original_stop_target_center_outside_90deg_fov"] += 1
            else:
                stop_diagnostics["original_stop_target_center_inside_90deg_fov"] += 1
        by_size[size]["trajectories"] += 1
        by_size[size][status] += 1
        if status != "clear_geometry_available":
            key = (scene, str(row.get("object_name")), str(row.get("true_name")))
            by_object[key]["affected_trajectories"] += 1
            by_object[key][str(refined_causes[str(row["trajectory_key"])] or "unknown")] += 1
            if not geometry["path_has_target_center_in_fov"]:
                path_problem_diagnostics["target_center_never_inside_90deg_fov"] += 1
            elif status == "no_detectable_target_pixels":
                path_problem_diagnostics[
                    "target_center_in_fov_but_instance_pixels_absent"
                ] += 1
            else:
                path_problem_diagnostics[
                    "instance_pixels_present_but_too_small_thin_or_clipped"
                ] += 1

    summary = {
        "definition": {
            "distance_threshold_m": 20.0,
            "path_problem": (
                "no expert-path frame within 20m satisfies the v4 geometry-only "
                "recognizable-view thresholds"
            ),
            "unresolved_object": (
                "no expert-path frame and no target-facing standoff candidate "
                "satisfies the same thresholds"
            ),
        },
        "trajectory_count": len(trajectory_rows),
        "actor_count": len(actor_rows),
        "error_trajectories": sum(row.get("status") == "error" for row in trajectory_rows),
        "original_stop_clear_geometry": sum(
            bool(row.get("original_stop_clear_geometry")) for row in trajectory_rows
        ),
        "original_stop_invalid": sum(
            not bool(row.get("original_stop_clear_geometry"))
            for row in trajectory_rows
            if row.get("status") == "ok"
        ),
        "original_stop_diagnostics": dict(stop_diagnostics),
        "path_status": dict(path_status),
        "path_problem_diagnostics": dict(path_problem_diagnostics),
        "problem_cause": dict(cause),
        "actor_status": dict(actor_status),
        "scene_breakdown": {scene: dict(counts) for scene, counts in sorted(by_scene.items())},
        "size_breakdown": {size: dict(counts) for size, counts in sorted(by_size.items())},
        "actor_pose_alignment_audited": len(alignment_by_actor),
        "sft_impact": count_sft_impact(
            rows_by_key,
            args.sft_jsonl,
            refined_causes,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    problem_rows = [
        row for row in trajectory_rows if row.get("path_status") != "clear_geometry_available"
    ]
    with (output_dir / "problem_trajectories.jsonl").open("w", encoding="utf-8") as output:
        for row in problem_rows:
            compact = {key: value for key, value in row.items() if key != "frames"}
            compact["refined_problem_cause"] = refined_causes[str(row["trajectory_key"])]
            output.write(json.dumps(compact, ensure_ascii=False) + "\n")
    with (output_dir / "problem_objects.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        fieldnames = [
            "scene_id",
            "object_name",
            "true_name",
            "affected_trajectories",
            "trajectory_viewpoint_distance_or_occlusion",
            "target_facing_standoff_repairable",
            "dataset_simulator_xy_coordinate_mismatch",
            "target_actor_at_expected_xy_but_no_pixels_in_path_or_standoff",
            "target_pixels_exist_but_below_clear_threshold_after_standoff",
            "object_mapping_or_capture_error",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for (scene, object_name, true_name), counts in sorted(
            by_object.items(),
            key=lambda item: (-item[1]["affected_trajectories"], item[0]),
        ):
            writer.writerow(
                {
                    "scene_id": scene,
                    "object_name": object_name,
                    "true_name": true_name,
                    **{name: counts[name] for name in fieldnames[3:]},
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit whether failed inference episodes ever rendered the target instance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit_inference_stop_visibility import (  # noqa: E402
    align_camera_to_saved_world_pose,
    camera_pose,
    quaternion_error_degrees,
    set_exact_vehicle_pose,
    vehicle_snapshot,
)
from capture_stop_visibility_cache import (  # noqa: E402
    SEGMENTATION_COLOR_BY_ID,
    SegmentationAirsimTrajRecorder,
    advance_simulation,
    capture_segmentation,
    identify_changed_segmentation_color,
    segmentation_color_mask,
    set_target_segmentation_id,
)
from vlm_baseline.stop_visibility import (  # noqa: E402
    VisibilityPolicy,
    assess_visibility_frame,
    canonical_stencil_difference_mask,
    mask_metrics,
    parse_size_bucket,
)


def normalize_target_positions(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if len(value) >= 3 and all(isinstance(item, (int, float)) for item in value[:3]):
        return [[float(item) for item in value[:3]]]
    return [
        [float(component) for component in item[:3]]
        for item in value
        if isinstance(item, list) and len(item) >= 3
    ]


def episode_key(row: dict[str, Any]) -> str:
    return f"{row['map_name']}::{row['episode_id']}"


def nearest_distance(position: list[float], targets: list[list[float]]) -> float:
    point = np.asarray(position[:3], dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(target_array - point[None, :], axis=1).min())


def in_conservative_horizontal_frustum(
    step: dict[str, Any],
    targets: list[list[float]],
    margin_degrees: float,
) -> bool:
    position = step.get("image_camera_pos")
    quaternion = step.get("image_quat_wb")
    if not (
        isinstance(position, list)
        and len(position) >= 3
        and isinstance(quaternion, list)
        and len(quaternion) >= 4
    ):
        return True
    relative_world = (
        np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        - np.asarray(position[:3], dtype=np.float64)[None, :]
    )
    relative_camera = Rotation.from_quat(quaternion[:4]).inv().apply(relative_world)
    fov = float(step.get("image_fov") or 90.0)
    half_angle = min(90.0, max(0.0, fov / 2.0 + margin_degrees))
    horizontal_angle = np.abs(
        np.degrees(np.arctan2(relative_camera[:, 1], relative_camera[:, 0]))
    )
    return bool(
        np.any(
            np.logical_and(
                relative_camera[:, 0] >= 0.0,
                horizontal_angle <= half_angle,
            )
        )
    )


def build_episode(row: dict[str, Any], frustum_margin_degrees: float) -> dict[str, Any]:
    targets = normalize_target_positions(row.get("pose"))
    if not targets:
        raise ValueError(f"missing target positions: {episode_key(row)}")
    candidates = []
    all_steps = row.get("step_records") or []
    for step in all_steps:
        if not in_conservative_horizontal_frustum(
            step, targets, frustum_margin_degrees
        ):
            continue
        snapshot = vehicle_snapshot(step)
        saved_position = step.get("image_camera_pos")
        saved_quaternion = step.get("image_quat_wb")
        candidates.append(
            {
                "step": int(step.get("step", len(candidates))),
                "action_id": int(step.get("action_id", -1)),
                "parsed_command": str(step.get("parsed_command") or ""),
                "distance_before_m": float(step.get("distance_before", math.nan)),
                "image_path": str(step.get("image_path") or ""),
                "vehicle_position": snapshot["position"],
                "vehicle_quat_xyzw": snapshot["quat_xyzw"],
                "saved_camera_position": (
                    [float(value) for value in saved_position[:3]]
                    if isinstance(saved_position, list) and len(saved_position) >= 3
                    else None
                ),
                "saved_camera_quat_xyzw": (
                    [float(value) for value in saved_quaternion[:4]]
                    if isinstance(saved_quaternion, list) and len(saved_quaternion) >= 4
                    else None
                ),
                "camera_target_distance_m": (
                    nearest_distance(saved_position, targets)
                    if isinstance(saved_position, list) and len(saved_position) >= 3
                    else float(step.get("distance_before", math.nan))
                ),
            }
        )
    return {
        "episode_key": episode_key(row),
        "scene_id": str(row["map_name"]),
        "episode_id": str(row["episode_id"]),
        "object_name": str(row.get("object_name") or "").strip(),
        "true_name": str(row.get("true_name") or "").strip(),
        "target_description": str(row.get("description") or "").strip(),
        "size": str(row.get("size") or ""),
        "size_bucket": parse_size_bucket(row.get("size")),
        "used_in_train": int(row.get("used-in-train", 0)),
        "termination_reason": str(row.get("termination_reason") or "unknown"),
        "collision": bool(row.get("collision")),
        "official_osr": bool(row.get("osr")),
        "oracle_success": bool(row.get("oracle_success")),
        "target_positions": targets,
        "total_step_count": len(all_steps),
        "candidate_step_count": len(candidates),
        "ever_stop_command": any(
            int(step.get("action_id", -1)) == 0 for step in all_steps
        ),
        "candidates": candidates,
    }


def select_visibility_mask(
    baseline: np.ndarray,
    marked: np.ndarray,
    target_object_id: int,
    size_bucket: str,
    policy: VisibilityPolicy,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    canonical_color = SEGMENTATION_COLOR_BY_ID[target_object_id]
    canonical_mask = canonical_stencil_difference_mask(
        baseline, marked, canonical_color
    )
    canonical_metrics = mask_metrics(canonical_mask)
    observed_color, baseline_color_count = identify_changed_segmentation_color(
        baseline, marked
    )
    changed_mask = np.any(marked != baseline, axis=2)
    if observed_color is None:
        dominant_mask = np.zeros_like(changed_mask)
    else:
        dominant_mask = np.logical_and(
            segmentation_color_mask(marked, observed_color),
            np.logical_not(segmentation_color_mask(baseline, observed_color)),
        )
    changed_metrics = mask_metrics(changed_mask)
    dominant_metrics = mask_metrics(dominant_mask)
    dominant_share = float(
        dominant_metrics["pixel_count"] / max(1, int(changed_metrics["pixel_count"]))
    )
    min_pixels = int(
        policy.min_pixels.get(size_bucket, policy.min_pixels["unknown"])
    )
    min_short_side = int(
        policy.min_bbox_short_side.get(
            size_bucket, policy.min_bbox_short_side["unknown"]
        )
    )
    dominant_short_side = min(
        int(dominant_metrics["bbox_width"]),
        int(dominant_metrics["bbox_height"]),
    )
    fallback_valid = bool(
        int(canonical_metrics["pixel_count"]) == 0
        and int(dominant_metrics["pixel_count"]) >= min_pixels
        and dominant_short_side >= min_short_side
        and dominant_share >= 0.5
        and int(baseline_color_count) == 0
    )
    if int(canonical_metrics["pixel_count"]) > 0:
        mask = canonical_mask
        source = "canonical_id42"
    elif fallback_valid:
        mask = dominant_mask
        source = "dominant_changed_color_fallback"
    else:
        mask = np.zeros_like(changed_mask)
        source = "none"
    diagnostics = {
        "canonical_pixel_count": int(canonical_metrics["pixel_count"]),
        "changed_pixel_count": int(changed_metrics["pixel_count"]),
        "dominant_changed_pixel_count": int(dominant_metrics["pixel_count"]),
        "dominant_changed_share": dominant_share,
        "fallback_valid": fallback_valid,
        "baseline_observed_color_pixels": int(baseline_color_count),
    }
    return mask, source, diagnostics


def classify_failure(
    target_present: bool,
    clear_target_visible: bool,
    ever_stop_command: bool,
) -> str:
    if not target_present:
        return (
            "never_target_visible_but_stopped"
            if ever_stop_command
            else "never_target_visible_no_stop"
        )
    if not clear_target_visible:
        return (
            "weak_target_glimpse_and_stopped"
            if ever_stop_command
            else "weak_target_glimpse_no_stop"
        )
    return (
        "clear_target_visible_and_stopped_but_failed"
        if ever_stop_command
        else "clear_target_visible_but_no_stop"
    )


def classify_failure_with_distance(row: dict[str, Any]) -> dict[str, Any]:
    frames = row.get("frames") or []
    visible = [frame for frame in frames if bool(frame.get("target_present"))]
    clear = [frame for frame in frames if bool(frame.get("geometry_clear"))]
    visible_within_20m = [
        frame
        for frame in visible
        if float(frame.get("camera_target_distance_m", math.inf)) <= 20.0
    ]
    clear_within_20m = [
        frame
        for frame in clear
        if float(frame.get("camera_target_distance_m", math.inf)) <= 20.0
    ]
    stopped = bool(row.get("ever_stop_command"))
    if not visible:
        category = (
            "never_target_visible_but_stopped"
            if stopped
            else "never_target_visible_no_stop"
        )
    elif clear_within_20m:
        category = (
            "clear_target_within_20m_and_stopped_but_failed"
            if stopped
            else "clear_target_within_20m_but_no_stop"
        )
    elif visible_within_20m:
        category = (
            "weak_target_within_20m_and_stopped"
            if stopped
            else "weak_target_within_20m_no_stop"
        )
    elif clear:
        category = (
            "clear_target_only_beyond_20m_and_stopped"
            if stopped
            else "clear_target_only_beyond_20m_no_stop"
        )
    else:
        category = (
            "weak_target_only_beyond_20m_and_stopped"
            if stopped
            else "weak_target_only_beyond_20m_no_stop"
        )
    return {
        "distance_aware_failure_class": category,
        "target_visible_within_20m_frame_count": len(visible_within_20m),
        "clear_target_within_20m_frame_count": len(clear_within_20m),
        "target_ever_visible_within_20m": bool(visible_within_20m),
        "clear_target_ever_visible_within_20m": bool(clear_within_20m),
        "first_visible_within_20m_step": (
            min(int(frame["step"]) for frame in visible_within_20m)
            if visible_within_20m
            else None
        ),
        "first_clear_within_20m_step": (
            min(int(frame["step"]) for frame in clear_within_20m)
            if clear_within_20m
            else None
        ),
    }


def capture_episode(
    env: SegmentationAirsimTrajRecorder,
    episode: dict[str, Any],
    args: argparse.Namespace,
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    client = env._client
    if not episode["object_name"]:
        raise ValueError("missing object_name")
    requested_names = [episode["object_name"]]
    if episode["object_name"].startswith("SM_MERGED_"):
        requested_names.append(episode["object_name"].removeprefix("SM_MERGED_"))
    if "Motorboat2" in episode["object_name"]:
        requested_names.append(episode["object_name"].replace("Motorboat2", "Motorboat02"))
    resolution_errors = []
    resolved = None
    for requested_name in requested_names:
        try:
            resolved = set_target_segmentation_id(
                client,
                requested_name,
                args.target_object_id,
                episode["target_positions"],
            )
            break
        except Exception as exc:
            resolution_errors.append(f"{requested_name}: {exc!r}")
    if resolved is None:
        raise RuntimeError("; ".join(resolution_errors))
    resolved_name, match_mode, pose_error = resolved
    if requested_names.index(requested_name) > 0:
        match_mode = f"inference_audit_alias:{match_mode}"
    frames = []
    try:
        for candidate in episode["candidates"]:
            if not client.simSetSegmentationObjectID(resolved_name, 0, False):
                raise RuntimeError(f"failed to clear target ID: {resolved_name}")
            set_exact_vehicle_pose(
                client,
                candidate["vehicle_position"],
                candidate["vehicle_quat_xyzw"],
                args.settle_frames,
            )
            for _ in range(args.camera_alignment_repeats):
                align_camera_to_saved_world_pose(
                    client,
                    args.camera_name,
                    candidate["saved_camera_position"],
                    candidate["saved_camera_quat_xyzw"],
                )
            replay_camera = camera_pose(client, args.camera_name)
            baseline = capture_segmentation(client, args.camera_name)
            if not client.simSetSegmentationObjectID(
                resolved_name, args.target_object_id, False
            ):
                raise RuntimeError(f"failed to mark target ID: {resolved_name}")
            marked = capture_segmentation(client, args.camera_name)
            mask, mask_source, diagnostics = select_visibility_mask(
                baseline,
                marked,
                args.target_object_id,
                episode["size_bucket"],
                policy,
            )
            metrics = mask_metrics(mask)
            saved_position = candidate["saved_camera_position"]
            saved_quaternion = candidate["saved_camera_quat_xyzw"]
            position_error = (
                float(
                    np.linalg.norm(
                        np.asarray(replay_camera["position"], dtype=np.float64)
                        - np.asarray(saved_position, dtype=np.float64)
                    )
                )
                if saved_position
                else None
            )
            orientation_error = (
                quaternion_error_degrees(
                    replay_camera["quat_xyzw"], saved_quaternion
                )
                if saved_quaternion
                else None
            )
            frames.append(
                {
                    **candidate,
                    "target_present": int(metrics["pixel_count"]) > 0,
                    "mask": metrics,
                    "mask_source": mask_source,
                    "segmentation_diagnostics": diagnostics,
                    "camera_position_error_m": position_error,
                    "camera_orientation_error_deg": orientation_error,
                    "camera_pose_match": bool(
                        (
                            position_error is None
                            or position_error <= args.camera_position_tolerance
                        )
                        and (
                            orientation_error is None
                            or orientation_error <= args.camera_orientation_tolerance
                        )
                    ),
                }
            )
    finally:
        client.simSetSegmentationObjectID(resolved_name, 0, False)

    peak_pixels = max(
        (int(frame["mask"]["pixel_count"]) for frame in frames), default=0
    )
    for frame in frames:
        assessment = assess_visibility_frame(
            {"frame_idx": frame["step"], "mask": frame["mask"]},
            episode["size_bucket"],
            peak_pixels,
            policy,
        )
        frame["geometry_clear"] = bool(assessment["clear"])
        frame["visibility_assessment"] = assessment
    visible_frames = [frame for frame in frames if frame["target_present"]]
    clear_frames = [frame for frame in frames if frame["geometry_clear"]]
    first_visible = min(visible_frames, key=lambda frame: frame["step"], default=None)
    first_clear = min(clear_frames, key=lambda frame: frame["step"], default=None)
    best_frame = max(
        frames,
        key=lambda frame: int(frame["mask"]["pixel_count"]),
        default=None,
    )
    target_present = bool(visible_frames)
    clear_target_visible = bool(clear_frames)
    result = {
        **{key: value for key, value in episode.items() if key != "candidates"},
        "status": "ok",
        "resolved_object_name": resolved_name,
        "object_name_match": match_mode,
        "object_name_pose_error_m": pose_error,
        "audited_frame_count": len(frames),
        "target_visible_frame_count": len(visible_frames),
        "clear_target_frame_count": len(clear_frames),
        "target_ever_visible": target_present,
        "clear_target_ever_visible": clear_target_visible,
        "peak_target_pixels": peak_pixels,
        "first_visible_step": first_visible["step"] if first_visible else None,
        "first_clear_step": first_clear["step"] if first_clear else None,
        "best_visible_step": best_frame["step"] if best_frame else None,
        "best_visible_distance_m": (
            best_frame["camera_target_distance_m"] if best_frame else None
        ),
        "camera_pose_mismatch_count": sum(
            not frame["camera_pose_match"] for frame in frames
        ),
        "failure_visibility_class": classify_failure(
            target_present,
            clear_target_visible,
            episode["ever_stop_command"],
        ),
        "frames": frames,
    }
    return result


def load_completed(path: Path, retry_errors: bool) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if retry_errors and row.get("status") == "error":
            continue
        completed.add(str(row["episode_key"]))
    return completed


def load_failure_episodes(
    path: Path,
    selected_scenes: set[str],
    frustum_margin_degrees: float,
) -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("acc")):
                continue
            scene = str(row["map_name"])
            if selected_scenes and scene not in selected_scenes:
                continue
            by_scene[scene].append(build_episode(row, frustum_margin_degrees))
    return by_scene


def summarize(output_dir: Path, expected_failures: int | None = None) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "captures").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                merged[str(row["episode_key"])] = row
    rows = list(merged.values())
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") != "ok"]
    for row in ok:
        row.update(classify_failure_with_distance(row))
    classes = Counter(row["distance_aware_failure_class"] for row in ok)
    terminations: dict[str, Counter] = defaultdict(Counter)
    sizes: dict[str, Counter] = defaultdict(Counter)
    scenes: dict[str, Counter] = defaultdict(Counter)
    for row in ok:
        category = row["distance_aware_failure_class"]
        terminations[row["termination_reason"]][category] += 1
        sizes[row["size_bucket"]][category] += 1
        scenes[row["scene_id"]][category] += 1
    summary = {
        "expected_failure_count": expected_failures,
        "captured_episode_count": len(rows),
        "ok_episode_count": len(ok),
        "error_episode_count": len(errors),
        "complete": expected_failures is None or len(rows) == expected_failures,
        "failure_visibility_classes": dict(classes),
        "derived": {
            "never_target_visible": sum(
                count for key, count in classes.items() if key.startswith("never_")
            ),
            "target_pixels_seen": sum(
                count for key, count in classes.items() if not key.startswith("never_")
            ),
            "clear_target_seen_but_no_stop": classes.get(
                "clear_target_within_20m_but_no_stop", 0
            ),
            "weak_glimpse_but_no_stop": classes.get(
                "weak_target_within_20m_no_stop", 0
            ),
            "target_seen_and_stop_attempted_but_failed": sum(
                count
                for key, count in classes.items()
                if ("stopped" in key and not key.startswith("never_"))
            ),
        },
        "by_termination": {
            key: dict(value) for key, value in sorted(terminations.items())
        },
        "by_size": {key: dict(value) for key, value in sorted(sizes.items())},
        "by_scene": {key: dict(value) for key, value in sorted(scenes.items())},
        "error_episode_keys": [row["episode_key"] for row in errors],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = [
            "episode_key",
            "scene_id",
            "episode_id",
            "true_name",
            "size_bucket",
            "used_in_train",
            "termination_reason",
            "official_osr",
            "ever_stop_command",
            "total_step_count",
            "candidate_step_count",
            "audited_frame_count",
            "target_visible_frame_count",
            "clear_target_frame_count",
            "target_visible_within_20m_frame_count",
            "clear_target_within_20m_frame_count",
            "target_ever_visible",
            "clear_target_ever_visible",
            "target_ever_visible_within_20m",
            "clear_target_ever_visible_within_20m",
            "first_visible_step",
            "first_clear_step",
            "first_visible_within_20m_step",
            "first_clear_within_20m_step",
            "best_visible_step",
            "best_visible_distance_m",
            "peak_target_pixels",
            "failure_visibility_class",
            "distance_aware_failure_class",
            "status",
            "error",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["episode_key"]):
            writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay failed inference trajectories and audit target visibility."
    )
    parser.add_argument("--all-episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", default="")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--base-port", type=int, default=61600)
    parser.add_argument("--target-object-id", type=int, default=42)
    parser.add_argument("--segmentation-width", type=int, default=512)
    parser.add_argument("--segmentation-height", type=int, default=512)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--segmentation-settle-frames", type=int, default=4)
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--camera-alignment-repeats", type=int, default=3)
    parser.add_argument("--camera-position-tolerance", type=float, default=0.25)
    parser.add_argument("--camera-orientation-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--frustum-margin-degrees",
        type=float,
        default=45.0,
        help="Extra horizontal margin beyond half FOV; 45 makes a 90-degree FOV test the full front hemisphere.",
    )
    parser.add_argument("--max-episodes-per-scene", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_failures = sum(
        not bool(json.loads(line).get("acc"))
        for line in args.all_episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if args.summarize_only:
        print(
            json.dumps(
                summarize(args.output_dir, total_failures),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.gpu < 0:
        parser.error("--gpu is required unless --summarize-only is used")

    selected_scenes = {item for item in args.scene_list.split(",") if item}
    by_scene = load_failure_episodes(
        args.all_episodes,
        selected_scenes,
        args.frustum_margin_degrees,
    )
    manifest = {
        "source": str(args.all_episodes.resolve()),
        "selected_scenes": sorted(by_scene),
        "gpu": args.gpu,
        "airsim_port": args.base_port + args.gpu,
        "frustum_margin_degrees": args.frustum_margin_degrees,
        "segmentation_size": [args.segmentation_width, args.segmentation_height],
        "visibility_policy": VisibilityPolicy().to_dict(),
        "episodes": sum(len(items) for items in by_scene.values()),
        "candidate_frames": sum(
            item["candidate_step_count"]
            for items in by_scene.values()
            for item in items
        ),
    }
    (args.output_dir / f"manifest_gpu{args.gpu}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    captures_dir = args.output_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    policy = VisibilityPolicy()
    for scene, episodes in sorted(by_scene.items()):
        if args.max_episodes_per_scene:
            episodes = episodes[: args.max_episodes_per_scene]
        output_path = captures_dir / f"{scene}.jsonl"
        if output_path.exists() and not args.resume:
            raise FileExistsError(output_path)
        completed = load_completed(output_path, args.retry_errors)
        pending_count = sum(
            episode["episode_key"] not in completed for episode in episodes
        )
        if pending_count == 0:
            print(
                json.dumps(
                    {
                        "scene_summary": {
                            "scene": scene,
                            "requested": len(episodes),
                            "skipped": len(episodes),
                            "pending": 0,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        env = SegmentationAirsimTrajRecorder(
            scene,
            airsim_port=args.base_port + args.gpu,
            device_id=args.gpu,
            segmentation_width=args.segmentation_width,
            segmentation_height=args.segmentation_height,
        )
        counts = Counter(requested=len(episodes), pending=pending_count)
        try:
            if not env._client.simSetSegmentationObjectID(".*", 0, True):
                raise RuntimeError("failed to initialize segmentation IDs")
            advance_simulation(env._client, args.segmentation_settle_frames)
            with output_path.open("a", encoding="utf-8") as output:
                for index, episode in enumerate(episodes, start=1):
                    if episode["episode_key"] in completed:
                        counts["skipped"] += 1
                        continue
                    try:
                        if not episode["candidates"]:
                            result = {
                                **{
                                    key: value
                                    for key, value in episode.items()
                                    if key != "candidates"
                                },
                                "status": "ok",
                                "audited_frame_count": 0,
                                "target_visible_frame_count": 0,
                                "clear_target_frame_count": 0,
                                "target_ever_visible": False,
                                "clear_target_ever_visible": False,
                                "peak_target_pixels": 0,
                                "first_visible_step": None,
                                "first_clear_step": None,
                                "best_visible_step": None,
                                "best_visible_distance_m": None,
                                "camera_pose_mismatch_count": 0,
                                "failure_visibility_class": classify_failure(
                                    False, False, episode["ever_stop_command"]
                                ),
                                "frames": [],
                            }
                        else:
                            result = capture_episode(env, episode, args, policy)
                    except Exception as exc:
                        result = {
                            **{
                                key: value
                                for key, value in episode.items()
                                if key != "candidates"
                            },
                            "status": "error",
                            "error": repr(exc),
                        }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    counts["written"] += 1
                    counts["errors"] += int(result["status"] == "error")
                    if result["status"] == "ok":
                        counts[result["failure_visibility_class"]] += 1
                    if index % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "scene": scene,
                                    "progress": index,
                                    "total": len(episodes),
                                    "episode_key": episode["episode_key"],
                                    "candidate_frames": episode["candidate_step_count"],
                                    "status": result["status"],
                                    "class": result.get("failure_visibility_class"),
                                    "visible_frames": result.get(
                                        "target_visible_frame_count"
                                    ),
                                    "clear_frames": result.get(
                                        "clear_target_frame_count"
                                    ),
                                    "error": result.get("error"),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
        finally:
            env.cleanup()
            env.air_runner.processes.clear()
            env.air_runner.settings_files.clear()
        print(
            json.dumps(
                {"scene_summary": {"scene": scene, **counts}},
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay inference Stop frames and measure exact target-instance visibility."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import airsim
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from capture_stop_visibility_cache import (  # noqa: E402
    SEGMENTATION_COLOR_BY_ID,
    SegmentationAirsimTrajRecorder,
    advance_simulation,
    capture_scene_rgb,
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


def parse_run_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=/path/to/all_episodes.jsonl")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"run file does not exist: {path}")
    return label.strip(), path


def normalize_target_positions(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if len(value) >= 3 and all(isinstance(item, (int, float)) for item in value[:3]):
        return [[float(item) for item in value[:3]]]
    positions = []
    for item in value:
        if isinstance(item, list) and len(item) >= 3:
            positions.append([float(component) for component in item[:3]])
    return positions


def nearest_distance(position: list[float], targets: list[list[float]]) -> float:
    point = np.asarray(position[:3], dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(target_array - point[None, :], axis=1).min())


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def resolve_source_image(
    row: dict[str, Any],
    step: dict[str, Any],
    run_root: Path,
) -> Path:
    image_value = str(step.get("image_path") or (row.get("image_paths") or [""])[-1])
    image_path = Path(image_value)
    if image_path.is_absolute() and image_path.is_file():
        return image_path.resolve()

    result_file = Path(str(row.get("_result_file") or ""))
    if not result_file.is_absolute():
        result_file = run_root / result_file
    candidates = []
    if result_file.is_file():
        candidates.append(result_file.parent / image_path)
        if result_file.parent.name == "temp":
            candidates.append(result_file.parent.parent / image_path)
    source_file = Path(str(row.get("source_file") or ""))
    if not source_file.is_absolute():
        source_file = run_root / source_file
    if source_file.is_file():
        candidates.append(source_file.parent / image_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Stop RGB not found: image={image_value!r}, result_file={str(result_file)!r}"
    )


def vehicle_snapshot(step: dict[str, Any]) -> dict[str, list[float]]:
    wait = step.get("pose_wait_before_capture") or {}
    candidates = [
        wait.get("vehicle_pose_after_level"),
        wait.get("vehicle_pose_after_command"),
        (wait.get("reset_status") or {}).get("vehicle_pose"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        position = candidate.get("position")
        quaternion = candidate.get("quat_xyzw")
        if (
            isinstance(position, list)
            and len(position) >= 3
            and isinstance(quaternion, list)
            and len(quaternion) >= 4
        ):
            return {
                "position": [float(value) for value in position[:3]],
                "quat_xyzw": [float(value) for value in quaternion[:4]],
            }

    pose = step.get("pose_before")
    if not isinstance(pose, list) or len(pose) < 4:
        raise ValueError("Stop step is missing pose_before and full vehicle pose")
    yaw = float(pose[3])
    quaternion = airsim.to_quaternion(0.0, 0.0, yaw)
    return {
        "position": [float(value) for value in pose[:3]],
        "quat_xyzw": [
            float(quaternion.x_val),
            float(quaternion.y_val),
            float(quaternion.z_val),
            float(quaternion.w_val),
        ],
    }


def build_queue(
    runs: list[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queue = []
    manifest: dict[str, Any] = {"runs": {}, "group_totals": {}}
    for label, path in runs:
        counts = Counter()
        group_totals: dict[str, Counter] = defaultdict(Counter)
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                size_bucket = parse_size_bucket(row.get("size"))
                seen_group = "seen" if int(row.get("used-in-train", 0)) else "unseen"
                group_key = f"{seen_group}_{size_bucket}"
                group_totals[group_key]["episodes"] += 1
                group_totals[group_key]["successes"] += int(bool(row.get("acc")))
                group_totals[group_key]["oracle_successes"] += int(bool(row.get("osr")))
                counts["episodes"] += 1
                if str(row.get("termination_reason")) != "stop":
                    continue

                steps = row.get("step_records") or []
                if not steps:
                    raise ValueError(
                        f"Stop result has no step_records: {label} "
                        f"{row.get('map_name')}/{row.get('episode_id')}"
                    )
                stop_step = steps[-1]
                if int(stop_step.get("action_id", -1)) != 0:
                    raise ValueError(
                        f"last step is not Stop: {label} "
                        f"{row.get('map_name')}/{row.get('episode_id')}"
                    )
                targets = normalize_target_positions(row.get("pose"))
                if not targets:
                    raise ValueError(
                        f"missing target positions: {label} "
                        f"{row.get('map_name')}/{row.get('episode_id')}"
                    )
                snapshot = vehicle_snapshot(stop_step)
                source_image = resolve_source_image(row, stop_step, path.parent)
                camera_position = stop_step.get("image_camera_pos")
                camera_quaternion = stop_step.get("image_quat_wb")
                item = {
                    "capture_key": (
                        f"{label}::{row['map_name']}::{row['episode_id']}"
                    ),
                    "run_label": label,
                    "run_file": str(path),
                    "scene_id": str(row["map_name"]),
                    "episode_id": str(row["episode_id"]),
                    "success": bool(row.get("acc")),
                    "oracle_success": bool(row.get("osr")),
                    "seen_group": seen_group,
                    "used_in_train": int(row.get("used-in-train", 0)),
                    "size": str(row.get("size") or ""),
                    "size_bucket": size_bucket,
                    "object_name": str(row.get("object_name") or "").strip(),
                    "true_name": str(row.get("true_name") or "").strip(),
                    "target_description": str(row.get("description") or "").strip(),
                    "target_positions": targets,
                    "stop_step": int(stop_step.get("step", len(steps) - 1)),
                    "stop_pose_before": [
                        float(value) for value in stop_step.get("pose_before", [])
                    ],
                    "stop_vehicle_position": snapshot["position"],
                    "stop_vehicle_quat_xyzw": snapshot["quat_xyzw"],
                    "saved_camera_position": (
                        [float(value) for value in camera_position[:3]]
                        if isinstance(camera_position, list) and len(camera_position) >= 3
                        else None
                    ),
                    "saved_camera_quat_xyzw": (
                        [float(value) for value in camera_quaternion[:4]]
                        if isinstance(camera_quaternion, list)
                        and len(camera_quaternion) >= 4
                        else None
                    ),
                    "distance_to_target_m": nearest_distance(
                        snapshot["position"], targets
                    ),
                    "reported_distance_before_m": float(
                        stop_step.get("distance_before", math.nan)
                    ),
                    "source_image_path": str(source_image),
                }
                queue.append(item)
                counts["stop_episodes"] += 1
                counts["successful_stops"] += int(item["success"])
                group_totals[group_key]["stops"] += 1
                group_totals[group_key]["successful_stops"] += int(item["success"])
        manifest["runs"][label] = {"source": str(path), **counts}
        manifest["group_totals"][label] = {
            key: dict(value) for key, value in sorted(group_totals.items())
        }
    queue.sort(
        key=lambda row: (
            row["scene_id"],
            row["run_label"],
            int(row["episode_id"]) if row["episode_id"].isdigit() else row["episode_id"],
        )
    )
    manifest["queue_size"] = len(queue)
    return queue, manifest


def set_exact_vehicle_pose(
    client: airsim.MultirotorClient,
    position: list[float],
    quat_xyzw: list[float],
    settle_frames: int,
) -> None:
    try:
        client.cancelLastTask()
    except Exception:
        pass
    state = airsim.KinematicsState()
    state.position = airsim.Vector3r(*[float(value) for value in position[:3]])
    state.orientation = airsim.Quaternionr(
        *[float(value) for value in quat_xyzw[:4]]
    )
    state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    client.simPause(False)
    client.simSetKinematics(state, True)
    if settle_frames > 0:
        client.simContinueForFrames(int(settle_frames))
    client.simPause(True)
    # Advancing one frame refreshes camera rendering after a large teleport, but
    # physics can slightly change roll/pitch. Reapply the recorded kinematics on
    # the paused frame so the audited camera matches the original Stop capture.
    if settle_frames > 0:
        client.simSetKinematics(state, True)
        client.simSetVehiclePose(
            airsim.Pose(state.position, state.orientation),
            True,
        )


def camera_pose(client: airsim.MultirotorClient, camera_name: str) -> dict[str, list[float]]:
    pose = client.simGetCameraInfo(camera_name).pose
    return {
        "position": [
            float(pose.position.x_val),
            float(pose.position.y_val),
            float(pose.position.z_val),
        ],
        "quat_xyzw": [
            float(pose.orientation.x_val),
            float(pose.orientation.y_val),
            float(pose.orientation.z_val),
            float(pose.orientation.w_val),
        ],
    }


def align_camera_to_saved_world_pose(
    client: airsim.MultirotorClient,
    camera_name: str,
    saved_position: list[float] | None,
    saved_quat_xyzw: list[float] | None,
) -> None:
    if saved_position is None or saved_quat_xyzw is None:
        return
    vehicle = client.simGetVehiclePose()
    vehicle_position = np.asarray(
        [
            vehicle.position.x_val,
            vehicle.position.y_val,
            vehicle.position.z_val,
        ],
        dtype=np.float64,
    )
    vehicle_quaternion = np.asarray(
        [
            vehicle.orientation.x_val,
            vehicle.orientation.y_val,
            vehicle.orientation.z_val,
            vehicle.orientation.w_val,
        ],
        dtype=np.float64,
    )
    vehicle_rotation = Rotation.from_quat(vehicle_quaternion)
    inverse_vehicle = vehicle_rotation.inv()
    relative_position = inverse_vehicle.apply(
        np.asarray(saved_position, dtype=np.float64) - vehicle_position
    )
    relative_quaternion = (
        inverse_vehicle * Rotation.from_quat(saved_quat_xyzw)
    ).as_quat()
    client.simSetCameraPose(
        camera_name,
        airsim.Pose(
            airsim.Vector3r(*[float(value) for value in relative_position]),
            airsim.Quaternionr(
                *[float(value) for value in relative_quaternion]
            ),
        ),
    )


def quaternion_error_degrees(first: list[float], second: list[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def save_review_images(
    item: dict[str, Any],
    replay_rgb: np.ndarray,
    mask: np.ndarray,
    dominant_changed_mask: np.ndarray,
    output_root: Path,
) -> dict[str, str]:
    stem = "__".join(
        [
            safe_name(item["run_label"]),
            safe_name(item["scene_id"]),
            safe_name(item["episode_id"]),
            safe_name(item["true_name"] or item["object_name"]),
            f"step{int(item['stop_step']):03d}",
        ]
    )
    directory = output_root / item["run_label"] / item["seen_group"] / item["size_bucket"]
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(item["source_image_path"]).convert("RGB")
    replay = Image.fromarray(replay_rgb, mode="RGB")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    metrics = mask_metrics(mask)

    def boxed(image: Image.Image) -> Image.Image:
        annotated = image.copy()
        bbox = metrics.get("bbox")
        if bbox:
            scale_x = image.width / max(1, mask_image.width)
            scale_y = image.height / max(1, mask_image.height)
            scaled = [
                int(round(bbox[0] * scale_x)),
                int(round(bbox[1] * scale_y)),
                int(round(bbox[2] * scale_x)),
                int(round(bbox[3] * scale_y)),
            ]
            ImageDraw.Draw(annotated).rectangle(scaled, outline=(255, 220, 0), width=4)
        return annotated

    paths = {
        "source_box_path": directory / f"{stem}__source_box.jpg",
        "replay_box_path": directory / f"{stem}__replay_box.jpg",
        "replay_path": directory / f"{stem}__replay.jpg",
        "mask_path": directory / f"{stem}__mask.png",
        "dominant_changed_mask_path": directory / f"{stem}__dominant_changed_mask.png",
        "dominant_changed_box_path": directory / f"{stem}__dominant_changed_box.jpg",
    }
    boxed(source).save(paths["source_box_path"], quality=92)
    boxed(replay).save(paths["replay_box_path"], quality=92)
    replay.save(paths["replay_path"], quality=95)
    mask_image.save(paths["mask_path"])
    dominant_image = Image.fromarray(dominant_changed_mask.astype(np.uint8) * 255)
    dominant_image.save(paths["dominant_changed_mask_path"])
    dominant_metrics = mask_metrics(dominant_changed_mask)
    dominant_box = replay.copy()
    if dominant_metrics.get("bbox"):
        ImageDraw.Draw(dominant_box).rectangle(
            dominant_metrics["bbox"], outline=(0, 255, 255), width=4
        )
    dominant_box.save(paths["dominant_changed_box_path"], quality=92)
    return {key: str(path.resolve()) for key, path in paths.items()}


def capture_item(
    env: SegmentationAirsimTrajRecorder,
    item: dict[str, Any],
    target_object_id: int,
    settle_frames: int,
    camera_name: str,
    policy: VisibilityPolicy,
    save_debug: bool,
    debug_root: Path,
    camera_position_tolerance: float,
    camera_orientation_tolerance: float,
) -> dict[str, Any]:
    client = env._client
    resolved_name, match_mode, pose_error = set_target_segmentation_id(
        client,
        item["object_name"],
        target_object_id,
        item["target_positions"],
    )
    try:
        if not client.simSetSegmentationObjectID(resolved_name, 0, False):
            raise RuntimeError(f"failed to clear target segmentation ID: {resolved_name}")
        set_exact_vehicle_pose(
            client,
            item["stop_vehicle_position"],
            item["stop_vehicle_quat_xyzw"],
            settle_frames,
        )
        for _ in range(3):
            align_camera_to_saved_world_pose(
                client,
                camera_name,
                item.get("saved_camera_position"),
                item.get("saved_camera_quat_xyzw"),
            )
        replay_camera = camera_pose(client, camera_name)
        baseline = capture_segmentation(client, camera_name)
        if not client.simSetSegmentationObjectID(
            resolved_name, target_object_id, False
        ):
            raise RuntimeError(f"failed to mark target segmentation ID: {resolved_name}")
        marked = capture_segmentation(client, camera_name)
        replay_rgb = capture_scene_rgb(client, camera_name)
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
            dominant_changed_mask = np.zeros_like(changed_mask)
        else:
            dominant_changed_mask = np.logical_and(
                segmentation_color_mask(marked, observed_color),
                np.logical_not(segmentation_color_mask(baseline, observed_color)),
            )
        changed_metrics = mask_metrics(changed_mask)
        dominant_changed_metrics = mask_metrics(dominant_changed_mask)
        dominant_share = float(
            dominant_changed_metrics["pixel_count"]
            / max(1, int(changed_metrics["pixel_count"]))
        )
        bucket = item["size_bucket"]
        min_pixels = int(policy.min_pixels.get(bucket, policy.min_pixels["unknown"]))
        min_short_side = int(
            policy.min_bbox_short_side.get(
                bucket, policy.min_bbox_short_side["unknown"]
            )
        )
        dominant_short_side = min(
            int(dominant_changed_metrics["bbox_width"]),
            int(dominant_changed_metrics["bbox_height"]),
        )
        fallback_valid = bool(
            int(canonical_metrics["pixel_count"]) == 0
            and int(dominant_changed_metrics["pixel_count"]) >= min_pixels
            and dominant_short_side >= min_short_side
            and dominant_share >= 0.5
            and int(baseline_color_count) == 0
        )
        if int(canonical_metrics["pixel_count"]) > 0:
            visibility_mask = canonical_mask
            mask_source = "canonical_id42"
        elif fallback_valid:
            visibility_mask = dominant_changed_mask
            mask_source = "dominant_changed_color_fallback"
        else:
            visibility_mask = np.zeros_like(changed_mask)
            mask_source = "none"
        metrics = mask_metrics(visibility_mask)
        assessment = assess_visibility_frame(
            {"frame_idx": int(item["stop_step"]), "mask": metrics},
            bucket,
            int(metrics["pixel_count"]),
            policy,
        )

        saved_position = item.get("saved_camera_position")
        saved_quaternion = item.get("saved_camera_quat_xyzw")
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
        camera_match = bool(
            (position_error is None or position_error <= camera_position_tolerance)
            and (
                orientation_error is None
                or orientation_error <= camera_orientation_tolerance
            )
        )
        result = {
            **item,
            "status": "ok",
            "resolved_object_name": resolved_name,
            "object_name_match": match_mode,
            "object_name_pose_error_m": pose_error,
            "replay_camera_position": replay_camera["position"],
            "replay_camera_quat_xyzw": replay_camera["quat_xyzw"],
            "camera_position_error_m": position_error,
            "camera_orientation_error_deg": orientation_error,
            "camera_pose_match": camera_match,
            "target_present": int(metrics["pixel_count"]) > 0,
            "geometry_clear": bool(assessment["clear"]),
            "mask": metrics,
            "mask_source": mask_source,
            "canonical_mask": canonical_metrics,
            "all_changed_mask": changed_metrics,
            "dominant_changed_mask": dominant_changed_metrics,
            "dominant_changed_share": dominant_share,
            "fallback_mask_valid": fallback_valid,
            "segmentation_ambiguous": bool(
                int(metrics["pixel_count"]) == 0
                and int(dominant_changed_metrics["pixel_count"]) > 0
            ),
            "visibility_assessment": assessment,
            "observed_segmentation_color_rgb": (
                observed_color.astype(int).tolist()
                if observed_color is not None
                else None
            ),
            "baseline_observed_color_pixels": int(baseline_color_count),
        }
        if save_debug:
            result["debug"] = save_review_images(
                item,
                replay_rgb,
                visibility_mask,
                dominant_changed_mask,
                debug_root,
            )
        return result
    finally:
        if not client.simSetSegmentationObjectID(resolved_name, 0, False):
            raise RuntimeError(f"failed to clear target segmentation ID: {resolved_name}")


def load_completed(path: Path, retry_errors: bool) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if retry_errors and row.get("status") == "error":
                continue
            completed.add(str(row["capture_key"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay inference Stop frames and capture exact target masks."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        help="Repeatable LABEL=/path/to/all_episodes.jsonl specification.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", default="")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=57400)
    parser.add_argument("--target-object-id", type=int, default=42)
    parser.add_argument("--segmentation-width", type=int, default=512)
    parser.add_argument("--segmentation-height", type=int, default=512)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--segmentation-settle-frames", type=int, default=4)
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--camera-position-tolerance", type=float, default=0.25)
    parser.add_argument("--camera-orientation-tolerance", type=float, default=2.0)
    parser.add_argument("--max-items-per-scene", type=int, default=0)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue, manifest = build_queue(args.run)
    selected_scenes = {value for value in args.scene_list.split(",") if value}
    if selected_scenes:
        queue = [row for row in queue if row["scene_id"] in selected_scenes]
    (args.output_dir / "queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
        encoding="utf-8",
    )
    manifest.update(
        {
            "selected_queue_size": len(queue),
            "selected_scenes": sorted(selected_scenes),
            "gpu": args.gpu,
            "airsim_port": args.base_port + args.gpu,
            "segmentation_size": [
                args.segmentation_width,
                args.segmentation_height,
            ],
            "settle_frames": args.settle_frames,
            "camera_position_tolerance_m": args.camera_position_tolerance,
            "camera_orientation_tolerance_deg": args.camera_orientation_tolerance,
            "visibility_policy": VisibilityPolicy().to_dict(),
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queue:
        by_scene[item["scene_id"]].append(item)

    captures_dir = args.output_dir / "captures"
    debug_root = args.output_dir / "review"
    captures_dir.mkdir(parents=True, exist_ok=True)
    policy = VisibilityPolicy()
    summaries = []
    for scene, items in sorted(by_scene.items()):
        if args.max_items_per_scene:
            items = items[: args.max_items_per_scene]
        output_path = captures_dir / f"{scene}.jsonl"
        if output_path.exists() and not args.resume:
            raise FileExistsError(output_path)
        completed = load_completed(output_path, args.retry_errors)
        env = SegmentationAirsimTrajRecorder(
            scene,
            airsim_port=args.base_port + args.gpu,
            device_id=args.gpu,
            segmentation_width=args.segmentation_width,
            segmentation_height=args.segmentation_height,
        )
        counts = Counter(requested=len(items))
        try:
            if not env._client.simSetSegmentationObjectID(".*", 0, True):
                raise RuntimeError("failed to initialize all segmentation IDs to zero")
            advance_simulation(env._client, args.segmentation_settle_frames)
            with output_path.open("a", encoding="utf-8") as output:
                for index, item in enumerate(items, start=1):
                    if item["capture_key"] in completed:
                        counts["skipped"] += 1
                        continue
                    try:
                        result = capture_item(
                            env=env,
                            item=item,
                            target_object_id=args.target_object_id,
                            settle_frames=args.settle_frames,
                            camera_name=args.camera_name,
                            policy=policy,
                            save_debug=args.save_debug,
                            debug_root=debug_root,
                            camera_position_tolerance=args.camera_position_tolerance,
                            camera_orientation_tolerance=args.camera_orientation_tolerance,
                        )
                    except Exception as exc:
                        result = {
                            **item,
                            "status": "error",
                            "error": repr(exc),
                        }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    counts["written"] += 1
                    counts["errors"] += int(result["status"] == "error")
                    counts["target_present"] += int(bool(result.get("target_present")))
                    counts["geometry_clear"] += int(bool(result.get("geometry_clear")))
                    counts["camera_pose_mismatch"] += int(
                        result.get("status") == "ok"
                        and not bool(result.get("camera_pose_match"))
                    )
                    print(
                        json.dumps(
                            {
                                "capture_key": item["capture_key"],
                                "scene": scene,
                                "progress": index,
                                "total": len(items),
                                "status": result["status"],
                                "success": item["success"],
                                "target_present": result.get("target_present"),
                                "geometry_clear": result.get("geometry_clear"),
                                "pixels": (result.get("mask") or {}).get("pixel_count"),
                                "camera_pose_match": result.get("camera_pose_match"),
                                "error": result.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            env.cleanup()
            # AirSim settings paths are reused per port. Prevent __del__ from
            # cleaning this recorder a second time after the next scene starts.
            env.air_runner.processes.clear()
            env.air_runner.settings_files.clear()
        summary = {"scene": scene, **counts, "output": str(output_path)}
        summaries.append(summary)
        print(json.dumps({"scene_summary": summary}, ensure_ascii=False), flush=True)

    (args.output_dir / "capture_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

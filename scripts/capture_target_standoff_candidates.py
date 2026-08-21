#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from capture_stop_visibility_cache import (  # noqa: E402
    SegmentationAirsimTrajRecorder,
    SEGMENTATION_COLOR_BY_ID,
    capture_scene_rgb,
    capture_segmentation,
    identify_changed_segmentation_color,
    load_requested_keys,
    set_target_segmentation_id,
)
from vlm_baseline.stop_visibility import (  # noqa: E402
    canonical_stencil_difference_mask,
    mask_metrics,
    parse_size_bucket,
)
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt  # noqa: E402


DEFAULT_RADII = {
    "small": (3.0, 5.0, 8.0, 12.0),
    "mid": (4.0, 6.0, 9.0, 13.0),
    "big": (6.0, 9.0, 13.0, 18.0),
    "unknown": (4.0, 7.0, 11.0, 16.0),
}


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    sources = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for jsonl in sources:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "trajectory_key" not in row:
                continue
            rows[str(row["trajectory_key"])] = row
    return rows


def angle_wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def standoff_poses(
    target_position: list[float],
    size_bucket: str,
    azimuth_count: int,
    height_offsets: tuple[float, ...],
    radii: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    target_x, target_y, target_z = (float(value) for value in target_position)
    poses = []
    for radius in radii or DEFAULT_RADII[size_bucket]:
        for height_offset in height_offsets:
            for azimuth_index in range(azimuth_count):
                azimuth = 2.0 * math.pi * azimuth_index / azimuth_count
                x = target_x + radius * math.cos(azimuth)
                y = target_y + radius * math.sin(azimuth)
                z = target_z - height_offset
                yaw = angle_wrap_radians(azimuth + math.pi)
                poses.append(
                    {
                        "pose": [x, y, z, yaw],
                        "radius_m": radius,
                        "height_offset_m": height_offset,
                        "azimuth_deg": math.degrees(azimuth),
                    }
                )
    return poses


def save_candidate_images(
    replay_rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    candidate_id: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(replay_rgb, mode="RGB")
    image_path = output_dir / f"{candidate_id}_replay.jpg"
    image.save(image_path, quality=95)

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_path = output_dir / f"{candidate_id}_mask.png"
    mask_image.save(mask_path)

    overlay = image.copy()
    metrics = mask_metrics(mask)
    if metrics["bbox"]:
        ImageDraw.Draw(overlay).rectangle(metrics["bbox"], outline=(255, 220, 0), width=4)
    overlay_path = output_dir / f"{candidate_id}_overlay.jpg"
    overlay.save(overlay_path, quality=95)
    return {
        "replay_image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
    }


def capture_trajectory(
    env: SegmentationAirsimTrajRecorder,
    trajectory: dict[str, Any],
    output_dir: Path,
    target_object_id: int,
    azimuth_count: int,
    height_offsets: tuple[float, ...],
    settle_frames: int,
    camera_name: str,
    radii_by_size: dict[str, tuple[float, ...]],
    depth_formatter: UAVONSingleViewDepthPrompt,
) -> dict[str, Any]:
    client = env._client
    target_positions = trajectory.get("target_positions") or []
    if not target_positions:
        raise ValueError(f"missing target positions: {trajectory['trajectory_key']}")
    object_name = str(trajectory.get("object_name") or "")
    resolved_name, match_mode, pose_error = set_target_segmentation_id(
        client,
        object_name,
        target_object_id,
        target_positions,
    )
    if not client.simSetSegmentationObjectID(resolved_name, 0, False):
        raise RuntimeError(f"failed to clear target segmentation ID: {resolved_name}")

    size_bucket = parse_size_bucket(trajectory.get("size"))
    candidates = []
    debug_dir = output_dir / "candidates" / str(trajectory["trajectory_key"]).replace(
        "::", "__"
    )
    try:
        candidate_index = 0
        for target_index, target_position in enumerate(target_positions):
            for spec in standoff_poses(
                target_position,
                size_bucket,
                azimuth_count,
                height_offsets,
                radii=radii_by_size[size_bucket],
            ):
                candidate_id = f"s{candidate_index:03d}"
                placement = env.zero_kinematics_at_pose(
                    spec["pose"], settle_frames=settle_frames
                )
                if not client.simSetSegmentationObjectID(resolved_name, 0, False):
                    raise RuntimeError(
                        f"failed to clear target segmentation ID: {resolved_name}"
                    )
                baseline = capture_segmentation(client, camera_name)
                if not client.simSetSegmentationObjectID(
                    resolved_name, target_object_id, False
                ):
                    raise RuntimeError(
                        f"failed to mark target segmentation ID: {resolved_name}"
                    )
                marked = capture_segmentation(client, camera_name)
                canonical_color = SEGMENTATION_COLOR_BY_ID.get(int(target_object_id))
                if canonical_color is None:
                    raise ValueError(
                        "missing canonical segmentation color for object ID "
                        f"{target_object_id}"
                    )
                mask = canonical_stencil_difference_mask(
                    baseline,
                    marked,
                    canonical_color,
                )
                replay_rgb = capture_scene_rgb(client, camera_name)
                capture = env._capture_images(
                    camera_names=[camera_name],
                    capture_depth=True,
                )
                depth = capture[camera_name]["depth"]
                depth_grid = depth_formatter.depth_to_grid(depth).tolist()
                color, _ = identify_changed_segmentation_color(baseline, marked)
                paths = save_candidate_images(
                    replay_rgb,
                    mask,
                    debug_dir,
                    candidate_id,
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "frame_idx": 100000 + candidate_index,
                        "source_type": "target_facing_standoff",
                        "target_position_index": target_index,
                        **spec,
                        "distance_to_target": float(spec["radius_m"]),
                        "collision_info": placement.get("collision_info"),
                        "actual_pose": placement.get("actual_pose"),
                        "mask": mask_metrics(mask),
                        "depth_grid": depth_grid,
                        "observed_segmentation_color_rgb": (
                            color.astype(int).tolist() if color is not None else None
                        ),
                        **paths,
                    }
                )
                candidate_index += 1
    finally:
        if not client.simSetSegmentationObjectID(resolved_name, 0, False):
            raise RuntimeError(f"failed to clear target segmentation ID: {resolved_name}")

    collision_free = [
        row
        for row in candidates
        if not bool((row.get("collision_info") or {}).get("has_collided"))
    ]
    visible = [row for row in collision_free if int(row["mask"]["pixel_count"]) > 0]
    return {
        "trajectory_key": trajectory["trajectory_key"],
        "status": "ok",
        "scene_id": trajectory["scene_id"],
        "episode_id": trajectory["episode_id"],
        "pose_idx": trajectory["pose_idx"],
        "true_name": trajectory.get("true_name"),
        "target_description": trajectory.get("target_description"),
        "size": trajectory.get("size"),
        "size_bucket": size_bucket,
        "target_positions": target_positions,
        "object_name": object_name,
        "resolved_object_name": resolved_name,
        "object_name_match": match_mode,
        "object_name_pose_error_m": pose_error,
        "candidate_count": len(candidates),
        "collision_free_candidate_count": len(collision_free),
        "visible_collision_free_candidate_count": len(visible),
        "depth_grid_size": depth_formatter.grid_size,
        "depth_max_meters": depth_formatter.max_meters,
        "radii_m": list(radii_by_size[size_bucket]),
        "height_offsets_m": list(height_offsets),
        "frames": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture target-facing standoff views for trajectories with no valid Stop."
    )
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--trajectory-keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=38400)
    parser.add_argument("--target-object-id", type=int, default=42)
    parser.add_argument("--azimuth-count", type=int, default=8)
    parser.add_argument("--height-offsets", default="1.5,3.0")
    parser.add_argument("--radii-small", default="3,5,8,12")
    parser.add_argument("--radii-mid", default="4,6,9,13")
    parser.add_argument("--radii-big", default="6,9,13,18")
    parser.add_argument("--radii-unknown", default="4,7,11,16")
    parser.add_argument("--depth-grid-size", type=int, default=3)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cache = load_cache(args.input_cache)
    requested = load_requested_keys(args.trajectory_keys) or set()
    scenes = {item for item in args.scene_list.split(",") if item}
    selected = [
        cache[key]
        for key in sorted(requested)
        if key in cache and str(cache[key]["scene_id"]) in scenes
    ]
    missing = sorted(
        key
        for key in requested
        if key.split("::", 1)[0] in scenes and key not in cache
    )
    if missing:
        raise KeyError(f"requested trajectories missing from input cache: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    height_offsets = tuple(float(value) for value in args.height_offsets.split(","))
    radii_by_size = {
        bucket: tuple(
            float(value)
            for value in getattr(args, f"radii_{bucket}").split(",")
            if value.strip()
        )
        for bucket in DEFAULT_RADII
    }
    if any(not values for values in radii_by_size.values()):
        raise ValueError("each --radii-* option must contain at least one radius")
    depth_formatter = UAVONSingleViewDepthPrompt(
        grid_size=args.depth_grid_size,
        max_meters=args.depth_max_meters,
    )

    for scene in sorted(scenes):
        scene_rows = [row for row in selected if row["scene_id"] == scene]
        if not scene_rows:
            continue
        output_path = args.output_dir / f"{scene}.jsonl"
        if output_path.exists() and not args.resume:
            raise FileExistsError(output_path)
        existing_keys = set()
        if args.resume and output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "ok" and row.get("trajectory_key"):
                    existing_keys.add(str(row["trajectory_key"]))
        scene_rows = [
            row
            for row in scene_rows
            if str(row["trajectory_key"]) not in existing_keys
        ]
        if not scene_rows:
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "event": "scene_skip_completed",
                        "existing_trajectories": len(existing_keys),
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
            segmentation_width=512,
            segmentation_height=512,
        )
        try:
            mode = "a" if args.resume and output_path.exists() else "w"
            with output_path.open(mode, encoding="utf-8") as output:
                for index, trajectory in enumerate(scene_rows, start=1):
                    result = capture_trajectory(
                        env,
                        trajectory,
                        args.output_dir,
                        args.target_object_id,
                        args.azimuth_count,
                        height_offsets,
                        args.settle_frames,
                        args.camera_name,
                        radii_by_size,
                        depth_formatter,
                    )
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "scene": scene,
                                "progress": f"{index}/{len(scene_rows)}",
                                "trajectory_key": result["trajectory_key"],
                                "candidates": result["candidate_count"],
                                "collision_free": result[
                                    "collision_free_candidate_count"
                                ],
                                "visible_collision_free": result[
                                    "visible_collision_free_candidate_count"
                                ],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            env.cleanup()


if __name__ == "__main__":
    main()

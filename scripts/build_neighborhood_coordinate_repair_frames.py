#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt  # noqa: E402


ACTION_ID = {
    "Stop": 0,
    "Move Forward": 1,
    "Turn Left": 2,
    "Turn Right": 3,
    "Ascend": 4,
    "Descend": 5,
}
ACTION_VECTORS = {
    "Stop": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Move Forward": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Turn Left": [0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Turn Right": [0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0],
    "Ascend": [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
    "Descend": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
}
UAVON_ACTION = {
    "Stop": "stop",
    "Move Forward": "forward",
    "Turn Left": "rotl",
    "Turn Right": "rotr",
    "Ascend": "ascend",
    "Descend": "descend",
}


def normalize_scene(row: dict[str, Any]) -> str:
    scene = str(row.get("scene_key") or row.get("map_name", ""))
    scene = scene.replace("_TrainSets", "")
    return {
        "NeighborhoodTrain": "Neighborhood",
        "ModularNeighborhood": "Neighborhood",
    }.get(scene, scene)


def load_metadata(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    descriptions: dict[str, str] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        scene = normalize_scene(row)
        episode_id = str(row["episode_id"])
        lookup[(scene, episode_id)] = row
        object_name = str(row.get("object_name") or "").strip()
        description = str(row.get("description") or "").strip()
        if object_name and description:
            descriptions.setdefault(object_name, description)
    return lookup, descriptions


def yaw_delta_degrees(current: float, following: float) -> float:
    delta = math.degrees(following - current)
    return (delta + 180.0) % 360.0 - 180.0


def derive_action(current: list[float], following: list[float] | None) -> str:
    if following is None:
        return "Stop"
    dx = float(following[0]) - float(current[0])
    dy = float(following[1]) - float(current[1])
    dz = float(following[2]) - float(current[2])
    xy = math.hypot(dx, dy)
    dyaw = yaw_delta_degrees(float(current[3]), float(following[3]))
    if abs(dz) >= 1.0 and abs(dz) >= xy:
        return "Descend" if dz > 0 else "Ascend"
    if abs(dyaw) >= 15.0 and abs(dyaw) >= xy * 3.0:
        return "Turn Right" if dyaw > 0 else "Turn Left"
    if xy >= 1.0:
        return "Move Forward"
    if abs(dyaw) >= 15.0:
        return "Turn Right" if dyaw > 0 else "Turn Left"
    return "Stop"


def numeric_sort(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (sys.maxsize, value)


def iter_records(record_root: Path, scene: str) -> Iterable[tuple[str, str, Path]]:
    root = record_root / "json" / scene
    for episode_dir in sorted(root.iterdir(), key=lambda path: numeric_sort(path.name)):
        if not episode_dir.is_dir():
            continue
        for path in sorted(episode_dir.glob("*.json"), key=lambda item: numeric_sort(item.stem)):
            yield episode_dir.name, path.stem, path


def resolve_capture_path(
    record_root: Path,
    scene: str,
    relative_path: str,
) -> Path:
    path = record_root / "images" / scene / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def target_description(
    metadata: dict[str, Any], descriptions: dict[str, str]
) -> str:
    description = str(metadata.get("description") or "").strip()
    object_name = str(metadata.get("object_name") or "").strip()
    true_name = str(metadata.get("true_name") or "").strip()
    if description:
        return description
    if descriptions.get(object_name):
        return descriptions[object_name]
    fallback = true_name or object_name.replace("_", " ")
    return f"object named {fallback}" if fallback else "target object"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build aligned Neighborhood frame labels and 3x3 DepthGrid records from "
            "the coordinate-repaired AirSim captures."
        )
    )
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default="Neighborhood")
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--depth-grid-size", type=int, default=3)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "frames": args.output_dir / "train_frames.jsonl",
        "depth_cache": args.output_dir / "depth_grid_cache" / f"{args.scene}.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    outputs["depth_cache"].parent.mkdir(parents=True, exist_ok=True)

    metadata_lookup, descriptions = load_metadata(args.metadata)
    formatter = UAVONSingleViewDepthPrompt(
        grid_size=args.depth_grid_size,
        max_meters=args.depth_max_meters,
    )
    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    episode_keys: set[str] = set()
    raw_depth_bytes = 0

    with (
        outputs["frames"].open("x", encoding="utf-8") as frame_output,
        outputs["depth_cache"].open("x", encoding="utf-8") as depth_output,
    ):
        for episode_id, pose_idx, record_path in iter_records(
            args.record_root, args.scene
        ):
            metadata = metadata_lookup.get((args.scene, episode_id))
            if metadata is None:
                raise KeyError(f"metadata missing for {args.scene}::{episode_id}")
            data = json.loads(record_path.read_text(encoding="utf-8"))
            poses = data.get("record_list") or []
            actions = data.get("action_list") or []
            camera_rows = (data.get("image_dict") or {}).get(args.camera_name) or []
            if not poses:
                raise ValueError(f"empty record_list: {record_path}")
            if len(poses) != len(actions):
                raise ValueError(
                    f"record/action mismatch in {record_path}: {len(poses)} != {len(actions)}"
                )
            if len(poses) != len(camera_rows):
                raise ValueError(
                    f"record/image mismatch in {record_path}: {len(poses)} != {len(camera_rows)}"
                )

            episode_key = f"{args.scene}::{episode_id}::{pose_idx}"
            episode_keys.add(episode_key)
            start_recovery = data.get("coordinate_repair_start_recovery")
            if start_recovery:
                stats["start_recovered_trajectories"] += 1
            for frame_idx, (pose, camera_row) in enumerate(zip(poses, camera_rows)):
                following = poses[frame_idx + 1] if frame_idx + 1 < len(poses) else None
                action_name = derive_action(pose, following)
                image_path = resolve_capture_path(
                    args.record_root,
                    args.scene,
                    str(camera_row.get("rgb") or ""),
                )
                depth_rel = str(camera_row.get("depth") or "")
                if not depth_rel:
                    raise KeyError(f"capture has no depth path: {record_path} frame={frame_idx}")
                depth_path = resolve_capture_path(
                    args.record_root,
                    args.scene,
                    depth_rel,
                )
                depth = np.load(depth_path, mmap_mode="r")
                if depth.ndim != 2 or not depth.size:
                    raise ValueError(f"invalid depth array {depth.shape}: {depth_path}")
                depth_grid = formatter.depth_to_grid(np.asarray(depth)).tolist()
                raw_depth_bytes += depth_path.stat().st_size
                key = f"{episode_key}::{frame_idx}"
                row = {
                    "planner": "octmen_agent_coordinate_repair_astar_v1",
                    "source_record": str(record_path.resolve()),
                    "episode_key": episode_key,
                    "scene_id": args.scene,
                    "map_name": metadata.get("map_name", args.scene),
                    "episode_id": episode_id,
                    "pose_idx": pose_idx,
                    "step_id": frame_idx,
                    "frame_idx": frame_idx,
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_grid": depth_grid,
                    "target_description": target_description(
                        metadata, descriptions
                    ).lower(),
                    "true_name": str(metadata.get("true_name") or "").strip(),
                    "object_name": str(metadata.get("object_name") or "").strip(),
                    "size": str(metadata.get("size") or "").strip(),
                    "action_name": action_name,
                    "action_id": ACTION_ID[action_name],
                    "uavon_action": UAVON_ACTION[action_name],
                    "action_vector": ACTION_VECTORS[action_name],
                    "pose": [float(value) for value in pose],
                    "next_pose": (
                        [float(value) for value in following]
                        if following is not None
                        else None
                    ),
                    "start_position": data.get("start_pos")
                    or (metadata.get("start_pose") or {}).get("start_position"),
                    "goal_position": data.get("goal_pos"),
                    "coordinate_repair": metadata.get("coordinate_repair"),
                    "coordinate_repair_start_recovery": start_recovery,
                }
                frame_output.write(json.dumps(row, ensure_ascii=False) + "\n")
                depth_output.write(
                    json.dumps(
                        {
                            "key": key,
                            "scene_id": args.scene,
                            "episode_id": episode_id,
                            "pose_idx": pose_idx,
                            "frame_idx": frame_idx,
                            "pose": row["pose"],
                            "depth_grid": depth_grid,
                            "depth_grid_size": args.depth_grid_size,
                            "depth_max_meters": args.depth_max_meters,
                            "source_depth_path": str(depth_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["frames"] += 1
                action_counts[action_name] += 1
            stats["trajectories"] += 1
            if stats["trajectories"] % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "build_progress",
                            "trajectories": stats["trajectories"],
                            "frames": stats["frames"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if action_counts["Stop"] != len(episode_keys):
        raise RuntimeError(
            f"expected one Stop per trajectory: {action_counts['Stop']} != {len(episode_keys)}"
        )
    manifest = {
        "format": "uavon_neighborhood_coordinate_repair_frames_v1",
        "record_root": str(args.record_root.resolve()),
        "metadata": str(args.metadata.resolve()),
        "scene": args.scene,
        "trajectories": len(episode_keys),
        "frames": stats["frames"],
        "action_counts": dict(action_counts),
        "start_recovered_trajectories": stats[
            "start_recovered_trajectories"
        ],
        "depth_grid_size": args.depth_grid_size,
        "depth_max_meters": args.depth_max_meters,
        "raw_depth_bytes": raw_depth_bytes,
        "label_rule": (
            "action is derived from record_list[i] -> record_list[i+1]; final frame is Stop"
        ),
        "coordinate_repair": {"dx": 130.0, "dy": 110.0, "dz": 0.0},
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

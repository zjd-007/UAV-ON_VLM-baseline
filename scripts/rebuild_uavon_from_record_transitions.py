#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


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


def normalize_scene(scene: str) -> str:
    aliases = {
        "NeighborhoodTrain": "Neighborhood",
        "ModularNeighborhood": "Neighborhood",
    }
    return aliases.get(scene, scene)


def load_raw_lookup(dataset_jsons: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    object_descriptions: dict[str, str] = {}
    for dataset_json in dataset_jsons:
        rows = json.loads(dataset_json.read_text(encoding="utf-8"))
        for row in rows:
            scene = row.get("scene_key") or row.get("map_name", "").replace("_TrainSets", "")
            scene = normalize_scene(scene)
            lookup.setdefault((scene, str(row["episode_id"])), row)
            object_name = str(row.get("object_name") or "").strip()
            description = str(row.get("description") or "").strip()
            if object_name and description:
                object_descriptions.setdefault(object_name, description)
    return lookup, object_descriptions


def yaw_delta_degrees(cur_yaw: float, next_yaw: float) -> float:
    delta = (next_yaw - cur_yaw) * 180.0 / math.pi
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def derive_action(cur_pose: list[float], next_pose: list[float]) -> str:
    dx = float(next_pose[0]) - float(cur_pose[0])
    dy = float(next_pose[1]) - float(cur_pose[1])
    dz = float(next_pose[2]) - float(cur_pose[2])
    xy = math.hypot(dx, dy)
    dyaw = yaw_delta_degrees(float(cur_pose[3]), float(next_pose[3]))

    if abs(dz) >= 1.0 and abs(dz) >= xy:
        return "Descend" if dz > 0 else "Ascend"

    if abs(dyaw) >= 15.0 and abs(dyaw) >= xy * 3.0:
        return "Turn Right" if dyaw > 0 else "Turn Left"

    if xy >= 1.0:
        return "Move Forward"

    if abs(dyaw) >= 15.0:
        return "Turn Right" if dyaw > 0 else "Turn Left"

    return "Stop"


def iter_record_files(record_json_root: Path):
    for record_path in sorted(
        record_json_root.glob("*/*/*.json"),
        key=lambda p: (
            p.relative_to(record_json_root).parts[0],
            int(p.relative_to(record_json_root).parts[1]),
            int(p.stem) if p.stem.isdigit() else p.stem,
        ),
    ):
        rel = record_path.relative_to(record_json_root)
        yield rel.parts[0], rel.parts[1], record_path.stem, record_path


def build_row(
    dataset_root: Path,
    raw_lookup: dict[tuple[str, str], dict[str, Any]],
    object_descriptions: dict[str, str],
    data: dict[str, Any],
    env: str,
    episode_id: str,
    pose_idx: str,
    record_path: Path,
    step_id: int,
    pose: list[float],
    next_pose: list[float] | None,
    image_rel: str,
) -> dict[str, Any]:
    raw = raw_lookup.get((env, str(episode_id)), {})
    action_name = "Stop" if next_pose is None else derive_action(pose, next_pose)
    image_path = dataset_root / "generated" / "record_output" / "images" / env / image_rel
    object_name = raw.get("object_name", "")
    true_name = str(raw.get("true_name") or "").strip()
    object_name_text = str(object_name or "").strip()
    target_description = str(raw.get("description") or "").strip() or object_descriptions.get(object_name_text, "")
    if not target_description:
        fallback_name = true_name or object_name_text.replace("_", " ")
        target_description = f"object named {fallback_name}" if fallback_name else "target object"
    return {
        "planner": "octmen_agent_record_transition",
        "source_record": str(record_path),
        "episode_key": f"{env}::{episode_id}::{pose_idx}",
        "scene_id": env,
        "map_name": raw.get("map_name", env),
        "episode_id": str(episode_id),
        "pose_idx": str(pose_idx),
        "step_id": step_id,
        "frame_idx": step_id,
        "action_name": action_name,
        "action_id": ACTION_ID[action_name],
        "uavon_action": UAVON_ACTION[action_name],
        "step_size": 0.0 if action_name == "Stop" else ACTION_VECTORS[action_name][ACTION_ID[action_name]],
        "action_vector": ACTION_VECTORS[action_name],
        "pose": pose,
        "next_pose": next_pose,
        "start_position": data.get("start_pos") or raw.get("start_pose", {}).get("start_position"),
        "goal_position": data.get("goal_pos"),
        "target_description": target_description,
        "true_name": true_name,
        "object_name": raw.get("object_name", ""),
        "image_path": str(image_path.resolve()),
    }


def rebuild(dataset_root: Path, metadata_jsons: list[Path]) -> dict[str, Any]:
    record_json_root = dataset_root / "generated" / "record_output" / "json"
    expert_dir = dataset_root / "expert" / "frame_index_jsonl"
    processed_dir = dataset_root / "processed" / "nomemory_baseline"
    brushify_path = expert_dir / "train_octmen_astar_BrushifyUrban_frames.jsonl"
    other_path = expert_dir / "train_octmen_astar_no_BrushifyUrban_frames.jsonl"
    processed_path = processed_dir / "train_frames.jsonl"
    manifest_path = processed_dir / "manifest.json"

    expert_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_lookup, object_descriptions = load_raw_lookup(metadata_jsons)
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    episodes: set[str] = set()
    rows = 0
    missing_images: list[str] = []
    bad_records: list[str] = []

    with (
        brushify_path.open("w", encoding="utf-8") as brushify_out,
        other_path.open("w", encoding="utf-8") as other_out,
        processed_path.open("w", encoding="utf-8") as processed_out,
    ):
        for env, episode_id, pose_idx, record_path in iter_record_files(record_json_root):
            data = json.loads(record_path.read_text(encoding="utf-8"))
            records = data.get("record_list") or []
            camera_rows = data.get("image_dict", {}).get("uav_on_0", [])
            if len(records) != len(camera_rows):
                bad_records.append(f"{record_path}: record_list={len(records)} images={len(camera_rows)}")
                continue

            for step_id, pose in enumerate(records):
                next_pose = records[step_id + 1] if step_id + 1 < len(records) else None
                image_rel = camera_rows[step_id].get("rgb", "")
                row = build_row(
                    dataset_root,
                    raw_lookup,
                    object_descriptions,
                    data,
                    env,
                    episode_id,
                    pose_idx,
                    record_path,
                    step_id,
                    pose,
                    next_pose,
                    image_rel,
                )
                if not Path(row["image_path"]).is_file():
                    missing_images.append(row["image_path"])
                    continue

                line = json.dumps(row, ensure_ascii=False) + "\n"
                if env == "BrushifyUrban":
                    brushify_out.write(line)
                else:
                    other_out.write(line)

                clean = {
                    "episode_key": row["episode_key"],
                    "scene_id": row["scene_id"],
                    "episode_id": row["episode_id"],
                    "pose_idx": row["pose_idx"],
                    "frame_idx": row["frame_idx"],
                    "image_path": row["image_path"],
                    "target_description": row["target_description"].strip().lower(),
                    "action_name": row["action_name"],
                    "action_vector": row["action_vector"],
                }
                processed_out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                action_counts[row["action_name"]] += 1
                scene_counts[env] += 1
                episodes.add(row["episode_key"])
                rows += 1

    if bad_records:
        raise RuntimeError(f"Found {len(bad_records)} bad record files. Examples: {bad_records[:5]}")
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images. Examples: {missing_images[:5]}")

    manifest = {
        "format": "uavon_nomemory_single_rgb_record_transition_v2",
        "source_record_json_root": str(record_json_root),
        "source_metadata_jsons": [str(path) for path in metadata_jsons],
        "expert_frame_jsonls": [str(brushify_path), str(other_path)],
        "prepared_jsonl": str(processed_path),
        "trajectories": len(episodes),
        "frames": rows,
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "label_rule": "action is derived from record_list[i] -> record_list[i+1]; final frame is Stop",
        "forbidden_model_inputs": ["depth", "history", "pose", "goal", "distance", "memory", "previous_action"],
        "normalization": "min_max",
        "action_mask": [True, True, True, True, True, True, False, False],
        "dataset_statistics": {
            "uavon_nomemory": {
                "action": {
                    "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "max": [1.0, 3.0, 30.0, 30.0, 3.0, 3.0, 0.0, 0.0],
                    "mask": [True, True, True, True, True, True, False, False],
                },
                "num_trajectories": len(episodes),
                "num_transitions": rows,
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild UAV-ON no-memory data from recorded pose transitions.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset"))
    parser.add_argument(
        "--metadata-json",
        type=Path,
        action="append",
        default=None,
        help="Metadata split JSON. Can be passed multiple times. Defaults to train/val/test.",
    )
    args = parser.parse_args()
    if args.metadata_json is None:
        split_root = args.dataset_root / "splits" / "uavon_raw_json"
        metadata_jsons = [split_root / "train.json", split_root / "val.json", split_root / "test.json"]
    else:
        metadata_jsons = args.metadata_json

    manifest = rebuild(args.dataset_root.resolve(), [path.resolve() for path in metadata_jsons])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

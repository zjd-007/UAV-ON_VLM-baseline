#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_METADATA = DATASET_ROOT / "splits" / "uavon_raw_json" / "train.json"
DEFAULT_ALIGNMENT = (
    DATASET_ROOT
    / "processed"
    / "stop_visible_full_audit"
    / "full_canonical_geometry_v1_20260812_153000"
    / "actor_pose_alignment"
    / "Neighborhood.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def shift_xyz(values: list[float], dx: float, dy: float) -> list[float]:
    shifted = [float(value) for value in values]
    shifted[0] += dx
    shifted[1] += dy
    return shifted


def select_episodes(
    metadata_rows: list[dict[str, Any]],
    aligned_root: Path,
    actor_rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        if row.get("scene_key") != "NeighborhoodTrain":
            continue
        record = aligned_root / "json" / "Neighborhood" / str(row["episode_id"]) / "0.json"
        if record.is_file():
            by_actor[str(row.get("object_name") or "")].append(row)

    selected = []
    used_true_names: Counter[str] = Counter()
    used_actors: set[str] = set()
    ordered_actors = sorted(
        actor_rows,
        key=lambda row: (
            abs(float(row.get("actor_to_target_error_xy_m") or 0.0) - 170.29386),
            str(row.get("object_name") or ""),
        ),
        reverse=True,
    )
    # First cover one episode per semantic class, retaining the strongest geometric
    # outliers when duplicate classes (notably Human) exist.
    for actor in ordered_actors:
        name = str(actor.get("object_name") or "")
        candidates = by_actor.get(name) or []
        if not candidates:
            continue
        true_name = str(candidates[0].get("true_name") or name)
        if used_true_names[true_name]:
            continue
        chosen = sorted(candidates, key=lambda row: int(row["episode_id"]))[0]
        selected.append(chosen)
        used_true_names[true_name] += 1
        used_actors.add(name)
        if len(selected) >= count:
            return selected

    for actor in ordered_actors:
        name = str(actor.get("object_name") or "")
        if name in used_actors:
            continue
        candidates = by_actor.get(name) or []
        if not candidates:
            continue
        selected.append(sorted(candidates, key=lambda row: int(row["episode_id"]))[0])
        used_actors.add(name)
        if len(selected) >= count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a non-destructive Neighborhood XY-coordinate smoke dataset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--dx", type=float, default=130.0)
    parser.add_argument("--dy", type=float, default=110.0)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    aligned_out = args.output_dir / "aligned_xy_shifted"
    json_out = aligned_out / "json" / "Neighborhood"
    images_link = aligned_out / "images"
    json_out.mkdir(parents=True)
    images_link.symlink_to((args.aligned_root / "images").resolve(), target_is_directory=True)

    metadata_rows = json.loads(args.metadata.read_text(encoding="utf-8"))
    actor_rows = read_jsonl(args.alignment)
    selected = select_episodes(
        metadata_rows,
        args.aligned_root,
        actor_rows,
        args.count,
    )
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count} trajectories, selected {len(selected)}")

    selected_ids = {str(row["episode_id"]) for row in selected}
    shifted_metadata = []
    for row in metadata_rows:
        copied = json.loads(json.dumps(row))
        if copied.get("scene_key") == "NeighborhoodTrain":
            copied["pose"] = [shift_xyz(position, args.dx, args.dy) for position in copied["pose"]]
            start_pose = copied.get("start_pose") or {}
            if start_pose.get("start_position"):
                start_pose["start_position"] = shift_xyz(
                    start_pose["start_position"], args.dx, args.dy
                )
        shifted_metadata.append(copied)

    metadata_out = args.output_dir / "train_xy_shifted.json"
    metadata_out.write_text(
        json.dumps(shifted_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sample_rows = []
    collision_rows = []
    for row in selected:
        episode_id = str(row["episode_id"])
        source_record = args.aligned_root / "json" / "Neighborhood" / episode_id / "0.json"
        data = json.loads(source_record.read_text(encoding="utf-8"))
        data["record_list"] = [
            shift_xyz(pose, args.dx, args.dy) for pose in data.get("record_list") or []
        ]
        if data.get("goal_pos"):
            data["goal_pos"] = shift_xyz(data["goal_pos"], args.dx, args.dy)
        shifted_actions = []
        for action in data.get("action_list") or []:
            action_copy = json.loads(json.dumps(action))
            if len(action_copy) > 1 and isinstance(action_copy[1], list):
                action_copy[1] = shift_xyz(action_copy[1], args.dx, args.dy)
            shifted_actions.append(action_copy)
        data["action_list"] = shifted_actions
        destination = json_out / episode_id / "0.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        frames = data.get("record_list") or []
        actions = data.get("action_list") or []
        camera_rows = data.get("image_dict", {}).get("uav_on_0", [])
        if not (len(frames) == len(actions) == len(camera_rows)):
            raise ValueError(
                f"record/action/image mismatch: {episode_id}: "
                f"{len(frames)}/{len(actions)}/{len(camera_rows)}"
            )
        true_name = str(row.get("true_name") or "unknown")
        sample_rows.append(
            {
                "trajectory_key": f"Neighborhood::{episode_id}::0",
                "episode_id": episode_id,
                "object_name": row.get("object_name"),
                "true_name": true_name,
                "size": row.get("size"),
                "frame_count": len(frames),
                "original_start": json.loads(source_record.read_text(encoding="utf-8"))["record_list"][0],
                "shifted_start": frames[0],
                "original_targets": row.get("pose"),
                "shifted_targets": [
                    shift_xyz(position, args.dx, args.dy) for position in row.get("pose") or []
                ],
            }
        )
        for frame_idx, (action, image_row) in enumerate(zip(actions, camera_rows)):
            action_name = str(action[0][0]).lower()
            action_map = {
                "turn right": "Turn Right",
                "turn left": "Turn Left",
                "go straight": "Move Forward",
                "move forward": "Move Forward",
                "ascend": "Ascend",
                "go up": "Ascend",
                "descend": "Descend",
                "go down": "Descend",
                "stop": "Stop",
            }
            collision_rows.append(
                {
                    "scene_id": "Neighborhood",
                    "episode_id": episode_id,
                    "pose_idx": "0",
                    "frame_idx": frame_idx,
                    "image_path": str(
                        args.aligned_root
                        / "images"
                        / "Neighborhood"
                        / str(image_row.get("rgb") or "")
                    ),
                    "target_description": row.get("description"),
                    "action_name": action_map.get(action_name, action[0][0]),
                }
            )

    sample_rows.sort(key=lambda row: int(row["episode_id"]))
    sample_path = args.output_dir / "sample_manifest.jsonl"
    with sample_path.open("w", encoding="utf-8") as output:
        for row in sample_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    actor_names = {str(row["object_name"]) for row in selected}
    selected_actors = [row for row in actor_rows if str(row["object_name"]) in actor_names]
    actor_path = args.output_dir / "selected_actor_alignment_before.jsonl"
    with actor_path.open("w", encoding="utf-8") as output:
        for row in selected_actors:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    targets_by_actor: dict[str, dict[tuple[float, float, float], list[float]]] = defaultdict(dict)
    true_names_by_actor: dict[str, set[str]] = defaultdict(set)
    for row in metadata_rows:
        if row.get("scene_key") != "NeighborhoodTrain":
            continue
        object_name = str(row.get("object_name") or "")
        true_names_by_actor[object_name].add(str(row.get("true_name") or ""))
        for position in row.get("pose") or []:
            shifted = shift_xyz(position, args.dx, args.dy)
            targets_by_actor[object_name][tuple(round(value, 5) for value in shifted)] = shifted
    alignment_input_dir = args.output_dir / "actor_alignment_input"
    alignment_input_dir.mkdir()
    with (alignment_input_dir / "Neighborhood_actors.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for actor in sorted(actor_rows, key=lambda row: str(row.get("object_name") or "")):
            object_name = str(actor.get("object_name") or "")
            output.write(
                json.dumps(
                    {
                        "scene_id": "Neighborhood",
                        "object_name": object_name,
                        "true_names": sorted(name for name in true_names_by_actor[object_name] if name),
                        "target_positions": list(targets_by_actor[object_name].values()),
                        "actor_audit_status": "coordinate_transform_candidate",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Split by trajectory and balance total frame count; each trajectory remains intact.
    lane_frames = [0, 0]
    lane_ids: list[set[str]] = [set(), set()]
    by_id = {str(row["episode_id"]): row for row in sample_rows}
    for episode_id, row in sorted(
        by_id.items(), key=lambda item: int(item[1]["frame_count"]), reverse=True
    ):
        lane = min(range(2), key=lambda index: lane_frames[index])
        lane_ids[lane].add(episode_id)
        lane_frames[lane] += int(row["frame_count"])

    for lane in range(2):
        key_path = args.output_dir / f"lane{lane}_trajectory_keys.txt"
        key_path.write_text(
            "".join(
                f"Neighborhood::{episode_id}::0\n"
                for episode_id in sorted(lane_ids[lane], key=int)
            ),
            encoding="utf-8",
        )
        source_path = args.output_dir / f"lane{lane}_collision_source.jsonl"
        with source_path.open("w", encoding="utf-8") as output:
            for row in collision_rows:
                if str(row["episode_id"]) in lane_ids[lane]:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format": "neighborhood_coordinate_xy_shift_smoke_v1",
        "source_aligned_root": str(args.aligned_root.resolve()),
        "temporary_aligned_root": str(aligned_out.resolve()),
        "source_metadata": str(args.metadata.resolve()),
        "temporary_metadata": str(metadata_out.resolve()),
        "transform": {"dx": args.dx, "dy": args.dy, "dz": 0.0},
        "trajectory_count": len(sample_rows),
        "actor_count": len(actor_names),
        "true_name_count": len({str(row["true_name"]) for row in sample_rows}),
        "frame_count": sum(int(row["frame_count"]) for row in sample_rows),
        "alignment_actor_count": len(actor_rows),
        "lane_frame_counts": lane_frames,
        "selected_episode_ids": sorted(selected_ids, key=int),
        "note": "Original images are symlinked for reference only; replay RGB is authoritative.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

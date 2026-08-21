#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path


def yaw_delta_degrees(cur_yaw: float, next_yaw: float) -> float:
    delta = (next_yaw - cur_yaw) * 180.0 / math.pi
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def derive_octmen_action(cur_pose: list[float], next_pose: list[float] | None) -> tuple[str, int]:
    if next_pose is None:
        return "stop", 0

    dx = float(next_pose[0]) - float(cur_pose[0])
    dy = float(next_pose[1]) - float(cur_pose[1])
    dz = float(next_pose[2]) - float(cur_pose[2])
    xy = math.hypot(dx, dy)
    dyaw = yaw_delta_degrees(float(cur_pose[3]), float(next_pose[3]))

    if abs(dz) >= 1.0 and abs(dz) >= xy:
        return ("go up", 3) if dz > 0 else ("go down", 3)

    if abs(dyaw) >= 15.0 and abs(dyaw) >= xy * 3.0:
        return ("turn left", 30) if dyaw > 0 else ("turn right", 30)

    if xy >= 1.0:
        return "go straight", 3

    if abs(dyaw) >= 15.0:
        return ("turn left", 30) if dyaw > 0 else ("turn right", 30)

    return "stop", 0


def iter_record_jsons(source_root: Path):
    json_root = source_root / "json"
    for path in sorted(
        json_root.glob("*/*/*.json"),
        key=lambda p: (
            p.relative_to(json_root).parts[0],
            int(p.relative_to(json_root).parts[1]) if p.relative_to(json_root).parts[1].isdigit() else p.relative_to(json_root).parts[1],
            int(p.stem) if p.stem.isdigit() else p.stem,
        ),
    ):
        yield path


def align_one(data: dict) -> tuple[dict, Counter[str]]:
    records = data.get("record_list") or []
    camera_rows = data.get("image_dict", {}).get("uav_on_0", [])
    if len(records) != len(camera_rows):
        raise ValueError(f"record_list/image_dict length mismatch: {len(records)} != {len(camera_rows)}")

    action_list = []
    action_type = []
    action = []
    pos = []
    yaw = []
    counts: Counter[str] = Counter()

    for idx, pose in enumerate(records):
        next_pose = records[idx + 1] if idx + 1 < len(records) else None
        name, value = derive_octmen_action(pose, next_pose)
        xyz = [float(pose[0]), float(pose[1]), float(pose[2])]
        action_list.append([[name, value], xyz])
        action_type.append(name)
        action.append(value)
        pos.append(xyz)
        yaw.append(float(pose[3]))
        counts[name] += 1

    aligned = dict(data)
    aligned["alignment_note"] = (
        "Aligned for inspection: action fields are derived from record_list[i] -> "
        "record_list[i+1]; final frame is stop."
    )
    aligned["index_list"] = list(range(len(records)))
    aligned["action_list"] = action_list
    aligned["action_type"] = action_type
    aligned["action"] = action
    aligned["pos"] = pos
    aligned["yaw"] = yaw
    return aligned, counts


def rebuild(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_json_root = output_root / "json"
    output_json_root.mkdir(parents=True, exist_ok=True)
    output_images = output_root / "images"
    source_images = source_root / "images"

    if not output_images.exists():
        os.symlink(source_images, output_images, target_is_directory=True)

    files = 0
    rows = 0
    action_counts: Counter[str] = Counter()
    examples = []

    source_json_root = source_root / "json"
    for source_path in iter_record_jsons(source_root):
        rel = source_path.relative_to(source_json_root)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        aligned, counts = align_one(data)
        target_path = output_json_root / rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")

        files += 1
        rows += len(aligned.get("record_list") or [])
        action_counts.update(counts)
        if len(examples) < 5:
            examples.append(
                {
                    "path": str(rel),
                    "record_list": len(aligned.get("record_list") or []),
                    "image_dict_uav_on_0": len(aligned.get("image_dict", {}).get("uav_on_0", [])),
                    "action_list": len(aligned.get("action_list") or []),
                }
            )

    manifest = {
        "format": "record_output_transition_aligned_for_inspection",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "images": f"symlink -> {source_images}",
        "json_files": files,
        "aligned_rows": rows,
        "action_counts": dict(action_counts),
        "examples": examples,
    }
    (output_root / "alignment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an inspection copy of record_output with action fields aligned.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/generated/record_output"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/generated/record_output_transition_aligned"),
    )
    args = parser.parse_args()
    manifest = rebuild(args.source_root, args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

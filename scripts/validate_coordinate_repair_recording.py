#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def plan_key(path: Path, plan_root: Path, scene: str) -> tuple[str, str]:
    relative = path.relative_to(plan_root / scene)
    return relative.parts[0], path.stem


def validate_capture(
    plan_path: Path,
    record_path: Path,
    image_root: Path,
    camera_name: str,
    require_depth: bool,
) -> tuple[bool, list[str], int]:
    errors = []
    frame_count = 0
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = plan.get("record_list") or []
        poses = record.get("record_list") or []
        actions = record.get("action_list") or []
        camera_rows = (record.get("image_dict") or {}).get(camera_name) or []
        frame_count = len(poses)
        if poses != expected:
            errors.append("record_list differs from A* plan")
        if len(poses) != len(actions):
            errors.append(f"record/action mismatch {len(poses)} != {len(actions)}")
        if len(poses) != len(camera_rows):
            errors.append(f"record/image mismatch {len(poses)} != {len(camera_rows)}")
        for index, row in enumerate(camera_rows):
            image_path = image_root / str(row.get("rgb") or "")
            if not image_path.is_file():
                errors.append(f"missing RGB f{index}: {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"invalid RGB f{index}: {exc!r}")
            if require_depth:
                depth_path = image_root / str(row.get("depth") or "")
                if not depth_path.is_file():
                    errors.append(f"missing depth f{index}: {depth_path}")
                    continue
                try:
                    depth = np.load(depth_path, mmap_mode="r")
                    if depth.ndim != 2 or not depth.size:
                        errors.append(f"invalid depth shape f{index}: {depth.shape}")
                except Exception as exc:
                    errors.append(f"invalid depth f{index}: {exc!r}")
    except Exception as exc:
        errors.append(f"capture parse error: {exc!r}")
    return not errors, errors, frame_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate coordinate-repaired RGB/depth trajectory recording."
    )
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--scene", default="Neighborhood")
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--require-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-invalid", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    plan_paths = sorted((args.plan_root / args.scene).glob("*/*.json"))
    invalid: list[dict[str, Any]] = []
    valid = 0
    frames = 0
    missing = 0
    for plan_path in plan_paths:
        episode_id, pose_idx = plan_key(plan_path, args.plan_root, args.scene)
        record_path = (
            args.record_root / "json" / args.scene / episode_id / f"{pose_idx}.json"
        )
        if not record_path.is_file():
            missing += 1
            invalid.append(
                {
                    "episode_id": episode_id,
                    "pose_idx": pose_idx,
                    "plan": str(plan_path.resolve()),
                    "record": str(record_path),
                    "errors": ["record JSON missing"],
                }
            )
            continue
        ok, errors, frame_count = validate_capture(
            plan_path,
            record_path,
            args.record_root / "images" / args.scene,
            args.camera_name,
            args.require_depth,
        )
        if ok:
            valid += 1
            frames += frame_count
            continue
        invalid.append(
            {
                "episode_id": episode_id,
                "pose_idx": pose_idx,
                "plan": str(plan_path.resolve()),
                "record": str(record_path.resolve()),
                "errors": errors[:20],
            }
        )
        if args.clean_invalid:
            record_path.unlink(missing_ok=True)
            shutil.rmtree(
                args.record_root / "images" / args.scene / episode_id / pose_idx,
                ignore_errors=True,
            )

    report = {
        "format": "coordinate_repair_recording_validation_v1",
        "plan_root": str(args.plan_root.resolve()),
        "record_root": str(args.record_root.resolve()),
        "scene": args.scene,
        "expected_trajectories": len(plan_paths),
        "valid_trajectories": valid,
        "valid_frames": frames,
        "missing_trajectories": missing,
        "invalid_trajectories": len(invalid) - missing,
        "clean_invalid": args.clean_invalid,
        "complete": valid == len(plan_paths),
        "problems": invalid[:100],
        "problems_truncated": max(0, len(invalid) - 100),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

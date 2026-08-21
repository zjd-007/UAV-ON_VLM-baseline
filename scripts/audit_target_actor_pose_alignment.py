#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from capture_stop_visibility_cache import (  # noqa: E402
    SegmentationAirsimTrajRecorder,
    set_target_segmentation_id,
)


def read_actor_rows(input_dir: Path, scene: str) -> list[dict[str, Any]]:
    path = input_dir / f"{scene}_actors.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def position_values(pose: Any) -> list[float]:
    return [
        float(pose.position.x_val),
        float(pose.position.y_val),
        float(pose.position.z_val),
    ]


def min_errors(
    actor_position: list[float],
    target_positions: list[list[float]],
) -> tuple[float, float, list[float] | None]:
    candidates = []
    for target in target_positions:
        dx = float(target[0]) - actor_position[0]
        dy = float(target[1]) - actor_position[1]
        dz = float(target[2]) - actor_position[2]
        candidates.append(
            (
                math.sqrt(dx * dx + dy * dy + dz * dz),
                math.sqrt(dx * dx + dy * dy),
                target,
            )
        )
    return min(candidates) if candidates else (math.inf, math.inf, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AirSim actor poses with UAV-ON target coordinates."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=39900)
    parser.add_argument("--only-nonclear-actors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scene in [item for item in args.scene_list.split(",") if item]:
        output_path = args.output_dir / f"{scene}.jsonl"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(output_path)
        rows = read_actor_rows(args.input_dir, scene)
        if args.only_nonclear_actors:
            rows = [
                row
                for row in rows
                if row.get("actor_audit_status")
                != "clear_on_at_least_one_expert_path"
            ]
        if not rows:
            output_path.write_text("", encoding="utf-8")
            continue
        env = SegmentationAirsimTrajRecorder(
            scene,
            airsim_port=args.base_port + args.gpu,
            device_id=args.gpu,
            segmentation_width=64,
            segmentation_height=64,
        )
        try:
            client = env._client
            with output_path.open("w", encoding="utf-8") as output:
                for index, row in enumerate(rows, start=1):
                    try:
                        resolved_name, match_mode, _ = set_target_segmentation_id(
                            client,
                            str(row["object_name"]),
                            42,
                            row.get("target_positions") or [],
                        )
                        actor_position = position_values(
                            client.simGetObjectPose(resolved_name)
                        )
                        finite = bool(np.isfinite(actor_position).all())
                        error_3d, error_xy, nearest = min_errors(
                            actor_position,
                            row.get("target_positions") or [],
                        )
                        result = {
                            "scene_id": scene,
                            "object_name": row["object_name"],
                            "true_names": row.get("true_names") or [],
                            "actor_audit_status": row.get("actor_audit_status"),
                            "resolved_object_name": resolved_name,
                            "object_name_match": match_mode,
                            "actor_position": actor_position,
                            "actor_position_finite": finite,
                            "nearest_target_position": nearest,
                            "actor_to_target_error_xy_m": error_xy,
                            "actor_to_target_error_3d_m": error_3d,
                            "status": "ok" if finite else "invalid_actor_pose",
                        }
                    except Exception as exc:
                        result = {
                            "scene_id": scene,
                            "object_name": row.get("object_name"),
                            "true_names": row.get("true_names") or [],
                            "actor_audit_status": row.get("actor_audit_status"),
                            "status": "error",
                            "error": repr(exc),
                        }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "scene": scene,
                                "progress": f"{index}/{len(rows)}",
                                "object_name": row.get("object_name"),
                                "status": result["status"],
                                "error_xy_m": result.get("actor_to_target_error_xy_m"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            env.cleanup()


if __name__ == "__main__":
    main()

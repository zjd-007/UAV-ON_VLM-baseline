#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
OCTMEN_ROOT = REPO_ROOT / "octmem_nomemory_repro" / "octmen-agent"
TRAJ_GEN_ROOT = OCTMEN_ROOT / "tools" / "traj_gen"
sys.path.insert(0, str(TRAJ_GEN_ROOT))

from collect_traj import (  # noqa: E402
    PathPlannerNode,
    save_path_result,
    to_eularian_angles,
)


LEGACY_UNAVAILABLE_KEYS = {"Neighborhood::373::3", "Neighborhood::374::3"}


def normalize_scene(row: dict[str, Any]) -> str:
    scene = str(row.get("scene_key") or row.get("map_name", ""))
    scene = scene.replace("_TrainSets", "")
    return {
        "NeighborhoodTrain": "Neighborhood",
        "ModularNeighborhood": "Neighborhood",
    }.get(scene, scene)


def build_tasks(metadata: Path) -> list[dict[str, Any]]:
    tasks = []
    for row in json.loads(metadata.read_text(encoding="utf-8")):
        if normalize_scene(row) != "Neighborhood":
            continue
        start_pose = row.get("start_pose") or {}
        start = [float(value) for value in start_pose["start_position"]]
        quaternion = [float(value) for value in start_pose["start_quaternionr"]]
        for pose_idx, goal in enumerate(row.get("pose") or []):
            episode_id = str(row["episode_id"])
            tasks.append(
                {
                    "key": f"Neighborhood::{episode_id}::{pose_idx}",
                    "episode_id": episode_id,
                    "pose_idx": str(pose_idx),
                    "start": start,
                    "quaternion": quaternion,
                    "goal": [float(value) for value in goal],
                    "object_name": str(row.get("object_name") or ""),
                    "true_name": str(row.get("true_name") or ""),
                }
            )
    return tasks


def plan_path(output_root: Path, task: dict[str, Any]) -> Path:
    return output_root / "Neighborhood" / task["episode_id"] / f"{task['pose_idx']}.json"


def candidate_offsets(max_horizontal_m: float) -> list[np.ndarray]:
    candidates = []
    # NED z: a more negative z raises the UAV. Preserve XY whenever possible.
    for dz in (-3.0, -6.0, -9.0, 3.0):
        candidates.append(np.asarray([0.0, 0.0, dz]))
    steps = range(
        -int(max_horizontal_m // 3.0),
        int(max_horizontal_m // 3.0) + 1,
    )
    for ix in steps:
        for iy in steps:
            if ix == 0 and iy == 0:
                continue
            dx = ix * 3.0
            dy = iy * 3.0
            if math.hypot(dx, dy) > max_horizontal_m + 1e-6:
                continue
            for dz in (0.0, -3.0, -6.0, 3.0):
                candidates.append(np.asarray([dx, dy, dz]))
    unique = {tuple(offset.tolist()): offset for offset in candidates}
    return sorted(
        unique.values(),
        key=lambda offset: (
            float(np.linalg.norm(offset)),
            0 if float(offset[2]) < 0 else 1,
            abs(float(offset[2])),
            abs(float(offset[0])) + abs(float(offset[1])),
        ),
    )


def recover_task(
    planner: PathPlannerNode,
    task: dict[str, Any],
    output_root: Path,
    offsets: list[np.ndarray],
    preferred_offset: np.ndarray | None,
) -> dict[str, Any]:
    started_from = np.asarray(task["start"], dtype=np.float64)
    goal = np.asarray(task["goal"], dtype=np.float64)
    _, _, initial_yaw = to_eularian_angles(*task["quaternion"])
    voxel_map, searcher = planner.build_cur_map(
        np.asarray([started_from, goal]), margin=50
    )
    del voxel_map
    ordered_offsets = list(offsets)
    if preferred_offset is not None:
        ordered_offsets = [preferred_offset] + [
            offset
            for offset in ordered_offsets
            if not np.allclose(offset, preferred_offset)
        ]
    occupied_candidates = 0
    attempts = 0
    for offset in ordered_offsets:
        candidate = started_from + offset
        if searcher.is_occupied(candidate):
            occupied_candidates += 1
            continue
        attempts += 1
        path_result = searcher.hybrid_a_star(candidate, goal, thr=5)
        if len(path_result) <= 1:
            continue
        records, actions = searcher.backtrack_path_with_yaw(
            path_result,
            initial_yaw=initial_yaw,
            with_stop=True,
        )
        if not records or len(records) != len(actions):
            continue
        path = plan_path(output_root, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_path_result(
            str(path),
            records,
            actions,
            candidate.tolist(),
            goal.tolist(),
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["coordinate_repair_start_recovery"] = {
            "version": "nearest_free_start_v1",
            "requested_start_position": started_from.tolist(),
            "recovered_start_position": candidate.tolist(),
            "offset_xyz_m": offset.tolist(),
            "offset_distance_m": float(np.linalg.norm(offset)),
            "reason": "corrected_start_was_occupied_or_not_astar_connected",
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            **task,
            "status": "recovered",
            "plan_path": str(path.resolve()),
            "requested_start_occupied": bool(searcher.is_occupied(started_from)),
            "offset_xyz_m": offset.tolist(),
            "offset_distance_m": float(np.linalg.norm(offset)),
            "candidate_attempts": attempts,
            "occupied_candidates_skipped": occupied_candidates,
        }
    return {
        **task,
        "status": "unresolved",
        "plan_path": None,
        "requested_start_occupied": bool(searcher.is_occupied(started_from)),
        "candidate_attempts": attempts,
        "occupied_candidates_skipped": occupied_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover coordinate-repaired Neighborhood A* failures from the nearest "
            "3m-grid free start while retaining an explicit provenance record."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizontal-m", type=float, default=9.0)
    parser.add_argument("--minimum-plans", type=int, default=848)
    parser.add_argument(
        "--trajectory-key",
        action="append",
        help="Optional exact trajectory key filter for smoke tests. Can be repeated.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    tasks = build_tasks(args.metadata)
    if args.trajectory_key:
        requested = set(args.trajectory_key)
        tasks = [task for task in tasks if task["key"] in requested]
        missing_requested = requested - {task["key"] for task in tasks}
        if missing_requested:
            raise KeyError(f"trajectory keys not found in metadata: {sorted(missing_requested)}")
    missing = [task for task in tasks if not plan_path(args.plan_root, task).is_file()]
    by_start: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in missing:
        key = tuple(round(value, 5) for value in [*task["start"], *task["quaternion"]])
        by_start[key].append(task)

    results = []
    offsets = candidate_offsets(args.max_horizontal_m)
    planner = PathPlannerNode("NeighborhoodRepairRecovery", "Neighborhood")
    for group_index, group in enumerate(by_start.values(), start=1):
        preferred_offset = None
        for task in group:
            result = recover_task(
                planner,
                task,
                args.plan_root,
                offsets,
                preferred_offset,
            )
            results.append(result)
            if result["status"] == "recovered":
                preferred_offset = np.asarray(result["offset_xyz_m"], dtype=np.float64)
            print(
                json.dumps(
                    {
                        "event": "recovery_progress",
                        "start_group": f"{group_index}/{len(by_start)}",
                        "key": task["key"],
                        "status": result["status"],
                        "requested_start_occupied": result[
                            "requested_start_occupied"
                        ],
                        "offset_xyz_m": result.get("offset_xyz_m"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    plan_count = len(list((args.plan_root / "Neighborhood").glob("*/*.json")))
    unresolved = [result for result in results if result["status"] == "unresolved"]
    unexpected_unresolved = [
        result for result in unresolved if result["key"] not in LEGACY_UNAVAILABLE_KEYS
    ]
    payload = {
        "format": "neighborhood_coordinate_repair_astar_nearest_free_start_v1",
        "metadata": str(args.metadata.resolve()),
        "plan_root": str(args.plan_root.resolve()),
        "tasks": len(tasks),
        "plans_before_recovery": len(tasks) - len(missing),
        "missing_before_recovery": len(missing),
        "recovered": sum(result["status"] == "recovered" for result in results),
        "unresolved": len(unresolved),
        "unresolved_keys": [result["key"] for result in unresolved],
        "legacy_unavailable_keys": sorted(LEGACY_UNAVAILABLE_KEYS),
        "unexpected_unresolved_keys": [
            result["key"] for result in unexpected_unresolved
        ],
        "plans_after_recovery": plan_count,
        "minimum_required_plans": args.minimum_plans,
        "max_horizontal_recovery_m": args.max_horizontal_m,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if plan_count < args.minimum_plans or unexpected_unresolved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

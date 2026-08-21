#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
OCTMEN_ROOT = REPO_ROOT / "octmem_nomemory_repro" / "octmen-agent"
TRAJ_GEN_ROOT = OCTMEN_ROOT / "tools" / "traj_gen"
DEFAULT_METADATA = DATASET_ROOT / "splits" / "uavon_raw_json" / "train.json"
DEFAULT_ALIGNMENT = (
    DATASET_ROOT
    / "processed"
    / "stop_visible_full_audit"
    / "full_canonical_geometry_v1_20260812_153000"
    / "actor_pose_alignment"
    / "Neighborhood.jsonl"
)

sys.path.insert(0, str(TRAJ_GEN_ROOT))

from collect_traj import PathPlannerNode, save_path_result  # noqa: E402


def normalize_scene(row: dict[str, Any]) -> str:
    scene = str(row.get("scene_key") or row.get("map_name", ""))
    scene = scene.replace("_TrainSets", "")
    return {
        "NeighborhoodTrain": "Neighborhood",
        "ModularNeighborhood": "Neighborhood",
    }.get(scene, scene)


def shift_xyz(values: list[float], dx: float, dy: float) -> list[float]:
    shifted = [float(value) for value in values]
    shifted[0] += dx
    shifted[1] += dy
    return shifted


def transform_metadata(
    source: Path,
    output: Path,
    dx: float,
    dy: float,
) -> list[dict[str, Any]]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    transformed = []
    for row in rows:
        copied = json.loads(json.dumps(row))
        if normalize_scene(copied) == "Neighborhood":
            copied["pose"] = [
                shift_xyz(position, dx, dy) for position in copied.get("pose") or []
            ]
            start_pose = copied.get("start_pose") or {}
            if start_pose.get("start_position"):
                start_pose["start_position"] = shift_xyz(
                    start_pose["start_position"], dx, dy
                )
            copied["coordinate_repair"] = {
                "source_scene": "NeighborhoodTrain",
                "transform": {"dx": dx, "dy": dy, "dz": 0.0},
                "reason": "dataset_simulator_xy_origin_mismatch",
            }
        transformed.append(copied)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(transformed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transformed


def validate_alignment(
    path: Path,
    metadata_path: Path,
    dx: float,
    dy: float,
    tolerance: float,
) -> dict[str, Any]:
    targets_by_actor: dict[str, list[list[float]]] = {}
    for metadata in json.loads(metadata_path.read_text(encoding="utf-8")):
        if normalize_scene(metadata) != "Neighborhood":
            continue
        targets_by_actor.setdefault(str(metadata.get("object_name") or ""), []).extend(
            metadata.get("pose") or []
        )
    residuals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        actor = row.get("actor_position")
        targets = targets_by_actor.get(str(row.get("object_name") or "")) or []
        if not actor or not targets:
            continue
        residuals.append(
            min(
                ((float(target[0]) + dx - float(actor[0])) ** 2
                 + (float(target[1]) + dy - float(actor[1])) ** 2) ** 0.5
                for target in targets
            )
        )
    if not residuals:
        raise RuntimeError(f"no usable actor alignment rows: {path}")
    maximum = max(residuals)
    if maximum > tolerance:
        raise RuntimeError(
            f"candidate transform failed actor alignment: max residual={maximum:.6f}m "
            f"> tolerance={tolerance:.6f}m"
        )
    ordered = sorted(residuals)
    return {
        "actors": len(residuals),
        "min_xy_residual_m": ordered[0],
        "median_xy_residual_m": ordered[len(ordered) // 2],
        "max_xy_residual_m": ordered[-1],
        "tolerance_m": tolerance,
    }


def build_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        if normalize_scene(row) != "Neighborhood":
            continue
        start_pose = row.get("start_pose") or {}
        start = start_pose.get("start_position")
        quaternion = start_pose.get("start_quaternionr")
        if not start or not quaternion:
            raise ValueError(f"missing start pose: episode={row.get('episode_id')}")
        for pose_idx, target in enumerate(row.get("pose") or []):
            tasks.append(
                {
                    "episode_id": str(row["episode_id"]),
                    "pose_idx": str(pose_idx),
                    "start_position": [float(value) for value in start],
                    "start_quaternion": [float(value) for value in quaternion],
                    "target_position": [float(value) for value in target],
                    "object_name": str(row.get("object_name") or ""),
                    "true_name": str(row.get("true_name") or ""),
                }
            )
    return sorted(
        tasks,
        key=lambda task: (
            tuple(task["start_position"]),
            tuple(task["target_position"]),
            int(task["episode_id"]),
            int(task["pose_idx"]),
        ),
    )


def plan_path(output_root: Path, task: dict[str, Any]) -> Path:
    return (
        output_root
        / "Neighborhood"
        / task["episode_id"]
        / f"{task['pose_idx']}.json"
    )


def valid_plan(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    records = data.get("record_list") or []
    actions = data.get("action_list") or []
    return bool(records) and len(records) == len(actions)


def run_worker(
    worker_id: int,
    tasks: list[dict[str, Any]],
    output_root: Path,
    progress: dict[str, int],
    lock: threading.Lock,
) -> list[dict[str, Any]]:
    planner = PathPlannerNode(f"NeighborhoodRepair-{worker_id}", "Neighborhood")
    results = []
    for task in tasks:
        path = plan_path(output_root, task)
        started = time.time()
        if valid_plan(path):
            status = "skipped_existing"
            error = None
        else:
            if path.exists():
                path.unlink()
            try:
                success, payload = planner.plan_path_direct(
                    task["start_position"],
                    task["start_quaternion"],
                    task["target_position"],
                )
                records, actions = payload
                if not success or not records or len(records) != len(actions):
                    status = "no_path"
                    error = None
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_path_result(
                        str(path),
                        records,
                        actions,
                        task["start_position"],
                        task["target_position"],
                    )
                    if not valid_plan(path):
                        raise RuntimeError(f"invalid plan written: {path}")
                    status = "success"
                    error = None
            except Exception as exc:
                status = "error"
                error = repr(exc)

        result = {
            **task,
            "worker_id": worker_id,
            "status": status,
            "plan_path": str(path.resolve()) if path.exists() else None,
            "elapsed_seconds": time.time() - started,
            "error": error,
        }
        results.append(result)
        with lock:
            progress["completed"] += 1
            progress[status] = progress.get(status, 0) + 1
            print(
                json.dumps(
                    {
                        "event": "plan_progress",
                        "completed": progress["completed"],
                        "total": progress["total"],
                        "success": progress.get("success", 0),
                        "skipped_existing": progress.get("skipped_existing", 0),
                        "no_path": progress.get("no_path", 0),
                        "errors": progress.get("error", 0),
                        "episode_id": task["episode_id"],
                        "pose_idx": task["pose_idx"],
                        "worker_id": worker_id,
                        "status": status,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replan all Neighborhood training tasks after the verified XY transform."
    )
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--dx", type=float, default=130.0)
    parser.add_argument("--dy", type=float, default=110.0)
    parser.add_argument("--alignment-tolerance", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    alignment = validate_alignment(
        args.alignment,
        args.source_metadata,
        args.dx,
        args.dy,
        args.alignment_tolerance,
    )
    rows = transform_metadata(
        args.source_metadata,
        args.output_metadata,
        args.dx,
        args.dy,
    )
    tasks = build_tasks(rows)
    if len(tasks) != 850:
        raise RuntimeError(f"expected 850 Neighborhood planning targets, got {len(tasks)}")
    all_task_count = len(tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    shards = [[] for _ in range(args.workers)]
    shard_load = [0 for _ in range(args.workers)]
    for task in tasks:
        worker_id = min(range(args.workers), key=lambda index: shard_load[index])
        shards[worker_id].append(task)
        shard_load[worker_id] += 1

    progress = {"completed": 0, "total": len(tasks)}
    lock = threading.Lock()
    all_results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_worker,
                worker_id,
                shard,
                args.output_root,
                progress,
                lock,
            ): worker_id
            for worker_id, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    all_results.sort(key=lambda row: (int(row["episode_id"]), int(row["pose_idx"])))
    results_path = args.output_root / "planning_results.jsonl"
    with results_path.open("w", encoding="utf-8") as output:
        for row in all_results:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    success_count = sum(
        row["status"] in {"success", "skipped_existing"} for row in all_results
    )
    manifest = {
        "format": "neighborhood_coordinate_repair_astar_v1",
        "source_metadata": str(args.source_metadata.resolve()),
        "transformed_metadata": str(args.output_metadata.resolve()),
        "output_root": str(args.output_root.resolve()),
        "pcd": str(
            (
                OCTMEN_ROOT
                / "tools"
                / "scene_data"
                / "pcd_map"
                / "Neighborhood.ply"
            ).resolve()
        ),
        "transform": {"dx": args.dx, "dy": args.dy, "dz": 0.0},
        "actor_alignment": alignment,
        "workers": args.workers,
        "expected_tasks": len(tasks),
        "all_available_tasks": all_task_count,
        "successful_plans": success_count,
        "no_path": sum(row["status"] == "no_path" for row in all_results),
        "errors": sum(row["status"] == "error" for row in all_results),
        "elapsed_seconds": time.time() - started,
        "results": str(results_path.resolve()),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "planning_complete", **manifest}, ensure_ascii=False), flush=True)
    if manifest["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault(
        "OCTMEN_ENV_ROOT",
        "/data/zhujd/Aerial-ObjectNav/octmem_nomemory_repro/TRAIN_ENVS",
    )
    main()

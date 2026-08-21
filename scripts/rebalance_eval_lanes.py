#!/usr/bin/env python3
"""Repartition only unfinished evaluation episodes onto a new GPU set."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from eval_lane_watchdog import (
    archive_incomplete_images,
    parse_lanes,
    read_completed_rows,
    start_lane,
    stop_lane,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve completed episodes and rebalance unfinished ones."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids")
    parser.add_argument("--reference-run-dir", type=Path)
    parser.add_argument("--lane-prefix", default="laneR")
    parser.add_argument("--conda-sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="octmem_openvla_nomemory")
    return parser.parse_args()


def row_key(row: dict) -> tuple[str, str]:
    return str(row["map_name"]), str(row["episode_id"])


def read_all_completed(run_dir: Path) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for lane_dir in sorted(run_dir.glob("lane*")):
        if not lane_dir.is_dir():
            continue
        for row in read_completed_rows(lane_dir):
            by_key[row_key(row)] = row
    return list(by_key.values())


def reference_costs(reference_run_dir: Path | None) -> dict[tuple[str, str], int]:
    costs: dict[tuple[str, str], int] = {}
    if reference_run_dir is None:
        return costs
    for lane_dir in sorted(reference_run_dir.glob("lane*")):
        if not lane_dir.is_dir():
            continue
        for row in read_completed_rows(lane_dir):
            steps = row.get("step_records")
            costs[row_key(row)] = max(1, len(steps) if isinstance(steps, list) else 1)
    return costs


def assign_scenes(
    rows: list[dict], costs: dict[tuple[str, str], int], lane_count: int, default_cost: int
) -> list[list[dict]]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["map_name"])].append(row)

    scene_groups = []
    for scene, scene_rows in by_scene.items():
        cost = sum(costs.get(row_key(row), default_cost) for row in scene_rows)
        scene_groups.append((cost, scene, scene_rows))

    assignments: list[list[dict]] = [[] for _ in range(lane_count)]
    assigned_cost = [0] * lane_count
    for cost, _scene, scene_rows in sorted(scene_groups, reverse=True):
        lane_index = min(range(lane_count), key=lambda index: (assigned_cost[index], index))
        assignments[lane_index].extend(scene_rows)
        assigned_cost[lane_index] += cost
    return assignments


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    log_dir = args.log_dir.resolve()
    config_path = run_dir / "run_config.json"
    lanes_path = run_dir / "lanes.tsv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_lanes = parse_lanes(lanes_path)
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError(f"GPU ids must be non-empty and unique: {args.gpus}")

    dataset_path = Path(config["eval_dataset"])
    dataset_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Stop only sessions and simulator ports owned by this run.
    for lane, lane_cfg in old_lanes.items():
        stop_lane(lane, int(lane_cfg["gpu"]), run_dir, config)

    completed_rows = read_all_completed(run_dir)
    completed = {row_key(row) for row in completed_rows}
    expected = {row_key(row) for row in dataset_rows}
    unexpected = completed - expected
    if unexpected:
        raise RuntimeError(f"run contains {len(unexpected)} results outside its dataset")
    remaining = [row for row in dataset_rows if row_key(row) not in completed]

    for lane in old_lanes:
        archived = archive_incomplete_images(
            run_dir / lane, read_completed_rows(run_dir / lane), stamp
        )
        if archived:
            print(f"{lane}: archived {len(archived)} incomplete image directories", flush=True)

    costs = reference_costs(args.reference_run_dir)
    assignments = assign_scenes(
        remaining,
        costs,
        len(gpus),
        int(config.get("eval_max_steps", 100)),
    )

    backup_lanes = run_dir / f"lanes_before_rebalance_{stamp}.tsv"
    backup_config = run_dir / f"run_config_before_rebalance_{stamp}.json"
    shutil.copy2(lanes_path, backup_lanes)
    shutil.copy2(config_path, backup_config)
    subset_dir = run_dir / "rebalance_datasets" / stamp
    subset_dir.mkdir(parents=True, exist_ok=False)

    new_lanes: dict[str, dict] = {}
    lane_lines = []
    for index, (gpu, lane_rows) in enumerate(zip(gpus, assignments)):
        lane = f"{args.lane_prefix}{index}"
        subset = subset_dir / f"{lane}.json"
        subset.write_text(json.dumps(lane_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        scenes = list(dict.fromkeys(str(row["map_name"]) for row in lane_rows))
        new_lanes[lane] = {
            "gpu": gpu,
            "scenes": scenes,
            "eval_dataset": str(subset),
        }
        lane_lines.append(f"{lane}\t{gpu}\t{','.join(scenes)}\t{subset}")

    lanes_path.write_text("\n".join(lane_lines) + "\n", encoding="utf-8")
    history = config.setdefault("rebalance_history", [])
    history.append(
        {
            "timestamp": stamp,
            "gpus": gpus,
            "completed_before_rebalance": len(completed),
            "remaining": len(remaining),
            "old_lanes_file": str(backup_lanes),
            "lane_datasets": {lane: cfg["eval_dataset"] for lane, cfg in new_lanes.items()},
        }
    )
    config["lane_count"] = len(gpus)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for lane, lane_cfg in new_lanes.items():
        start_lane(
            lane,
            lane_cfg,
            config,
            run_dir,
            log_dir,
            args.conda_sh,
            args.conda_env,
        )
        estimated_cost = sum(costs.get(row_key(row), int(config.get("eval_max_steps", 100))) for row in assignments[int(lane.removeprefix(args.lane_prefix))])
        print(
            f"{lane}: gpu={lane_cfg['gpu']} tasks={len(assignments[int(lane.removeprefix(args.lane_prefix))])} "
            f"estimated_steps={estimated_cost} scenes={','.join(lane_cfg['scenes'])}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "completed_preserved": len(completed),
                "remaining_started": len(remaining),
                "gpus": gpus,
                "lanes": list(new_lanes),
                "lanes_backup": str(backup_lanes),
                "config_backup": str(backup_config),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

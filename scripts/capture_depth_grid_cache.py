#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed" / "depth_grid_cache" / "train"

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval_utils import AirsimTrajRecorder  # noqa: E402
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt  # noqa: E402


def sample_key(row: dict) -> str:
    return f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}"


def load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key and row.get("depth_grid") is not None:
                keys.add(str(key))
    return keys


def load_rows(source: Path, scenes: set[str] | None, limit: int) -> list[dict]:
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if scenes and row["scene_id"] not in scenes:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def load_episode_pose(aligned_root: Path, row: dict, episode_cache: dict[tuple[str, str, str], dict]) -> list[float]:
    scene = str(row["scene_id"])
    episode_id = str(row["episode_id"])
    pose_idx = str(row["pose_idx"])
    key = (scene, episode_id, pose_idx)
    if key not in episode_cache:
        path = aligned_root / "json" / scene / episode_id / f"{pose_idx}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Aligned transition JSON not found: {path}")
        episode_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    data = episode_cache[key]
    frame_idx = int(row["frame_idx"])
    record_list = data.get("record_list") or []
    if frame_idx >= len(record_list):
        raise IndexError(f"{key} frame_idx={frame_idx} out of range for record_list len={len(record_list)}")
    pose = record_list[frame_idx]
    if len(pose) < 4:
        raise ValueError(f"{key} frame_idx={frame_idx} pose has fewer than 4 values: {pose}")
    return [float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])]


def capture_scene(
    scene: str,
    rows: list[dict],
    output_path: Path,
    aligned_root: Path,
    gpu: int,
    port: int,
    grid_size: int,
    max_meters: float,
    settle_frames: int,
    sleep_seconds: float,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path)
    formatter = UAVONSingleViewDepthPrompt(grid_size=grid_size, max_meters=max_meters)
    episode_cache: dict[tuple[str, str, str], dict] = {}
    written = 0
    skipped = 0
    failed = 0

    env = AirsimTrajRecorder(scene, airsim_port=port, device_id=gpu)
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for row in rows:
                key = sample_key(row)
                if key in existing:
                    skipped += 1
                    continue
                try:
                    pose = load_episode_pose(aligned_root, row, episode_cache)
                    env.zero_kinematics_at_pose(pose, settle_frames=settle_frames)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    image_dict = env._capture_images(camera_names=["uav_on_0"], capture_depth=True)
                    depth = image_dict["uav_on_0"]["depth"]
                    grid = formatter.depth_to_grid(depth).tolist()
                    record = {
                        "key": key,
                        "scene_id": row["scene_id"],
                        "episode_id": str(row["episode_id"]),
                        "pose_idx": str(row["pose_idx"]),
                        "frame_idx": int(row["frame_idx"]),
                        "pose": pose,
                        "depth_grid": grid,
                        "depth_grid_size": grid_size,
                        "depth_max_meters": max_meters,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    existing.add(key)
                    written += 1
                except Exception as exc:
                    failed += 1
                    error_record = {
                        "key": key,
                        "scene_id": row.get("scene_id"),
                        "episode_id": str(row.get("episode_id")),
                        "pose_idx": str(row.get("pose_idx")),
                        "frame_idx": int(row.get("frame_idx", -1)),
                        "depth_grid": None,
                        "error": str(exc),
                    }
                    out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    out.flush()
    finally:
        env.cleanup()

    return {"scene": scene, "written": written, "skipped_existing": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UAV-ON train-frame depth grids from AirSim.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene-list", type=str, default="", help="Comma-separated train scenes. Empty means all scenes.")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=36200)
    parser.add_argument("--limit", type=int, default=0, help="Total source rows to consider, for smoke tests.")
    parser.add_argument("--max-rows-per-scene", type=int, default=0)
    parser.add_argument("--depth-grid-size", type=int, default=3)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    scenes = {x for x in args.scene_list.split(",") if x} or None
    rows = load_rows(args.source, scenes, args.limit)
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_scene[row["scene_id"]].append(row)

    summaries = []
    for scene in sorted(rows_by_scene):
        scene_rows = rows_by_scene[scene]
        if args.max_rows_per_scene:
            scene_rows = scene_rows[: args.max_rows_per_scene]
        output_path = args.output_dir / f"{scene}.jsonl"
        summary = capture_scene(
            scene=scene,
            rows=scene_rows,
            output_path=output_path,
            aligned_root=args.aligned_root,
            gpu=args.gpu,
            port=args.base_port + args.gpu,
            grid_size=args.depth_grid_size,
            max_meters=args.depth_max_meters,
            settle_frames=args.settle_frames,
            sleep_seconds=args.sleep_seconds,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

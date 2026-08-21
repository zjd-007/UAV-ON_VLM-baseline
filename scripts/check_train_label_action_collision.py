#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed" / "label_action_collision_check" / "train"

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval_utils import AirsimTrajRecorder, getPoseAfterMakeAction  # noqa: E402
from vlm_baseline.actions import ACTION_IDS, action_name_to_command  # noqa: E402


def sample_key(row: dict[str, Any]) -> str:
    return f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}"


def load_existing(path: Path, retry_errors: bool) -> set[str]:
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
            if key and (not row.get("error") or not retry_errors):
                keys.add(str(key))
    return keys


def load_rows(source: Path, scenes: set[str] | None, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def load_episode_pose(
    aligned_root: Path,
    row: dict[str, Any],
    episode_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> list[float]:
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


def collision_timestamp(info: dict[str, Any] | None) -> int | None:
    if not info:
        return None
    try:
        return int(info.get("time_stamp", 0))
    except Exception:
        return None


def check_scene(
    scene: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    aligned_root: Path,
    gpu: int,
    port: int,
    retry_errors: bool,
    settle_frames: int,
    action_velocity: float,
    action_move_timeout: float,
    action_rotate_timeout: float,
    fix_vertical_actions: bool,
    fix_yaw_actions: bool,
    sleep_seconds: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path, retry_errors=retry_errors)
    episode_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    summary = Counter()
    action_counts = Counter()
    action_new_collisions = Counter()
    action_after_collisions = Counter()

    env = AirsimTrajRecorder(scene, airsim_port=port, device_id=gpu)
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for row_idx, row in enumerate(rows, start=1):
                key = sample_key(row)
                if key in existing:
                    summary["skipped_existing"] += 1
                    continue

                record: dict[str, Any]
                try:
                    pose_before = load_episode_pose(aligned_root, row, episode_cache)
                    command = action_name_to_command(str(row["action_name"]))
                    action_id = ACTION_IDS[command]
                    action_counts[command] += 1

                    reset_status = env.zero_kinematics_at_pose(pose_before, settle_frames=settle_frames)
                    initial_collision = reset_status.get("collision_info") or env.get_collision_info()
                    initial_ts = collision_timestamp(initial_collision)

                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

                    target_pose = getPoseAfterMakeAction(
                        pose_before,
                        action_id,
                        fix_vertical_actions=fix_vertical_actions,
                        fix_yaw_actions=fix_yaw_actions,
                    )
                    action_status = env.execute_action_to_pose_join(
                        action_id,
                        target_pose,
                        velocity=action_velocity,
                        move_timeout=action_move_timeout,
                        rotate_timeout=action_rotate_timeout,
                        level_after_action=False,
                        level_settle_frames=1,
                    )
                    collision_after = action_status.get("collision_info_after_command") or env.get_collision_info()
                    after_ts = collision_timestamp(collision_after)
                    collided_after = bool(collision_after.get("has_collided"))
                    initial_collided = bool(initial_collision.get("has_collided"))
                    new_collision = bool(
                        collided_after
                        and (not initial_collided or after_ts is None or initial_ts is None or after_ts != initial_ts)
                    )

                    if collided_after:
                        action_after_collisions[command] += 1
                    if new_collision:
                        action_new_collisions[command] += 1

                    record = {
                        "key": key,
                        "scene_id": row["scene_id"],
                        "episode_id": str(row["episode_id"]),
                        "pose_idx": str(row["pose_idx"]),
                        "frame_idx": int(row["frame_idx"]),
                        "image_path": row.get("image_path"),
                        "target_description": row.get("target_description"),
                        "label_action_name": row.get("action_name"),
                        "label_command": command,
                        "action_id": action_id,
                        "pose_before": pose_before,
                        "target_pose_after_label": [float(v) for v in target_pose],
                        "actual_pose_after_action": action_status.get("actual_pose"),
                        "position_error": action_status.get("position_error"),
                        "yaw_error": action_status.get("yaw_error"),
                        "initial_collision": initial_collision,
                        "collision_after_action": collision_after,
                        "initial_collided": initial_collided,
                        "collided_after_action": collided_after,
                        "new_collision_after_action": new_collision,
                        "action_status": action_status,
                    }
                    summary["checked"] += 1
                    if initial_collided:
                        summary["initial_collided"] += 1
                    if collided_after:
                        summary["collided_after_action"] += 1
                    if new_collision:
                        summary["new_collision_after_action"] += 1
                except Exception as exc:
                    summary["failed"] += 1
                    record = {
                        "key": key,
                        "scene_id": row.get("scene_id"),
                        "episode_id": str(row.get("episode_id")),
                        "pose_idx": str(row.get("pose_idx")),
                        "frame_idx": int(row.get("frame_idx", -1)),
                        "image_path": row.get("image_path"),
                        "label_action_name": row.get("action_name"),
                        "error": repr(exc),
                    }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                existing.add(key)

                if row_idx % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "scene": scene,
                                "progress_rows_seen": row_idx,
                                **summary,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        env.cleanup()

    total_checked = int(summary["checked"])
    result = {
        "scene": scene,
        "rows_requested": len(rows),
        "checked": total_checked,
        "skipped_existing": int(summary["skipped_existing"]),
        "failed": int(summary["failed"]),
        "initial_collided": int(summary["initial_collided"]),
        "collided_after_action": int(summary["collided_after_action"]),
        "new_collision_after_action": int(summary["new_collision_after_action"]),
        "initial_collision_rate": (summary["initial_collided"] / total_checked) if total_checked else None,
        "collision_after_action_rate": (summary["collided_after_action"] / total_checked) if total_checked else None,
        "new_collision_after_action_rate": (summary["new_collision_after_action"] / total_checked) if total_checked else None,
        "action_counts": dict(action_counts),
        "action_collided_after_action": dict(action_after_collisions),
        "action_new_collision_after_action": dict(action_new_collisions),
        "output_path": str(output_path),
    }
    return result


def write_summary(output_dir: Path, summaries: list[dict[str, Any]], filename: str = "summary.json") -> None:
    totals = Counter()
    action_counts = Counter()
    action_after = Counter()
    action_new = Counter()
    for item in summaries:
        for key in [
            "rows_requested",
            "checked",
            "skipped_existing",
            "failed",
            "initial_collided",
            "collided_after_action",
            "new_collision_after_action",
        ]:
            totals[key] += int(item.get(key) or 0)
        action_counts.update(item.get("action_counts") or {})
        action_after.update(item.get("action_collided_after_action") or {})
        action_new.update(item.get("action_new_collision_after_action") or {})

    checked = int(totals["checked"])
    payload = {
        "totals": {
            **dict(totals),
            "initial_collision_rate": (totals["initial_collided"] / checked) if checked else None,
            "collision_after_action_rate": (totals["collided_after_action"] / checked) if checked else None,
            "new_collision_after_action_rate": (totals["new_collision_after_action"] / checked) if checked else None,
            "action_counts": dict(action_counts),
            "action_collided_after_action": dict(action_after),
            "action_new_collision_after_action": dict(action_new),
        },
        "scenes": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay train label actions in AirSim and count collisions.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene-list", type=str, default="", help="Comma-separated train scenes. Empty means all scenes.")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=39800)
    parser.add_argument("--limit", type=int, default=0, help="Total source rows to consider before scene grouping.")
    parser.add_argument("--max-rows-per-scene", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true", help="Retry rows that already have error records.")
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--action-velocity", type=float, default=2.0)
    parser.add_argument("--action-move-timeout", type=float, default=5.0)
    parser.add_argument("--action-rotate-timeout", type=float, default=3.0)
    parser.add_argument("--fix-vertical-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-yaw-actions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    scenes = {x for x in args.scene_list.split(",") if x} or None
    rows = load_rows(args.source, scenes, args.limit)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for scene in sorted(rows_by_scene):
        scene_rows = rows_by_scene[scene]
        if args.max_rows_per_scene:
            scene_rows = scene_rows[: args.max_rows_per_scene]
        output_path = args.output_dir / f"{scene}.jsonl"
        print(
            json.dumps(
                {
                    "event": "start_scene",
                    "scene": scene,
                    "rows": len(scene_rows),
                    "gpu": args.gpu,
                    "port": args.base_port + args.gpu,
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        summary = check_scene(
            scene=scene,
            rows=scene_rows,
            output_path=output_path,
            aligned_root=args.aligned_root,
            gpu=args.gpu,
            port=args.base_port + args.gpu,
            retry_errors=args.retry_errors,
            settle_frames=args.settle_frames,
            action_velocity=args.action_velocity,
            action_move_timeout=args.action_move_timeout,
            action_rotate_timeout=args.action_rotate_timeout,
            fix_vertical_actions=args.fix_vertical_actions,
            fix_yaw_actions=args.fix_yaw_actions,
            sleep_seconds=args.sleep_seconds,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        write_summary(args.output_dir, summaries)

    write_summary(args.output_dir, summaries)
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

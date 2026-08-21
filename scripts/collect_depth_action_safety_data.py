#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_ORIGINAL_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048"
)
DEFAULT_REPAIR_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048_repair_lane3"
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed" / "depth_action_safety" / "train"

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_utils import AirsimTrajRecorder, getPoseAfterMakeAction  # noqa: E402
from prepare_collision_filtered_depth_sft_data import load_collision_filter  # noqa: E402
from vlm_baseline.actions import ACTION_IDS, action_name_to_command  # noqa: E402


LABEL_ORDER = [
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
]
SIMULATED_ACTIONS = LABEL_ORDER[1:]


def sample_key(row: dict[str, Any]) -> str:
    return f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}"


def load_filtered_rows(
    source: Path,
    excluded_keys: set[str],
    scenes: set[str] | None,
    max_rows_per_scene: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["source_rows_seen"] += 1
            scene = str(row["scene_id"])
            if scenes and scene not in scenes:
                continue
            stats["requested_scene_rows"] += 1
            if sample_key(row) in excluded_keys:
                stats["excluded_collision_rows"] += 1
                continue
            stats["retained_collision_filtered_rows_seen"] += 1
            if max_rows_per_scene and scene_counts[scene] >= max_rows_per_scene:
                continue
            selected.append(row)
            scene_counts[scene] += 1
            if limit and len(selected) >= limit:
                break
            if scenes and max_rows_per_scene and all(
                scene_counts[name] >= max_rows_per_scene for name in scenes
            ):
                break

    stats["selected_rows"] = len(selected)
    return selected, dict(stats)


def load_episode_pose(
    aligned_root: Path,
    row: dict[str, Any],
    cache: dict[tuple[str, str, str], dict[str, Any]],
) -> list[float]:
    scene = str(row["scene_id"])
    episode_id = str(row["episode_id"])
    pose_idx = str(row["pose_idx"])
    cache_key = (scene, episode_id, pose_idx)
    if cache_key not in cache:
        path = aligned_root / "json" / scene / episode_id / f"{pose_idx}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Aligned transition JSON not found: {path}")
        cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))

    record_list = cache[cache_key].get("record_list") or []
    frame_idx = int(row["frame_idx"])
    if frame_idx >= len(record_list):
        raise IndexError(
            f"{cache_key} frame_idx={frame_idx} is outside record_list len={len(record_list)}"
        )
    pose = record_list[frame_idx]
    if len(pose) < 4:
        raise ValueError(f"{cache_key} frame_idx={frame_idx} has invalid pose: {pose}")
    return [float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])]


def collision_timestamp(info: dict[str, Any] | None) -> int | None:
    if not info:
        return None
    try:
        return int(info.get("time_stamp", 0))
    except Exception:
        return None


def reset_client(
    env: AirsimTrajRecorder,
    settle_seconds: float,
    diagnostic: bool = False,
    diagnostic_context: str = "",
) -> dict[str, Any]:
    status: dict[str, Any] = {"ok": False, "elapsed": 0.0, "error": ""}
    start = time.time()
    try:
        try:
            if diagnostic:
                print(json.dumps({"event": "diagnostic_cancel_start", "context": diagnostic_context}), flush=True)
            env._client.cancelLastTask()
            if diagnostic:
                print(json.dumps({"event": "diagnostic_cancel_complete", "context": diagnostic_context}), flush=True)
        except Exception:
            pass
        if diagnostic:
            print(json.dumps({"event": "diagnostic_reset_start", "context": diagnostic_context}), flush=True)
        env._client.reset()
        if diagnostic:
            print(json.dumps({"event": "diagnostic_reset_complete", "context": diagnostic_context}), flush=True)
        env._client.enableApiControl(True)
        env._client.armDisarm(True)
        if diagnostic:
            print(json.dumps({"event": "diagnostic_control_ready", "context": diagnostic_context}), flush=True)
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        status["ok"] = True
    except Exception as exc:
        status["error"] = repr(exc)
    status["elapsed"] = time.time() - start
    return status


def place_at_expert_pose(
    env: AirsimTrajRecorder,
    pose: list[float],
    settle_frames: int,
    wait_timeout: float,
    position_tolerance: float,
    yaw_tolerance: float,
    poll_interval: float,
    client_reset: bool,
    client_reset_settle_seconds: float,
    retries: int,
    diagnostic: bool = False,
    diagnostic_context: str = "",
) -> dict[str, Any]:
    reset_status = (
        reset_client(
            env,
            settle_seconds=client_reset_settle_seconds,
            diagnostic=diagnostic,
            diagnostic_context=diagnostic_context,
        )
        if client_reset
        else {"ok": True, "enabled": False}
    )
    if not reset_status.get("ok"):
        return {
            "valid": False,
            "client_reset": reset_status,
            "error": "client reset failed",
        }

    attempts: list[dict[str, Any]] = []
    for attempt_index in range(max(1, retries)):
        if diagnostic:
            print(json.dumps({"event": "diagnostic_zero_pose_start", "context": diagnostic_context}), flush=True)
        pose_status = env.zero_kinematics_at_pose(
            pose,
            settle_frames=settle_frames,
            diagnostic=diagnostic,
            diagnostic_context=diagnostic_context,
        )
        if diagnostic:
            print(json.dumps({"event": "diagnostic_zero_pose_complete", "context": diagnostic_context}), flush=True)
            print(json.dumps({"event": "diagnostic_wait_pose_start", "context": diagnostic_context}), flush=True)
        wait_status = env.wait_until_pose(
            pose,
            position_tol=position_tolerance,
            yaw_tol=yaw_tolerance,
            timeout=wait_timeout,
            poll_interval=poll_interval,
        )
        if diagnostic:
            print(json.dumps({"event": "diagnostic_wait_pose_complete", "context": diagnostic_context}), flush=True)
        collision = env.get_collision_info()
        if diagnostic:
            print(json.dumps({"event": "diagnostic_collision_query_complete", "context": diagnostic_context}), flush=True)
        reached = wait_status.get("reached")
        reached_ok = True if reached is None else bool(reached)
        collision_free = not bool(collision.get("has_collided"))
        attempt_status = {
            "attempt": attempt_index + 1,
            "valid": bool(reached_ok and collision_free),
            "reached_ok": reached_ok,
            "collision_free": collision_free,
            "pose_status": pose_status,
            "wait_status": wait_status,
            "collision": collision,
        }
        attempts.append(attempt_status)
        if attempt_status["valid"]:
            return {
                **attempt_status,
                "client_reset": reset_status,
                "attempts": attempts,
                "error": "",
            }

    return {
        **attempts[-1],
        "client_reset": reset_status,
        "attempts": attempts,
        "error": "initial pose is invalid after retries",
    }


def min_pool_depth(depth: np.ndarray, output_size: int, max_meters: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"expected a non-empty 2-D depth image, got {arr.shape}")
    finite = np.isfinite(arr) & (arr >= 0.0)
    clipped = np.where(finite, np.clip(arr, 0.0, max_meters), max_meters)
    height, width = clipped.shape

    if height % output_size == 0 and width % output_size == 0:
        row_factor = height // output_size
        col_factor = width // output_size
        pooled = clipped.reshape(output_size, row_factor, output_size, col_factor).min(axis=(1, 3))
        valid = finite.reshape(output_size, row_factor, output_size, col_factor).any(axis=(1, 3))
    else:
        pooled = cv2.resize(clipped, (output_size, output_size), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(finite.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_AREA) > 0
    return pooled.astype(np.float32), valid


def save_depth_png(
    depth: np.ndarray,
    path: Path,
    output_size: int | None,
    max_meters: float,
    scale_meters: float,
    invalid_value: int,
) -> dict[str, Any]:
    source = np.asarray(depth, dtype=np.float32)
    if output_size is None:
        if source.ndim != 2 or source.size == 0:
            raise ValueError(f"expected a non-empty 2-D depth image, got {source.shape}")
        valid = np.isfinite(source) & (source >= 0.0)
        prepared = np.where(valid, np.clip(source, 0.0, max_meters), max_meters)
    else:
        prepared, valid = min_pool_depth(
            source,
            output_size=output_size,
            max_meters=max_meters,
        )
    encoded = np.rint(prepared / scale_meters).astype(np.uint16)
    encoded[~valid] = np.uint16(invalid_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), encoded):
        raise OSError(f"failed to write depth image: {path}")
    return {
        "path": str(path),
        "source_shape": list(source.shape),
        "encoded_shape": list(encoded.shape),
        "encoding": "uint16_png",
        "scale_meters": scale_meters,
        "invalid_value": invalid_value,
        "max_meters": max_meters,
        "valid_fraction": float(valid.mean()),
        "min_meters": float(np.min(prepared[valid])) if valid.any() else None,
        "max_valid_meters": float(np.max(prepared[valid])) if valid.any() else None,
    }


def depth_path_for_row(output_dir: Path, row: dict[str, Any], directory: str) -> Path:
    return (
        output_dir
        / directory
        / str(row["scene_id"])
        / str(row["episode_id"])
        / str(row["pose_idx"])
        / f"{int(row['frame_idx']):05d}.png"
    )


def write_progress(
    output_dir: Path,
    scene: str,
    requested: int,
    seen: int,
    completed_keys: set[str],
    summary: Counter[str],
    last_key: str,
    last_complete_unix: float | None,
) -> None:
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    path = progress_dir / f"{scene}.json"
    tmp_path = path.with_suffix(".json.tmp")
    payload = {
        "scene": scene,
        "requested": requested,
        "seen_this_process": seen,
        "complete_unique": len(completed_keys),
        "last_key": last_key,
        "last_activity_unix": time.time(),
        "last_complete_unix": last_complete_unix,
        **dict(summary),
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def shuffled_actions(key: str, enabled: bool) -> list[str]:
    actions = list(SIMULATED_ACTIONS)
    if enabled:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
        random.Random(seed).shuffle(actions)
    return actions


def simulate_action(
    env: AirsimTrajRecorder,
    pose: list[float],
    command: str,
    args: argparse.Namespace,
    diagnostic: bool = False,
) -> dict[str, Any]:
    if diagnostic:
        print(
            json.dumps({"event": "diagnostic_action_place_start", "action": command}),
            flush=True,
        )
    initial = place_at_expert_pose(
        env,
        pose,
        settle_frames=args.settle_frames,
        wait_timeout=args.pose_wait_timeout,
        position_tolerance=args.pose_position_tolerance,
        yaw_tolerance=args.pose_yaw_tolerance,
        poll_interval=args.pose_poll_interval,
        client_reset=args.client_reset_per_action,
        client_reset_settle_seconds=args.client_reset_settle_seconds,
        retries=args.initial_pose_retries,
        diagnostic=diagnostic,
        diagnostic_context=f"action:{command}",
    )
    if diagnostic:
        print(
            json.dumps(
                {
                    "event": "diagnostic_action_place_complete",
                    "action": command,
                    "valid": initial.get("valid"),
                }
            ),
            flush=True,
        )
    if not initial.get("valid"):
        return {
            "valid": False,
            "safe": None,
            "simulated": False,
            "initial": initial,
            "error": initial.get("error") or "invalid initial pose",
        }

    action_id = ACTION_IDS[command]
    target_pose = getPoseAfterMakeAction(
        list(pose),
        action_id,
        fix_vertical_actions=args.fix_vertical_actions,
        fix_yaw_actions=args.fix_yaw_actions,
    )
    initial_collision = initial.get("collision") or env.get_collision_info()
    initial_timestamp = collision_timestamp(initial_collision)
    try:
        if diagnostic:
            print(
                json.dumps({"event": "diagnostic_action_execute_start", "action": command}),
                flush=True,
            )
        action_status = env.execute_action_to_pose_join(
            action_id,
            target_pose,
            velocity=args.action_velocity,
            move_timeout=args.action_move_timeout,
            rotate_timeout=args.action_rotate_timeout,
            level_after_action=False,
            level_settle_frames=0,
        )
        if diagnostic:
            print(
                json.dumps({"event": "diagnostic_action_execute_complete", "action": command}),
                flush=True,
            )
        collision_after = action_status.get("collision_info_after_command") or env.get_collision_info()
        after_timestamp = collision_timestamp(collision_after)
        collided_after = bool(collision_after.get("has_collided"))
        new_collision = bool(
            collided_after
            and (
                not bool(initial_collision.get("has_collided"))
                or initial_timestamp is None
                or after_timestamp is None
                or after_timestamp != initial_timestamp
            )
        )
        position_error = action_status.get("position_error")
        yaw_error = action_status.get("yaw_error")
        target_reached = bool(
            position_error is not None
            and yaw_error is not None
            and float(position_error) <= args.action_position_tolerance
            and float(yaw_error) <= args.action_yaw_tolerance
        )
        return {
            "valid": True,
            "safe": not new_collision,
            "simulated": True,
            "action_id": action_id,
            "target_pose": [float(value) for value in target_pose],
            "initial": initial,
            "action_status": action_status,
            "collision_after_action": collision_after,
            "new_collision": new_collision,
            "target_reached": target_reached,
            "error": "",
        }
    except Exception as exc:
        return {
            "valid": False,
            "safe": None,
            "simulated": True,
            "action_id": action_id,
            "target_pose": [float(value) for value in target_pose],
            "initial": initial,
            "error": repr(exc),
        }


def load_completed_keys(path: Path, retry_incomplete: bool) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key"):
                latest[str(row["key"])] = row
    if retry_incomplete:
        return {key for key, row in latest.items() if row.get("complete")}
    return set(latest)


def cleanup_scene_env(env: AirsimTrajRecorder) -> None:
    """Make cleanup idempotent before the same port is reused by the next scene."""
    env.cleanup()
    try:
        env.air_runner.processes.clear()
        env.air_runner.settings_files.clear()
    except Exception:
        pass


def collect_scene(
    scene: str,
    rows: list[dict[str, Any]],
    env: AirsimTrajRecorder,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = output_dir / "labels" / f"{scene}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_load_started = time.time()
    completed_keys = load_completed_keys(output_path, retry_incomplete=args.retry_incomplete)
    print(
        json.dumps(
            {
                "event": "resume_state_loaded",
                "scene": scene,
                "complete_keys": len(completed_keys),
                "elapsed_seconds": round(time.time() - resume_load_started, 3),
            }
        ),
        flush=True,
    )
    pose_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    summary: Counter[str] = Counter()
    action_summary: dict[str, Counter[str]] = defaultdict(Counter)
    last_complete_unix: float | None = None
    first_pending = True

    with output_path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            key = sample_key(row)
            if key in completed_keys:
                summary["skipped_existing"] += 1
                continue

            diagnostic = first_pending
            first_pending = False
            if diagnostic:
                print(
                    json.dumps(
                        {
                            "event": "diagnostic_first_pending",
                            "scene": scene,
                            "key": key,
                            "index": index,
                        }
                    ),
                    flush=True,
                )
            record: dict[str, Any]
            try:
                pose = load_episode_pose(args.aligned_root, row, pose_cache)
                if diagnostic:
                    print(json.dumps({"event": "diagnostic_capture_place_start"}), flush=True)
                capture_initial = place_at_expert_pose(
                    env,
                    pose,
                    settle_frames=args.settle_frames,
                    wait_timeout=args.pose_wait_timeout,
                    position_tolerance=args.pose_position_tolerance,
                    yaw_tolerance=args.pose_yaw_tolerance,
                    poll_interval=args.pose_poll_interval,
                    client_reset=True,
                    client_reset_settle_seconds=args.client_reset_settle_seconds,
                    retries=args.initial_pose_retries,
                    diagnostic=diagnostic,
                    diagnostic_context="capture",
                )
                if diagnostic:
                    print(
                        json.dumps(
                            {
                                "event": "diagnostic_capture_place_complete",
                                "valid": capture_initial.get("valid"),
                            }
                        ),
                        flush=True,
                    )
                if not capture_initial.get("valid"):
                    raise RuntimeError(f"invalid pose before depth capture: {capture_initial}")

                if diagnostic:
                    print(json.dumps({"event": "diagnostic_depth_capture_start"}), flush=True)
                captured = env._capture_images(camera_names=["uav_on_0"], capture_depth=True)["uav_on_0"]
                if diagnostic:
                    print(json.dumps({"event": "diagnostic_depth_capture_complete"}), flush=True)
                depth = captured.get("depth")
                if depth is None:
                    raise RuntimeError("AirSim returned no depth image")
                if args.save_full_resolution_depth and np.asarray(depth).shape != (
                    args.expected_full_depth_size,
                    args.expected_full_depth_size,
                ):
                    raise RuntimeError(
                        "unexpected AirSim depth shape: "
                        f"{np.asarray(depth).shape}, expected "
                        f"{args.expected_full_depth_size}x{args.expected_full_depth_size}"
                    )
                training_depth_info = save_depth_png(
                    depth,
                    depth_path_for_row(output_dir, row, f"depth_{args.depth_output_size}"),
                    output_size=args.depth_output_size,
                    max_meters=args.depth_max_meters,
                    scale_meters=args.depth_scale_meters,
                    invalid_value=args.depth_invalid_value,
                )
                full_depth_info = None
                if args.save_full_resolution_depth:
                    full_depth_info = save_depth_png(
                        depth,
                        depth_path_for_row(output_dir, row, "depth_512"),
                        output_size=None,
                        max_meters=args.depth_max_meters,
                        scale_meters=args.depth_scale_meters,
                        invalid_value=args.depth_invalid_value,
                    )
                depth_info = {
                    "camera": "uav_on_0",
                    "camera_metadata": captured.get("metainfo") or {},
                    "airsim_source_dtype": str(np.asarray(depth).dtype),
                    "airsim_source_shape": list(np.asarray(depth).shape),
                    "full_resolution": full_depth_info,
                    "training_resolution": training_depth_info,
                }

                expert_command = action_name_to_command(str(row["action_name"]))
                action_results: dict[str, dict[str, Any]] = {
                    "stop": {
                        "valid": True,
                        "safe": True,
                        "simulated": False,
                        "action_id": ACTION_IDS["stop"],
                        "reason": "stop is fixed collision-safe and is not replayed",
                    }
                }
                if args.reuse_collision_audited_expert_action and expert_command != "stop":
                    action_results[expert_command] = {
                        "valid": True,
                        "safe": True,
                        "simulated": False,
                        "action_id": ACTION_IDS[expert_command],
                        "reason": (
                            "expert action is reused from the complete collision audit; "
                            "collision/error samples were removed by the source filter"
                        ),
                        "label_source": "collision_audit",
                    }
                    action_summary[expert_command]["reused_collision_audit"] += 1
                for command in shuffled_actions(key, enabled=args.shuffle_actions):
                    if command in action_results:
                        continue
                    result = simulate_action(env, pose, command, args, diagnostic=diagnostic)
                    action_results[command] = result
                    action_summary[command]["attempted"] += 1
                    if not result.get("valid"):
                        action_summary[command]["invalid"] += 1
                    elif result.get("safe"):
                        action_summary[command]["safe"] += 1
                    else:
                        action_summary[command]["unsafe"] += 1

                safe_mask = [
                    1 if action_results[command].get("safe") is True else 0
                    for command in LABEL_ORDER
                ]
                valid_mask = [
                    1 if action_results[command].get("valid") is True else 0
                    for command in LABEL_ORDER
                ]
                complete = all(valid_mask)
                record = {
                    "schema_version": "depth_action_safety_v3",
                    "key": key,
                    "scene_id": scene,
                    "episode_id": str(row["episode_id"]),
                    "pose_idx": str(row["pose_idx"]),
                    "frame_idx": int(row["frame_idx"]),
                    "episode_key": row.get("episode_key"),
                    "expert_pose": pose,
                    "expert_label_action": expert_command,
                    "expert_action_label_source": (
                        "fixed_stop"
                        if expert_command == "stop"
                        else (
                            "collision_audit"
                            if args.reuse_collision_audited_expert_action
                            else "current_replay"
                        )
                    ),
                    "source_rgb_path": row.get("image_path"),
                    "target_description": row.get("target_description"),
                    "collision_filtered_source": True,
                    "depth": depth_info,
                    "label_order": LABEL_ORDER,
                    "safe_mask": safe_mask,
                    "valid_mask": valid_mask,
                    "action_results": action_results,
                    "capture_initial": capture_initial,
                    "complete": complete,
                }
                summary["written"] += 1
                summary["complete"] += int(complete)
                summary["incomplete"] += int(not complete)
                if complete:
                    completed_keys.add(key)
                    last_complete_unix = time.time()
            except Exception as exc:
                summary["failed_rows"] += 1
                record = {
                    "schema_version": "depth_action_safety_v3",
                    "key": key,
                    "scene_id": scene,
                    "episode_id": str(row.get("episode_id")),
                    "pose_idx": str(row.get("pose_idx")),
                    "frame_idx": int(row.get("frame_idx", -1)),
                    "collision_filtered_source": True,
                    "complete": False,
                    "error": repr(exc),
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if diagnostic:
                print(
                    json.dumps(
                        {
                            "event": "diagnostic_first_pending_written",
                            "key": key,
                            "complete": record.get("complete"),
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )
            write_progress(
                output_dir=output_dir,
                scene=scene,
                requested=len(rows),
                seen=index,
                completed_keys=completed_keys,
                summary=summary,
                last_key=key,
                last_complete_unix=last_complete_unix,
            )
            if index % args.progress_interval == 0 or index == len(rows):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "scene": scene,
                            "seen": index,
                            "requested": len(rows),
                            **summary,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    return {
        "scene": scene,
        "rows_requested": len(rows),
        **dict(summary),
        "actions": {command: dict(counts) for command, counts in action_summary.items()},
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect depth images and five-action collision-safety labels at collision-filtered expert poses."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--original-collision-dir", type=Path, default=DEFAULT_ORIGINAL_COLLISION_DIR)
    parser.add_argument("--repair-collision-dir", type=Path, default=DEFAULT_REPAIR_COLLISION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene-list", type=str, default="")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=47000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-rows-per-scene", type=int, default=0)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--summary-name", type=str, default="")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--pose-wait-timeout", type=float, default=1.0)
    parser.add_argument("--pose-position-tolerance", type=float, default=0.2)
    parser.add_argument("--pose-yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--pose-poll-interval", type=float, default=0.05)
    parser.add_argument("--initial-pose-retries", type=int, default=3)
    parser.add_argument("--action-position-tolerance", type=float, default=0.75)
    parser.add_argument("--action-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--action-velocity", type=float, default=2.0)
    parser.add_argument("--action-move-timeout", type=float, default=5.0)
    parser.add_argument("--action-rotate-timeout", type=float, default=3.0)
    parser.add_argument("--fix-vertical-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-yaw-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client-reset-per-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client-reset-settle-seconds", type=float, default=0.05)
    parser.add_argument("--shuffle-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-collision-audited-expert-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--depth-output-size", type=int, default=128)
    parser.add_argument(
        "--save-full-resolution-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expected-full-depth-size", type=int, default=512)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    parser.add_argument("--depth-scale-meters", type=float, default=0.01)
    parser.add_argument("--depth-invalid-value", type=int, default=65535)
    args = parser.parse_args()

    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if not 0 < args.depth_invalid_value <= np.iinfo(np.uint16).max:
        parser.error("--depth-invalid-value must fit uint16 and be non-zero")
    max_encoded = args.depth_max_meters / args.depth_scale_meters
    if max_encoded >= args.depth_invalid_value:
        parser.error("valid depth encoding overlaps --depth-invalid-value")

    scenes = {name for name in args.scene_list.split(",") if name} or None
    excluded_keys, filter_stats = load_collision_filter(
        original_dir=args.original_collision_dir,
        repair_dir=args.repair_collision_dir,
    )
    rows, selection_stats = load_filtered_rows(
        source=args.source,
        excluded_keys=excluded_keys,
        scenes=scenes,
        max_rows_per_scene=args.max_rows_per_scene,
        limit=args.limit,
    )
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene_id"])].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "selection_complete",
                "gpu": args.gpu,
                "port": args.base_port + args.gpu,
                "scenes": sorted(rows_by_scene),
                "selection": selection_stats,
                "collision_filter": {
                    "excluded_keys": filter_stats["excluded_keys"],
                    "repair_override_keys": filter_stats["repair_override_keys"],
                    "unresolved_error_keys": filter_stats["unresolved_error_keys"],
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for scene in sorted(rows_by_scene):
        scene_rows = rows_by_scene[scene]
        print(
            json.dumps(
                {
                    "event": "start_scene",
                    "scene": scene,
                    "rows": len(scene_rows),
                    "gpu": args.gpu,
                    "port": args.base_port + args.gpu,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        env = AirsimTrajRecorder(scene, airsim_port=args.base_port + args.gpu, device_id=args.gpu)
        try:
            summaries.append(collect_scene(scene, scene_rows, env, args.output_dir, args))
        finally:
            cleanup_scene_env(env)
            del env
            time.sleep(1.0)

    payload = {
        "schema_version": "depth_action_safety_v3",
        "source": str(args.source),
        "aligned_root": str(args.aligned_root),
        "output_dir": str(args.output_dir),
        "gpu": args.gpu,
        "port": args.base_port + args.gpu,
        "selection": selection_stats,
        "collision_filter": filter_stats,
        "label_order": LABEL_ORDER,
        "action_execution": {
            "mode": "apex_join",
            "velocity": args.action_velocity,
            "move_timeout": args.action_move_timeout,
            "rotate_timeout": args.action_rotate_timeout,
            "fix_vertical_actions": args.fix_vertical_actions,
            "fix_yaw_actions": args.fix_yaw_actions,
            "client_reset_per_action": args.client_reset_per_action,
            "client_reset_settle_seconds": args.client_reset_settle_seconds,
            "initial_pose_retries": args.initial_pose_retries,
            "reuse_collision_audited_expert_action": (
                args.reuse_collision_audited_expert_action
            ),
        },
        "depth_encoding": {
            "camera": "uav_on_0",
            "training_output_size": args.depth_output_size,
            "save_full_resolution": args.save_full_resolution_depth,
            "full_resolution_shape": [
                args.expected_full_depth_size,
                args.expected_full_depth_size,
            ],
            "max_meters": args.depth_max_meters,
            "scale_meters": args.depth_scale_meters,
            "invalid_value": args.depth_invalid_value,
        },
        "scenes": summaries,
    }
    summary_name = args.summary_name or f"summary_gpu{args.gpu}.json"
    summary_path = args.output_dir / summary_name
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "summary": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

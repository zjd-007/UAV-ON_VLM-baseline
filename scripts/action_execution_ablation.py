#!/usr/bin/env python3
"""Replay recorded first actions under different AirSim execution modes.

This diagnostic is intentionally model-free: it reuses the first action that was
already produced in an eval run, then compares how AirSim executes that action.
It is meant to separate policy errors from simulator/action-execution effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import airsim
import numpy as np
from scipy.spatial.transform import Rotation as R

from eval_utils import AirsimTrajRecorder, getPoseAfterMakeAction


def dist3(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.array(a[:3], dtype=float) - np.array(b[:3], dtype=float)))


def angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi)


def quat_from_yaw(yaw: float) -> airsim.Quaternionr:
    x, y, z, w = R.from_euler("ZYX", [float(yaw), 0.0, 0.0]).as_quat()
    return airsim.Quaternionr(float(x), float(y), float(z), float(w))


def pose_obj(pose: list[float]) -> airsim.Pose:
    return airsim.Pose(
        airsim.Vector3r(float(pose[0]), float(pose[1]), float(pose[2])),
        quat_from_yaw(float(pose[3])),
    )


def kinematics_state(pose: list[float]) -> airsim.KinematicsState:
    state = airsim.KinematicsState()
    state.position = airsim.Vector3r(float(pose[0]), float(pose[1]), float(pose[2]))
    state.orientation = quat_from_yaw(float(pose[3]))
    state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
    return state


def get_state(env: AirsimTrajRecorder) -> dict[str, Any]:
    kin = env._client.getMultirotorState().kinematics_estimated
    return {
        "pose": env.get_vehicle_pose_xyzyaw(),
        "rpy": env.get_vehicle_pose_xyzrpyyaw(),
        "collision": env.get_collision_info(),
        "linear_velocity": [
            float(kin.linear_velocity.x_val),
            float(kin.linear_velocity.y_val),
            float(kin.linear_velocity.z_val),
        ],
        "angular_velocity": [
            float(kin.angular_velocity.x_val),
            float(kin.angular_velocity.y_val),
            float(kin.angular_velocity.z_val),
        ],
    }


def reset_zero(env: AirsimTrajRecorder, pose: list[float], settle_frames: int) -> dict[str, Any]:
    env.zero_kinematics_at_pose(pose, settle_frames=settle_frames)
    return get_state(env)


def reset_airsim_then_zero(env: AirsimTrajRecorder, pose: list[float], settle_frames: int) -> dict[str, Any]:
    env._client.reset()
    env._client.enableApiControl(True)
    env._client.armDisarm(True)
    time.sleep(0.1)
    env.zero_kinematics_at_pose(pose, settle_frames=settle_frames)
    return get_state(env)


def teleport_to_pose(
    env: AirsimTrajRecorder,
    pose: list[float],
    pause_after: bool = True,
    ignore_collision: bool = True,
    settle_frames: int = 1,
) -> dict[str, Any]:
    env._client.cancelLastTask()
    env._client.simPause(False)
    env._client.simSetKinematics(kinematics_state(pose), bool(ignore_collision))
    if settle_frames > 0:
        env._client.simContinueForFrames(int(settle_frames))
    if not pause_after:
        env._client.simPause(False)
    else:
        env._client.simPause(True)
    return {
        "mode": "teleport_ignore_collision" if ignore_collision else "teleport_collision_checked",
        "actual_pose": env.get_vehicle_pose_xyzyaw(),
        "vehicle_pose_after_command": env.get_vehicle_pose_xyzrpyyaw(),
        "collision_info_after_command": env.get_collision_info(),
        "elapsed": 0.0,
    }


def execute_mode(
    env: AirsimTrajRecorder,
    action_id: int,
    target_pose: list[float],
    mode: str,
    velocity: float,
    move_timeout: float,
    rotate_timeout: float,
) -> dict[str, Any]:
    if mode in {"apex_join", "reset_then_apex_join"}:
        return env.execute_action_to_pose_join(
            action_id,
            target_pose,
            velocity=velocity,
            move_timeout=move_timeout,
            rotate_timeout=rotate_timeout,
            level_after_action=False,
            level_settle_frames=0,
        )
    if mode == "turn_teleport_move_apex":
        if action_id in {2, 3}:
            return teleport_to_pose(env, target_pose)
        return env.execute_action_to_pose_join(
            action_id,
            target_pose,
            velocity=velocity,
            move_timeout=move_timeout,
            rotate_timeout=rotate_timeout,
            level_after_action=False,
            level_settle_frames=0,
        )
    if mode == "all_teleport":
        return teleport_to_pose(env, target_pose)
    if mode == "all_teleport_collision_checked":
        return teleport_to_pose(env, target_pose, ignore_collision=False, settle_frames=1)
    raise ValueError(f"unknown mode: {mode}")


def load_cases(review_dir: Path, limit: int | None, scenes: set[str] | None) -> list[dict[str, Any]]:
    cases = []
    for path in sorted((review_dir / "json").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if scenes and data.get("map_name") not in scenes:
            continue
        steps = data.get("step_records") or []
        if not steps:
            continue
        first = steps[0]
        cases.append(
            {
                "scene": data.get("map_name"),
                "episode_id": str(data.get("episode_id")),
                "source_json": str(path),
                "pose_before": list(first["pose_before"]),
                "action_id": int(first["action_id"]),
                "action": first.get("parsed_command"),
                "source_pose_after": first.get("pose_after"),
                "source_collision": first.get("collision_info"),
                "source_target": first.get("target_pose_after"),
            }
        )
    cases.sort(key=lambda r: (r["scene"], int(r["episode_id"]) if r["episode_id"].isdigit() else r["episode_id"]))
    if limit is not None:
        cases = cases[:limit]
    return cases


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    summary = {"total_trials": len(rows), "by_mode": {}}
    for mode, items in sorted(by_mode.items()):
        n = len(items)
        if not n:
            continue
        summary["by_mode"][mode] = {
            "n": n,
            "collisions": sum(1 for r in items if r["collided"]),
            "collision_rate": sum(1 for r in items if r["collided"]) / n,
            "target_error_gt1m": sum(1 for r in items if (r["target_error"] or 0) > 1.0),
            "target_error_gt1m_rate": sum(1 for r in items if (r["target_error"] or 0) > 1.0) / n,
            "turn_big_drift_gt1m": sum(1 for r in items if r["is_turn"] and (r["drift_from_start"] or 0) > 1.0),
            "turn_big_drift_gt1m_rate": sum(1 for r in items if r["is_turn"] and (r["drift_from_start"] or 0) > 1.0) / n,
            "mean_target_error": float(np.mean([r["target_error"] for r in items if r["target_error"] is not None] or [0.0])),
            "median_target_error": float(np.median([r["target_error"] for r in items if r["target_error"] is not None] or [0.0])),
            "by_action": dict(Counter(r["action"] for r in items)),
            "collisions_by_action": dict(Counter(r["action"] for r in items if r["collided"])),
            "by_scene": dict(Counter(r["scene"] for r in items)),
            "collisions_by_scene": dict(Counter(r["scene"] for r in items if r["collided"])),
        }
    return summary


def run(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_dir)
    output_dir = Path(args.output_dir) if args.output_dir else review_dir / "action_execution_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trials.jsonl"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if not (args.retry_errors and row.get("error")):
                completed.add((str(row.get("scene")), str(row.get("episode_id")), str(row.get("mode"))))
        print(f"resume: loaded {len(rows)} existing rows, {len(completed)} completed trial keys", flush=True)
    else:
        jsonl_path.write_text("", encoding="utf-8")

    scenes = set(args.scenes.split(",")) if args.scenes else None
    cases = load_cases(review_dir, args.limit, scenes)
    cases_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_scene[case["scene"]].append(case)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for scene, scene_cases in sorted(cases_by_scene.items()):
        print(f"=== scene {scene}: {len(scene_cases)} cases ===", flush=True)
        env = None
        try:
            env = AirsimTrajRecorder(scene, airsim_port=args.airsim_port, device_id=args.gpu)
            if args.rpc_timeout > 0:
                env._client = airsim.MultirotorClient(
                    port=args.airsim_port,
                    timeout_value=float(args.rpc_timeout),
                )
                env._client.confirmConnection()
                env._client.enableApiControl(True)
                env._client.armDisarm(True)
            for case in scene_cases:
                pose_before = case["pose_before"]
                action_id = case["action_id"]
                target_pose = getPoseAfterMakeAction(
                    list(pose_before),
                    action_id,
                    fix_vertical_actions=True,
                    fix_yaw_actions=True,
                )
                for mode in modes:
                    trial_key = (str(case["scene"]), str(case["episode_id"]), str(mode))
                    if trial_key in completed:
                        print(f"{case['scene']}-{case['episode_id']} {mode}: skip existing", flush=True)
                        continue
                    try:
                        if mode == "reset_then_apex_join":
                            reset_state = reset_airsim_then_zero(env, pose_before, args.settle_frames)
                        else:
                            reset_state = reset_zero(env, pose_before, args.settle_frames)
                        baseline = env.get_collision_info()
                        before = get_state(env)
                        status = execute_mode(
                            env,
                            action_id,
                            target_pose,
                            mode,
                            args.velocity,
                            args.move_timeout,
                            args.rotate_timeout,
                        )
                        after = get_state(env)
                        collision = status.get("collision_info_after_command") or after["collision"]
                        collided = bool(collision.get("has_collided"))
                        actual_pose = status.get("actual_pose") or after["pose"]
                        row = {
                            **case,
                            "mode": mode,
                            "target_pose": target_pose,
                            "reset_pose": reset_state["pose"],
                            "reset_collision_has_collided": reset_state["collision"].get("has_collided"),
                            "reset_collision_object": reset_state["collision"].get("object_name"),
                            "baseline_collision_has_collided": baseline.get("has_collided"),
                            "baseline_collision_object": baseline.get("object_name"),
                            "baseline_collision_ts": baseline.get("time_stamp"),
                            "before_pose": before["pose"],
                            "before_linear_velocity": before["linear_velocity"],
                            "before_angular_velocity": before["angular_velocity"],
                            "actual_pose": actual_pose,
                            "after_pose": after["pose"],
                            "after_linear_velocity": after["linear_velocity"],
                            "after_angular_velocity": after["angular_velocity"],
                            "collision_has_collided": collided,
                            "collision_object": collision.get("object_name"),
                            "collision_ts": collision.get("time_stamp"),
                            "same_ts_as_baseline": collision.get("time_stamp") == baseline.get("time_stamp"),
                            "collided": collided,
                            "is_turn": action_id in {2, 3},
                            "drift_from_start": dist3(pose_before, actual_pose),
                            "target_error": dist3(target_pose, actual_pose),
                            "yaw_error": angle_diff(actual_pose[3], target_pose[3]) if actual_pose else None,
                            "source_drift_from_start": dist3(pose_before, case.get("source_pose_after")),
                            "source_target_error": dist3(case.get("source_target"), case.get("source_pose_after")),
                            "error": "",
                        }
                    except Exception as exc:
                        row = {
                            **case,
                            "mode": mode,
                            "target_pose": target_pose,
                            "error": repr(exc),
                            "collided": None,
                            "is_turn": action_id in {2, 3},
                        }
                    rows.append(row)
                    if not (args.retry_errors and row.get("error")):
                        completed.add(trial_key)
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    print(
                        f"{scene}-{case['episode_id']} {mode}: "
                        f"collided={row.get('collided')} target_error={row.get('target_error')} "
                        f"drift={row.get('drift_from_start')} obj={row.get('collision_object')} err={row.get('error')}",
                        flush=True,
                    )
        finally:
            if env is not None:
                env.cleanup()
                time.sleep(args.scene_teardown_sleep)

    csv_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize([row for row in rows if not row.get("error")])
    summary.update(
        {
            "num_cases": len(cases),
            "modes": modes,
            "review_dir": str(review_dir),
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review_dir",
        default="/data/zhujd/Aerial-ObjectNav/VLM-baseline/results/phi35_fixed_full_eval_20260702_213705/immediate_collision_review",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--airsim_port", type=int, default=32434)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenes", default=None)
    parser.add_argument(
        "--modes",
        default="apex_join,turn_teleport_move_apex,reset_then_apex_join,all_teleport",
    )
    parser.add_argument("--settle_frames", type=int, default=1)
    parser.add_argument("--velocity", type=float, default=2.0)
    parser.add_argument("--move_timeout", type=float, default=5.0)
    parser.add_argument("--rotate_timeout", type=float, default=3.0)
    parser.add_argument("--scene_teardown_sleep", type=float, default=2.0)
    parser.add_argument("--rpc_timeout", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        help="When resuming, retry trial keys whose existing rows contain an error.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

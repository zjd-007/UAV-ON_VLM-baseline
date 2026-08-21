#!/usr/bin/env python3
"""AirSim action-speed sanity check without loading the VLM."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "eval"))

from eval_utils import AirsimTrajRecorder, getPoseAfterMakeAction, to_eularian_angles  # noqa: E402


ACTION_NAMES = {
    1: "forward 3m",
    2: "turn left 30 degree",
    3: "turn right 30 degree",
    4: "ascend 3m",
    5: "descend 3m",
}


def load_episode(split: Path, scene: str, episode_id: str | None) -> dict:
    rows = json.loads(split.read_text(encoding="utf-8"))
    for row in rows:
        if row["map_name"] != scene:
            continue
        if episode_id is None or str(row["episode_id"]) == str(episode_id):
            return row
    raise ValueError(f"episode not found: scene={scene}, episode_id={episode_id}")


def start_pose_from_episode(item: dict) -> list[float]:
    start_pose = item["start_pose"]
    pos = start_pose["start_position"]
    quat = start_pose["start_quaternionr"]
    _, _, yaw = to_eularian_angles(quat[0], quat[1], quat[2], quat[3])
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)]


def reset_to_pose(env: AirsimTrajRecorder, pose: list[float], settle_frames: int) -> dict:
    env._set_camera_pose(pose[0], pose[1], pose[2], pose[3], 0.0, 0.0, settle_frames=settle_frames)
    return {
        "actual_pose": env.get_vehicle_pose_xyzyaw(),
        "vehicle_pose": env.get_vehicle_pose_xyzrpyyaw(),
        "collision_info": env.get_collision_info(),
    }


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for key_fn, key_name in [
        (lambda r: f"v{r['velocity']}", "by_velocity"),
        (lambda r: r["action_name"], "by_action"),
        (lambda r: f"v{r['velocity']}|{r['action_name']}", "by_velocity_action"),
    ]:
        group: dict[str, list[dict]] = {}
        for row in rows:
            group.setdefault(key_fn(row), []).append(row)
        summary[key_name] = {}
        for key, vals in sorted(group.items()):
            elapsed = [float(v["exec"]["elapsed"]) for v in vals]
            pos_err = [float(v["exec"]["position_error"]) for v in vals]
            yaw_err = [abs(float(v["exec"]["yaw_error"])) for v in vals]
            pitch_abs = [
                abs(float(v["exec"].get("vehicle_pose_after_command", {}).get("pitch", 0.0)))
                for v in vals
            ]
            roll_abs = [
                abs(float(v["exec"].get("vehicle_pose_after_command", {}).get("roll", 0.0)))
                for v in vals
            ]
            collisions = [
                bool(v["exec"].get("collision_info_after_command", {}).get("has_collided"))
                for v in vals
            ]
            summary[key_name][key] = {
                "n": len(vals),
                "elapsed_mean": float(np.mean(elapsed)),
                "elapsed_median": float(np.median(elapsed)),
                "position_error_mean": float(np.mean(pos_err)),
                "position_error_max": float(np.max(pos_err)),
                "yaw_error_mean": float(np.mean(yaw_err)),
                "abs_pitch_deg_mean": float(math.degrees(np.mean(pitch_abs))),
                "abs_roll_deg_mean": float(math.degrees(np.mean(roll_abs))),
                "collision_count": int(sum(collisions)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="BrushifyRoad_test")
    parser.add_argument("--episode_id", default=None)
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "UAV-ON_dataset/splits/uavon_raw_json/test.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/diagnostics/action_speed_sanity")
    parser.add_argument("--airsim_port", type=int, default=32602)
    parser.add_argument("--simulator_gpu", type=int, default=2)
    parser.add_argument("--velocities", default="1.0,2.0,3.0")
    parser.add_argument("--actions", default="1,4,5,2,3")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--move_timeout", type=float, default=5.0)
    parser.add_argument("--rotate_timeout", type=float, default=3.0)
    parser.add_argument("--settle_frames", type=int, default=1)
    parser.add_argument("--fix_vertical_actions", action="store_true", default=True)
    parser.add_argument("--fix_yaw_actions", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    velocities = [float(x) for x in args.velocities.split(",") if x.strip()]
    actions = [int(x) for x in args.actions.split(",") if x.strip()]
    item = load_episode(args.split, args.scene, args.episode_id)
    start_pose = start_pose_from_episode(item)

    run_config = {
        "scene": args.scene,
        "episode_id": str(item["episode_id"]),
        "split": str(args.split),
        "airsim_port": args.airsim_port,
        "simulator_gpu": args.simulator_gpu,
        "velocities": velocities,
        "actions": actions,
        "repeats": args.repeats,
        "move_timeout": args.move_timeout,
        "rotate_timeout": args.rotate_timeout,
        "settle_frames": args.settle_frames,
        "start_pose": start_pose,
        "fix_vertical_actions": bool(args.fix_vertical_actions),
        "fix_yaw_actions": bool(args.fix_yaw_actions),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = AirsimTrajRecorder(args.scene, airsim_port=args.airsim_port, device_id=args.simulator_gpu)
    rows: list[dict] = []
    try:
        for velocity in velocities:
            for action_id in actions:
                for repeat in range(args.repeats):
                    reset_info = reset_to_pose(env, start_pose, args.settle_frames)
                    target_pose = getPoseAfterMakeAction(
                        list(start_pose),
                        action_id,
                        fix_vertical_actions=args.fix_vertical_actions,
                        fix_yaw_actions=args.fix_yaw_actions,
                    )
                    exec_info = env.execute_action_to_pose_join(
                        action_id,
                        target_pose,
                        velocity=velocity,
                        move_timeout=args.move_timeout,
                        rotate_timeout=args.rotate_timeout,
                        level_after_action=False,
                    )
                    row = {
                        "timestamp": time.time(),
                        "velocity": velocity,
                        "action_id": action_id,
                        "action_name": ACTION_NAMES.get(action_id, str(action_id)),
                        "repeat": repeat,
                        "reset": reset_info,
                        "target_pose": target_pose,
                        "exec": exec_info,
                    }
                    rows.append(row)
                    print(
                        f"v={velocity:.1f} action={row['action_name']} repeat={repeat} "
                        f"elapsed={exec_info['elapsed']:.3f}s pos_err={exec_info['position_error']:.3f} "
                        f"collision={exec_info.get('collision_info_after_command', {}).get('has_collided')}",
                        flush=True,
                    )
                    (args.output_dir / "records.jsonl").write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8",
                    )
        summary = summarize(rows)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary["by_velocity"], ensure_ascii=False, indent=2))
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()

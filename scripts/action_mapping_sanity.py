#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from eval_utils import AirsimTrajRecorder, getPoseAfterMakeAction, to_eularian_angles  # noqa: E402


ACTION_NAMES = {
    0: "stop",
    1: "forward_3m",
    2: "turn_left_30deg",
    3: "turn_right_30deg",
    4: "ascend_3m",
    5: "descend_3m",
}


def find_episode(split_path: Path, scene: str, episode_id: str) -> dict:
    items = json.loads(split_path.read_text(encoding="utf-8"))
    for item in items:
        if item["map_name"] == scene and str(item["episode_id"]) == str(episode_id):
            return item
    raise ValueError(f"Episode not found: scene={scene}, episode_id={episode_id}")


def save_rgb(image_array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).convert("RGB").save(path, quality=92)


def capture(env: AirsimTrajRecorder, out_dir: Path, step_name: str) -> dict:
    data = env._capture_images(camera_names=["uav_on_0"], capture_depth=True)["uav_on_0"]
    image_path = out_dir / f"{step_name}.jpg"
    save_rgb(data["rgb"], image_path)
    metainfo = data.get("metainfo", {})
    return {
        "image": str(image_path),
        "camera_pos": metainfo.get("camera_pos"),
        "quat_wb": metainfo.get("quat_wb"),
        "fov": metainfo.get("fov"),
    }


def run_sequence(env: AirsimTrajRecorder, start_pose: list[float], action_id: int, steps: int, out_dir: Path) -> dict:
    action_name = ACTION_NAMES.get(action_id, f"action_{action_id}")
    seq_dir = out_dir / f"action_{action_id}_{action_name}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    pose = list(start_pose)
    env._set_camera_pose(pose[0], pose[1], pose[2], pose[3], 0, 0)
    time.sleep(0.2)

    records = []
    cap = capture(env, seq_dir, "step_000_before")
    records.append(
        {
            "step": 0,
            "phase": "before",
            "pose": list(pose),
            **cap,
        }
    )

    for step in range(1, steps + 1):
        before = list(pose)
        pose = getPoseAfterMakeAction(pose, action_id)
        env._set_camera_pose(pose[0], pose[1], pose[2], pose[3], 0, 0)
        time.sleep(0.2)
        cap = capture(env, seq_dir, f"step_{step:03d}_after")
        records.append(
            {
                "step": step,
                "phase": "after",
                "action_id": action_id,
                "action_name": action_name,
                "pose_before": before,
                "pose": list(pose),
                "delta_pose": [pose[i] - before[i] for i in range(4)],
                **cap,
            }
        )

    return {
        "action_id": action_id,
        "action_name": action_name,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check UAV-ON action id to AirSim motion mapping.")
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT.parent / "UAV-ON_dataset" / "splits" / "uavon_raw_json" / "test.json",
    )
    parser.add_argument("--scene", default="WinterTown_test")
    parser.add_argument("--episode_id", default="77")
    parser.add_argument("--actions", default="1,4,5")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "action_mapping_sanity")
    parser.add_argument("--airsim_port", type=int, default=31606)
    parser.add_argument("--simulator_gpu", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item = find_episode(args.split, args.scene, args.episode_id)
    start_position = item["start_pose"]["start_position"]
    start_quaternion = item["start_pose"]["start_quaternionr"]
    _, _, yaw = to_eularian_angles(
        start_quaternion[0],
        start_quaternion[1],
        start_quaternion[2],
        start_quaternion[3],
    )
    start_pose = [start_position[0], start_position[1], start_position[2], yaw]
    action_ids = [int(part.strip()) for part in args.actions.split(",") if part.strip()]

    run_dir = args.output_dir / f"{args.scene}_episode_{args.episode_id}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scene": args.scene,
        "episode_id": str(args.episode_id),
        "start_pose": start_pose,
        "airsim_port": args.airsim_port,
        "simulator_gpu": args.simulator_gpu,
        "actions": action_ids,
        "steps": args.steps,
        "sequences": [],
    }

    env = AirsimTrajRecorder(args.scene, airsim_port=args.airsim_port, device_id=args.simulator_gpu)
    try:
        for action_id in action_ids:
            summary["sequences"].append(run_sequence(env, start_pose, action_id, args.steps, run_dir))
    finally:
        env.cleanup()

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(run_dir)
    for seq in summary["sequences"]:
        first_after = seq["records"][1]
        last = seq["records"][-1]
        print(
            f"action {seq['action_id']} {seq['action_name']}: "
            f"first_delta={first_after['delta_pose']} final_pose={last['pose']} "
            f"final_camera_pos={last.get('camera_pos')}"
        )


if __name__ == "__main__":
    main()

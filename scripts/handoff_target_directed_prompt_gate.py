#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CONFIG_ENV_KEYS = {
    "model_path": "MODEL_PATH",
    "eval_dataset": "EVAL_DATASET",
    "eval_max_steps": "EVAL_MAX_STEPS",
    "max_new_tokens": "MAX_NEW_TOKENS",
    "save_step_images": "SAVE_STEP_IMAGES",
    "image_save_stride": "IMAGE_SAVE_STRIDE",
    "image_quality": "IMAGE_QUALITY",
    "inference_mode": "INFERENCE_MODE",
    "depth_avoidance": "DEPTH_AVOIDANCE",
    "depth_grid_size": "DEPTH_GRID_SIZE",
    "depth_max_meters": "DEPTH_MAX_METERS",
    "depth_forward_threshold": "DEPTH_FORWARD_THRESHOLD",
    "depth_turn_threshold": "DEPTH_TURN_THRESHOLD",
    "depth_descend_threshold": "DEPTH_DESCEND_THRESHOLD",
    "depth_ascend_top_threshold": "DEPTH_ASCEND_TOP_THRESHOLD",
    "action_redirect": "ACTION_REDIRECT",
    "action_redirect_search_radius": "ACTION_REDIRECT_SEARCH_RADIUS",
    "action_redirect_near_obstacle_threshold": "ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD",
    "memory_history_size": "MEMORY_HISTORY_SIZE",
    "memory_search_radius": "MEMORY_SEARCH_RADIUS",
    "memory_include_search_bounds": "MEMORY_INCLUDE_SEARCH_BOUNDS",
    "memory_pose_yaw_unit": "MEMORY_POSE_YAW_UNIT",
    "fix_vertical_actions": "FIX_VERTICAL_ACTIONS",
    "fix_yaw_actions": "FIX_YAW_ACTIONS",
    "pose_wait_timeout": "POSE_WAIT_TIMEOUT",
    "pose_wait_position_tol": "POSE_WAIT_POSITION_TOL",
    "pose_wait_yaw_tol": "POSE_WAIT_YAW_TOL",
    "pose_wait_poll_interval": "POSE_WAIT_POLL_INTERVAL",
    "render_settle_seconds": "RENDER_SETTLE_SECONDS",
    "action_execution_mode": "ACTION_EXECUTION_MODE",
    "action_sim_frames": "ACTION_SIM_FRAMES",
    "action_velocity": "ACTION_VELOCITY",
    "action_move_timeout": "ACTION_MOVE_TIMEOUT",
    "action_rotate_timeout": "ACTION_ROTATE_TIMEOUT",
    "level_after_action": "LEVEL_AFTER_ACTION",
    "level_settle_frames": "LEVEL_SETTLE_FRAMES",
    "initial_pose_retries": "INITIAL_POSE_RETRIES",
    "initial_pose_settle_frames": "INITIAL_POSE_SETTLE_FRAMES",
    "zero_kinematics_reset": "ZERO_KINEMATICS_RESET",
    "client_reset_per_episode": "CLIENT_RESET_PER_EPISODE",
    "nvidia_compat_lib": "NVIDIA_COMPAT_LIB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch target_directed_v1_1 after the original policy gate completes."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-log-dir", type=Path)
    parser.add_argument(
        "--target-prefix",
        default="phi35_target_directed_v1_1_targeted100",
    )
    parser.add_argument(
        "--target-memory-context",
        default="uavon_pose_history_target_directed_v1_1",
    )
    parser.add_argument("--gpus", default="0,1,3")
    parser.add_argument("--base-port", type=int, default=56400)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--watchdog-interval-seconds", type=float, default=60.0)
    parser.add_argument("--watchdog-stale-seconds", type=float, default=300.0)
    parser.add_argument("--state-file", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_keys(run_dir: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in run_dir.glob("lane*/temp/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        keys.add((str(row.get("map_name")), str(row.get("episode_id"))))
    return keys


def expected_keys(config: dict) -> set[tuple[str, str]]:
    rows = load_rows(Path(config["eval_dataset"]))
    limit = config.get("eval_samples_per_env")
    if limit is None:
        return {(str(row["map_name"]), str(row["episode_id"])) for row in rows}

    per_scene: dict[str, int] = {}
    selected: set[tuple[str, str]] = set()
    for row in rows:
        scene = str(row["map_name"])
        count = per_scene.get(scene, 0)
        if count >= int(limit):
            continue
        per_scene[scene] = count + 1
        selected.add((scene, str(row["episode_id"])))
    return selected


def stop_screen(session: str) -> None:
    subprocess.run(
        ["screen", "-S", session, "-X", "quit"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_watchdog(
    run_id: str,
    run_dir: Path,
    log_dir: Path,
    interval_seconds: float,
    stale_seconds: float,
) -> None:
    command = " && ".join(
        [
            "source /data/zhujd/miniconda3/etc/profile.d/conda.sh",
            "conda activate octmem_openvla_nomemory",
            f"cd {shlex.quote(str(ROOT))}",
            (
                "PYTHONUNBUFFERED=1 "
                f"PYTHONPATH={shlex.quote(str(ROOT / 'src') + ':' + str(ROOT) + ':' + str(ROOT / 'eval'))} "
                "python -u scripts/eval_lane_watchdog.py "
                f"--run_dir {shlex.quote(str(run_dir))} "
                f"--log_dir {shlex.quote(str(log_dir))} "
                f"--interval_seconds {interval_seconds} "
                f"--stale_seconds {stale_seconds} --restart_if_no_activity "
                f"2>&1 | tee {shlex.quote(str(log_dir / 'watchdog.log'))}"
            ),
        ]
    )
    subprocess.run(
        ["screen", "-dmS", f"{run_id}_watchdog", "bash", "-lc", command],
        check=True,
    )


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    source_config = json.loads(
        (source_run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    source_run_id = str(source_config["run_id"])
    source_log_dir = (
        args.source_log_dir.resolve()
        if args.source_log_dir
        else (ROOT / "logs" / source_run_id)
    )
    state_file = (
        args.state_file.resolve()
        if args.state_file
        else source_log_dir / "target_directed_handoff.json"
    )
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("status") == "launched":
            print(json.dumps(state, ensure_ascii=True), flush=True)
            return

    expected = expected_keys(source_config)
    while True:
        completed = completed_keys(source_run_dir)
        missing = expected - completed
        print(
            f"[{time.strftime('%F %T')}] source progress "
            f"{len(completed & expected)}/{len(expected)} missing={len(missing)}",
            flush=True,
        )
        if not missing:
            break
        time.sleep(args.poll_seconds)

    stop_screen(f"{source_run_id}_watchdog")
    time.sleep(10)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    target_run_id = f"{args.target_prefix}_{stamp}"
    target_run_dir = ROOT / "results" / target_run_id
    target_log_dir = ROOT / "logs" / target_run_id
    if target_run_dir.exists() or target_log_dir.exists():
        raise FileExistsError(f"target run already exists: {target_run_id}")

    env = os.environ.copy()
    for config_key, env_key in CONFIG_ENV_KEYS.items():
        value = source_config.get(config_key)
        if value is not None:
            env[env_key] = str(value)
    env.update(
        {
            "RUN_ID": target_run_id,
            "RUN_DIR": str(target_run_dir),
            "LOG_DIR": str(target_log_dir),
            "LANE_GPUS": args.gpus,
            "BASE_PORT": str(args.base_port),
            "MEMORY_CONTEXT": args.target_memory_context,
            "KILL_ENV_PROCESS": "0",
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval_phi35_uavon_parallel.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    start_watchdog(
        target_run_id,
        target_run_dir,
        target_log_dir,
        args.watchdog_interval_seconds,
        args.watchdog_stale_seconds,
    )

    state = {
        "status": "launched",
        "source_run_id": source_run_id,
        "source_completed": len(expected),
        "target_run_id": target_run_id,
        "target_run_dir": str(target_run_dir),
        "target_log_dir": str(target_log_dir),
        "target_memory_context": args.target_memory_context,
        "gpus": args.gpus,
        "base_port": args.base_port,
        "launched_at": time.strftime("%F %T"),
    }
    state_file.write_text(
        json.dumps(state, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

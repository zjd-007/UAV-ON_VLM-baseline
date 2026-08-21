#!/usr/bin/env python3
"""Launch a Phi evaluation after another run completes and its GPUs are idle."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lane_watchdog import parse_lanes, read_completed_rows, screen_exists, stop_lane  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, **updates) -> dict:
    state = load_json(path) if path.exists() else {}
    state.update(updates)
    state["updated_at"] = time.strftime("%F %T")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return state


def task_key(row: dict) -> tuple[str, str]:
    return str(row.get("map_name")), str(row.get("episode_id"))


def source_coverage(run_dir: Path) -> dict:
    config = load_json(run_dir / "run_config.json")
    expected_rows = json.loads(Path(config["eval_dataset"]).read_text(encoding="utf-8"))
    expected = {task_key(row) for row in expected_rows}

    observed_counts: Counter = Counter()
    lane_counts: dict[str, int] = {}
    for lane_dir in sorted(path for path in run_dir.glob("lane*") if path.is_dir()):
        rows = read_completed_rows(lane_dir)
        lane_counts[lane_dir.name] = len(rows)
        observed_counts.update(task_key(row) for row in rows)

    observed = set(observed_counts)
    duplicate_keys = sorted(key for key, count in observed_counts.items() if count > 1)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "expected": len(expected),
        "unique": len(observed & expected),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": len(duplicate_keys),
        "lane_counts": lane_counts,
        "missing_examples": missing[:5],
        "duplicate_examples": duplicate_keys[:5],
        "complete": not missing and not extra and not duplicate_keys,
    }


def format_coverage(coverage: dict) -> str:
    lanes = " ".join(f"{lane}={count}" for lane, count in coverage["lane_counts"].items())
    return (
        f"unique={coverage['unique']}/{coverage['expected']} missing={coverage['missing']} "
        f"extra={coverage['extra']} duplicates={coverage['duplicates']} {lanes}"
    )


def wait_until_source_complete(run_dir: Path, poll_seconds: float) -> dict:
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    run_id = str(config["run_id"])
    while True:
        coverage = source_coverage(run_dir)
        log(f"source progress {format_coverage(coverage)}")
        if coverage["complete"]:
            return coverage

        lane_alive = any(screen_exists(f"{run_id}_{lane}") for lane in lanes)
        watchdog_alive = screen_exists(f"{run_id}_watchdog")
        if not lane_alive and not watchdog_alive:
            raise RuntimeError(f"source run {run_id} is incomplete but has no active lane or watchdog")
        time.sleep(poll_seconds)


def wait_for_source_lanes_to_exit(run_dir: Path, timeout_seconds: float = 300.0) -> None:
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    run_id = str(config["run_id"])
    deadline = time.time() + timeout_seconds
    while True:
        alive = [lane for lane in lanes if screen_exists(f"{run_id}_{lane}")]
        if not alive:
            log("all source evaluator screens exited naturally")
            return
        if time.time() >= deadline:
            log(f"source evaluator cleanup timeout; residual lanes will be stopped: {alive}")
            return
        log(f"source results are complete; waiting for evaluator cleanup: {alive}")
        time.sleep(5)


def stop_source_run(run_dir: Path) -> None:
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    run_id = str(config["run_id"])
    subprocess.run(
        ["screen", "-S", f"{run_id}_watchdog", "-X", "quit"],
        text=True,
        capture_output=True,
        check=False,
    )
    log(f"stopped source watchdog {run_id}_watchdog")
    for lane, lane_cfg in sorted(lanes.items()):
        stop_lane(lane, int(lane_cfg["gpu"]), run_dir, config)
        log(f"cleaned source lane {run_id}_{lane}")


def gpu_status(gpus: list[int], compat_lib: str) -> dict[int, dict]:
    env = dict(os.environ)
    if compat_lib:
        env["LD_LIBRARY_PATH"] = compat_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    status: dict[int, dict] = {}
    uuid_to_index: dict[str, int] = {}
    for raw in gpu_query.stdout.splitlines():
        index_text, uuid, memory_text = (part.strip() for part in raw.split(",", 2))
        index = int(index_text)
        if index not in gpus:
            continue
        status[index] = {"memory_mib": int(memory_text), "processes": []}
        uuid_to_index[uuid] = index

    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    for raw in process_query.stdout.splitlines():
        if not raw.strip():
            continue
        uuid, pid, name, memory = (part.strip() for part in raw.split(",", 3))
        index = uuid_to_index.get(uuid)
        if index is None:
            continue
        status[index]["processes"].append(
            {"pid": int(pid), "name": name, "memory_mib": int(memory)}
        )
    return status


def format_gpu_status(status: dict[int, dict]) -> str:
    parts = []
    for gpu, item in sorted(status.items()):
        pids = ",".join(str(proc["pid"]) for proc in item["processes"]) or "none"
        parts.append(f"gpu{gpu}={item['memory_mib']}MiB,pids={pids}")
    return " ".join(parts)


def wait_for_idle_gpus(
    gpus: list[int],
    compat_lib: str,
    poll_seconds: float,
    memory_threshold_mib: int,
    confirmations: int,
) -> dict[int, dict]:
    consecutive = 0
    while True:
        status = gpu_status(gpus, compat_lib)
        missing = set(gpus) - set(status)
        idle = not missing and all(
            not item["processes"] and item["memory_mib"] <= memory_threshold_mib
            for item in status.values()
        )
        consecutive = consecutive + 1 if idle else 0
        log(
            f"GPU idle check {consecutive}/{confirmations}: {format_gpu_status(status)}"
            + (f" missing={sorted(missing)}" if missing else "")
        )
        if consecutive >= confirmations:
            return status
        time.sleep(poll_seconds)


def config_env(config: dict, run_id: str, model_path: Path, gpus: list[int], base_port: int) -> dict:
    values = {
        "RUN_ID": run_id,
        "MODEL_PATH": str(model_path),
        "EVAL_DATASET": str(config["eval_dataset"]),
        "EVAL_SAMPLES_PER_ENV": (
            "" if config.get("eval_samples_per_env") is None else str(config["eval_samples_per_env"])
        ),
        "LANE_GPUS": ",".join(str(gpu) for gpu in gpus),
        "NVIDIA_COMPAT_LIB": str(config.get("nvidia_compat_lib", "")),
        "BASE_PORT": str(base_port),
        "EVAL_MAX_STEPS": str(config["eval_max_steps"]),
        "MAX_NEW_TOKENS": str(config["max_new_tokens"]),
        "SAVE_STEP_IMAGES": str(config.get("save_step_images", 1)),
        "IMAGE_SAVE_STRIDE": str(config.get("image_save_stride", 1)),
        "IMAGE_QUALITY": str(config.get("image_quality", 85)),
        "INFERENCE_MODE": str(config.get("inference_mode", "generate")),
        "DEPTH_AVOIDANCE": str(config.get("depth_avoidance", "uavon_single_view_prompt")),
        "DEPTH_GRID_SIZE": str(config.get("depth_grid_size", 3)),
        "DEPTH_MAX_METERS": str(config.get("depth_max_meters", 100.0)),
        "DEPTH_FORWARD_THRESHOLD": str(config.get("depth_forward_threshold", 4.0)),
        "DEPTH_TURN_THRESHOLD": str(config.get("depth_turn_threshold", 1.5)),
        "DEPTH_DESCEND_THRESHOLD": str(config.get("depth_descend_threshold", 6.0)),
        "DEPTH_ASCEND_TOP_THRESHOLD": str(config.get("depth_ascend_top_threshold", 8.0)),
        "ACTION_REDIRECT": str(config.get("action_redirect", "none")),
        "ACTION_REDIRECT_SEARCH_RADIUS": str(config.get("action_redirect_search_radius", 50.0)),
        "ACTION_REDIRECT_NEAR_OBSTACLE_THRESHOLD": str(
            config.get("action_redirect_near_obstacle_threshold", 2.0)
        ),
        "MEMORY_CONTEXT": str(config.get("memory_context", "uavon_pose_history")),
        "MEMORY_HISTORY_SIZE": str(config.get("memory_history_size", 5)),
        "MEMORY_SEARCH_RADIUS": str(config.get("memory_search_radius", 50.0)),
        "MEMORY_INCLUDE_SEARCH_BOUNDS": str(config.get("memory_include_search_bounds", 0)),
        "MEMORY_POSE_YAW_UNIT": str(config.get("memory_pose_yaw_unit", "radians")),
        "FIX_VERTICAL_ACTIONS": str(config.get("fix_vertical_actions", 1)),
        "FIX_YAW_ACTIONS": str(config.get("fix_yaw_actions", 1)),
        "POSE_WAIT_TIMEOUT": str(config.get("pose_wait_timeout", 1.0)),
        "POSE_WAIT_POSITION_TOL": str(config.get("pose_wait_position_tol", 0.2)),
        "POSE_WAIT_YAW_TOL": str(config.get("pose_wait_yaw_tol", 0.05)),
        "POSE_WAIT_POLL_INTERVAL": str(config.get("pose_wait_poll_interval", 0.05)),
        "RENDER_SETTLE_SECONDS": str(config.get("render_settle_seconds", 0.0)),
        "ACTION_EXECUTION_MODE": str(config.get("action_execution_mode", "apex_join")),
        "ACTION_SIM_FRAMES": str(config.get("action_sim_frames", 150)),
        "ACTION_VELOCITY": str(config.get("action_velocity", 2.0)),
        "ACTION_MOVE_TIMEOUT": str(config.get("action_move_timeout", 5.0)),
        "ACTION_ROTATE_TIMEOUT": str(config.get("action_rotate_timeout", 3.0)),
        "LEVEL_AFTER_ACTION": str(config.get("level_after_action", 0)),
        "LEVEL_SETTLE_FRAMES": str(config.get("level_settle_frames", 1)),
        "INITIAL_POSE_RETRIES": str(config.get("initial_pose_retries", 3)),
        "INITIAL_POSE_SETTLE_FRAMES": str(config.get("initial_pose_settle_frames", 1)),
        "ZERO_KINEMATICS_RESET": str(config.get("zero_kinematics_reset", 1)),
        "CLIENT_RESET_PER_EPISODE": str(config.get("client_reset_per_episode", 1)),
        "KILL_ENV_PROCESS": "0",
    }
    env = dict(os.environ)
    env.update(values)
    return env


def start_watchdog(run_id: str, run_dir: Path, log_dir: Path, args: argparse.Namespace) -> None:
    session = f"{run_id}_watchdog"
    command = (
        f"source {shlex.quote(args.conda_sh)} && "
        f"conda activate {shlex.quote(args.conda_env)} && "
        f"cd {shlex.quote(str(ROOT))} && "
        "export PYTHONUNBUFFERED=1 && "
        "python -u scripts/eval_lane_watchdog.py "
        f"--run_dir {shlex.quote(str(run_dir))} "
        f"--log_dir {shlex.quote(str(log_dir))} "
        f"--interval_seconds {args.watchdog_interval_seconds} "
        f"--stale_seconds {args.watchdog_stale_seconds} "
        "--restart_if_no_activity "
        f"2>&1 | tee -a {shlex.quote(str(log_dir / 'watchdog.log'))}"
    )
    result = subprocess.run(
        ["screen", "-dmS", session, "bash", "-lc", command],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to start watchdog: {result.stderr.strip()}")


def launch_target(args: argparse.Namespace, source_config: dict, run_id: str) -> Path:
    env = config_env(source_config, run_id, args.target_model_path, args.gpus, args.base_port)
    subprocess.run(
        ["bash", "scripts/eval_phi35_uavon_parallel.sh"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    run_dir = ROOT / "results" / run_id
    log_dir = ROOT / "logs" / run_id
    target_config_path = run_dir / "run_config.json"
    target_config = load_json(target_config_path)
    target_config.update(
        {
            "parent_run_id": source_config["run_id"],
            "handoff_controller": Path(__file__).name,
            "handoff_require_idle_gpus": args.gpus,
            "handoff_created_at": time.strftime("%F %T"),
        }
    )
    target_config_path.write_text(
        json.dumps(target_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    start_watchdog(run_id, run_dir, log_dir, args)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument("--target_model_path", type=Path, required=True)
    parser.add_argument("--target_prefix", default="phi35_cfmem_v2_ckpt14000")
    parser.add_argument("--gpus", default="0,1,3")
    parser.add_argument("--base_port", type=int, default=53300)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--gpu_memory_threshold_mib", type=int, default=512)
    parser.add_argument("--gpu_idle_confirmations", type=int, default=2)
    parser.add_argument("--watchdog_interval_seconds", type=int, default=60)
    parser.add_argument("--watchdog_stale_seconds", type=int, default=300)
    parser.add_argument("--conda_sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda_env", default="octmem_openvla_nomemory")
    parser.add_argument("--state_dir", type=Path)
    parser.add_argument("--check_once", action="store_true")
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus.split(",") if value]
    if len(args.gpus) != 3:
        raise ValueError(f"expected exactly three GPUs, got {args.gpus}")
    return args


def main() -> None:
    args = parse_args()
    source_run_dir = ROOT / "results" / args.source_run_id
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    if not args.target_model_path.is_dir():
        raise FileNotFoundError(args.target_model_path)

    source_config = load_json(source_run_dir / "run_config.json")
    coverage = source_coverage(source_run_dir)
    status = gpu_status(args.gpus, str(source_config.get("nvidia_compat_lib", "")))
    if args.check_once:
        log(f"source {format_coverage(coverage)}")
        log(f"GPU status {format_gpu_status(status)}")
        return

    state_dir = args.state_dir or ROOT / "logs" / (
        f"handoff_{args.source_run_id}_to_{args.target_prefix}"
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    if state.get("stage") == "full_launched":
        log(f"target already launched: {state.get('target_run_id')}")
        return

    if state.get("stage") != "source_stopped":
        save_state(state_path, stage="waiting_source", source_run_id=args.source_run_id)
        coverage = wait_until_source_complete(source_run_dir, args.poll_seconds)
        save_state(state_path, stage="source_complete", source_coverage=coverage)
        wait_for_source_lanes_to_exit(source_run_dir)
        stop_source_run(source_run_dir)
        save_state(state_path, stage="source_stopped")

    wait_for_idle_gpus(
        args.gpus,
        str(source_config.get("nvidia_compat_lib", "")),
        args.poll_seconds,
        args.gpu_memory_threshold_mib,
        args.gpu_idle_confirmations,
    )
    target_run_id = f"{args.target_prefix}_full_{time.strftime('%Y%m%d_%H%M%S')}"
    save_state(state_path, stage="launching_full", target_run_id=target_run_id)
    run_dir = launch_target(args, source_config, target_run_id)
    time.sleep(5)
    expected_sessions = [f"{target_run_id}_lane{i}" for i in range(3)]
    missing_sessions = [session for session in expected_sessions if not screen_exists(session)]
    if missing_sessions:
        raise RuntimeError(f"target lane screens missing after launch: {missing_sessions}")
    if not screen_exists(f"{target_run_id}_watchdog"):
        raise RuntimeError("target watchdog screen missing after launch")
    save_state(state_path, stage="full_launched", target_run_dir=str(run_dir))
    log(f"handoff complete: launched {target_run_id} on GPUs {args.gpus}")


if __name__ == "__main__":
    main()

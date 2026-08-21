#!/usr/bin/env python3
"""Wait for one Qwen eval, stop it cleanly, smoke-test, and launch its successor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lane_watchdog import (  # noqa: E402
    expected_count_for_scenes,
    lane_scene_counts,
    parse_lanes,
    read_completed_rows,
    screen_exists,
    stop_lane,
)


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


def run_progress(run_dir: Path) -> tuple[dict[str, tuple[int, int]], bool]:
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    scene_count_cache = {}
    progress: dict[str, tuple[int, int]] = {}
    for lane, lane_cfg in sorted(lanes.items()):
        rows = read_completed_rows(run_dir / lane)
        scenes = set(lane_cfg["scenes"])
        completed_keys = {
            (row.get("map_name"), str(row.get("episode_id")))
            for row in rows
            if row.get("map_name") in scenes
        }
        scene_counts = lane_scene_counts(lane_cfg, config, scene_count_cache)
        expected = expected_count_for_scenes(scene_counts, lane_cfg["scenes"])
        sample_limit = config.get("eval_samples_per_env")
        if sample_limit is not None:
            expected = sum(
                min(scene_counts[scene], int(sample_limit)) for scene in lane_cfg["scenes"]
            )
        progress[lane] = (len(completed_keys), expected)
    return progress, all(done >= expected for done, expected in progress.values())


def format_progress(progress: dict[str, tuple[int, int]]) -> str:
    return " ".join(f"{lane}={done}/{expected}" for lane, (done, expected) in progress.items())


def wait_until_complete(run_dir: Path, poll_seconds: float, timeout_seconds: float = 0) -> None:
    started = time.time()
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    while True:
        progress, complete = run_progress(run_dir)
        log(f"progress run={config['run_id']} {format_progress(progress)}")
        if complete:
            return
        if timeout_seconds > 0 and time.time() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for {config['run_id']}")
        alive = [
            lane
            for lane in lanes
            if screen_exists(f"{config['run_id']}_{lane}")
        ]
        if not alive and not screen_exists(f"{config['run_id']}_watchdog"):
            raise RuntimeError(
                f"{config['run_id']} is incomplete but has no lane or watchdog screens"
            )
        time.sleep(poll_seconds)


def stop_run(run_dir: Path) -> None:
    config = load_json(run_dir / "run_config.json")
    lanes = parse_lanes(run_dir / "lanes.tsv")
    run_id = str(config["run_id"])
    subprocess.run(
        ["screen", "-S", f"{run_id}_watchdog", "-X", "quit"],
        text=True,
        capture_output=True,
    )
    log(f"stopped watchdog {run_id}_watchdog")
    for lane, lane_cfg in sorted(lanes.items()):
        stop_lane(lane, int(lane_cfg["gpu"]), run_dir, config)
        log(f"stopped residual processes for {run_id}_{lane}")


def gpu_memory_mib(gpus: list[int], compat_lib: str) -> dict[int, int]:
    env = dict(os.environ)
    if compat_lib:
        env["LD_LIBRARY_PATH"] = compat_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    values = {}
    for line in result.stdout.splitlines():
        index, used = (part.strip() for part in line.split(",", 1))
        if int(index) in gpus:
            values[int(index)] = int(used)
    return values


def wait_for_free_gpus(
    gpus: list[int],
    compat_lib: str,
    poll_seconds: float,
    threshold_mib: int,
) -> None:
    while True:
        used = gpu_memory_mib(gpus, compat_lib)
        log("GPU memory " + " ".join(f"gpu{gpu}={used.get(gpu, -1)}MiB" for gpu in gpus))
        if all(used.get(gpu, threshold_mib + 1) <= threshold_mib for gpu in gpus):
            return
        time.sleep(poll_seconds)


def launch_qwen_run(
    run_id: str,
    source_config: dict,
    gpus: list[int],
    simulator_gpus: list[int],
    base_port: int,
    memory_context: str,
    memory_history_size: int,
    eval_samples_per_env: int | None,
    eval_max_steps: int,
    start_watchdog: bool,
) -> Path:
    env = dict(os.environ)
    env.update(
        {
            "RUN_ID": run_id,
            "MODEL_PATH": str(source_config["model_path"]),
            "EVAL_DATASET": str(source_config["eval_dataset"]),
            "LANE_GPUS": ",".join(str(gpu) for gpu in gpus),
            "SIMULATOR_GPUS": ",".join(str(gpu) for gpu in simulator_gpus),
            "NVIDIA_COMPAT_LIB": str(source_config.get("nvidia_compat_lib", "")),
            "BASE_PORT": str(base_port),
            "EVAL_SAMPLES_PER_ENV": (
                "" if eval_samples_per_env is None else str(eval_samples_per_env)
            ),
            "EVAL_MAX_STEPS": str(eval_max_steps),
            "MAX_NEW_TOKENS": str(source_config.get("max_new_tokens", 8)),
            "SAVE_STEP_IMAGES": str(source_config.get("save_step_images", 1)),
            "IMAGE_SAVE_STRIDE": str(source_config.get("image_save_stride", 1)),
            "IMAGE_QUALITY": str(source_config.get("image_quality", 85)),
            "DEPTH_AVOIDANCE": str(
                source_config.get("depth_avoidance", "uavon_single_view_prompt")
            ),
            "ACTION_REDIRECT": str(source_config.get("action_redirect", "none")),
            "MEMORY_CONTEXT": memory_context,
            "MEMORY_HISTORY_SIZE": str(memory_history_size),
            "MEMORY_POSE_YAW_UNIT": str(
                source_config.get("memory_pose_yaw_unit", "radians")
            ),
            "START_WATCHDOG": "1" if start_watchdog else "0",
            "WATCHDOG_INTERVAL_SECONDS": "60",
            "WATCHDOG_STALE_SECONDS": "300",
            "PYTORCH_CUDA_ALLOC_CONF": str(
                source_config.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
            ),
            "QWEN_MERGE_ADAPTER_FOR_INFERENCE": str(
                source_config.get("qwen_merge_adapter_for_inference", 0)
            ),
        }
    )
    subprocess.run(
        ["bash", "scripts/eval_qwen25_vl_uavon_parallel.sh"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return ROOT / "results" / run_id


def validate_smoke(run_dir: Path, expected_memory_module: str, history_size: int) -> None:
    progress, complete = run_progress(run_dir)
    if not complete:
        raise RuntimeError(f"smoke incomplete: {format_progress(progress)}")
    row_count = 0
    evaluated_rows = 0
    initial_pose_failures: list[str] = []
    for lane_dir in sorted(run_dir.glob("lane*")):
        for row in read_completed_rows(lane_dir):
            row_count += 1
            status = row.get("initial_pose_wait_status") or {}
            sample_key = f"{row.get('map_name')}-{row.get('episode_id')}"
            if not status.get("reached_ok"):
                if (
                    row.get("termination_reason") != "initial_pose_reset_failed"
                    or not status.get("collision_free")
                    or row.get("step_records")
                ):
                    raise RuntimeError(
                        f"malformed initial pose failure in smoke: {sample_key}"
                    )
                initial_pose_failures.append(sample_key)
                continue
            if not status.get("collision_free"):
                raise RuntimeError(f"initial collision in smoke: {sample_key}")
            evaluated_rows += 1
            for step in row.get("step_records", []):
                memory = step.get("memory_context") or {}
                summary = memory.get("summary") or {}
                if memory.get("module") != expected_memory_module:
                    raise RuntimeError(f"unexpected memory module: {memory.get('module')}")
                step_index = int(step.get("step", 0))
                expected_poses = min(step_index + 1, history_size)
                expected_actions = min(step_index, history_size)
                if len(summary.get("recent_poses", [])) != expected_poses:
                    raise RuntimeError(
                        f"unexpected recent pose count in {sample_key} step {step_index}"
                    )
                if len(summary.get("recent_actions", [])) != expected_actions:
                    raise RuntimeError(
                        f"unexpected recent action count in {sample_key} step {step_index}"
                    )
            if not all(row.get("parse_matched", [])):
                raise RuntimeError(f"unparsed action in smoke: {sample_key}")
    if row_count != 14:
        raise RuntimeError(f"expected 14 smoke rows, got {row_count}")
    if evaluated_rows == 0:
        raise RuntimeError("smoke did not execute any model steps")
    for log_file in sorted((ROOT / "logs" / run_dir.name).glob("lane*.log")):
        text = log_file.read_text(encoding="utf-8", errors="replace")
        for marker in ["Traceback", "CUDA out of memory"]:
            if marker in text:
                raise RuntimeError(f"{marker} found in {log_file}")
    log(
        f"smoke validation passed: rows={row_count} evaluated={evaluated_rows} "
        f"initial_pose_failures={initial_pose_failures}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument("--target_prefix", default="qwen25vl_lora_cfmem_real5_action5")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--simulator_gpus", default="4,5,6,7")
    parser.add_argument("--memory_context", default="uavon_pose_action_history")
    parser.add_argument("--memory_history_size", type=int, default=5)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--smoke_timeout_seconds", type=float, default=3600.0)
    parser.add_argument("--smoke_base_port", type=int, default=48400)
    parser.add_argument("--full_base_port", type=int, default=48500)
    parser.add_argument("--gpu_free_threshold_mib", type=int, default=512)
    parser.add_argument("--state_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    simulator_gpus = [int(value) for value in args.simulator_gpus.split(",") if value]
    if len(gpus) != 4:
        raise ValueError(f"expected four GPUs, got {gpus}")
    if len(simulator_gpus) != len(gpus):
        raise ValueError(
            f"expected {len(gpus)} simulator GPUs, got {simulator_gpus}"
        )
    source_run_dir = ROOT / "results" / args.source_run_id
    source_config = load_json(source_run_dir / "run_config.json")
    state_dir = args.state_dir or ROOT / "logs" / (
        f"handoff_{args.source_run_id}_to_{args.target_prefix}"
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    if state.get("stage") == "full_launched":
        log(f"full run already launched: {state.get('full_run_id')}")
        return

    compat_lib = str(source_config.get("nvidia_compat_lib", ""))
    if state.get("stage") == "smoke_running":
        smoke_run_dir = ROOT / "results" / str(state["smoke_run_id"])
        wait_until_complete(
            smoke_run_dir,
            min(args.poll_seconds, 30.0),
            timeout_seconds=args.smoke_timeout_seconds,
        )
        validate_smoke(smoke_run_dir, args.memory_context, args.memory_history_size)
        stop_run(smoke_run_dir)
        save_state(state_path, stage="smoke_passed")
    elif state.get("stage") != "smoke_passed":
        save_state(state_path, stage="waiting_source", source_run_id=args.source_run_id)
        wait_until_complete(source_run_dir, args.poll_seconds)
        save_state(state_path, stage="source_complete")
        stop_run(source_run_dir)
        save_state(state_path, stage="source_stopped")

        wait_for_free_gpus(
            gpus,
            compat_lib,
            args.poll_seconds,
            args.gpu_free_threshold_mib,
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        smoke_run_id = f"{args.target_prefix}_smoke_{stamp}"
        save_state(state_path, stage="launching_smoke", smoke_run_id=smoke_run_id)
        smoke_run_dir = launch_qwen_run(
            smoke_run_id,
            source_config,
            gpus,
            simulator_gpus,
            args.smoke_base_port,
            args.memory_context,
            args.memory_history_size,
            eval_samples_per_env=1,
            eval_max_steps=2,
            start_watchdog=False,
        )
        save_state(state_path, stage="smoke_running")
        wait_until_complete(
            smoke_run_dir,
            min(args.poll_seconds, 30.0),
            timeout_seconds=args.smoke_timeout_seconds,
        )
        validate_smoke(smoke_run_dir, args.memory_context, args.memory_history_size)
        stop_run(smoke_run_dir)
        save_state(state_path, stage="smoke_passed")

    wait_for_free_gpus(
        gpus,
        compat_lib,
        args.poll_seconds,
        args.gpu_free_threshold_mib,
    )
    full_run_id = f"{args.target_prefix}_full_{time.strftime('%Y%m%d_%H%M%S')}"
    save_state(state_path, stage="launching_full", full_run_id=full_run_id)
    launch_qwen_run(
        full_run_id,
        source_config,
        gpus,
        simulator_gpus,
        args.full_base_port,
        args.memory_context,
        args.memory_history_size,
        eval_samples_per_env=None,
        eval_max_steps=100,
        start_watchdog=True,
    )
    save_state(state_path, stage="full_launched")
    log(f"handoff complete: launched {full_run_id}")


if __name__ == "__main__":
    main()

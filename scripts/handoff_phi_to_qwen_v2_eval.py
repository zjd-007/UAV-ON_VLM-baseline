#!/usr/bin/env python3
"""Wait for a Phi run, smoke-test Qwen with its prompt, then launch full eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lane_watchdog import read_completed_rows, screen_exists  # noqa: E402
from handoff_phi_checkpoint_eval import wait_for_source_lanes_to_exit  # noqa: E402
from handoff_qwen_memory_eval import (  # noqa: E402
    format_progress,
    launch_qwen_run,
    load_json,
    run_progress,
    save_state,
    stop_run,
    validate_smoke,
    wait_for_free_gpus,
    wait_until_complete,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def task_key(row: dict) -> tuple[str, str]:
    return str(row.get("map_name")), str(row.get("episode_id"))


def validate_global_coverage(run_dir: Path) -> dict:
    config = load_json(run_dir / "run_config.json")
    expected_rows = json.loads(Path(config["eval_dataset"]).read_text(encoding="utf-8"))
    expected = {task_key(row) for row in expected_rows}
    counts: Counter = Counter()
    lane_counts = {}
    for lane_dir in sorted(path for path in run_dir.glob("lane*") if path.is_dir()):
        rows = read_completed_rows(lane_dir)
        lane_counts[lane_dir.name] = len(rows)
        counts.update(task_key(row) for row in rows)
    observed = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    summary = {
        "expected": len(expected),
        "unique": len(observed & expected),
        "duplicates": len(duplicates),
        "missing": len(missing),
        "extra": len(extra),
        "lane_counts": lane_counts,
        "duplicate_examples": duplicates[:5],
        "missing_examples": missing[:5],
        "extra_examples": extra[:5],
    }
    if missing or extra or duplicates:
        raise RuntimeError(f"source coverage validation failed: {summary}")
    log(f"source global coverage passed: {summary}")
    return summary


def validate_reference_config(config: dict) -> None:
    expected = {
        "eval_max_steps": 100,
        "max_new_tokens": 8,
        "depth_avoidance": "uavon_single_view_prompt",
        "depth_grid_size": 3,
        "action_redirect": "none",
        "memory_context": "uavon_pose_history_v2",
        "memory_history_size": 5,
        "memory_include_search_bounds": 0,
        "memory_pose_yaw_unit": "radians",
        "fix_vertical_actions": 1,
        "fix_yaw_actions": 1,
        "action_execution_mode": "apex_join",
        "zero_kinematics_reset": 1,
        "client_reset_per_episode": 1,
    }
    differences = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if differences:
        raise RuntimeError(f"reference config is not the expected Phi V2 setup: {differences}")


def validate_qwen_config(
    run_dir: Path,
    reference: dict,
    model_path: Path,
    gpus: list[int],
    eval_samples_per_env: int | None,
    eval_max_steps: int,
) -> None:
    config = load_json(run_dir / "run_config.json")
    if config.get("evaluator_script") != "eval/eval_qwen25_vl_uavon.py":
        raise RuntimeError(f"unexpected evaluator: {config.get('evaluator_script')}")
    if config.get("model_path") != str(model_path):
        raise RuntimeError(f"unexpected model: {config.get('model_path')}")
    if config.get("eval_samples_per_env") != eval_samples_per_env:
        raise RuntimeError(
            f"unexpected eval_samples_per_env: {config.get('eval_samples_per_env')}"
        )
    if int(config.get("eval_max_steps")) != int(eval_max_steps):
        raise RuntimeError(f"unexpected eval_max_steps: {config.get('eval_max_steps')}")
    if int(config.get("lane_count")) != len(gpus):
        raise RuntimeError(f"unexpected lane_count: {config.get('lane_count')}")
    simulator_gpus = config.get("simulator_gpu_by_lane", {})
    expected_simulator_gpus = {f"lane{i}": gpu for i, gpu in enumerate(gpus)}
    if simulator_gpus != expected_simulator_gpus:
        raise RuntimeError(
            f"unexpected simulator GPU mapping: {simulator_gpus}, "
            f"expected {expected_simulator_gpus}"
        )

    same_keys = [
        "eval_dataset",
        "max_new_tokens",
        "save_step_images",
        "image_save_stride",
        "image_quality",
        "inference_mode",
        "depth_avoidance",
        "depth_grid_size",
        "depth_max_meters",
        "depth_forward_threshold",
        "depth_turn_threshold",
        "depth_descend_threshold",
        "depth_ascend_top_threshold",
        "action_redirect",
        "action_redirect_search_radius",
        "action_redirect_near_obstacle_threshold",
        "memory_context",
        "memory_history_size",
        "memory_search_radius",
        "memory_include_search_bounds",
        "memory_pose_yaw_unit",
        "fix_vertical_actions",
        "fix_yaw_actions",
        "pose_wait_timeout",
        "pose_wait_position_tol",
        "pose_wait_yaw_tol",
        "pose_wait_poll_interval",
        "render_settle_seconds",
        "action_execution_mode",
        "action_sim_frames",
        "action_velocity",
        "action_move_timeout",
        "action_rotate_timeout",
        "level_after_action",
        "level_settle_frames",
        "initial_pose_retries",
        "initial_pose_settle_frames",
        "zero_kinematics_reset",
        "client_reset_per_episode",
        "kill_env_process",
    ]
    differences = {
        key: {"reference": reference.get(key), "qwen": config.get(key)}
        for key in same_keys
        if reference.get(key) != config.get(key)
    }
    if differences:
        raise RuntimeError(f"Qwen config differs from Phi V2 reference: {differences}")


def validate_v2_smoke(run_dir: Path, reference: dict, args: argparse.Namespace) -> None:
    validate_qwen_config(
        run_dir,
        reference,
        args.target_model_path,
        args.gpus,
        eval_samples_per_env=1,
        eval_max_steps=2,
    )
    validate_smoke(run_dir, "uavon_pose_history_v2", 5)
    steps = []
    for lane_dir in sorted(run_dir.glob("lane*")):
        for row in read_completed_rows(lane_dir):
            steps.extend(row.get("step_records", []))
    if not steps:
        raise RuntimeError("Qwen smoke produced no model steps")
    for step in steps:
        memory = step.get("memory_context") or {}
        summary = memory.get("summary") or {}
        prompt = str(memory.get("prompt_text", ""))
        depth = step.get("depth_avoidance") or {}
        redirect = step.get("action_redirect") or {}
        grid = depth.get("depth_grid")
        if summary.get("prompt_version") != "uavon_pose_history_v2":
            raise RuntimeError("smoke did not save the V2 prompt version")
        if summary.get("include_search_bounds") or "SearchBounds" in prompt:
            raise RuntimeError("SearchBounds unexpectedly present in Qwen smoke prompt")
        if depth.get("module") != "uavon_single_view_prompt":
            raise RuntimeError(f"unexpected depth module: {depth.get('module')}")
        if not isinstance(grid, list) or len(grid) != 3 or any(len(row) != 3 for row in grid):
            raise RuntimeError(f"unexpected depth grid: {grid}")
        if depth.get("depth_summary") is not None:
            raise RuntimeError("DepthSummary unexpectedly present in Qwen smoke")
        if redirect.get("enabled") or redirect.get("changed"):
            raise RuntimeError(f"action redirect unexpectedly active: {redirect}")
    log(f"Qwen V2 smoke validation passed: steps={len(steps)}")


def wait_for_smoke_with_restarts(
    run_dir: Path,
    poll_seconds: float,
    timeout_seconds: float,
    restart_attempts: int,
) -> None:
    """Restart only missing smoke lanes when an evaluator exits early."""
    log_dir = ROOT / "logs" / run_dir.name
    for attempt in range(restart_attempts + 1):
        try:
            wait_until_complete(run_dir, poll_seconds, timeout_seconds=timeout_seconds)
            return
        except RuntimeError as exc:
            if "incomplete but has no lane or watchdog screens" not in str(exc):
                raise
            if attempt >= restart_attempts:
                raise RuntimeError(
                    f"smoke remained incomplete after {restart_attempts} restart attempts"
                ) from exc
            log(f"smoke lane exited early; restarting missing work ({attempt + 1}/{restart_attempts})")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "eval_lane_watchdog.py"),
                    "--run_dir",
                    str(run_dir),
                    "--log_dir",
                    str(log_dir),
                    "--stale_seconds",
                    "300",
                    "--once",
                ],
                cwd=ROOT,
                check=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument("--reference_run_id", required=True)
    parser.add_argument("--target_model_path", type=Path, required=True)
    parser.add_argument("--target_prefix", default="qwen25vl_langonly_cfmem_v2_ckpt20997")
    parser.add_argument("--gpus", default="0,1,3")
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--smoke_timeout_seconds", type=float, default=3600.0)
    parser.add_argument("--smoke_restart_attempts", type=int, default=3)
    parser.add_argument("--smoke_base_port", type=int, default=54100)
    parser.add_argument("--full_base_port", type=int, default=54200)
    parser.add_argument("--gpu_prelaunch_memory_mib", type=int, default=4096)
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
    reference_run_dir = ROOT / "results" / args.reference_run_id
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    if not reference_run_dir.is_dir():
        raise FileNotFoundError(reference_run_dir)
    if not args.target_model_path.is_dir():
        raise FileNotFoundError(args.target_model_path)
    if not (args.target_model_path / "adapter_config.json").is_file():
        raise FileNotFoundError(args.target_model_path / "adapter_config.json")

    reference = load_json(reference_run_dir / "run_config.json")
    validate_reference_config(reference)
    target_config = dict(reference)
    target_config["model_path"] = str(args.target_model_path)
    target_config["pytorch_cuda_alloc_conf"] = "expandable_segments:True"
    target_config["qwen_merge_adapter_for_inference"] = 1

    progress, complete = run_progress(source_run_dir)
    if args.check_once:
        log(f"source progress {format_progress(progress)} complete={complete}")
        log(
            "planned Qwen config: "
            f"model={args.target_model_path} memory={reference['memory_context']} "
            f"history={reference['memory_history_size']} GPUs={args.gpus}"
        )
        return

    state_dir = args.state_dir or ROOT / "logs" / (
        f"handoff_{args.source_run_id}_to_{args.target_prefix}"
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    if state.get("stage") == "full_launched":
        log(f"full run already launched: {state.get('full_run_id')}")
        return

    if state.get("stage") == "smoke_running":
        smoke_run_dir = ROOT / "results" / str(state["smoke_run_id"])
        wait_for_smoke_with_restarts(
            smoke_run_dir,
            min(args.poll_seconds, 30.0),
            args.smoke_timeout_seconds,
            args.smoke_restart_attempts,
        )
        validate_v2_smoke(smoke_run_dir, reference, args)
        stop_run(smoke_run_dir)
        save_state(state_path, stage="smoke_passed")
    elif state.get("stage") != "smoke_passed":
        if state.get("stage") != "source_stopped":
            save_state(
                state_path,
                stage="waiting_source",
                source_run_id=args.source_run_id,
                reference_run_id=args.reference_run_id,
            )
            wait_until_complete(source_run_dir, args.poll_seconds)
            coverage = validate_global_coverage(source_run_dir)
            save_state(state_path, stage="source_complete", source_coverage=coverage)
            wait_for_source_lanes_to_exit(source_run_dir)
            stop_run(source_run_dir)
            save_state(state_path, stage="source_stopped")

        wait_for_free_gpus(
            args.gpus,
            str(reference.get("nvidia_compat_lib", "")),
            args.poll_seconds,
            args.gpu_prelaunch_memory_mib,
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        smoke_run_id = f"{args.target_prefix}_smoke_{stamp}"
        save_state(state_path, stage="launching_smoke", smoke_run_id=smoke_run_id)
        smoke_run_dir = launch_qwen_run(
            smoke_run_id,
            target_config,
            args.gpus,
            args.gpus,
            args.smoke_base_port,
            str(reference["memory_context"]),
            int(reference["memory_history_size"]),
            eval_samples_per_env=1,
            eval_max_steps=2,
            start_watchdog=False,
        )
        save_state(state_path, stage="smoke_running")
        wait_for_smoke_with_restarts(
            smoke_run_dir,
            min(args.poll_seconds, 30.0),
            args.smoke_timeout_seconds,
            args.smoke_restart_attempts,
        )
        validate_v2_smoke(smoke_run_dir, reference, args)
        stop_run(smoke_run_dir)
        save_state(state_path, stage="smoke_passed")

    wait_for_free_gpus(
        args.gpus,
        str(reference.get("nvidia_compat_lib", "")),
        args.poll_seconds,
        args.gpu_prelaunch_memory_mib,
    )
    full_run_id = f"{args.target_prefix}_full_{time.strftime('%Y%m%d_%H%M%S')}"
    save_state(state_path, stage="launching_full", full_run_id=full_run_id)
    full_run_dir = launch_qwen_run(
        full_run_id,
        target_config,
        args.gpus,
        args.gpus,
        args.full_base_port,
        str(reference["memory_context"]),
        int(reference["memory_history_size"]),
        eval_samples_per_env=None,
        eval_max_steps=int(reference["eval_max_steps"]),
        start_watchdog=True,
    )
    validate_qwen_config(
        full_run_dir,
        reference,
        args.target_model_path,
        args.gpus,
        eval_samples_per_env=None,
        eval_max_steps=int(reference["eval_max_steps"]),
    )
    time.sleep(5)
    expected_sessions = [f"{full_run_id}_lane{i}" for i in range(len(args.gpus))]
    missing = [session for session in expected_sessions if not screen_exists(session)]
    if missing:
        raise RuntimeError(f"Qwen full lane screens missing after launch: {missing}")
    if not screen_exists(f"{full_run_id}_watchdog"):
        raise RuntimeError("Qwen full watchdog screen missing after launch")
    save_state(state_path, stage="full_launched", full_run_dir=str(full_run_dir))
    log(f"handoff complete: launched {full_run_id} on GPUs {args.gpus}")


if __name__ == "__main__":
    main()

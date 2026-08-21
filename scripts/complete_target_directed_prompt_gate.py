#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from handoff_target_directed_prompt_gate import (
    CONFIG_ENV_KEYS,
    completed_keys,
    expected_keys,
    start_watchdog,
    stop_screen,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the paired prompt gate and conditionally launch full evaluation."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--handoff-state", type=Path, required=True)
    parser.add_argument("--analysis-prefix", default="phi35_target_directed_prompt_gate")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--launch-full-on-pass", action="store_true")
    parser.add_argument("--full-dataset", type=Path)
    parser.add_argument("--full-prefix", default="phi35_target_directed_v1_1_full")
    parser.add_argument("--full-gpus", default="0,1,2,3")
    parser.add_argument("--full-base-port", type=int, default=56500)
    parser.add_argument("--gpu-idle-threshold-mb", type=int, default=1000)
    parser.add_argument("--watchdog-interval-seconds", type=float, default=60.0)
    parser.add_argument("--watchdog-stale-seconds", type=float, default=300.0)
    parser.add_argument("--state-file", type=Path)
    return parser.parse_args()


def wait_for_file(path: Path, poll_seconds: float) -> dict:
    while not path.is_file():
        print(f"[{time.strftime('%F %T')}] waiting for {path}", flush=True)
        time.sleep(poll_seconds)
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_completion(run_dir: Path, poll_seconds: float) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    expected = expected_keys(config)
    while True:
        completed = completed_keys(run_dir)
        done = len(expected & completed)
        print(
            f"[{time.strftime('%F %T')}] revised progress "
            f"{done}/{len(expected)} missing={len(expected - completed)}",
            flush=True,
        )
        if expected <= completed:
            return config
        time.sleep(poll_seconds)


def gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    values = {}
    for line in result.stdout.splitlines():
        index_text, memory_text = [part.strip() for part in line.split(",", 1)]
        values[int(index_text)] = int(memory_text)
    return values


def wait_for_gpus(gpus: list[int], threshold_mb: int, poll_seconds: float) -> None:
    while True:
        memory = gpu_memory_used()
        selected = {gpu: memory.get(gpu, -1) for gpu in gpus}
        if all(0 <= used <= threshold_mb for used in selected.values()):
            print(f"[{time.strftime('%F %T')}] GPUs ready: {selected}", flush=True)
            return
        print(
            f"[{time.strftime('%F %T')}] waiting for GPUs <= {threshold_mb}MB: {selected}",
            flush=True,
        )
        time.sleep(poll_seconds)


def launch_full(
    revised_config: dict,
    dataset: Path,
    prefix: str,
    gpu_text: str,
    base_port: int,
    watchdog_interval: float,
    watchdog_stale: float,
) -> dict:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}_{stamp}"
    run_dir = ROOT / "results" / run_id
    log_dir = ROOT / "logs" / run_id
    if run_dir.exists() or log_dir.exists():
        raise FileExistsError(f"full run already exists: {run_id}")

    env = os.environ.copy()
    for config_key, env_key in CONFIG_ENV_KEYS.items():
        value = revised_config.get(config_key)
        if value is not None:
            env[env_key] = str(value)
    env.update(
        {
            "RUN_ID": run_id,
            "RUN_DIR": str(run_dir),
            "LOG_DIR": str(log_dir),
            "EVAL_DATASET": str(dataset.resolve()),
            "LANE_GPUS": gpu_text,
            "BASE_PORT": str(base_port),
            "MEMORY_CONTEXT": "uavon_pose_history_target_directed_v1_1",
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
        run_id,
        run_dir,
        log_dir,
        watchdog_interval,
        watchdog_stale,
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "gpus": gpu_text,
        "base_port": base_port,
    }


def main() -> None:
    args = parse_args()
    state_file = (
        args.state_file.resolve()
        if args.state_file
        else args.original_dir.resolve() / "target_directed_gate_completion.json"
    )
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("status") in {"gate_failed", "full_launched"}:
            print(json.dumps(state, ensure_ascii=True), flush=True)
            return

    handoff = wait_for_file(args.handoff_state.resolve(), args.poll_seconds)
    revised_dir = Path(handoff["target_run_dir"])
    revised_config = wait_for_completion(revised_dir, args.poll_seconds)
    stop_screen(f"{revised_config['run_id']}_watchdog")
    time.sleep(10)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_dir = ROOT / "results" / f"{args.analysis_prefix}_{stamp}"
    subprocess.run(
        [
            "python",
            str(ROOT / "scripts" / "analyze_target_directed_prompt_gate.py"),
            "--dataset",
            str(args.dataset.resolve()),
            "--baseline-dir",
            str(args.baseline_dir.resolve()),
            "--original-dir",
            str(args.original_dir.resolve()),
            "--revised-dir",
            str(revised_dir.resolve()),
            "--output-dir",
            str(analysis_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    gate_passed = bool(summary["gate"]["pass_for_full_eval"])

    state = {
        "status": "gate_passed" if gate_passed else "gate_failed",
        "original_run_dir": str(args.original_dir.resolve()),
        "revised_run_dir": str(revised_dir.resolve()),
        "analysis_dir": str(analysis_dir),
        "gate": summary["gate"],
        "completed_at": time.strftime("%F %T"),
    }
    if gate_passed and args.launch_full_on_pass:
        if args.full_dataset is None:
            raise ValueError("--full-dataset is required with --launch-full-on-pass")
        gpus = [int(value) for value in args.full_gpus.split(",")]
        wait_for_gpus(gpus, args.gpu_idle_threshold_mb, args.poll_seconds)
        state["full_run"] = launch_full(
            revised_config,
            args.full_dataset,
            args.full_prefix,
            args.full_gpus,
            args.full_base_port,
            args.watchdog_interval_seconds,
            args.watchdog_stale_seconds,
        )
        state["status"] = "full_launched"
        state["full_launched_at"] = time.strftime("%F %T")

    state_file.write_text(
        json.dumps(state, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

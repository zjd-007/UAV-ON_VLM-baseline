#!/usr/bin/env python3
"""Wait for a Phi training run, then launch a matching full evaluation.

The controller never stops the training job. It waits for the final checkpoint,
validates the adapter, waits for the requested GPUs to be released, and starts
the evaluator plus its lane watchdog with settings copied from a reference run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_lane_watchdog import screen_exists  # noqa: E402
from handoff_phi_checkpoint_eval import (  # noqa: E402
    config_env,
    format_gpu_status,
    gpu_status,
    load_json,
    save_state,
    start_watchdog,
    wait_for_idle_gpus,
)


PROJECTOR_KEYS = {
    "base_model.model.model.vision_embed_tokens.img_projection.0.lora_A.weight",
    "base_model.model.model.vision_embed_tokens.img_projection.0.lora_B.weight",
    "base_model.model.model.vision_embed_tokens.img_projection.2.lora_A.weight",
    "base_model.model.model.vision_embed_tokens.img_projection.2.lora_B.weight",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def read_tail(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def training_progress(log_path: Path, expected_step: int) -> tuple[int | None, bool]:
    text = read_tail(log_path)
    matches = re.findall(rf"(\d+)\s*/\s*{expected_step}\b", text)
    step = max((int(value) for value in matches), default=None)
    completed = "Training completed." in text and "train_runtime" in text
    return step, completed


def training_process_alive(pattern: str) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        text=True,
        capture_output=True,
        check=True,
    )
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != own_pid and pattern in command and "handoff_phi_training_to_eval.py" not in command:
            return True
    return False


def validate_adapter(checkpoint: Path) -> dict:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
        raise FileNotFoundError(f"incomplete checkpoint; missing {missing}")
    if weights_path.stat().st_size < 1024:
        raise RuntimeError(f"adapter weights are unexpectedly small: {weights_path}")

    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    missing_projector = sorted(PROJECTOR_KEYS - keys)
    if missing_projector:
        raise RuntimeError(f"projector LoRA keys missing from adapter: {missing_projector}")

    adapter_config = load_json(config_path)
    return {
        "checkpoint": str(checkpoint),
        "adapter_bytes": weights_path.stat().st_size,
        "adapter_key_count": len(keys),
        "projector_keys": sorted(PROJECTOR_KEYS),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
    }


def wait_for_training(args: argparse.Namespace, state_path: Path) -> dict:
    missing_process_checks = 0
    while True:
        step, completed_marker = training_progress(args.training_log, args.expected_final_step)
        active = training_process_alive(args.training_process_pattern)
        checkpoint_exists = args.final_checkpoint.is_dir()
        progress = 0.0 if step is None else 100.0 * step / args.expected_final_step
        log(
            f"training step={step or 'unknown'}/{args.expected_final_step} "
            f"({progress:.1f}%) active={active} completed_marker={completed_marker} "
            f"checkpoint={checkpoint_exists}"
        )
        save_state(
            state_path,
            stage="waiting_training",
            training_step=step,
            training_expected_steps=args.expected_final_step,
            training_process_active=active,
            training_completed_marker=completed_marker,
            final_checkpoint_exists=checkpoint_exists,
        )

        if completed_marker and checkpoint_exists:
            validation = validate_adapter(args.final_checkpoint)
            log(
                "final adapter validated: "
                f"keys={validation['adapter_key_count']} "
                f"size={validation['adapter_bytes'] / 1024**3:.2f}GiB projector_keys=4/4"
            )
            return validation

        missing_process_checks = 0 if active else missing_process_checks + 1
        if missing_process_checks >= args.failed_training_confirmations:
            raise RuntimeError(
                "training process disappeared before a completed final checkpoint was available"
            )
        if args.check_once:
            return {}
        time.sleep(args.poll_seconds)


def ports_available(base_port: int, gpus: list[int]) -> None:
    unavailable = []
    for gpu in gpus:
        port = base_port + gpu
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            unavailable.append(port)
        finally:
            sock.close()
    if unavailable:
        raise RuntimeError(f"AirSim ports are already in use: {unavailable}")


def launch_eval(args: argparse.Namespace, reference_config: dict, run_id: str) -> Path:
    ports_available(args.base_port, args.gpus)
    env = config_env(reference_config, run_id, args.final_checkpoint, args.gpus, args.base_port)
    subprocess.run(
        ["bash", "scripts/eval_phi35_uavon_parallel.sh"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    run_dir = ROOT / "results" / run_id
    log_dir = ROOT / "logs" / run_id
    config_path = run_dir / "run_config.json"
    config = load_json(config_path)
    config.update(
        {
            "reference_run_id": reference_config["run_id"],
            "reference_run_config": str(args.reference_run_dir / "run_config.json"),
            "training_log": str(args.training_log),
            "handoff_controller": Path(__file__).name,
            "handoff_created_at": time.strftime("%F %T"),
            "checkpoint_validation": validate_adapter(args.final_checkpoint),
        }
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start_watchdog(run_id, run_dir, log_dir, args)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_log", type=Path, required=True)
    parser.add_argument("--final_checkpoint", type=Path, required=True)
    parser.add_argument("--training_process_pattern", required=True)
    parser.add_argument("--reference_run_dir", type=Path, required=True)
    parser.add_argument("--expected_final_step", type=int, default=20997)
    parser.add_argument("--target_prefix", default="phi35_vlp_cfmem_v2_ckpt20997")
    parser.add_argument("--gpus", default="0,2,4,5")
    parser.add_argument("--base_port", type=int, default=57400)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--failed_training_confirmations", type=int, default=2)
    parser.add_argument("--gpu_memory_threshold_mib", type=int, default=512)
    parser.add_argument("--gpu_idle_confirmations", type=int, default=2)
    parser.add_argument("--watchdog_interval_seconds", type=int, default=60)
    parser.add_argument("--watchdog_stale_seconds", type=int, default=300)
    parser.add_argument("--conda_sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda_env", default="octmem_openvla_nomemory")
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument("--check_once", action="store_true")
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus.split(",") if value]
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError(f"expected four distinct GPUs, got {args.gpus}")
    return args


def main() -> None:
    args = parse_args()
    if not args.training_log.is_file():
        raise FileNotFoundError(args.training_log)
    if not (args.reference_run_dir / "run_config.json").is_file():
        raise FileNotFoundError(args.reference_run_dir / "run_config.json")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    if state.get("stage") == "full_launched":
        log(f"evaluation already launched: {state.get('target_run_id')}")
        return

    reference_config = load_json(args.reference_run_dir / "run_config.json")
    if args.check_once:
        wait_for_training(args, state_path)
        status = gpu_status(args.gpus, str(reference_config.get("nvidia_compat_lib", "")))
        log(f"GPU status {format_gpu_status(status)}")
        return

    try:
        validation = wait_for_training(args, state_path)
        save_state(state_path, stage="training_complete", checkpoint_validation=validation)
        wait_for_idle_gpus(
            args.gpus,
            str(reference_config.get("nvidia_compat_lib", "")),
            args.poll_seconds,
            args.gpu_memory_threshold_mib,
            args.gpu_idle_confirmations,
        )
        run_id = f"{args.target_prefix}_full_{time.strftime('%Y%m%d_%H%M%S')}"
        save_state(state_path, stage="launching_full", target_run_id=run_id)
        run_dir = launch_eval(args, reference_config, run_id)
        time.sleep(15)
        expected = [f"{run_id}_lane{index}" for index in range(len(args.gpus))]
        missing = [session for session in expected if not screen_exists(session)]
        if missing:
            raise RuntimeError(f"evaluation lane screens missing after launch: {missing}")
        if not screen_exists(f"{run_id}_watchdog"):
            raise RuntimeError("evaluation watchdog screen missing after launch")
        save_state(state_path, stage="full_launched", target_run_dir=str(run_dir))
        log(f"handoff complete: launched {run_id} on GPUs {args.gpus}")
    except Exception as exc:
        save_state(state_path, stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def parse_lanes(path: Path) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        gpu_text, scenes_text = line.split("\t")[:2]
        gpu = int(gpu_text)
        lanes[f"gpu{gpu}"] = {
            "gpu": gpu,
            "scenes": [scene for scene in scenes_text.split(",") if scene],
        }
    return lanes


def sample_key(row: dict[str, Any]) -> str:
    return f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}"


def expected_scene_counts(config: dict[str, Any]) -> Counter[str]:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_collision_filtered_depth_sft_data import load_collision_filter

    excluded, _ = load_collision_filter(
        original_dir=Path(config["original_collision_dir"]),
        repair_dir=Path(config["repair_collision_dir"]),
    )
    counts: Counter[str] = Counter()
    with Path(config["source"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_key(row) not in excluded:
                counts[str(row["scene_id"])] += 1
    return counts


def screen_exists(session: str) -> bool:
    result = run(["screen", "-ls"])
    return session in (result.stdout + result.stderr)


def process_table() -> list[tuple[int, str]]:
    result = run(["ps", "-eo", "pid=,args="])
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        try:
            rows.append((int(pid_text), command))
        except ValueError:
            continue
    return rows


def pids_matching(patterns: list[str]) -> list[int]:
    return [
        pid
        for pid, command in process_table()
        if pid != os.getpid() and all(pattern in command for pattern in patterns)
    ]


def kill_pids(pids: Iterable[int]) -> None:
    targets = sorted({pid for pid in pids if pid > 1 and pid != os.getpid()})
    if not targets:
        return
    log(f"killing pids={targets}")
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in targets:
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def build_command(config: dict[str, Any], lane: dict[str, Any], conda_sh: str, conda_env: str) -> str:
    gpu = int(lane["gpu"])
    scenes = ",".join(lane["scenes"])
    summary_suffix = str(config.get("summary_suffix", ""))
    args = [
        "python", "-u", str(config["script"]),
        "--source", str(config["source"]),
        "--aligned-root", str(config["aligned_root"]),
        "--original-collision-dir", str(config["original_collision_dir"]),
        "--repair-collision-dir", str(config["repair_collision_dir"]),
        "--output-dir", str(config["output_dir"]),
        "--scene-list", scenes,
        "--gpu", str(gpu),
        "--base-port", str(config["base_port"]),
        "--progress-interval", str(config["progress_interval"]),
        "--depth-output-size", str(config["depth_output_size"]),
        "--expected-full-depth-size", str(config["full_resolution_depth_shape"][0]),
        "--save-full-resolution-depth",
        "--reuse-collision-audited-expert-action",
        "--retry-incomplete",
        "--summary-name", f"summary_gpu{gpu}{summary_suffix}.json",
    ]
    command = " ".join(shlex.quote(value) for value in args)
    log_file = Path(config["log_dir"]) / f"gpu{gpu}.log"
    setup = [
        f"source {shlex.quote(conda_sh)}",
        f"conda activate {shlex.quote(conda_env)}",
        f"export PYTHONPATH={shlex.quote(str(ROOT / 'src') + ':' + str(ROOT / 'eval') + ':' + str(ROOT / 'scripts'))}:${{PYTHONPATH:-}}",
    ]
    compat_root = config.get("nvidia_compat_root")
    if compat_root:
        compat_root = Path(compat_root)
        setup.extend([
            f"export LD_LIBRARY_PATH={shlex.quote(str(compat_root / 'lib'))}:${{LD_LIBRARY_PATH:-}}",
            "export __EGL_VENDOR_LIBRARY_FILENAMES="
            + shlex.quote(str(compat_root / "extracted" / "10_nvidia.json")),
            "export VK_ICD_FILENAMES="
            + shlex.quote(str(compat_root / "extracted" / "nvidia_icd.json")),
        ])
    setup.extend([
        f"cd {shlex.quote(str(ROOT))}",
        f"{command} 2>&1 | tee -a {shlex.quote(str(log_file))}",
    ])
    return "\n".join(setup)


def start_lane(config: dict[str, Any], name: str, lane: dict[str, Any], args: argparse.Namespace) -> None:
    session = f"{config['run_id']}_{name}"
    command = build_command(config, lane, args.conda_sh, args.conda_env)
    result = run(["screen", "-dmS", session, "bash", "-lc", command])
    if result.returncode:
        raise RuntimeError(result.stderr)
    log(f"{name}: started screen={session}")


def stop_lane(config: dict[str, Any], name: str, lane: dict[str, Any]) -> None:
    session = f"{config['run_id']}_{name}"
    run(["screen", "-S", session, "-X", "quit"])
    time.sleep(2)
    gpu = int(lane["gpu"])
    scenes = ",".join(lane["scenes"])
    port = int(config["base_port"]) + gpu
    pids = pids_matching([str(config["script"]), "--scene-list", scenes])
    pids += pids_matching([f"settings_512_{port}.json"])
    kill_pids(pids)


def read_progress(output_dir: Path, scenes: list[str]) -> tuple[int, float | None, int]:
    complete = 0
    latest_complete = None
    failures = 0
    for scene in scenes:
        path = output_dir / "progress" / f"{scene}.json"
        if not path.is_file():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        complete += int(row.get("complete_unique", 0))
        failures += int(row.get("failed_rows", 0)) + int(row.get("incomplete", 0))
        timestamp = row.get("last_complete_unix")
        if timestamp is not None:
            timestamp = float(timestamp)
            latest_complete = timestamp if latest_complete is None else max(latest_complete, timestamp)
    return complete, latest_complete, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    parser.add_argument("--conda-sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="octmem_openvla_nomemory")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    lanes = parse_lanes(args.run_dir / "lanes.tsv")
    expected = expected_scene_counts(config)
    output_dir = Path(config["output_dir"])
    launch_time = float(config["launch_unix"])
    last_restart: dict[str, float] = {}
    log(f"watchdog start run={config['run_id']} interval={args.interval_seconds}s stale={args.stale_seconds}s")

    while True:
        all_complete = True
        now = time.time()
        for name, lane in sorted(lanes.items()):
            expected_lane = sum(expected[scene] for scene in lane["scenes"])
            complete, latest_complete, failures = read_progress(output_dir, lane["scenes"])
            if complete >= expected_lane:
                log(f"{name}: complete={complete}/{expected_lane} failures_this_process={failures}")
                continue
            all_complete = False
            session = f"{config['run_id']}_{name}"
            screen_alive = screen_exists(session)
            scenes = ",".join(lane["scenes"])
            process_alive = bool(pids_matching([str(config["script"]), "--scene-list", scenes]))
            freshness_anchor = max(
                latest_complete if latest_complete is not None else launch_time,
                last_restart.get(name, launch_time),
            )
            stale = now - freshness_anchor
            log(
                f"{name}: complete={complete}/{expected_lane} failures_this_process={failures} "
                f"screen={screen_alive} process={process_alive} stale={stale:.0f}s"
            )
            reason = None
            if not screen_alive or not process_alive:
                reason = "screen_or_process_missing"
            elif stale > args.stale_seconds:
                reason = f"no_complete_progress_{int(stale)}s"
            if reason:
                log(f"{name}: restarting reason={reason}")
                with (Path(config["log_dir"]) / f"{name}.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"\n===== watchdog restart {time.strftime('%F %T')} reason={reason} =====\n")
                stop_lane(config, name, lane)
                start_lane(config, name, lane, args)
                last_restart[name] = time.time()
        if args.once or all_complete:
            if all_complete:
                log("all lanes complete")
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

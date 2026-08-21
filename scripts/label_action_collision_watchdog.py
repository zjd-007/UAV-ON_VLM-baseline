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
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def parse_lanes(path: Path) -> dict[str, dict]:
    lanes: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lane, gpu, scenes = line.split("\t")[:3]
        lanes[lane] = {
            "gpu": int(gpu),
            "scenes": [scene for scene in scenes.split(",") if scene],
        }
    return lanes


def load_scene_counts(source: Path) -> Counter:
    counts: Counter = Counter()
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row["scene_id"])] += 1
    return counts


def count_output_rows(output_dir: Path, scenes: list[str]) -> tuple[int, int, int]:
    total = 0
    errors = 0
    new_collisions = 0
    for scene in scenes:
        path = output_dir / f"{scene}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue
                if row.get("error"):
                    errors += 1
                if row.get("new_collision_after_action"):
                    new_collisions += 1
    return total, errors, new_collisions


def latest_mtime(paths: Iterable[Path]) -> float | None:
    latest = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    return latest


def lane_activity_time(output_dir: Path, log_dir: Path, lane: str, scenes: list[str]) -> float | None:
    # Treat only result writes as progress. A broken AirSim client can keep
    # appending connection errors to the lane log forever, which must not reset
    # the stale timer.
    candidates: list[Path] = [output_dir / f"{scene}.jsonl" for scene in scenes]
    return latest_mtime(candidates)


def screen_exists(session: str) -> bool:
    res = run(["screen", "-ls"])
    return session in (res.stdout + res.stderr)


def lane_process_exists(config: dict, lane_cfg: dict) -> bool:
    scenes = ",".join(lane_cfg["scenes"])
    patterns = [
        str(config["script"]),
        "--scene-list",
        scenes,
    ]
    return bool(pids_matching(patterns))


def process_table() -> list[tuple[int, str]]:
    res = run(["ps", "-eo", "pid=,args="])
    rows: list[tuple[int, str]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        rows.append((pid, cmd))
    return rows


def pids_matching(patterns: list[str]) -> list[int]:
    self_pid = os.getpid()
    matches: list[int] = []
    for pid, cmd in process_table():
        if pid == self_pid:
            continue
        if all(pattern in cmd for pattern in patterns):
            matches.append(pid)
    return matches


def kill_pids(pids: Iterable[int]) -> None:
    pids = sorted(set(pid for pid in pids if pid > 1 and pid != os.getpid()))
    if not pids:
        return
    log(f"killing pids: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def build_lane_command(config: dict, lane: str, lane_cfg: dict, conda_sh: str, conda_env: str) -> str:
    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    script = str(config["script"])
    gpu = int(lane_cfg["gpu"])
    scenes = ",".join(lane_cfg["scenes"])

    args = [
        "python",
        "-u",
        script,
        "--source",
        str(config["source"]),
        "--aligned-root",
        str(config["aligned_root"]),
        "--scene-list",
        scenes,
        "--gpu",
        str(gpu),
        "--base-port",
        str(config["base_port"]),
        "--output-dir",
        str(output_dir),
        "--settle-frames",
        str(config.get("settle_frames", 1)),
        "--action-velocity",
        str(config.get("action_velocity", 2.0)),
        "--action-move-timeout",
        str(config.get("action_move_timeout", 5.0)),
        "--action-rotate-timeout",
        str(config.get("action_rotate_timeout", 3.0)),
    ]
    args.append("--fix-vertical-actions" if config.get("fix_vertical_actions", True) else "--no-fix-vertical-actions")
    args.append("--fix-yaw-actions" if config.get("fix_yaw_actions", True) else "--no-fix-yaw-actions")

    command = " ".join(shlex.quote(x) for x in args)
    log_file = log_dir / f"{lane}.log"
    return "\n".join(
        [
            f"cd {shlex.quote(str(ROOT))} &&",
            f"source {shlex.quote(conda_sh)} &&",
            f"conda activate {shlex.quote(conda_env)} &&",
            "export PYTHONUNBUFFERED=1 &&",
            f"{command} 2>&1 | tee -a {shlex.quote(str(log_file))}",
        ]
    )


def stop_lane(config: dict, lane: str, lane_cfg: dict) -> None:
    session = f"{config['run_id']}_{lane}"
    run(["screen", "-S", session, "-X", "quit"])
    time.sleep(2)
    gpu = int(lane_cfg["gpu"])
    port = int(config["base_port"]) + gpu
    settings_name = f"settings_512_{port}.json"
    pids = []
    pids += pids_matching([str(config["script"]), "--scene-list", ",".join(lane_cfg["scenes"])])
    pids += pids_matching([settings_name])
    kill_pids(pids)


def start_lane(config: dict, lane: str, lane_cfg: dict, conda_sh: str, conda_env: str) -> None:
    session = f"{config['run_id']}_{lane}"
    command = build_lane_command(config, lane, lane_cfg, conda_sh, conda_env)
    res = run(["screen", "-dmS", session, "bash", "-lc", command])
    if res.returncode != 0:
        raise RuntimeError(f"failed to start {lane}: {res.stderr}")
    log(f"started {lane} in screen {session}")


def monitor_once(args, config: dict, lanes: dict, scene_counts: Counter) -> None:
    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    now = time.time()
    for lane, lane_cfg in sorted(lanes.items()):
        scenes = lane_cfg["scenes"]
        expected = sum(scene_counts[scene] for scene in scenes)
        completed, errors, new_collisions = count_output_rows(output_dir, scenes)
        if completed >= expected:
            log(f"{lane}: complete {completed}/{expected} errors={errors} new_collisions={new_collisions}")
            continue

        session = f"{config['run_id']}_{lane}"
        screen_alive = screen_exists(session)
        process_alive = lane_process_exists(config, lane_cfg)
        alive = screen_alive or process_alive
        activity = lane_activity_time(output_dir, log_dir, lane, scenes)
        stale = None if activity is None else now - activity
        stale_s = "none" if stale is None else f"{stale:.0f}s"
        log(
            f"{lane}: alive={alive} screen={screen_alive} process={process_alive} completed={completed}/{expected} "
            f"errors={errors} new_collisions={new_collisions} stale={stale_s}"
        )

        should_restart = False
        reason = ""
        if not alive:
            should_restart = True
            reason = "screen_missing"
        elif stale is not None and stale > args.stale_seconds:
            should_restart = True
            reason = f"stale_{int(stale)}s"
        elif activity is None and args.restart_if_no_activity:
            should_restart = True
            reason = "no_activity"

        if should_restart and args.no_restart:
            log(f"{lane}: would restart due to {reason}, but --no-restart is set")
            continue

        if not should_restart:
            continue

        log(f"{lane}: restarting due to {reason}")
        with (log_dir / f"{lane}.log").open("a", encoding="utf-8") as f:
            f.write(f"\n===== watchdog restart at {time.strftime('%F %T')} reason={reason} =====\n")
        stop_lane(config, lane, lane_cfg)
        start_lane(config, lane, lane_cfg, args.conda_sh, args.conda_env)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    parser.add_argument("--conda-sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="octmem_openvla_nomemory")
    parser.add_argument("--restart-if-no-activity", action="store_true")
    parser.add_argument("--no-restart", action="store_true", help="Only report progress; do not restart lanes.")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    lanes = parse_lanes(args.run_dir / "lanes.tsv")
    scene_counts = load_scene_counts(Path(config["source"]))
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    log(
        f"watchdog start run={config['run_id']} interval={args.interval_seconds}s "
        f"stale={args.stale_seconds}s lanes={list(lanes)}"
    )
    while True:
        monitor_once(args, config, lanes, scene_counts)
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

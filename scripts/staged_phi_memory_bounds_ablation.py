#!/usr/bin/env python3
"""Stage a no-SearchBounds Phi evaluation around a busy GPU0."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from eval_lane_watchdog import read_completed_rows, screen_exists, start_lane


ROOT = Path(__file__).resolve().parents[1]
LANES = {
    "lane0": {
        "gpu": 0,
        "scenes": ["NYC_test", "WinterTown_test", "UrbanJapan_test", "WesternTown_test"],
    },
    "lane1": {
        "gpu": 2,
        "scenes": ["Slum_test", "Barnyard_test", "BrushifyUrban_test", "CabinLake_test", "DownTown_test"],
    },
    "lane2": {
        "gpu": 3,
        "scenes": ["CityStreet_test", "BrushifyRoad_test", "CityPark_test", "ModularNeighborhood_test", "Venice_test"],
    },
}


def log(message: str, log_file: Path) -> None:
    line = f"[{time.strftime('%F %T')}] {message}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def stop_screen(session: str) -> None:
    run(["screen", "-S", session, "-X", "quit"])


def write_lanes(path: Path, lane_names: list[str]) -> None:
    lines = []
    for lane in lane_names:
        cfg = LANES[lane]
        lines.append(f"{lane}\t{cfg['gpu']}\t{','.join(cfg['scenes'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sessions(path: Path, run_id: str, run_dir: Path, log_dir: Path) -> None:
    lines = []
    for lane, cfg in LANES.items():
        port = int(json.loads((run_dir / "run_config.json").read_text())["base_port"]) + cfg["gpu"]
        lines.append(
            f"{run_id}_{lane}\tgpu={cfg['gpu']}\tport={port}"
            f"\tout={run_dir / lane}\tlog={log_dir / (lane + '.log')}"
            f"\tscenes={','.join(cfg['scenes'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scene_counts(dataset: Path) -> Counter:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    return Counter(row["map_name"] for row in rows)


def lane_progress(run_dir: Path, lane: str, scenes: list[str], counts: Counter) -> tuple[int, int]:
    rows = read_completed_rows(run_dir / lane)
    scene_set = set(scenes)
    completed = sum(1 for row in rows if row.get("map_name") in scene_set)
    expected = sum(counts[scene] for scene in scenes)
    return completed, expected


def source_complete(source_dir: Path, counts: Counter) -> tuple[bool, list[str]]:
    lines = []
    complete = True
    for raw in (source_dir / "lanes.tsv").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        lane, _, scene_text, *_ = raw.split("\t")
        scenes = [scene for scene in scene_text.split(",") if scene]
        completed, expected = lane_progress(source_dir, lane, scenes, counts)
        lines.append(f"{lane}={completed}/{expected}")
        complete &= completed >= expected
    return complete, lines


def start_watchdog(
    run_id: str,
    run_dir: Path,
    log_dir: Path,
    conda_sh: str,
    conda_env: str,
    interval_seconds: int,
    stale_seconds: int,
) -> None:
    session = f"{run_id}_watchdog"
    if screen_exists(session):
        return
    command = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(conda_env)} && "
        "export PYTHONUNBUFFERED=1 && "
        "python -u scripts/eval_lane_watchdog.py "
        f"--run_dir {shlex.quote(str(run_dir))} "
        f"--log_dir {shlex.quote(str(log_dir))} "
        f"--interval_seconds {interval_seconds} "
        f"--stale_seconds {stale_seconds} "
        "--restart_if_no_activity "
        f"2>&1 | tee -a {shlex.quote(str(log_dir / 'watchdog.log'))}"
    )
    result = run(["screen", "-dmS", session, "bash", "-lc", command])
    if result.returncode != 0:
        raise RuntimeError(f"failed to start watchdog: {result.stderr}")


def ensure_target_run(args, log_file: Path) -> tuple[dict, Path, Path]:
    source_dir = ROOT / "results" / args.source_run_id
    source_config = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    run_dir = ROOT / "results" / args.run_id
    log_dir = ROOT / "logs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config.get("memory_include_search_bounds", 1)) != 0:
            raise RuntimeError(f"existing target run is not a no-bounds run: {config_path}")
    else:
        config = dict(source_config)
        config.update(
            {
                "run_id": args.run_id,
                "parent_run_id": args.source_run_id,
                "ablation": "memory_search_bounds_removed",
                "base_port": args.base_port,
                "lane_count": 3,
                "memory_include_search_bounds": 0,
            }
        )
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_lanes(run_dir / "lanes_full.tsv", list(LANES))
        write_lanes(run_dir / "lanes.tsv", ["lane1", "lane2"])
        write_sessions(run_dir / "sessions.tsv", args.run_id, run_dir, log_dir)
        log(f"created target run {args.run_id}", log_file)
    return config, run_dir, log_dir


def start_initial_lanes(args, config: dict, run_dir: Path, log_dir: Path, log_file: Path) -> None:
    counts = scene_counts(Path(config["eval_dataset"]))
    for lane in ("lane1", "lane2"):
        completed, expected = lane_progress(run_dir, lane, LANES[lane]["scenes"], counts)
        session = f"{args.run_id}_{lane}"
        if completed >= expected:
            log(f"{lane} already complete {completed}/{expected}", log_file)
            continue
        if screen_exists(session):
            log(f"{lane} already alive {completed}/{expected}", log_file)
            continue
        start_lane(
            lane,
            LANES[lane],
            config,
            run_dir,
            log_dir,
            args.conda_sh,
            args.conda_env,
        )
        log(f"started {lane} on GPU{LANES[lane]['gpu']} {completed}/{expected}", log_file)
    start_watchdog(
        args.run_id,
        run_dir,
        log_dir,
        args.conda_sh,
        args.conda_env,
        args.watchdog_interval,
        args.stale_seconds,
    )


def activate_lane0(args, run_dir: Path, log_dir: Path, log_file: Path) -> None:
    write_lanes(run_dir / "lanes.tsv", list(LANES))
    watchdog = f"{args.run_id}_watchdog"
    stop_screen(watchdog)
    time.sleep(3)
    start_watchdog(
        args.run_id,
        run_dir,
        log_dir,
        args.conda_sh,
        args.conda_env,
        args.watchdog_interval,
        args.stale_seconds,
    )
    log("activated lane0 on GPU0 and reloaded target watchdog", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--base_port", type=int, default=49700)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--watchdog_interval", type=int, default=60)
    parser.add_argument("--stale_seconds", type=int, default=300)
    parser.add_argument("--conda_sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda_env", default="octmem_openvla_nomemory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ROOT / "results" / args.source_run_id
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    control_dir = ROOT / "logs" / f"{args.run_id}_handoff"
    control_dir.mkdir(parents=True, exist_ok=True)
    log_file = control_dir / "handoff.log"
    state_file = control_dir / "state.json"

    config, run_dir, log_dir = ensure_target_run(args, log_file)
    start_initial_lanes(args, config, run_dir, log_dir, log_file)
    state_file.write_text(
        json.dumps({"stage": "lane1_lane2_running", "run_id": args.run_id}, indent=2) + "\n",
        encoding="utf-8",
    )

    counts = scene_counts(Path(config["eval_dataset"]))
    while True:
        complete, progress = source_complete(source_dir, counts)
        log("source progress " + " ".join(progress), log_file)
        if complete:
            source_assist = f"{args.source_run_id}_lane1_assist"
            if screen_exists(source_assist):
                log("source results complete; waiting for GPU0 evaluator cleanup", log_file)
                time.sleep(args.poll_seconds)
                continue
            stop_screen(f"{args.source_run_id}_watchdog")
            activate_lane0(args, run_dir, log_dir, log_file)
            break
        time.sleep(args.poll_seconds)

    deadline = time.time() + 180
    lane0_session = f"{args.run_id}_lane0"
    while time.time() < deadline and not screen_exists(lane0_session):
        time.sleep(5)
    if not screen_exists(lane0_session):
        raise RuntimeError("lane0 was not started by the target watchdog")

    state_file.write_text(
        json.dumps({"stage": "all_lanes_active", "run_id": args.run_id}, indent=2) + "\n",
        encoding="utf-8",
    )
    log("handoff complete; lane0, lane1, and lane2 are under the target watchdog", log_file)


if __name__ == "__main__":
    main()

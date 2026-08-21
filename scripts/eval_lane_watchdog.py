#!/usr/bin/env python3
"""Watch and restart stalled UAV-ON eval lanes.

The eval script writes one JSON per completed episode. This watchdog treats the
temp JSONs as the source of truth, archives stale in-progress images, and
restarts only the affected lane.
"""

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
REPO_ROOT = ROOT.parent


def now() -> float:
    return time.time()


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def parse_lanes(path: Path) -> dict[str, dict]:
    lanes: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) not in {3, 4}:
            raise ValueError(f"expected 3 or 4 tab-separated columns in {path}, got: {line!r}")
        lane, gpu, scenes = parts[:3]
        eval_dataset = parts[3] if len(parts) == 4 and parts[3] else None
        lanes[lane] = {
            "gpu": int(gpu),
            "scenes": [scene for scene in scenes.split(",") if scene],
            "eval_dataset": eval_dataset,
        }
    return lanes


def load_scene_counts(dataset: Path) -> Counter:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    return Counter(row["map_name"] for row in rows)


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return (p for p in path.rglob("*") if p.is_file())


def latest_mtime(paths: Iterable[Path]) -> float | None:
    latest = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    return latest


def read_completed_rows(lane_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((lane_dir / "temp").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"warning: failed to read {path}: {exc}")
            continue
        row["_json_path"] = str(path)
        rows.append(row)
    return rows


def screen_exists(session: str) -> bool:
    res = run(["screen", "-ls"])
    return session in (res.stdout + res.stderr)


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


def completed_count_for_scenes(rows: list[dict], scenes: list[str]) -> int:
    scene_set = set(scenes)
    return sum(1 for row in rows if row.get("map_name") in scene_set)


def expected_count_for_scenes(scene_counts: Counter, scenes: list[str]) -> int:
    return sum(scene_counts[scene] for scene in scenes)


def lane_scene_counts(lane_cfg: dict, config: dict, cache: dict[str, Counter]) -> Counter:
    dataset = str(lane_cfg.get("eval_dataset") or config["eval_dataset"])
    if dataset not in cache:
        cache[dataset] = load_scene_counts(Path(dataset))
    return cache[dataset]


def archive_incomplete_images(lane_dir: Path, completed_rows: list[dict], stamp: str) -> list[Path]:
    completed = {(row.get("map_name"), str(row.get("episode_id"))) for row in completed_rows}
    image_root = lane_dir / "images"
    archived: list[Path] = []
    if not image_root.exists():
        return archived

    for scene_dir in sorted(image_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
            continue
        for ep_dir in sorted(scene_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            key = (scene_dir.name, ep_dir.name)
            if key in completed:
                continue
            if not any(ep_dir.iterdir()):
                continue
            archive_root = image_root / "_incomplete_archives" / scene_dir.name
            archive_root.mkdir(parents=True, exist_ok=True)
            dst = archive_root / f"{ep_dir.name}_stuck_{stamp}"
            suffix = 1
            while dst.exists():
                dst = archive_root / f"{ep_dir.name}_stuck_{stamp}_{suffix}"
                suffix += 1
            ep_dir.rename(dst)
            archived.append(dst)
    return archived


def build_eval_command(
    config: dict,
    lane: str,
    gpu: int,
    scenes: list[str],
    lane_eval_dataset: str | None,
    run_dir: Path,
    log_dir: Path,
    conda_sh: str,
    conda_env: str,
) -> str:
    out_dir = run_dir / lane
    log_file = log_dir / f"{lane}.log"
    port = int(config["base_port"]) + int(gpu)
    simulator_gpu = int(config.get("simulator_gpu_by_lane", {}).get(lane, gpu))
    eval_dataset = lane_eval_dataset or str(config["eval_dataset"])

    evaluator_script = str(config.get("evaluator_script", "eval/eval_phi35_uavon.py"))
    args = [
        "python",
        "-u",
        evaluator_script,
        "--model_path",
        str(config["model_path"]),
        "--eval_dataset",
        str(eval_dataset),
    ]
    eval_samples_per_env = config.get("eval_samples_per_env")
    if eval_samples_per_env is not None:
        args.extend(["--eval_samples_per_env", str(eval_samples_per_env)])
    args.extend(
        [
        "--output_foler",
        str(out_dir),
        "--eval_max_steps",
        str(config["eval_max_steps"]),
        "--airsim_default_port",
        str(port),
        "--simulator_gpu",
        str(simulator_gpu),
        "--device",
        "cuda:0",
        "--max_new_tokens",
        str(config["max_new_tokens"]),
        "--inference_mode",
        str(config["inference_mode"]),
        "--depth_avoidance",
        str(config.get("depth_avoidance", "uavon_single_view_prompt")),
        "--depth_grid_size",
        str(config.get("depth_grid_size", 3)),
        "--depth_max_meters",
        str(config.get("depth_max_meters", 100.0)),
        "--depth_forward_threshold",
        str(config.get("depth_forward_threshold", 4.0)),
        "--depth_turn_threshold",
        str(config.get("depth_turn_threshold", 1.5)),
        "--depth_descend_threshold",
        str(config.get("depth_descend_threshold", 6.0)),
        "--depth_ascend_top_threshold",
        str(config.get("depth_ascend_top_threshold", 8.0)),
        "--action_redirect",
        str(config.get("action_redirect", "none")),
        "--action_redirect_search_radius",
        str(config.get("action_redirect_search_radius", 50.0)),
        "--action_redirect_near_obstacle_threshold",
        str(config.get("action_redirect_near_obstacle_threshold", 2.0)),
        "--memory_context",
        str(config.get("memory_context", "uavon_pose_history")),
        "--memory_history_size",
        str(config.get("memory_history_size", 5)),
        "--memory_search_radius",
        str(config.get("memory_search_radius", 50.0)),
        "--memory_include_search_bounds",
        str(config.get("memory_include_search_bounds", 0)),
        "--memory_pose_yaw_unit",
        str(config.get("memory_pose_yaw_unit", "radians")),
        "--scene_list",
        ",".join(scenes),
        "--skip_kill_env_process",
        "--pose_wait_timeout",
        str(config["pose_wait_timeout"]),
        "--pose_wait_position_tol",
        str(config["pose_wait_position_tol"]),
        "--pose_wait_yaw_tol",
        str(config["pose_wait_yaw_tol"]),
        "--pose_wait_poll_interval",
        str(config["pose_wait_poll_interval"]),
        "--render_settle_seconds",
        str(config["render_settle_seconds"]),
        "--action_execution_mode",
        str(config["action_execution_mode"]),
        "--action_sim_frames",
        str(config["action_sim_frames"]),
        "--action_velocity",
        str(config["action_velocity"]),
        "--action_move_timeout",
        str(config["action_move_timeout"]),
        "--action_rotate_timeout",
        str(config["action_rotate_timeout"]),
        "--level_settle_frames",
        str(config["level_settle_frames"]),
        "--initial_pose_retries",
        str(config["initial_pose_retries"]),
        "--initial_pose_settle_frames",
        str(config["initial_pose_settle_frames"]),
        ]
    )
    if int(config.get("fix_vertical_actions", 0)):
        args.append("--fix_vertical_actions")
    if int(config.get("fix_yaw_actions", 0)):
        args.append("--fix_yaw_actions")
    if int(config.get("level_after_action", 0)):
        args.append("--level_after_action")
    if int(config.get("zero_kinematics_reset", 1)):
        args.append("--zero_kinematics_reset")
    else:
        args.append("--no-zero_kinematics_reset")
    if int(config.get("client_reset_per_episode", 0)):
        args.append("--client_reset_per_episode")
    if int(config.get("save_step_images", 0)):
        args.extend(
            [
                "--save_step_images",
                "--image_save_stride",
                str(config["image_save_stride"]),
                "--image_format",
                "jpg",
                "--image_quality",
                str(config["image_quality"]),
            ]
        )

    command = " ".join(shlex.quote(arg) for arg in args)
    setup = [
        f"cd {shlex.quote(str(ROOT))} &&",
        f"source {shlex.quote(conda_sh)} &&",
        f"conda activate {shlex.quote(conda_env)} &&",
    ]
    nvidia_compat_lib = str(config.get("nvidia_compat_lib", "")).strip()
    if nvidia_compat_lib:
        setup.append(
            f"export LD_LIBRARY_PATH={shlex.quote(nvidia_compat_lib)}:${{LD_LIBRARY_PATH:-}} &&"
        )
    setup.extend(
        [
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} &&",
            "export PYTHONUNBUFFERED=1 &&",
            "export TOKENIZERS_PARALLELISM=false &&",
            (
                "export PYTORCH_CUDA_ALLOC_CONF="
                f"{shlex.quote(str(config.get('pytorch_cuda_alloc_conf', 'expandable_segments:True')))} &&"
            ),
            (
                "export QWEN_MERGE_ADAPTER_FOR_INFERENCE="
                f"{shlex.quote(str(config.get('qwen_merge_adapter_for_inference', 0)))} &&"
            ),
            (
                "export PYTHONPATH="
                f"{shlex.quote(str(ROOT / 'src') + ':' + str(ROOT) + ':' + str(ROOT / 'eval'))}:"
                "${PYTHONPATH:-} &&"
            ),
            f"{command} 2>&1 | tee -a {shlex.quote(str(log_file))}",
        ]
    )
    return "\n".join(setup)


def stop_lane(lane: str, gpu: int, run_dir: Path, config: dict) -> None:
    session = f"{config['run_id']}_{lane}"
    run(["screen", "-S", session, "-X", "quit"])
    time.sleep(3)
    port = int(config["base_port"]) + int(gpu)
    lane_dir = run_dir / lane
    settings_name = f"settings_512_{port}.json"
    pids = []
    evaluator_script = str(config.get("evaluator_script", "eval/eval_phi35_uavon.py"))
    pids += pids_matching([evaluator_script, "--output_foler", str(lane_dir)])
    pids += pids_matching([settings_name])
    pids += pids_matching([f":{port}"])
    kill_pids(pids)


def start_lane(
    lane: str,
    lane_cfg: dict,
    config: dict,
    run_dir: Path,
    log_dir: Path,
    conda_sh: str,
    conda_env: str,
) -> None:
    session = f"{config['run_id']}_{lane}"
    command = build_eval_command(
        config,
        lane,
        int(lane_cfg["gpu"]),
        lane_cfg["scenes"],
        lane_cfg.get("eval_dataset"),
        run_dir,
        log_dir,
        conda_sh,
        conda_env,
    )
    res = run(["screen", "-dmS", session, "bash", "-lc", command])
    if res.returncode != 0:
        raise RuntimeError(f"failed to start {lane}: {res.stderr}")
    log(f"started {lane} in screen {session}")


def lane_activity_time(lane_dir: Path, log_file: Path) -> float | None:
    candidates: list[float] = []
    if lane_dir.exists():
        candidates.append(lane_dir.stat().st_mtime)
    if log_file.exists():
        candidates.append(log_file.stat().st_mtime)
    for sub in ["temp", "images"]:
        mtime = latest_mtime(iter_files(lane_dir / sub))
        if mtime is not None:
            candidates.append(mtime)
    return max(candidates) if candidates else None


def lane_output_time(lane_dir: Path) -> float | None:
    candidates: list[float] = []
    for sub in ["temp", "images"]:
        mtime = latest_mtime(iter_files(lane_dir / sub))
        if mtime is not None:
            candidates.append(mtime)
    return max(candidates) if candidates else None


def restart_reason(
    *,
    alive: bool,
    stale: float | None,
    stale_seconds: float,
    restart_if_no_activity: bool,
) -> str | None:
    """Return a restart reason based on aggregate lane activity.

    A result JSON is only written after an episode finishes, so output age can
    legitimately exceed the timeout while a long episode is still producing
    fresh log or image activity.
    """
    if not alive:
        return "screen_missing"
    if stale is not None and stale > stale_seconds:
        return f"stale_{int(stale)}s"
    if stale is None and restart_if_no_activity:
        return "no_activity"
    return None


def monitor_once(
    args,
    lanes: dict,
    config: dict,
    scene_count_cache: dict[str, Counter],
) -> None:
    run_dir = args.run_dir
    log_dir = args.log_dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for lane, lane_cfg in sorted(lanes.items()):
        lane_dir = run_dir / lane
        log_file = log_dir / f"{lane}.log"
        rows = read_completed_rows(lane_dir)
        completed = completed_count_for_scenes(rows, lane_cfg["scenes"])
        scene_counts = lane_scene_counts(lane_cfg, config, scene_count_cache)
        expected = expected_count_for_scenes(scene_counts, lane_cfg["scenes"])
        eval_samples_per_env = config.get("eval_samples_per_env")
        if eval_samples_per_env is not None:
            sample_limit = int(eval_samples_per_env)
            expected = sum(min(scene_counts[scene], sample_limit) for scene in lane_cfg["scenes"])
        current_time = now()

        if completed >= expected:
            log(f"{lane}: complete {completed}/{expected}")
            continue

        session = f"{config['run_id']}_{lane}"
        alive = screen_exists(session)
        activity = lane_activity_time(lane_dir, log_file)
        stale = None if activity is None else current_time - activity
        stale_s = "none" if stale is None else f"{stale:.0f}s"
        output_activity = lane_output_time(lane_dir)
        output_idle = None if output_activity is None else current_time - output_activity
        output_idle_s = "none" if output_idle is None else f"{output_idle:.0f}s"
        log(
            f"{lane}: alive={alive} completed={completed}/{expected} "
            f"stale={stale_s} output_idle={output_idle_s}"
        )

        reason = restart_reason(
            alive=alive,
            stale=stale,
            stale_seconds=args.stale_seconds,
            restart_if_no_activity=args.restart_if_no_activity,
        )
        if reason is None:
            continue

        log(f"{lane}: restarting due to {reason}")
        stop_lane(lane, int(lane_cfg["gpu"]), run_dir, config)
        rows = read_completed_rows(lane_dir)
        archived = archive_incomplete_images(lane_dir, rows, stamp)
        if archived:
            log(f"{lane}: archived {len(archived)} incomplete dirs: {[str(p) for p in archived]}")
        with (log_dir / f"{lane}.log").open("a", encoding="utf-8") as f:
            f.write(f"\n===== watchdog restart at {time.strftime('%F %T')} reason={reason} =====\n")
            if archived:
                f.write("archived=" + json.dumps([str(p) for p in archived], ensure_ascii=False) + "\n")
        completed_by_scene = Counter(row.get("map_name") for row in rows)
        sample_limit = config.get("eval_samples_per_env")
        remaining_scenes = []
        for scene in lane_cfg["scenes"]:
            scene_expected = scene_counts[scene]
            if sample_limit is not None:
                scene_expected = min(scene_expected, int(sample_limit))
            if completed_by_scene[scene] < scene_expected:
                remaining_scenes.append(scene)
        restart_cfg = dict(lane_cfg)
        restart_cfg["scenes"] = remaining_scenes
        start_lane(lane, restart_cfg, config, run_dir, log_dir, args.conda_sh, args.conda_env)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=ROOT / "results" / "phi35_llamafactory_v4_apex_join_eval")
    parser.add_argument("--log_dir", type=Path, default=ROOT / "logs" / "phi35_llamafactory_v4_apex_join_eval")
    parser.add_argument("--interval_seconds", type=float, default=120.0)
    parser.add_argument("--stale_seconds", type=float, default=1200.0)
    parser.add_argument("--conda_sh", default="/data/zhujd/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda_env", default="octmem_openvla_nomemory")
    parser.add_argument("--restart_if_no_activity", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    lanes = parse_lanes(args.run_dir / "lanes.tsv")
    scene_count_cache: dict[str, Counter] = {}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"watchdog start run={config['run_id']} interval={args.interval_seconds}s "
        f"stale={args.stale_seconds}s lanes={list(lanes)}"
    )
    while True:
        monitor_once(args, lanes, config, scene_count_cache)
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

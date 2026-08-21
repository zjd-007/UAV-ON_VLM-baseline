#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "results" / "phi35_cfmem_v2_ckpt20997_full_20260802_215558"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phi35_uavon_lora_r256_depth_grid_collision_filtered_20260719_001301"
    / "checkpoint-20997"
)
TURN_ACTIONS = {"turn left 30 degree", "turn right 30 degree"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved loop-recovery observations with the V2 and V3 memory prompts "
            "without starting AirSim."
        )
    )
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--forward_clearance", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--rerun_v2",
        action="store_true",
        help="Also replay the exact logged V2 prompt as a saved-image control.",
    )
    parser.add_argument(
        "--append_forward_safe_directive",
        action="store_true",
        help=(
            "Append an oracle directive for this diagnostic set, whose selected observations "
            "already satisfy the configured forward-clearance test."
        ),
    )
    return parser.parse_args()


def load_all_episodes(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_forward_clear(grid: Any, threshold: float) -> bool:
    try:
        return float(grid[1][1]) > threshold and float(grid[2][1]) > threshold
    except (IndexError, TypeError, ValueError):
        return False


def resolve_image(result_dir: Path, relative_path: str) -> Path:
    matches = sorted(result_dir.glob(f"lane*/{relative_path}"))
    if not matches:
        raise FileNotFoundError(f"Could not resolve {relative_path!r} below {result_dir}")
    return matches[0]


def collect_candidates(
    result_dir: Path,
    sample_count: int,
    forward_clearance: float,
    build_episodic_memory,
) -> list[dict[str, Any]]:
    source_path = result_dir / "all_episodes.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing merged episode file: {source_path}")

    by_scene: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for episode in load_all_episodes(source_path):
        steps = episode.get("step_records") or []
        if not steps:
            continue
        memory = build_episodic_memory(
            "uavon_pose_history_v3",
            start_pose=steps[0]["pose_before"],
            history_size=5,
            pose_yaw_unit="radians",
            include_search_bounds=False,
            max_steps=100,
        )
        for step in steps:
            v3_context = memory.build_context()
            logged_memory = step.get("memory_context") or {}
            logged_summary = logged_memory.get("summary") or {}
            depth = step.get("depth_avoidance") or {}
            grid = depth.get("depth_grid")
            if (
                logged_summary.get("recent_turn_state") == "full_rotation_loop"
                and is_forward_clear(grid, forward_clearance)
            ):
                recent_actions = logged_summary.get("recent_actions") or []
                previous_action = recent_actions[-1] if recent_actions else None
                relative_image = step.get("image_path")
                if not relative_image:
                    break
                try:
                    image_path = resolve_image(result_dir, relative_image)
                except FileNotFoundError:
                    break
                by_scene[episode["scene_key"]].append(
                    {
                        "scene_key": episode["scene_key"],
                        "episode_id": str(episode["episode_id"]),
                        "step": int(step["step"]),
                        "target_description": episode.get("description", ""),
                        "image_path": str(image_path),
                        "depth_grid": grid,
                        "logged_action": step.get("parsed_command"),
                        "previous_action": previous_action,
                        "logged_v2_prompt": logged_memory.get("prompt_text", ""),
                        "v3_prompt": v3_context.prompt_text,
                        "v3_summary": v3_context.summary,
                    }
                )
                break
            memory.update(step["pose_before"], step["pose_after"], step["parsed_command"])

    selected: list[dict[str, Any]] = []
    scene_names = sorted(by_scene)
    while len(selected) < sample_count and scene_names:
        remaining = []
        for scene in scene_names:
            if by_scene[scene] and len(selected) < sample_count:
                selected.append(by_scene[scene].popleft())
            if by_scene[scene]:
                remaining.append(scene)
        scene_names = remaining
    return selected


def action_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    actions = Counter(row.get(key) or "unmatched" for row in rows)
    total = len(rows)
    repeated = sum(
        row.get(key) == row.get("previous_action") and row.get(key) in TURN_ACTIONS
        for row in rows
    )
    turns = sum(row.get(key) in TURN_ACTIONS for row in rows)
    forwards = sum(row.get(key) == "forward 3m" for row in rows)
    return {
        "count": total,
        "action_distribution": dict(actions),
        "repeat_previous_turn_count": repeated,
        "repeat_previous_turn_rate": repeated / total if total else 0.0,
        "turn_count": turns,
        "turn_rate": turns / total if total else 0.0,
        "forward_count": forwards,
        "forward_rate": forwards / total if total else 0.0,
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    rerun_v2: bool,
    append_forward_safe_directive: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "counterfactual_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "sample_count": len(rows),
        "append_forward_safe_directive": append_forward_safe_directive,
        "logged_v2": action_stats(rows, "logged_action"),
        "v3": action_stats(rows, "v3_action"),
        "changed_count": sum(row["v3_action"] != row["logged_action"] for row in rows),
        "changed_rate": (
            sum(row["v3_action"] != row["logged_action"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "v3_parse_matched_count": sum(row["v3_parse_matched"] for row in rows),
    }
    if rerun_v2:
        summary["replayed_v2"] = action_stats(rows, "replayed_v2_action")
        summary["replayed_v2_matches_logged_count"] = sum(
            row["replayed_v2_action"] == row["logged_action"] for row in rows
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    logged = summary["logged_v2"]
    v3 = summary["v3"]
    lines = [
        "# V3 Memory Prompt Counterfactual",
        "",
        f"- samples: {len(rows)}",
        f"- logged V2 repeat-previous-turn: {logged['repeat_previous_turn_rate']:.1%}",
        f"- V3 repeat-previous-turn: {v3['repeat_previous_turn_rate']:.1%}",
        f"- logged V2 any-turn: {logged['turn_rate']:.1%}",
        f"- V3 any-turn: {v3['turn_rate']:.1%}",
        f"- logged V2 forward: {logged['forward_rate']:.1%}",
        f"- V3 forward: {v3['forward_rate']:.1%}",
        f"- action changed: {summary['changed_rate']:.1%}",
        f"- V3 parse matched: {summary['v3_parse_matched_count']}/{len(rows)}",
    ]
    if rerun_v2:
        lines.append(
            "- replayed V2 matches logged action: "
            f"{summary['replayed_v2_matches_logged_count']}/{len(rows)}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from eval_phi35_uavon import (  # noqa: PLC0415
        generate_action_text,
        load_model_and_processor,
        patch_transformers_cache_compat,
    )
    from vlm_baseline.actions import parse_action_text  # noqa: PLC0415
    from vlm_baseline.depth_avoidance import build_depth_avoidance  # noqa: PLC0415
    from vlm_baseline.memory_context import build_episodic_memory  # noqa: PLC0415

    candidates = collect_candidates(
        args.result_dir,
        args.sample_count,
        args.forward_clearance,
        build_episodic_memory,
    )
    if len(candidates) < args.sample_count:
        raise RuntimeError(
            f"Only found {len(candidates)} eligible observations, requested {args.sample_count}"
        )

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"memory_v3_counterfactual_{timestamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    patch_transformers_cache_compat()
    model, processor = load_model_and_processor(
        str(args.model_path),
        base_model_path=None,
        device=args.device,
    )
    depth_module = build_depth_avoidance("uavon_single_view_prompt")

    for index, row in enumerate(candidates, start=1):
        image = Image.open(row["image_path"]).convert("RGB")
        depth_prompt = depth_module.format_prompt(np.asarray(row["depth_grid"], dtype=np.float32))
        if args.rerun_v2:
            raw_v2 = generate_action_text(
                model,
                processor,
                image,
                row["target_description"],
                args.device,
                args.max_new_tokens,
                depth_context=depth_prompt,
                memory_context=row["logged_v2_prompt"],
            )
            parsed_v2 = parse_action_text(raw_v2)
            row["replayed_v2_raw_text"] = raw_v2
            row["replayed_v2_action"] = parsed_v2.command
            row["replayed_v2_parse_matched"] = parsed_v2.matched

        v3_prompt = row["v3_prompt"]
        if args.append_forward_safe_directive:
            observation_directive = (
                "CurrentLoopRecoveryDecision:\n"
                "ForwardPathSafe = true\n"
                "Unless the target is clearly confirmed in RGB, RequiredNextCommand = forward 3m.\n"
                "Do not choose a turning command for this decision."
            )
            v3_prompt = f"{v3_prompt}\n\n{observation_directive}"
            row["v3_observation_directive"] = observation_directive

        raw_v3 = generate_action_text(
            model,
            processor,
            image,
            row["target_description"],
            args.device,
            args.max_new_tokens,
            depth_context=depth_prompt,
            memory_context=v3_prompt,
        )
        parsed_v3 = parse_action_text(raw_v3)
        row["v3_raw_text"] = raw_v3
        row["v3_action"] = parsed_v3.command
        row["v3_parse_matched"] = parsed_v3.matched
        print(
            f"[{index}/{len(candidates)}] {row['scene_key']}/{row['episode_id']} "
            f"step={row['step']} logged={row['logged_action']!r} v3={row['v3_action']!r}",
            flush=True,
        )

    write_outputs(
        output_dir,
        candidates,
        args.rerun_v2,
        args.append_forward_safe_directive,
    )
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()

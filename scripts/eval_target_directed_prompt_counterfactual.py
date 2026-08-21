#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vlm_baseline.memory_context import TARGET_DIRECTED_V1_POLICY  # noqa: E402


DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "results" / "phi35_cfmem_v2_ckpt20997_full_20260802_215558"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phi35_uavon_lora_r256_depth_grid_collision_filtered_20260719_001301"
    / "checkpoint-20997"
)
CATEGORIES = (
    "successful_stop",
    "false_stop_far",
    "osr_only_near",
    "non_osr_loop",
)
TURN_ACTIONS = {"turn left 30 degree", "turn right 30 degree"}

TARGET_DIRECTED_POLICY = TARGET_DIRECTED_V1_POLICY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the logged V2 memory policy with a target-directed policy on "
            "identical saved RGB, DepthGrid, and trajectory memory."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "phi35", "qwen25_vl"),
        default="auto",
        help="Inference backend. Auto detects Qwen adapters from adapter_config.json.",
    )
    parser.add_argument(
        "--qwen-merge-adapter-for-inference",
        action="store_true",
        help="Merge the Qwen LoRA adapter before inference; disabled by default.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def resolve_backend(model_path: Path, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend

    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.is_file():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = str(adapter_config.get("base_model_name_or_path", "")).lower()
        if "qwen" in base_model:
            return "qwen25_vl"

    model_config_path = model_path / "config.json"
    if model_config_path.is_file():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        model_type = str(model_config.get("model_type", "")).lower()
        if "qwen" in model_type:
            return "qwen25_vl"
    return "phi35"


def load_all_episodes(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_image(result_dir: Path, relative_path: str) -> Path:
    matches = sorted(result_dir.glob(f"lane*/{relative_path}"))
    if not matches:
        raise FileNotFoundError(f"Could not resolve {relative_path!r} below {result_dir}")
    return matches[0]


def is_forward_clear(grid: Any, threshold: float = 4.0) -> bool:
    try:
        return float(grid[1][1]) > threshold and float(grid[2][1]) > threshold
    except (IndexError, TypeError, ValueError):
        return False


def replace_memory_policy(logged_prompt: str) -> str:
    marker = "\nMemory rules:\n"
    prefix, separator, _ = logged_prompt.partition(marker)
    if not separator:
        raise ValueError("Logged memory prompt does not contain the V2 Memory rules marker")
    return f"{prefix}\n{TARGET_DIRECTED_POLICY}"


def candidate_from_step(
    result_dir: Path,
    episode: dict[str, Any],
    step: dict[str, Any],
    category: str,
) -> dict[str, Any] | None:
    relative_image = step.get("image_path")
    logged_memory = step.get("memory_context") or {}
    depth = step.get("depth_avoidance") or {}
    if not relative_image or not logged_memory.get("prompt_text") or depth.get("depth_grid") is None:
        return None
    try:
        image_path = resolve_image(result_dir, relative_image)
        new_memory_prompt = replace_memory_policy(logged_memory["prompt_text"])
    except (FileNotFoundError, ValueError):
        return None
    return {
        "category": category,
        "scene_key": episode["scene_key"],
        "episode_id": str(episode["episode_id"]),
        "step": int(step["step"]),
        "target_description": episode.get("description", ""),
        "true_name": episode.get("true_name"),
        "object_name": episode.get("object_name"),
        "episode_acc": int(bool(episode.get("acc"))),
        "episode_osr": int(bool(episode.get("osr"))),
        "episode_termination": episode.get("termination_reason"),
        "distance_before": step.get("distance_before"),
        "distance_after": step.get("distance_after"),
        "image_path": str(image_path),
        "source_image_path": relative_image,
        "depth_grid": depth["depth_grid"],
        "forward_clear_4m": is_forward_clear(depth["depth_grid"]),
        "logged_action": step.get("parsed_command"),
        "logged_raw_text": step.get("raw_action_text"),
        "logged_memory_prompt": logged_memory["prompt_text"],
        "new_memory_prompt": new_memory_prompt,
        "logged_memory_summary": logged_memory.get("summary"),
    }


def collect_candidates(result_dir: Path, per_category: int) -> list[dict[str, Any]]:
    source_path = result_dir / "all_episodes.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing merged episode file: {source_path}")

    by_category_scene: dict[str, dict[str, deque[dict[str, Any]]]] = {
        category: defaultdict(deque) for category in CATEGORIES
    }
    for episode in load_all_episodes(source_path):
        steps = episode.get("step_records") or []
        if not steps:
            continue

        category = None
        selected_step = None
        if episode.get("acc") and episode.get("termination_reason") == "stop":
            category = "successful_stop"
            selected_step = steps[-1]
        elif not episode.get("acc") and episode.get("termination_reason") == "stop":
            category = "false_stop_far"
            selected_step = steps[-1]
        elif (
            episode.get("osr")
            and not episode.get("acc")
            and episode.get("termination_reason") == "step_limit"
        ):
            near_steps = [
                step
                for step in steps
                if step.get("distance_after") is not None
                and float(step["distance_after"]) <= 20.0
            ]
            if near_steps:
                category = "osr_only_near"
                selected_step = min(near_steps, key=lambda step: float(step["distance_after"]))
        elif (
            not episode.get("osr")
            and not episode.get("acc")
            and episode.get("termination_reason") == "step_limit"
        ):
            for step in steps:
                summary = ((step.get("memory_context") or {}).get("summary") or {})
                grid = (step.get("depth_avoidance") or {}).get("depth_grid")
                if (
                    summary.get("recent_turn_state") == "full_rotation_loop"
                    and is_forward_clear(grid)
                    and float(step.get("distance_after") or 0.0) > 20.0
                ):
                    category = "non_osr_loop"
                    selected_step = step
                    break

        if category is None or selected_step is None:
            continue
        candidate = candidate_from_step(result_dir, episode, selected_step, category)
        if candidate is not None:
            by_category_scene[category][episode["scene_key"]].append(candidate)

    selected: list[dict[str, Any]] = []
    for category in CATEGORIES:
        scene_queues = by_category_scene[category]
        scene_names = sorted(scene_queues)
        category_rows = []
        while len(category_rows) < per_category and scene_names:
            remaining = []
            for scene in scene_names:
                if scene_queues[scene] and len(category_rows) < per_category:
                    category_rows.append(scene_queues[scene].popleft())
                if scene_queues[scene]:
                    remaining.append(scene)
            scene_names = remaining
        if len(category_rows) < per_category:
            raise RuntimeError(
                f"Only found {len(category_rows)} candidates for {category}, "
                f"requested {per_category}"
            )
        selected.extend(category_rows)
    return selected


def action_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    actions = Counter(row.get(field) or "unmatched" for row in rows)
    total = len(rows)
    return {
        "count": total,
        "distribution": dict(actions),
        "stop_count": actions.get("stop", 0),
        "stop_rate": actions.get("stop", 0) / total if total else 0.0,
        "turn_count": sum(actions[action] for action in TURN_ACTIONS),
        "turn_rate": sum(actions[action] for action in TURN_ACTIONS) / total if total else 0.0,
        "forward_count": actions.get("forward 3m", 0),
        "forward_rate": actions.get("forward 3m", 0) / total if total else 0.0,
    }


def paired_stop_transitions(rows: list[dict[str, Any]]) -> dict[str, int]:
    transitions = Counter(
        (
            "stop" if row.get("replayed_old_action") == "stop" else "move",
            "stop" if row.get("new_action") == "stop" else "move",
        )
        for row in rows
    )
    return {f"{old}_to_{new}": count for (old, new), count in transitions.items()}


def write_review_bundle(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir()
    labels_path = output_dir / "manual_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "category",
                "target_visibility",
                "target_screen_region",
                "target_appears_near",
                "promising_direction",
                "preferred_action",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow([row["sample_id"], row["category"], "", "", "", "", "", ""])

    cards = []
    for row in rows:
        suffix = Path(row["image_path"]).suffix or ".jpg"
        image_name = f"{row['sample_id']}{suffix}"
        target = image_dir / image_name
        if not target.exists():
            target.symlink_to(Path(row["image_path"]).resolve())
        cards.append(
            f"""
<article>
  <h2>{html.escape(row['sample_id'])}</h2>
  <img src="images/{html.escape(image_name)}" loading="lazy">
  <p><b>Target:</b> {html.escape(row['target_description'])}</p>
  <p><b>Scene/task:</b> {html.escape(row['scene_key'])}/{html.escape(row['episode_id'])}, step {row['step']}</p>
  <p><b>Distance:</b> {float(row['distance_after']):.2f}m; <b>forward clear:</b> {row['forward_clear_4m']}</p>
  <p><b>Logged:</b> {html.escape(str(row['logged_action']))}; <b>old replay:</b> {html.escape(str(row['replayed_old_action']))}; <b>new:</b> {html.escape(str(row['new_action']))}</p>
</article>"""
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Target-directed prompt review</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; color: #222; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
article {{ background: white; border: 1px solid #ccc; padding: 12px; }}
img {{ width: 100%; height: 280px; object-fit: contain; background: #111; }}
h2 {{ font-size: 16px; }} p {{ font-size: 13px; line-height: 1.35; }}
</style></head><body>
<h1>Fixed-frame target-directed prompt A/B</h1>
<p>Fill manual_labels.csv using target_visibility = clear/partial/absent/uncertain and preferred_action from the official action space.</p>
<main>{''.join(cards)}</main></body></html>"""
    (output_dir / "review.html").write_text(page, encoding="utf-8")


def write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> None:
    by_category = {}
    for category in CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "logged": action_stats(subset, "logged_action"),
            "replayed_old": action_stats(subset, "replayed_old_action"),
            "target_directed_v1": action_stats(subset, "new_action"),
            "old_to_new_stop_transitions": paired_stop_transitions(subset),
            "action_changed_count": sum(
                row["replayed_old_action"] != row["new_action"] for row in subset
            ),
        }
    summary = {
        "sample_count": len(rows),
        "samples_per_category": len(rows) // len(CATEGORIES),
        "categories": by_category,
        "old_replay_matches_logged": sum(
            row["replayed_old_action"] == row["logged_action"] for row in rows
        ),
        "old_parse_matched": sum(row["replayed_old_parse_matched"] for row in rows),
        "new_parse_matched": sum(row["new_parse_matched"] for row in rows),
        "action_changed_count": sum(
            row["replayed_old_action"] != row["new_action"] for row in rows
        ),
        "policy_name": "target_directed_v1",
        "policy_text": TARGET_DIRECTED_POLICY,
        "run_metadata": run_metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Fixed-frame Target-directed Prompt A/B",
        "",
        f"- backend: {run_metadata['backend']}",
        f"- model: {run_metadata['model_path']}",
        f"- Qwen adapter merged: {run_metadata['qwen_merge_adapter_for_inference']}",
        f"- samples: {len(rows)}",
        f"- V2 replay matches source Phi log: {summary['old_replay_matches_logged']}/{len(rows)}",
        f"- old/new parse matched: {summary['old_parse_matched']}/{summary['new_parse_matched']}",
        f"- changed actions: {summary['action_changed_count']}/{len(rows)}",
        "",
        "| category | old stop | new stop | old forward | new forward | changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        item = by_category[category]
        old = item["replayed_old"]
        new = item["target_directed_v1"]
        lines.append(
            f"| {category} | {old['stop_count']}/{old['count']} | "
            f"{new['stop_count']}/{new['count']} | {old['forward_count']}/{old['count']} | "
            f"{new['forward_count']}/{new['count']} | {item['action_changed_count']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_category <= 0:
        raise ValueError("samples-per-category must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    backend = resolve_backend(args.model_path, args.backend)
    os.environ["QWEN_MERGE_ADAPTER_FOR_INFERENCE"] = (
        "1" if args.qwen_merge_adapter_for_inference else "0"
    )

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    from eval_phi35_uavon import patch_transformers_cache_compat  # noqa: PLC0415

    if backend == "qwen25_vl":
        from eval_qwen25_vl_uavon import (  # noqa: PLC0415
            generate_qwen_action_text as generate_action_text,
        )
        from eval_qwen25_vl_uavon import (  # noqa: PLC0415
            load_qwen_model_and_processor as load_model_and_processor,
        )
    else:
        from eval_phi35_uavon import (  # noqa: PLC0415
            generate_action_text,
            load_model_and_processor,
        )
    from vlm_baseline.actions import parse_action_text  # noqa: PLC0415
    from vlm_baseline.depth_avoidance import build_depth_avoidance  # noqa: PLC0415

    rows = collect_candidates(args.result_dir, args.samples_per_category)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"target_prompt_fixed_frame_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    run_metadata = {
        "backend": backend,
        "model_path": str(args.model_path.resolve()),
        "base_model_path": (
            str(args.base_model_path.resolve()) if args.base_model_path else None
        ),
        "source_result_dir": str(args.result_dir.resolve()),
        "gpu": args.gpu,
        "device": args.device,
        "samples_per_category": args.samples_per_category,
        "max_new_tokens": args.max_new_tokens,
        "qwen_merge_adapter_for_inference": bool(
            args.qwen_merge_adapter_for_inference
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    patch_transformers_cache_compat()
    model, processor = load_model_and_processor(
        str(args.model_path),
        base_model_path=str(args.base_model_path) if args.base_model_path else None,
        device=args.device,
    )
    depth_module = build_depth_avoidance("uavon_single_view_prompt")
    records_path = output_dir / "counterfactual_records.jsonl"
    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(rows, start=1):
            row["sample_id"] = f"{index:03d}_{row['category']}_{row['scene_key']}_{row['episode_id']}_s{row['step']}"
            image = Image.open(row["image_path"]).convert("RGB")
            depth_prompt = depth_module.format_prompt(
                np.asarray(row["depth_grid"], dtype=np.float32)
            )
            raw_old = generate_action_text(
                model,
                processor,
                image,
                row["target_description"],
                args.device,
                args.max_new_tokens,
                depth_context=depth_prompt,
                memory_context=row["logged_memory_prompt"],
            )
            parsed_old = parse_action_text(raw_old)
            raw_new = generate_action_text(
                model,
                processor,
                image,
                row["target_description"],
                args.device,
                args.max_new_tokens,
                depth_context=depth_prompt,
                memory_context=row["new_memory_prompt"],
            )
            parsed_new = parse_action_text(raw_new)
            row["replayed_old_raw_text"] = raw_old
            row["replayed_old_action"] = parsed_old.command
            row["replayed_old_parse_matched"] = parsed_old.matched
            row["new_raw_text"] = raw_new
            row["new_action"] = parsed_new.command
            row["new_parse_matched"] = parsed_new.matched
            records.write(json.dumps(row, ensure_ascii=True) + "\n")
            records.flush()
            print(
                f"[{index}/{len(rows)}] {row['category']} "
                f"{row['scene_key']}/{row['episode_id']} step={row['step']} "
                f"logged={row['logged_action']!r} old={parsed_old.command!r} "
                f"new={parsed_new.command!r}",
                flush=True,
            )

    write_summary(output_dir, rows, run_metadata)
    write_review_bundle(output_dir, rows)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()

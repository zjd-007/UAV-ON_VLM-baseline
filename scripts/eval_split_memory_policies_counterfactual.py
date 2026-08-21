#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "results" / "target_prompt_fixed_frame_100_20260808_164145"
)
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

STOP_CONFIRMATION_RULES = """Memory rules:
- Memory contains executed history only. Current RGB and CurrentViewDepth describe the present state.
- Stop only when the current RGB clearly shows the target: its category or shape and at least one distinctive described attribute must both match.
- Do not stop for a partial, tiny, edge-clipped, heavily occluded, or merely similar object.
- If a target-like object is visible but not yet clear and close, keep it in view and approach or recenter only when depth-safe.
- If the target is not clearly confirmed, never stop because of step count, stagnation, repeated actions, or shallow depth.
- A short same-direction turn sequence is normal visual scanning. Change strategy only when RecentTurnState is oscillating or full_rotation_loop, or RecentMotionState is stagnant or revisiting.
- For full_rotation_loop or revisiting, do not repeat the same turn pattern. Choose forward only when CurrentViewDepth indicates that the complete 3m path is safe.
- For oscillating, do not reverse the most recent turn again; either repeat the most recent turn once to break the alternation, or choose forward when safe.
- If a translation action moved less than 1m, do not immediately repeat that translation; inspect another direction first.
- Never use descend only to break a loop. Use ascend only when the upper depth cells are clear."""

EXPLORATION_RULES = """Memory rules:
- Memory contains executed history only. Current RGB and CurrentViewDepth describe the present state.
- First inspect the current RGB image. If the target clearly matches the target description, choose stop immediately; memory must not override clear target evidence.
- If the target is not clearly confirmed, never choose stop because of step count, stagnation, or repeated actions.
- Navigation objective: maximize the chance that the next observations reveal the target; changing the recent action pattern is not itself a goal.
- Use the target description and visible scene context to prefer a region likely to contain the target.
- Turn toward a promising uninspected region. If that region is centered and the complete 3m path is depth-safe, prefer forward to obtain a new viewpoint.
- If no region is visibly more promising, prefer a safe new viewpoint with high visual novelty over revisiting or repeated turns.
- A full_rotation_loop means the current viewpoint is exhausted. Never reverse turn direction merely to break the pattern.
- After a full rotation, move forward only when the complete path is safe; otherwise inspect one safer unobserved direction and reassess.
- If a translation action moved less than 1m, do not immediately repeat it. Never use descend only to break a loop; use ascend only when upper depth cells are clear."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate separate compact stop-confirmation and target-exploration memory "
            "policies on the existing fixed-frame set."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=0,
        help="Limit stop and exploration groups independently for smoke testing; 0 means all.",
    )
    return parser.parse_args()


def replace_memory_rules(prompt: str, rules: str) -> str:
    marker = "\nMemory rules:\n"
    prefix, separator, _ = prompt.partition(marker)
    if not separator:
        raise ValueError("Source prompt does not contain V2 Memory rules")
    return f"{prefix}\n{rules}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["metric_within_20m"] = int(row["metric_within_20m"])
            row["desired_stop"] = int(row["desired_stop"])
            labels[row["sample_id"]] = row
    return labels


def wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def action_heading_delta(action: str) -> float:
    if action == "turn right 30 degree":
        return math.radians(30.0)
    if action == "turn left 30 degree":
        return math.radians(-30.0)
    return 0.0


def add_oracle_geometry(
    rows: list[dict[str, Any]],
    result_dir: Path,
) -> None:
    wanted = {(row["scene_key"], str(row["episode_id"])): row for row in rows}
    merged_path = result_dir / "all_episodes.jsonl"
    with merged_path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            key = (episode["scene_key"], str(episode["episode_id"]))
            row = wanted.get(key)
            if row is None:
                continue
            step = (episode.get("step_records") or [])[int(row["step"])]
            pose = step["pose_before"]
            target = episode["pose"][0]
            target_yaw = math.atan2(float(target[1]) - pose[1], float(target[0]) - pose[0])
            relative = wrap_radians(target_yaw - pose[3])
            row["oracle_geometry"] = {
                "target_relative_bearing_deg": round(math.degrees(relative), 3),
                "absolute_bearing_error_deg": round(abs(math.degrees(relative)), 3),
                "target_in_nominal_horizontal_fov": abs(math.degrees(relative)) <= 45.0,
            }
    missing = [row["sample_id"] for row in rows if "oracle_geometry" not in row]
    if missing:
        raise RuntimeError(f"Missing oracle geometry for {len(missing)} rows")


def heading_error_after(row: dict[str, Any], action: str) -> float:
    relative = math.radians(row["oracle_geometry"]["target_relative_bearing_deg"])
    return abs(math.degrees(wrap_radians(relative - action_heading_delta(action))))


def binary_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        desired = bool(row["manual_label"]["desired_stop"])
        predicted = row.get(field) == "stop"
        if desired and predicted:
            tp += 1
        elif desired:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def stop_paired_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = losses = 0
    for row in rows:
        desired = bool(row["manual_label"]["desired_stop"])
        old_ok = (row["replayed_old_action"] == "stop") == desired
        new_ok = (row["stop_confirmation_action"] == "stop") == desired
        gains += int(not old_ok and new_ok)
        losses += int(old_ok and not new_ok)
    discordant = gains + losses
    return {
        "gains": gains,
        "losses": losses,
        "exact_p": binomtest(gains, discordant, 0.5).pvalue if discordant else 1.0,
    }


def action_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(row.get(field) or "not_run" for row in rows))


def exploration_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    errors = [heading_error_after(row, row[field]) for row in rows]
    before = [row["oracle_geometry"]["absolute_bearing_error_deg"] for row in rows]
    return {
        "n": len(rows),
        "action_distribution": action_distribution(rows, field),
        "turn_rate": sum(row[field] in TURN_ACTIONS for row in rows) / len(rows),
        "forward_rate": sum(row[field] == "forward 3m" for row in rows) / len(rows),
        "mean_heading_error_after_deg": float(np.mean(errors)),
        "median_heading_error_after_deg": float(np.median(errors)),
        "mean_heading_improvement_deg": float(np.mean(np.asarray(before) - np.asarray(errors))),
    }


def write_review(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir()
    cards = []
    for row in rows:
        suffix = Path(row["image_path"]).suffix or ".jpg"
        image_name = f"{row['sample_id']}{suffix}"
        link = image_dir / image_name
        if not link.exists():
            link.symlink_to(Path(row["image_path"]).resolve())
        label = row.get("manual_label") or {}
        cards.append(
            f"""<article><h2>{html.escape(row['sample_id'])}</h2>
<img src="images/{html.escape(image_name)}" loading="lazy">
<p><b>Target:</b> {html.escape(row['target_description'])}</p>
<p><b>Distance:</b> {float(row['distance_after']):.2f}m; <b>bearing:</b> {row['oracle_geometry']['target_relative_bearing_deg']:.1f}deg</p>
<p><b>Visibility:</b> {html.escape(str(label.get('target_visibility', 'unlabeled')))}; <b>desired stop:</b> {html.escape(str(label.get('desired_stop', 'unlabeled')))}</p>
<p><b>Old:</b> {html.escape(row['replayed_old_action'])}; <b>combined:</b> {html.escape(row['new_action'])}; <b>stop-only:</b> {html.escape(str(row.get('stop_confirmation_action')))}; <b>explore-only:</b> {html.escape(str(row.get('exploration_action')))}</p></article>"""
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Split policy review</title>
<style>body{{font-family:sans-serif;margin:20px;background:#f5f5f5}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}}article{{background:white;border:1px solid #ccc;padding:12px}}img{{width:100%;height:280px;object-fit:contain;background:#111}}h2{{font-size:15px}}p{{font-size:13px}}</style></head>
<body><h1>Split stop-confirmation / exploration prompt review</h1><main>{''.join(cards)}</main></body></html>"""
    (output_dir / "review.html").write_text(page, encoding="utf-8")


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    labeled = [
        row
        for row in rows
        if row.get("manual_label") and row.get("stop_confirmation_action") is not None
    ]
    exploration = [row for row in rows if row.get("exploration_action")]
    old_stop = binary_metrics(labeled, "replayed_old_action")
    combined_stop = binary_metrics(labeled, "new_action")
    split_stop = binary_metrics(labeled, "stop_confirmation_action")
    old_explore = exploration_metrics(exploration, "replayed_old_action")
    combined_explore = exploration_metrics(exploration, "new_action")
    split_explore = exploration_metrics(exploration, "exploration_action")
    old_errors = [heading_error_after(row, row["replayed_old_action"]) for row in exploration]
    split_errors = [heading_error_after(row, row["exploration_action"]) for row in exploration]
    differences = np.asarray(split_errors) - np.asarray(old_errors)
    heading_p = (
        float(wilcoxon(differences).pvalue)
        if np.any(np.abs(differences) > 1e-9)
        else 1.0
    )
    summary = {
        "sample_count": len(rows),
        "manual_visibility_labeled_count": len(labeled),
        "stop_confirmation": {
            "replayed_v2": old_stop,
            "combined_target_directed_v1": combined_stop,
            "stop_confirmation_v1": split_stop,
            "paired_v2_vs_stop_confirmation": stop_paired_test(labeled),
        },
        "target_exploration": {
            "replayed_v2": old_explore,
            "combined_target_directed_v1": combined_explore,
            "target_exploration_v1": split_explore,
            "paired_heading_error_wilcoxon_p": heading_p,
            "changed_vs_v2": sum(
                row["exploration_action"] != row["replayed_old_action"]
                for row in exploration
            ),
        },
        "policy_text": {
            "stop_confirmation_v1": STOP_CONFIRMATION_RULES,
            "target_exploration_v1": EXPLORATION_RULES,
        },
        "oracle_note": (
            "Target bearing is used only for offline evaluation and is never included "
            "in either model prompt."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Split Memory Policy Fixed-frame A/B",
        "",
        f"- manually visibility-labeled frames: {len(labeled)}",
        f"- exploration frames: {len(exploration)}",
        "",
        "## Stop confirmation",
        "",
        "| policy | accuracy | precision | recall | false-positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in [
        ("V2 replay", old_stop),
        ("combined v1", combined_stop),
        ("stop confirmation v1", split_stop),
    ]:
        lines.append(
            f"| {name} | {item['accuracy']:.1%} | {item['precision']:.1%} | "
            f"{item['recall']:.1%} | {item['false_positive_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Target exploration",
            "",
            "| policy | forward | turn | mean heading error | mean improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in [
        ("V2 replay", old_explore),
        ("combined v1", combined_explore),
        ("target exploration v1", split_explore),
    ]:
        lines.append(
            f"| {name} | {item['forward_rate']:.1%} | {item['turn_rate']:.1%} | "
            f"{item['mean_heading_error_after_deg']:.2f}deg | "
            f"{item['mean_heading_improvement_deg']:.2f}deg |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels_path = args.labels or args.source_dir / "manual_visibility_labels_codex.csv"
    source_records = args.source_dir / "counterfactual_records.jsonl"
    rows = read_jsonl(source_records)
    labels = load_labels(labels_path)
    for row in rows:
        if row["sample_id"] in labels:
            row["manual_label"] = labels[row["sample_id"]]
    add_oracle_geometry(rows, args.result_dir)

    stop_rows = [row for row in rows if row.get("manual_label")]
    exploration_rows = [
        row for row in rows if row["category"] in {"osr_only_near", "non_osr_loop"}
    ]
    if args.limit_per_group > 0:
        stop_rows = stop_rows[: args.limit_per_group]
        exploration_rows = exploration_rows[: args.limit_per_group]
    selected_ids = {row["sample_id"] for row in stop_rows + exploration_rows}
    rows = [row for row in rows if row["sample_id"] in selected_ids]

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"split_prompt_fixed_frame_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

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

    patch_transformers_cache_compat()
    model, processor = load_model_and_processor(
        str(args.model_path), base_model_path=None, device=args.device
    )
    depth_module = build_depth_avoidance("uavon_single_view_prompt")
    stop_ids = {row["sample_id"] for row in stop_rows}
    exploration_ids = {row["sample_id"] for row in exploration_rows}
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["image_path"]).convert("RGB")
        depth_prompt = depth_module.format_prompt(
            np.asarray(row["depth_grid"], dtype=np.float32)
        )
        if row["sample_id"] in stop_ids:
            prompt = replace_memory_rules(
                row["logged_memory_prompt"], STOP_CONFIRMATION_RULES
            )
            raw = generate_action_text(
                model,
                processor,
                image,
                row["target_description"],
                args.device,
                args.max_new_tokens,
                depth_context=depth_prompt,
                memory_context=prompt,
            )
            parsed = parse_action_text(raw)
            row["stop_confirmation_raw_text"] = raw
            row["stop_confirmation_action"] = parsed.command
            row["stop_confirmation_parse_matched"] = parsed.matched
        if row["sample_id"] in exploration_ids:
            prompt = replace_memory_rules(row["logged_memory_prompt"], EXPLORATION_RULES)
            raw = generate_action_text(
                model,
                processor,
                image,
                row["target_description"],
                args.device,
                args.max_new_tokens,
                depth_context=depth_prompt,
                memory_context=prompt,
            )
            parsed = parse_action_text(raw)
            row["exploration_raw_text"] = raw
            row["exploration_action"] = parsed.command
            row["exploration_parse_matched"] = parsed.matched
        print(
            f"[{index}/{len(rows)}] {row['sample_id']} old={row['replayed_old_action']!r} "
            f"stop={row.get('stop_confirmation_action')!r} "
            f"explore={row.get('exploration_action')!r}",
            flush=True,
        )

    with (output_dir / "counterfactual_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    write_summary(output_dir, rows)
    write_review(output_dir, rows)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()

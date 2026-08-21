#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_depth_sft_data import (
    DATASET_ROOT,
    DEFAULT_ALIGNED_ROOT,
    DEFAULT_CACHE,
    DEFAULT_SOURCE,
    PROJECT_ROOT,
    far_depth_grid,
    load_depth_cache,
    resolve_image_path,
    row_to_command,
    sample_key,
)
from vlm_baseline.actions import ACTION_COMMANDS
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt
from vlm_baseline.prompting import DEPTH_AUGMENTED_PROMPT_TEMPLATE, build_prompt


DEFAULT_ORIGINAL_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048"
)
DEFAULT_REPAIR_COLLISION_DIR = (
    DATASET_ROOT
    / "processed"
    / "label_action_collision_check"
    / "label_action_collision_full_20260715_175048_repair_lane3"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "uavon_phi35_sft_depth_grid_collision_filtered.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "uavon_phi35_sft_depth_grid_collision_filtered_manifest.json"


def collision_key(row: dict) -> str:
    key = row.get("key")
    if key:
        return key
    scene_id = row.get("scene_id") or row.get("scene")
    episode_id = row.get("episode_id")
    pose_idx = row.get("pose_idx")
    frame_idx = int(row.get("frame_idx"))
    return f"{scene_id}::{episode_id}::{pose_idx}::{frame_idx}"


def is_error_row(row: dict) -> bool:
    return bool(row.get("error") or row.get("failed") or row.get("status") == "error")


def iter_collision_rows(directory: Path):
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("failed_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield path, json.loads(line)


def load_collision_filter(
    original_dir: Path,
    repair_dir: Path | None,
    exclude_unresolved_errors: bool = True,
) -> tuple[set[str], dict]:
    repair_decisions: dict[str, bool] = {}
    repair_stats: Counter[str] = Counter()
    if repair_dir and repair_dir.exists():
        for _, row in iter_collision_rows(repair_dir):
            key = collision_key(row)
            repair_stats["rows"] += 1
            if is_error_row(row):
                repair_stats["errors"] += 1
                continue
            repair_stats["checked"] += 1
            repair_decisions[key] = bool(row.get("new_collision_after_action"))
            if repair_decisions[key]:
                repair_stats["new_collisions"] += 1

    excluded: set[str] = set()
    original_stats: Counter[str] = Counter()
    scene_stats: dict[str, Counter[str]] = {}
    skipped_original_repaired = 0
    unresolved_error_keys: set[str] = set()

    for path, row in iter_collision_rows(original_dir):
        key = collision_key(row)
        scene = row.get("scene_id") or row.get("scene") or path.stem
        scene_stats.setdefault(scene, Counter())
        original_stats["rows"] += 1
        scene_stats[scene]["rows"] += 1

        if key in repair_decisions:
            skipped_original_repaired += 1
            continue

        if is_error_row(row):
            original_stats["errors"] += 1
            scene_stats[scene]["errors"] += 1
            unresolved_error_keys.add(key)
            if exclude_unresolved_errors:
                excluded.add(key)
            continue

        original_stats["checked"] += 1
        scene_stats[scene]["checked"] += 1
        if row.get("new_collision_after_action"):
            excluded.add(key)
            original_stats["new_collisions"] += 1
            scene_stats[scene]["new_collisions"] += 1

    for key, collided in repair_decisions.items():
        if collided:
            excluded.add(key)

    filter_stats = {
        "original_dir": str(original_dir),
        "repair_dir": str(repair_dir) if repair_dir else None,
        "original_stats": dict(original_stats),
        "repair_stats": dict(repair_stats),
        "repair_override_keys": len(repair_decisions),
        "skipped_original_repaired": skipped_original_repaired,
        "unresolved_error_keys": len(unresolved_error_keys),
        "exclude_unresolved_errors": exclude_unresolved_errors,
        "excluded_keys": len(excluded),
        "scene_stats_without_repair_overrides": {
            scene: dict(counts) for scene, counts in sorted(scene_stats.items())
        },
    }
    return excluded, filter_stats


def convert(
    source: Path,
    output: Path,
    manifest_path: Path,
    dataset_root: Path,
    aligned_root: Path,
    depth_cache_path: Path,
    original_collision_dir: Path,
    repair_collision_dir: Path | None,
    missing_depth_policy: str,
    limit: int = 0,
    grid_size: int = 3,
    max_meters: float = 100.0,
    overwrite: bool = False,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    excluded_keys, filter_stats = load_collision_filter(
        original_dir=original_collision_dir,
        repair_dir=repair_collision_dir,
    )

    if depth_cache_path.exists():
        depth_cache = load_depth_cache(depth_cache_path)
    elif missing_depth_policy == "error":
        raise FileNotFoundError(f"Depth cache path does not exist: {depth_cache_path}")
    else:
        depth_cache = {}

    depth_formatter = UAVONSingleViewDepthPrompt(grid_size=grid_size, max_meters=max_meters)

    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    episodes: set[str] = set()
    rows_read = 0
    rows_written = 0
    skipped_collision = 0
    retained_relabelled_stop = 0
    missing_depth = 0
    skipped_missing_depth = 0

    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_read += 1
            key = sample_key(row)
            stop_visibility = row.get("stop_visibility") or row.get("stop_visible_v1") or {}
            relabelled_stop = bool(
                stop_visibility.get("selected")
                and str(row.get("action_name") or "").strip().lower() == "stop"
            )
            if key in excluded_keys and not relabelled_stop:
                skipped_collision += 1
                continue
            if key in excluded_keys and relabelled_stop:
                retained_relabelled_stop += 1

            grid = row.get("depth_grid")
            if grid is None:
                grid = depth_cache.get(key)
            if grid is None:
                missing_depth += 1
                if missing_depth_policy == "error":
                    raise KeyError(f"Missing depth grid for {key}.")
                if missing_depth_policy == "skip":
                    skipped_missing_depth += 1
                    continue
                grid = far_depth_grid(grid_size, max_meters)

            command = row_to_command(row)
            image_path = resolve_image_path(row["image_path"], dataset_root, aligned_root)
            depth_context = depth_formatter.format_prompt(np.asarray(grid, dtype=np.float32))
            sample = {
                "conversations": [
                    {"from": "human", "value": build_prompt(row["target_description"], depth_context=depth_context)},
                    {"from": "gpt", "value": command},
                ],
                "images": [str(image_path)],
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            rows_written += 1
            action_counts[command] += 1
            scene_counts[row["scene_id"]] += 1
            episodes.add(row["episode_key"])
            if limit and rows_written >= limit:
                break

    expected_commands = set(ACTION_COMMANDS.values())
    unknown_commands = set(action_counts) - expected_commands
    if unknown_commands:
        raise ValueError(f"Unexpected commands generated: {sorted(unknown_commands)}")

    manifest = {
        "format": "llamafactory_sharegpt_multimodal_jsonl",
        "source": str(source),
        "output": str(output),
        "dataset_root": str(dataset_root),
        "aligned_root": str(aligned_root),
        "depth_cache": str(depth_cache_path),
        "rows_read": rows_read,
        "rows": rows_written,
        "episodes": len(episodes),
        "skipped_collision": skipped_collision,
        "retained_relabelled_stop_from_collision_filter": retained_relabelled_stop,
        "missing_depth": missing_depth,
        "skipped_missing_depth": skipped_missing_depth,
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "prompt_template": DEPTH_AUGMENTED_PROMPT_TEMPLATE.replace("<image>\n", ""),
        "commands": sorted(expected_commands),
        "depth_prompt": {
            "module": "uavon_single_view_prompt",
            "grid_size": grid_size,
            "max_meters": max_meters,
            "summary_in_prompt": False,
        },
        "collision_filter": filter_stats,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build collision-filtered depth-grid Phi-3.5-Vision SFT data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--original-collision-dir", type=Path, default=DEFAULT_ORIGINAL_COLLISION_DIR)
    parser.add_argument("--repair-collision-dir", type=Path, default=DEFAULT_REPAIR_COLLISION_DIR)
    parser.add_argument("--missing-depth-policy", choices=["error", "skip", "far"], default="error")
    parser.add_argument("--limit", type=int, default=0, help="Write only the first N retained rows.")
    parser.add_argument("--depth-grid-size", type=int, default=3)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = convert(
        source=args.source,
        output=args.output,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        aligned_root=args.aligned_root,
        depth_cache_path=args.depth_cache,
        original_collision_dir=args.original_collision_dir,
        repair_collision_dir=args.repair_collision_dir,
        missing_depth_policy=args.missing_depth_policy,
        limit=args.limit,
        grid_size=args.depth_grid_size,
        max_meters=args.depth_max_meters,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

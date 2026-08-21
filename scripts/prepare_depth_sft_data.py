#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from vlm_baseline.actions import ACTION_COMMANDS, action_name_to_command, action_vector_to_command
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt
from vlm_baseline.prompting import DEPTH_AUGMENTED_PROMPT_TEMPLATE, build_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_CACHE = DATASET_ROOT / "processed" / "depth_grid_cache" / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "uavon_phi35_sft_depth_grid.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "uavon_phi35_sft_depth_grid_manifest.json"


def sample_key(row: dict) -> str:
    return f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::{int(row['frame_idx'])}"


def row_to_command(row: dict) -> str:
    if row.get("action_name"):
        return action_name_to_command(row["action_name"])
    return action_vector_to_command(row["action_vector"])


def iter_cache_files(cache_path: Path) -> Iterable[Path]:
    if cache_path.is_file():
        yield cache_path
    elif cache_path.is_dir():
        yield from sorted(cache_path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Depth cache path does not exist: {cache_path}")


def load_depth_cache(cache_path: Path) -> dict[str, list[list[float]]]:
    cache: dict[str, list[list[float]]] = {}
    for path in iter_cache_files(cache_path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                grid = row.get("depth_grid")
                if grid is None:
                    continue
                cache[row.get("key") or sample_key(row)] = grid
    return cache


def resolve_image_path(raw_path: str, dataset_root: Path, aligned_root: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    marker = "record_output/images/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        aligned_candidate = aligned_root / "images" / suffix
        if aligned_candidate.is_file():
            return aligned_candidate.absolute()
        raw_candidate = dataset_root / "generated" / "record_output" / "images" / suffix
        if raw_candidate.is_file():
            return raw_candidate.absolute()

    path = Path(raw_path)
    if path.is_file():
        return path.absolute()
    raise FileNotFoundError(f"Image path does not exist and could not be remapped: {raw_path}")


def far_depth_grid(grid_size: int, value: float) -> list[list[float]]:
    return [[float(value) for _ in range(grid_size)] for _ in range(grid_size)]


def convert(
    source: Path,
    output: Path,
    manifest_path: Path,
    dataset_root: Path,
    aligned_root: Path,
    depth_cache_path: Path,
    missing_depth_policy: str,
    limit: int = 0,
    grid_size: int = 3,
    max_meters: float = 100.0,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
    rows = 0
    missing_depth = 0
    skipped_missing_depth = 0

    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            key = sample_key(row)
            grid = depth_cache.get(key)
            if grid is None:
                missing_depth += 1
                if missing_depth_policy == "error":
                    raise KeyError(f"Missing depth grid for {key}. Build cache first or use --missing-depth-policy skip.")
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
            rows += 1
            action_counts[command] += 1
            scene_counts[row["scene_id"]] += 1
            episodes.add(row["episode_key"])
            if limit and rows >= limit:
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
        "rows": rows,
        "episodes": len(episodes),
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
        "note": "Formal training should use a real AirSim depth cache. missing_depth_policy=far is for smoke tests only.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build depth-grid Phi-3.5-Vision SFT data for UAV-ON.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--depth-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--missing-depth-policy", choices=["error", "skip", "far"], default="error")
    parser.add_argument("--limit", type=int, default=0, help="Write only the first N rows for smoke tests.")
    parser.add_argument("--depth-grid-size", type=int, default=3)
    parser.add_argument("--depth-max-meters", type=float, default=100.0)
    args = parser.parse_args()

    manifest = convert(
        source=args.source,
        output=args.output,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        aligned_root=args.aligned_root,
        depth_cache_path=args.depth_cache,
        missing_depth_policy=args.missing_depth_policy,
        limit=args.limit,
        grid_size=args.depth_grid_size,
        max_meters=args.depth_max_meters,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

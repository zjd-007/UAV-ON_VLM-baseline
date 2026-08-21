#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vlm_baseline.actions import ACTION_COMMANDS, action_name_to_command, action_vector_to_command
from vlm_baseline.prompting import PROMPT_TEMPLATE, build_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "uavon_phi35_sft.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "uavon_phi35_sft_manifest.json"
EXPECTED_FULL_FRAMES: int | None = None


def resolve_image_path(raw_path: str, dataset_root: Path) -> Path:
    marker = "record_output/images/"
    normalized = raw_path.replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = dataset_root / "generated" / "record_output" / "images" / suffix
        if candidate.is_file():
            return candidate.resolve()

    path = Path(raw_path)
    if path.is_file():
        return path.resolve()

    raise FileNotFoundError(f"Image path does not exist and could not be remapped: {raw_path}")


def row_to_command(row: dict) -> str:
    if row.get("action_name"):
        return action_name_to_command(row["action_name"])
    return action_vector_to_command(row["action_vector"])


def convert(source: Path, output: Path, manifest_path: Path, dataset_root: Path, limit: int = 0) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    episodes: set[str] = set()
    rows = 0

    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            command = row_to_command(row)
            image_path = resolve_image_path(row["image_path"], dataset_root)
            sample = {
                "conversations": [
                    {"from": "human", "value": build_prompt(row["target_description"])},
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
    if limit == 0 and EXPECTED_FULL_FRAMES is not None and rows != EXPECTED_FULL_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FULL_FRAMES} full frames, wrote {rows}")

    manifest = {
        "format": "llamafactory_sharegpt_multimodal_jsonl",
        "source": str(source),
        "output": str(output),
        "dataset_root": str(dataset_root),
        "rows": rows,
        "episodes": len(episodes),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "prompt_template": PROMPT_TEMPLATE.replace("<image>\n", ""),
        "commands": sorted(expected_commands),
        "forbidden_model_inputs": ["depth", "history", "pose", "goal", "distance", "memory", "previous_action"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phi-3.5-Vision SFT data for UAV-ON.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Write only the first N rows for smoke tests.")
    args = parser.parse_args()

    manifest = convert(args.source, args.output, args.manifest, args.dataset_root, args.limit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

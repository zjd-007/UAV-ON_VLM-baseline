#!/usr/bin/env python3
"""Build ablation A: old small-target SFT plus new mid/big-target SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path


ROOT = Path("/data/zhujd/Aerial-ObjectNav")
VLM_ROOT = ROOT / "VLM-baseline"
DATASET_ROOT = ROOT / "UAV-ON_dataset"

DEFAULT_OLD_SOURCE = DATASET_ROOT / "processed/nomemory_baseline/train_frames.jsonl"
DEFAULT_OLD_SFT = VLM_ROOT / "data/uavon_phi35_sft_depth_grid_collision_filtered.jsonl"
DEFAULT_NEW_DIR = (
    DATASET_ROOT
    / "processed/neighborhood_coordinate_repair_v1_20260812_194807"
    / "final_dataset_per_frame_safe_stopbank_v1"
)
DEFAULT_NEW_SOURCE = DEFAULT_NEW_DIR / "train_frames.jsonl"
DEFAULT_NEW_SFT = (
    DEFAULT_NEW_DIR
    / "uavon_phi35_sft_depth_grid_stop_visible_v4_per_frame_safe_stopbank_v1.jsonl"
)
DEFAULT_ANNOTATION_ROOT = DATASET_ROOT / "annotations/UAV-ON-data/trainset"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed/size_hybrid_a_oldsmall_newmidbig_v1_20260818"

DATASET_NAME = "uavon_phi35_sft_depth_grid_size_hybrid_a_v1"
OUTPUT_NAME = f"{DATASET_NAME}.jsonl"
SMOKE_NAME = f"{DATASET_NAME}_smoke.jsonl"

ANNOTATION_FILES = {
    "BrushifyUrban": "BrushifyUrban_train.json",
    "CabinLake": "CabinLake_train.json",
    "CityPark": "CityPark_train.json",
    "DownTown": "DownTown_train.json",
    "Neighborhood": "NeighborhoodTrain.json",
    "Slum": "Slum_train.json",
    "UrbanJapan": "UrbanJapan_train.json",
    "Venice": "Venice_train.json",
    "WesternTown": "WesternTown_train.json",
    "WinterTown": "WinterTown_train.json",
}

ACTION_TO_COMMAND = {
    "ascend": "ascend 3m",
    "ascend 3m": "ascend 3m",
    "descend": "descend 3m",
    "descend 3m": "descend 3m",
    "move forward": "forward 3m",
    "forward": "forward 3m",
    "forward 3m": "forward 3m",
    "stop": "stop",
    "turn left": "turn left 30 degree",
    "turn left 30 degree": "turn left 30 degree",
    "turn right": "turn right 30 degree",
    "turn right 30 degree": "turn right 30 degree",
}
COMMANDS = set(ACTION_TO_COMMAND.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", type=Path, default=DEFAULT_OLD_SOURCE)
    parser.add_argument("--old-sft", type=Path, default=DEFAULT_OLD_SFT)
    parser.add_argument("--new-source", type=Path, default=DEFAULT_NEW_SOURCE)
    parser.add_argument("--new-sft", type=Path, default=DEFAULT_NEW_SFT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_size(value: object) -> str | None:
    match = re.search(r"\b(small|mid|big)\b", str(value or "").strip().lower())
    return match.group(1) if match else None


def command_from_source(row: dict) -> str:
    value = str(row.get("action_name") or row.get("action") or "").strip().lower()
    if value not in ACTION_TO_COMMAND:
        raise ValueError(f"Unknown source action: {value!r}")
    return ACTION_TO_COMMAND[value]


def conversation_parts(sample: dict) -> tuple[str, str]:
    human = [x.get("value", "") for x in sample.get("conversations", []) if x.get("from") == "human"]
    assistant = [x.get("value", "") for x in sample.get("conversations", []) if x.get("from") == "gpt"]
    if len(human) != 1 or len(assistant) != 1 or len(sample.get("images", [])) != 1:
        raise ValueError("Expected one human turn, one gpt turn, and one image")
    command = str(assistant[0]).strip()
    if command not in COMMANDS:
        raise ValueError(f"Unknown SFT command: {command!r}")
    return str(human[0]), command


def normalize_description(value: str) -> str:
    return " ".join(value.strip().lower().split()).rstrip(" .,!;:")


def source_sample_key(row: dict) -> str:
    return "::".join(
        [
            str(row["scene_id"]),
            str(row["episode_id"]),
            str(row["pose_idx"]),
            str(int(row["frame_idx"])),
        ]
    )


def image_sample_key(image_path: str) -> str:
    parts = Path(image_path).parts
    positions = [i for i, part in enumerate(parts) if part == "images"]
    for i in reversed(positions):
        tail = parts[i + 1 :]
        if len(tail) >= 5 and tail[3] == "uav_on_0":
            return "::".join([tail[0], tail[1], tail[2], str(int(Path(tail[4]).stem))])
    raise ValueError(f"Cannot derive sample key from image path: {image_path}")


def episode_key(row: dict) -> str:
    return str(row.get("episode_key") or f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}")


def load_annotations(root: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for scene, filename in ANNOTATION_FILES.items():
        path = root / filename
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            key = f"{scene}::{row['episode_id']}::0"
            if key in result:
                raise ValueError(f"Duplicate annotation episode: {key}")
            result[key] = {
                "size": normalize_size(row.get("size")),
                "size_raw": row.get("size", ""),
                "true_name": str(row.get("true_name") or "").strip(),
                "object_name": str(row.get("object_name") or "").strip(),
            }
    return result


def load_new_episode_sizes(path: Path) -> tuple[dict[str, str], Counter]:
    sizes: dict[str, str] = {}
    stats: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = episode_key(row)
            size = normalize_size(row.get("size"))
            if size is None:
                raise ValueError(f"New source has no recognized size: {key}: {row.get('size')!r}")
            if key in sizes and sizes[key] != size:
                raise ValueError(f"New source changes size inside episode: {key}")
            sizes[key] = size
            stats[f"rows_{size}"] += 1
    return sizes, stats


def load_old_source(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    samples: dict[str, dict] = {}
    episodes: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_key = source_sample_key(row)
            if sample_key in samples:
                raise ValueError(f"Duplicate old source sample: {sample_key}")
            ep_key = episode_key(row)
            meta = {
                "episode_key": ep_key,
                "scene_id": str(row["scene_id"]),
                "episode_id": str(row["episode_id"]),
                "pose_idx": str(row["pose_idx"]),
                "frame_idx": int(row["frame_idx"]),
                "target_description": str(row.get("target_description") or ""),
                "command": command_from_source(row),
            }
            samples[sample_key] = meta
            episodes.setdefault(ep_key, meta)
    return samples, episodes


def validate_sample(sample: dict, source_row: dict) -> tuple[str, str, str]:
    human, command = conversation_parts(sample)
    expected = command_from_source(source_row) if "action_name" in source_row else source_row["command"]
    if command != expected:
        raise ValueError(
            f"Action mismatch for {source_sample_key(source_row)}: SFT={command!r}, source={expected!r}"
        )
    description = normalize_description(str(source_row.get("target_description") or ""))
    if description and description not in normalize_description(human):
        raise ValueError(f"Target description missing from prompt: {source_sample_key(source_row)}")
    if "DepthGrid:" not in human or "DepthSummary" in human:
        raise ValueError(f"Unexpected depth prompt: {source_sample_key(source_row)}")
    image = str(sample["images"][0])
    return human, command, image


def add_nested(counter: dict[str, Counter], key: str, value: str) -> None:
    counter[key][value] += 1


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / OUTPUT_NAME
    provenance = args.output_dir / "source_index.jsonl"
    smoke = args.output_dir / SMOKE_NAME
    manifest_path = args.output_dir / "manifest.json"
    excluded_path = args.output_dir / "excluded_old_only_unknown_size.jsonl"
    conflicts_path = args.output_dir / "conflicting_decisions.jsonl"
    targets = [output, provenance, smoke, manifest_path, excluded_path, conflicts_path]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    annotations = load_annotations(args.annotation_root)
    new_episode_sizes, new_source_size_rows = load_new_episode_sizes(args.new_source)
    old_samples, old_episodes = load_old_source(args.old_source)

    old_episode_sizes: dict[str, str | None] = {}
    old_size_resolution: Counter = Counter()
    size_conflicts: list[dict] = []
    unknown_old_only: list[dict] = []
    for key, row in old_episodes.items():
        annotation = annotations.get(key, {})
        annotation_size = annotation.get("size")
        if key in new_episode_sizes:
            size = new_episode_sizes[key]
            old_size_resolution["new_train_frames_overlap"] += 1
            if annotation_size and annotation_size != size:
                size_conflicts.append(
                    {"episode_key": key, "new_size": size, "annotation_size": annotation_size}
                )
        else:
            size = annotation_size
            old_size_resolution["original_train_annotation"] += int(size is not None)
            old_size_resolution["missing"] += int(size is None)
            if size is None:
                unknown_old_only.append(
                    {
                        "episode_key": key,
                        "target_description": row["target_description"],
                        "true_name": annotation.get("true_name", ""),
                        "object_name": annotation.get("object_name", ""),
                        "size_raw": annotation.get("size_raw", ""),
                        "reason": "old-only episode has no authoritative size annotation",
                    }
                )
        old_episode_sizes[key] = size

    selected_episodes: dict[str, set[str]] = defaultdict(set)
    source_available_rows: dict[str, Counter] = defaultdict(Counter)
    output_rows_by_partition: dict[str, Counter] = defaultdict(Counter)
    action_counts: dict[str, Counter] = defaultdict(Counter)
    scene_counts: dict[str, Counter] = defaultdict(Counter)
    output_size_counts: Counter = Counter()
    output_action_counts: Counter = Counter()
    output_scene_counts: Counter = Counter()
    smoke_samples: dict[str, list[dict]] = defaultdict(list)
    missing_images = 0
    total_rows = 0
    exact_hashes: set[bytes] = set()
    duplicate_exact_rows = 0
    decision_labels: dict[bytes, dict] = {}
    conflict_records: list[dict] = []
    conflicting_decisions = 0

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_provenance = provenance.with_suffix(provenance.suffix + ".tmp")

    def emit(sample: dict, meta: dict, size: str, partition: str, dst, idx) -> None:
        nonlocal total_rows, missing_images, duplicate_exact_rows, conflicting_decisions
        human, command, image = validate_sample(sample, meta)
        if not Path(image).is_file():
            missing_images += 1
        encoded = json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8")
        exact_hash = hashlib.blake2b(encoded, digest_size=16).digest()
        if exact_hash in exact_hashes:
            duplicate_exact_rows += 1
        exact_hashes.add(exact_hash)
        decision_hash = hashlib.blake2b(
            (human + "\0" + image).encode("utf-8"), digest_size=16
        ).digest()
        decision = {
            "sample_key": source_sample_key(meta),
            "episode_key": episode_key(meta),
            "partition": partition,
            "action": command,
            "image": image,
        }
        previous = decision_labels.setdefault(decision_hash, decision)
        if previous["action"] != command:
            conflicting_decisions += 1
            conflict_records.append({"previous": previous, "current": decision})

        dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
        ep_key = episode_key(meta)
        record = {
            "row_index": total_rows,
            "sample_key": source_sample_key(meta),
            "episode_key": ep_key,
            "scene_id": str(meta["scene_id"]),
            "episode_id": str(meta["episode_id"]),
            "pose_idx": str(meta["pose_idx"]),
            "frame_idx": int(meta["frame_idx"]),
            "target_size": size,
            "source_partition": partition,
            "source_sft": str(args.old_sft if partition == "old_small" else args.new_sft),
            "action": command,
            "image": image,
        }
        idx.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_rows += 1
        selected_episodes[partition].add(ep_key)
        output_rows_by_partition[partition][size] += 1
        action_counts[partition][command] += 1
        scene_counts[partition][str(meta["scene_id"])] += 1
        output_size_counts[size] += 1
        output_action_counts[command] += 1
        output_scene_counts[str(meta["scene_id"])] += 1
        if len(smoke_samples[partition]) < 10:
            smoke_samples[partition].append(sample)

    try:
        with tmp_output.open("w", encoding="utf-8") as dst, tmp_provenance.open(
            "w", encoding="utf-8"
        ) as idx:
            with args.old_sft.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    key = image_sample_key(str(sample["images"][0]))
                    if key not in old_samples:
                        raise KeyError(f"Old SFT sample missing from old source: {key}")
                    meta = old_samples[key]
                    size = old_episode_sizes[meta["episode_key"]]
                    source_available_rows["old_collision_filtered"][str(size or "unknown")] += 1
                    if size == "small":
                        emit(sample, meta, size, "old_small", dst, idx)

            with args.new_source.open("r", encoding="utf-8") as src, args.new_sft.open(
                "r", encoding="utf-8"
            ) as sft:
                for line_number, pair in enumerate(zip_longest(src, sft), start=1):
                    source_line, sft_line = pair
                    if source_line is None or sft_line is None:
                        raise ValueError("New source and new SFT have different line counts")
                    if not source_line.strip() or not sft_line.strip():
                        raise ValueError(f"Blank line in aligned new data at line {line_number}")
                    meta = json.loads(source_line)
                    sample = json.loads(sft_line)
                    size = normalize_size(meta.get("size"))
                    if size is None:
                        raise ValueError(f"Missing new size at line {line_number}")
                    source_available_rows["new_stopbank_v1"][size] += 1
                    sample_image = str(sample["images"][0])
                    if sample_image != str(meta["image_path"]):
                        raise ValueError(f"New source/SFT image mismatch at line {line_number}")
                    if size in {"mid", "big"}:
                        emit(sample, meta, size, f"new_{size}", dst, idx)

        if missing_images:
            raise ValueError(f"Selected output contains {missing_images} missing image files")
        overlap = selected_episodes["old_small"] & (
            selected_episodes["new_mid"] | selected_episodes["new_big"]
        )
        if overlap:
            raise ValueError(f"Source partitions overlap on {len(overlap)} episodes")

        tmp_output.replace(output)
        tmp_provenance.replace(provenance)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        tmp_provenance.unlink(missing_ok=True)
        raise

    with smoke.open("w", encoding="utf-8") as handle:
        for partition in ("old_small", "new_mid", "new_big"):
            for sample in smoke_samples[partition]:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with excluded_path.open("w", encoding="utf-8") as handle:
        for row in unknown_old_only:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with conflicts_path.open("w", encoding="utf-8") as handle:
        for row in conflict_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "format": "llamafactory_sharegpt_multimodal_jsonl",
        "version": "size_hybrid_a_oldsmall_newmidbig_v1",
        "policy": {
            "unit": "episode",
            "small": "copy exact SFT rows from old collision-filtered dataset",
            "mid": "copy exact SFT rows from new per-frame-safe StopBank v1 dataset",
            "big": "copy exact SFT rows from new per-frame-safe StopBank v1 dataset",
            "unknown_size": "exclude; never infer size from appearance or object name",
            "sample_content": "copied verbatim; prompts and DepthGrid text are not regenerated",
        },
        "sources": {
            "old_small": {
                "source_frames": str(args.old_source),
                "sft": str(args.old_sft),
                "version": "depth_grid_collision_filtered_20260719",
            },
            "new_mid_big": {
                "source_frames": str(args.new_source),
                "sft": str(args.new_sft),
                "version": "stop_visible_v4_per_frame_safe_stopbank_v1",
            },
            "size_metadata": {
                "overlap": str(args.new_source),
                "old_only": str(args.annotation_root),
            },
        },
        "outputs": {
            "sft": str(output),
            "source_index": str(provenance),
            "smoke": str(smoke),
            "excluded_old_only_unknown_size": str(excluded_path),
            "conflicting_decisions": str(conflicts_path),
        },
        "rows": total_rows,
        "episodes": len(set().union(*selected_episodes.values())),
        "partition_rows": {
            key: dict(value) for key, value in sorted(output_rows_by_partition.items())
        },
        "partition_episodes": {
            key: len(value) for key, value in sorted(selected_episodes.items())
        },
        "size_rows": dict(output_size_counts),
        "size_episodes": {
            "small": len(selected_episodes["old_small"]),
            "mid": len(selected_episodes["new_mid"]),
            "big": len(selected_episodes["new_big"]),
        },
        "action_counts": dict(output_action_counts),
        "scene_counts": dict(output_scene_counts),
        "partition_action_counts": {
            key: dict(value) for key, value in sorted(action_counts.items())
        },
        "partition_scene_counts": {
            key: dict(value) for key, value in sorted(scene_counts.items())
        },
        "source_available_size_rows": {
            key: dict(value) for key, value in sorted(source_available_rows.items())
        },
        "new_source_size_rows": dict(new_source_size_rows),
        "old_episodes": len(old_episodes),
        "new_episodes": len(new_episode_sizes),
        "overlap_episodes": len(set(old_episodes) & set(new_episode_sizes)),
        "old_only_episodes": len(set(old_episodes) - set(new_episode_sizes)),
        "old_size_resolution": dict(old_size_resolution),
        "old_only_unknown_size_episodes": len(unknown_old_only),
        "size_conflicts_between_new_and_annotation": size_conflicts,
        "integrity": {
            "missing_images": missing_images,
            "conflicting_decisions": conflicting_decisions,
            "exact_duplicate_rows": duplicate_exact_rows,
            "partition_episode_overlap": 0,
            "source_index_rows": total_rows,
            "smoke_rows": sum(len(value) for value in smoke_samples.values()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

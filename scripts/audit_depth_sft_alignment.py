#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from prepare_depth_sft_data import (
    load_depth_cache,
    resolve_image_path,
    row_to_command,
    sample_key,
)
from vlm_baseline.actions import ACTION_COMMANDS
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt
from vlm_baseline.prompting import build_prompt


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield json.loads(line), line_number


def grids_match(left: Any, right: Any) -> bool:
    try:
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return (
        left_array.shape == (3, 3)
        and right_array.shape == (3, 3)
        and bool(np.allclose(left_array, right_array, rtol=0.0, atol=1e-4))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one-to-one alignment between UAV-ON frame rows and SFT rows."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--decode-image-stride",
        type=int,
        default=250,
        help="Decode every Nth movement image; every Stop image is always decoded.",
    )
    args = parser.parse_args()

    depth_cache = load_depth_cache(args.depth_cache)
    depth_formatter = UAVONSingleViewDepthPrompt(grid_size=3, max_meters=100.0)
    allowed_commands = set(ACTION_COMMANDS.values())

    counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_keys: set[str] = set()
    seen_images: set[str] = set()
    input_labels: dict[bytes, str] = {}
    input_first_rows: dict[bytes, dict[str, Any]] = {}
    input_occurrences: Counter[bytes] = Counter()
    episodes: dict[str, dict[str, Any]] = {}

    def record_error(kind: str, payload: dict[str, Any]) -> None:
        errors[kind] += 1
        if len(examples[kind]) < 10:
            examples[kind].append(payload)

    frame_iter = iter_jsonl(args.frames)
    sft_iter = iter_jsonl(args.sft)
    for pair_index, pair in enumerate(zip_longest(frame_iter, sft_iter), start=1):
        frame_item, sft_item = pair
        if frame_item is None:
            errors["extra_sft_rows"] += 1
            continue
        if sft_item is None:
            errors["missing_sft_rows"] += 1
            continue

        row, frame_line = frame_item
        sample, sft_line = sft_item
        counts["rows"] += 1
        key = sample_key(row)
        episode_key = str(row.get("episode_key") or "")
        payload = {
            "pair_index": pair_index,
            "frame_line": frame_line,
            "sft_line": sft_line,
            "sample_key": key,
            "episode_key": episode_key,
        }

        if key in sample_keys:
            record_error("duplicate_sample_key", payload)
        sample_keys.add(key)

        description = str(row.get("target_description") or "").strip()
        if not description:
            record_error("empty_target_description", payload)

        try:
            command = row_to_command(row)
        except Exception as exc:
            record_error("invalid_source_action", {**payload, "error": str(exc)})
            continue
        action_counts[command] += 1
        scene_counts[str(row.get("scene_id"))] += 1
        if command not in allowed_commands:
            record_error("unknown_command", {**payload, "command": command})

        grid = row.get("depth_grid")
        if grid is None:
            grid = depth_cache.get(key)
        if grid is None:
            record_error("missing_depth_grid", payload)
            continue
        grid_array = np.asarray(grid, dtype=np.float32)
        if grid_array.shape != (3, 3):
            record_error("invalid_depth_shape", {**payload, "shape": list(grid_array.shape)})
            continue
        if not np.isfinite(grid_array).all():
            record_error("nonfinite_depth", payload)
            continue
        if float(grid_array.min()) < 0.0 or float(grid_array.max()) > 100.0:
            record_error(
                "depth_out_of_range",
                {**payload, "min": float(grid_array.min()), "max": float(grid_array.max())},
            )

        try:
            expected_image = resolve_image_path(
                str(row["image_path"]), args.dataset_root, args.aligned_root
            ).resolve()
        except Exception as exc:
            record_error("unresolvable_source_image", {**payload, "error": str(exc)})
            continue

        conversations = sample.get("conversations")
        images = sample.get("images")
        if not isinstance(conversations, list) or len(conversations) != 2:
            record_error("invalid_conversation_structure", payload)
            continue
        if not isinstance(images, list) or len(images) != 1:
            record_error("invalid_image_list", payload)
            continue
        human, assistant = conversations
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            record_error("invalid_conversation_roles", payload)

        depth_context = depth_formatter.format_prompt(grid_array)
        expected_prompt = build_prompt(description, depth_context=depth_context)
        actual_prompt = str(human.get("value") or "")
        actual_command = str(assistant.get("value") or "")
        if actual_prompt != expected_prompt:
            record_error("prompt_mismatch", payload)
        if actual_command != command:
            record_error(
                "action_mismatch",
                {**payload, "expected": command, "actual": actual_command},
            )

        actual_image = Path(str(images[0])).resolve()
        if actual_image != expected_image:
            record_error(
                "image_path_mismatch",
                {**payload, "expected": str(expected_image), "actual": str(actual_image)},
            )
        if not actual_image.is_file():
            record_error("missing_image_file", {**payload, "image": str(actual_image)})
        elif actual_image.stat().st_size <= 0:
            record_error("empty_image_file", {**payload, "image": str(actual_image)})

        image_name = str(actual_image)
        is_stop = command == "stop"
        should_decode = is_stop or (
            args.decode_image_stride > 0 and pair_index % args.decode_image_stride == 0
        )
        if should_decode and image_name not in seen_images and actual_image.is_file():
            seen_images.add(image_name)
            try:
                with Image.open(actual_image) as image:
                    image.verify()
                counts["decoded_images"] += 1
            except Exception as exc:
                record_error(
                    "unreadable_image",
                    {**payload, "image": image_name, "error": str(exc)},
                )

        digest = hashlib.blake2b(digest_size=16)
        digest.update(image_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(actual_prompt.encode("utf-8"))
        input_key = digest.digest()
        previous_label = input_labels.get(input_key)
        if previous_label is not None and previous_label != actual_command:
            record_error(
                "identical_input_conflicting_label",
                {
                    **payload,
                    "image": image_name,
                    "first": input_first_rows[input_key],
                    "current_label": actual_command,
                },
            )
        input_labels.setdefault(input_key, actual_command)
        input_first_rows.setdefault(
            input_key,
            {
                "sample_key": key,
                "episode_key": episode_key,
                "frame_line": frame_line,
                "label": actual_command,
            },
        )
        input_occurrences[input_key] += 1

        episode = episodes.setdefault(
            episode_key,
            {
                "rows": 0,
                "stops": 0,
                "last_command": None,
                "last_frame_idx": None,
                "scene_id": str(row.get("scene_id")),
            },
        )
        frame_idx = int(row["frame_idx"])
        if episode["last_frame_idx"] is not None and frame_idx <= episode["last_frame_idx"]:
            record_error(
                "nonincreasing_episode_frame_idx",
                {**payload, "previous_frame_idx": episode["last_frame_idx"]},
            )
        episode["rows"] += 1
        episode["stops"] += int(is_stop)
        episode["last_command"] = command
        episode["last_frame_idx"] = frame_idx

        visibility = row.get("stop_visibility") or {}
        source_type = str(visibility.get("source_type") or "source_frame")
        source_type_counts[source_type] += 1
        if source_type == "actor_stop_bank_appended_to_original_episode":
            counts["actor_stop_bank_rows"] += 1
            source_image = Path(str(visibility.get("source_image_path") or "")).resolve()
            if source_image != expected_image:
                record_error(
                    "actor_bank_source_image_mismatch",
                    {**payload, "source_image": str(source_image), "image": str(expected_image)},
                )
            source_episode = str(visibility.get("source_stop_episode_key") or "")
            source_frame_idx = visibility.get("source_stop_frame_idx")
            source_depth = depth_cache.get(f"{source_episode}::{int(source_frame_idx)}")
            if source_depth is None:
                record_error("actor_bank_source_depth_missing", payload)
            elif not grids_match(grid, source_depth):
                record_error("actor_bank_source_depth_mismatch", payload)

    for episode_key, episode in episodes.items():
        if episode["stops"] == 0:
            counts["navigation_only_episodes"] += 1
        elif episode["stops"] > 1:
            record_error(
                "multiple_stops_in_episode",
                {"episode_key": episode_key, "stops": episode["stops"]},
            )
        if episode["stops"] and episode["last_command"] != "stop":
            record_error(
                "stop_not_final_row",
                {"episode_key": episode_key, "last_command": episode["last_command"]},
            )

    duplicate_groups = sum(count > 1 for count in input_occurrences.values())
    duplicate_rows = sum(count - 1 for count in input_occurrences.values() if count > 1)
    report = {
        "format": "uavon_depth_sft_alignment_audit_v1",
        "frames": str(args.frames.resolve()),
        "sft": str(args.sft.resolve()),
        "depth_cache": str(args.depth_cache.resolve()),
        "valid": not errors,
        "rows": counts["rows"],
        "episodes": len(episodes),
        "unique_sample_keys": len(sample_keys),
        "unique_model_inputs": len(input_occurrences),
        "identical_input_duplicate_groups": duplicate_groups,
        "identical_input_duplicate_rows": duplicate_rows,
        "decoded_images": counts["decoded_images"],
        "decode_policy": "all Stop images plus every Nth row",
        "decode_image_stride": args.decode_image_stride,
        "actor_stop_bank_rows": counts["actor_stop_bank_rows"],
        "navigation_only_episodes": counts["navigation_only_episodes"],
        "action_counts": dict(sorted(action_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "error_count": sum(errors.values()),
        "errors": dict(sorted(errors.items())),
        "error_examples": dict(sorted(examples.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()

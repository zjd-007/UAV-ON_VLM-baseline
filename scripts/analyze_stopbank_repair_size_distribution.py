#!/usr/bin/env python3
"""Summarize Stop repair categories by target-size bucket."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORY_GROUPS = {
    "reselected_existing_path": [
        "clear_candidate_stop_reselected",
        "neighborhood_coordinate_repaired__clear_candidate_stop_reselected",
    ],
    "unchanged_existing_stop": [
        "clear_candidate_stop_unchanged",
        "neighborhood_coordinate_repaired__clear_candidate_stop_unchanged",
    ],
    "direct_standoff_recaptured": [
        "target_facing_repairable_appended",
        "below_threshold_rescue_appended",
        "neighborhood_coordinate_repaired__below_threshold_rescue_appended",
    ],
    "actor_bank_reused_appended": ["actor_stop_bank_appended"],
}


def size_bucket(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    for bucket in ("small", "mid", "big"):
        if normalized.startswith(bucket):
            return bucket
    return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def image_sample_key(path: str) -> str | None:
    match = re.search(
        r"/images/([^/]+)/([^/]+)/([^/]+)/uav_on_0/(\d+)\.[A-Za-z0-9]+$",
        path,
    )
    if not match:
        return None
    scene, episode_id, pose_idx, frame_idx = match.groups()
    return f"{scene}::{episode_id}::{pose_idx}::{int(frame_idx)}"


def canonical_action(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "move forward": "forward",
        "forward 3m": "forward",
        "turn left": "turn_left",
        "turn left 30 degree": "turn_left",
        "turn right": "turn_right",
        "turn right 30 degree": "turn_right",
        "ascend": "ascend",
        "ascend 3m": "ascend",
        "descend": "descend",
        "descend 3m": "descend",
        "stop": "stop",
    }
    return aliases.get(normalized, normalized or "unknown")


def summarize(rows: list[dict[str, Any]], total_episodes: int) -> dict[str, Any]:
    counts = Counter(size_bucket(row.get("size")) for row in rows)
    image_counts = Counter(
        str(row.get("stop_image") or row.get("image_path"))
        for row in rows
        if row.get("stop_image") or row.get("image_path")
    )
    images_by_size = {
        bucket: {
            str(row.get("stop_image") or row.get("image_path"))
            for row in rows
            if size_bucket(row.get("size")) == bucket
            and (row.get("stop_image") or row.get("image_path"))
        }
        for bucket in ("small", "mid", "big", "unknown")
    }
    total = len(rows)
    return {
        "episodes": total,
        "percent_of_all_episodes": round(100.0 * total / total_episodes, 4),
        "size_counts": dict(counts),
        "size_composition_pct": {
            bucket: round(100.0 * counts[bucket] / total, 4) if total else 0.0
            for bucket in ("small", "mid", "big", "unknown")
        },
        "unique_stop_images": len(image_counts),
        "reused_stop_rows": sum(count - 1 for count in image_counts.values()),
        "maximum_single_image_reuse": max(image_counts.values(), default=0),
        "unique_stop_images_by_size": {
            bucket: len(images) for bucket, images in images_by_size.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--old-source-frames", type=Path)
    parser.add_argument("--old-sft", type=Path)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    episode_sizes: dict[str, str] = {}
    description_sizes: dict[tuple[str, str], str] = {}
    new_action_by_size: dict[str, Counter[str]] = defaultdict(Counter)
    with (dataset_dir / "train_frames.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            bucket = size_bucket(row.get("size"))
            episode_sizes[str(row["episode_key"])] = bucket
            description_sizes[
                (
                    str(row.get("scene_id")),
                    str(row.get("target_description") or "").strip().lower(),
                )
            ] = bucket
            new_action_by_size[bucket][canonical_action(row.get("action_name"))] += 1
    total_episodes = len(episode_sizes)
    population_counts = Counter(episode_sizes.values())

    audit_summary = json.loads(
        (dataset_dir / "stop_pair_audit" / "summary.json").read_text(encoding="utf-8")
    )
    category_rows = {
        category: read_jsonl(Path(path))
        for category, path in audit_summary["category_indexes"].items()
    }
    categories = {
        category: summarize(rows, total_episodes)
        for category, rows in category_rows.items()
    }
    groups: dict[str, dict[str, Any]] = {}
    classified_episode_keys = set()
    for group, category_names in CATEGORY_GROUPS.items():
        rows = [row for category in category_names for row in category_rows[category]]
        classified_episode_keys.update(str(row["episode_key"]) for row in rows)
        groups[group] = summarize(rows, total_episodes)

    appended_categories = (
        CATEGORY_GROUPS["direct_standoff_recaptured"]
        + CATEGORY_GROUPS["actor_bank_reused_appended"]
    )
    changed_categories = CATEGORY_GROUPS["reselected_existing_path"] + appended_categories
    groups["all_appended"] = summarize(
        [row for category in appended_categories for row in category_rows[category]],
        total_episodes,
    )
    groups["all_changed_or_appended"] = summarize(
        [row for category in changed_categories for row in category_rows[category]],
        total_episodes,
    )
    groups["all_stop_episodes"] = summarize(
        [row for rows in category_rows.values() for row in rows],
        total_episodes,
    )

    unresolved_keys = set(episode_sizes) - classified_episode_keys
    unresolved_rows = [{"size": episode_sizes[key]} for key in unresolved_keys]
    groups["unresolved_navigation_only"] = summarize(unresolved_rows, total_episodes)

    for group in groups.values():
        group["within_size_population_pct"] = {
            bucket: round(
                100.0 * group["size_counts"].get(bucket, 0) / population_counts[bucket],
                4,
            )
            if population_counts[bucket]
            else None
            for bucket in ("small", "mid", "big")
        }

    report = {
        "dataset_dir": str(dataset_dir),
        "total_episodes": total_episodes,
        "episode_size_counts": dict(population_counts),
        "episode_size_composition_pct": {
            bucket: round(100.0 * population_counts[bucket] / total_episodes, 4)
            for bucket in ("small", "mid", "big", "unknown")
        },
        "categories": categories,
        "groups": groups,
    }
    if args.old_source_frames and args.old_sft:
        source_by_key: dict[str, dict[str, Any]] = {}
        with args.old_source_frames.resolve().open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_path = str(row.get("image_path") or "")
                sample_key = image_sample_key(raw_path)
                if sample_key:
                    source_by_key[sample_key] = row
        old_stops = []
        missing_source_metadata = 0
        old_action_by_size: dict[str, Counter[str]] = defaultdict(Counter)
        with args.old_sft.resolve().open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                conversations = row.get("conversations") or []
                answer = str(conversations[-1].get("value", "")).strip().lower()
                image_path = str((row.get("images") or [""])[0])
                sample_key = image_sample_key(image_path)
                source_row = source_by_key.get(str(sample_key))
                if source_row is None:
                    bucket = "unknown"
                else:
                    bucket = episode_sizes.get(str(source_row.get("episode_key")))
                    if bucket is None:
                        bucket = description_sizes.get(
                            (
                                str(source_row.get("scene_id")),
                                str(source_row.get("target_description") or "")
                                .strip()
                                .lower(),
                            ),
                            "unknown",
                        )
                old_action_by_size[bucket][canonical_action(answer)] += 1
                if answer != "stop":
                    continue
                if source_row is None:
                    missing_source_metadata += 1
                    old_stops.append({"image_path": image_path, "size": "unknown"})
                else:
                    old_stops.append(
                        {"image_path": image_path, "size": bucket}
                    )
        report["old_training_stop_distribution"] = {
            **summarize(old_stops, total_episodes),
            "source_frames": str(args.old_source_frames.resolve()),
            "sft": str(args.old_sft.resolve()),
            "missing_source_metadata": missing_source_metadata,
        }
        report["training_action_counts_by_size"] = {
            "old": {bucket: dict(counts) for bucket, counts in old_action_by_size.items()},
            "new": {bucket: dict(counts) for bucket, counts in new_action_by_size.items()},
        }
    output_path = args.output or dataset_dir / "stop_repair_size_distribution.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

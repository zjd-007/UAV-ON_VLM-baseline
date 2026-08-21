#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTION_IDS = {
    "stop": 0,
    "forward 3m": 1,
    "turn left 30 degree": 2,
    "turn right 30 degree": 3,
    "ascend 3m": 4,
    "descend 3m": 5,
}

ACTION_NAMES = {
    "Stop": "stop",
    "Move Forward": "forward 3m",
    "Turn Left": "turn left 30 degree",
    "Turn Right": "turn right 30 degree",
    "Ascend": "ascend 3m",
    "Descend": "descend 3m",
}

GROUP_LIMITS = {
    "clear_stop": 60,
    "clear_motion": 80,
    "visible_weak_motion": 60,
    "absent_motion": 40,
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def size_bucket(value: Any) -> str:
    return str(value or "unknown").split("(", 1)[0].strip().lower() or "unknown"


def pixel_bin(pixel_count: int) -> str:
    if pixel_count <= 0:
        return "p00000"
    if pixel_count < 128:
        return "p00001_00127"
    if pixel_count < 512:
        return "p00128_00511"
    if pixel_count < 2048:
        return "p00512_02047"
    if pixel_count < 8192:
        return "p02048_08191"
    return "p08192_plus"


def distance_bin(distance: float | None) -> str:
    if distance is None:
        return "distance_unknown"
    if distance < 8.0:
        return "d00_08"
    if distance < 12.0:
        return "d08_12"
    if distance < 16.0:
        return "d12_16"
    return "d16_20"


def center_bin(center_offset: float | None) -> str:
    if center_offset is None:
        return "center_unknown"
    if center_offset <= 0.15:
        return "centered"
    if center_offset <= 0.35:
        return "off_center"
    return "edge"


def completeness_bin(assessment: dict[str, Any], pixels: int) -> str:
    if pixels <= 0:
        return "absent"
    reasons = {str(value) for value in assessment.get("reasons", [])}
    if "severely_clipped_or_too_close" in reasons:
        return "severely_clipped"
    clipping = (assessment.get("quality_components") or {}).get("clipping")
    if clipping is not None and float(clipping) < 0.7:
        return "partial"
    return "complete"


def visibility_group(clear: bool, pixels: int, action: str) -> str:
    is_stop = action == "stop"
    if clear:
        return "clear_stop" if is_stop else "clear_motion"
    if pixels > 0:
        return "visible_weak_stop" if is_stop else "visible_weak_motion"
    return "absent_stop" if is_stop else "absent_motion"


def stable_order_key(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}:{row['episode_key']}:{row['frame_idx']}:{row['image']}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def load_small_selections(paths: list[Path]) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if size_bucket(row.get("size")) != "small":
                continue
            key = str(row["trajectory_key"])
            # The repaired Neighborhood audit supersedes the original one.
            if str(row.get("scene_id")) == "Neighborhood" and "neighborhood_coordinate_repair" not in str(path):
                continue
            selections[key] = row
    return selections


def load_training_rows(
    frames_path: Path,
    wanted_episodes: set[str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    total = 0
    for line_index, row in enumerate(read_jsonl(frames_path)):
        total += 1
        episode_key = str(row.get("episode_key"))
        if episode_key not in wanted_episodes:
            continue
        frame_idx = int(row.get("frame_idx", -1))
        action = ACTION_NAMES.get(str(row.get("action_name")))
        if action is None:
            continue
        rows[(episode_key, frame_idx)] = {
            "line_index": line_index,
            "episode_key": episode_key,
            "frame_idx": frame_idx,
            "image": str(row.get("image_path")),
            "target_description": str(row.get("target_description") or ""),
            "true_name": str(row.get("true_name") or ""),
            "object_name": str(row.get("object_name") or ""),
            "scene_id": str(row.get("scene_id") or ""),
            "label": action,
            "label_id": ACTION_IDS[action],
        }
    return rows, total


def attach_prompts(samples: list[dict[str, Any]], sft_path: Path) -> None:
    by_line = {int(row["source_index"]): row for row in samples}
    pending = set(by_line)
    for line_index, row in enumerate(read_jsonl(sft_path)):
        if line_index not in pending:
            continue
        conversations = row.get("conversations") or []
        if not conversations:
            raise ValueError(f"SFT row {line_index} has no conversations")
        by_line[line_index]["prompt"] = str(conversations[0]["value"])
        pending.remove(line_index)
        if not pending:
            break
    if pending:
        raise ValueError(f"Missing prompts for {len(pending)} sampled rows")


def stratified_sample(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["visibility_group"] in GROUP_LIMITS:
            by_group[row["visibility_group"]].append(row)

    sampled: list[dict[str, Any]] = []
    for group, limit in GROUP_LIMITS.items():
        candidates = by_group[group]
        candidates.sort(
            key=lambda row: (
                row["pixel_bin"],
                row["distance_bin"],
                row["center_bin"],
                stable_order_key(seed, row),
            )
        )
        strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            strata[(row["pixel_bin"], row["distance_bin"], row["center_bin"])].append(row)
        keys = sorted(strata)
        while keys and sum(1 for row in sampled if row["visibility_group"] == group) < limit:
            next_keys = []
            for key in keys:
                bucket = strata[key]
                if bucket:
                    sampled.append(bucket.pop(0))
                    if sum(1 for row in sampled if row["visibility_group"] == group) >= limit:
                        break
                if bucket:
                    next_keys.append(key)
            keys = next_keys
    sampled.sort(key=lambda row: (row["visibility_group"], stable_order_key(seed, row)))
    for source_index, row in enumerate(sampled):
        row["source_index"] = int(row.pop("line_index"))
        row["sample_index"] = source_index
    return sampled


def write_bucket_counts(path: Path, counters: dict[str, Counter]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "bucket", "count"])
        for dimension, counts in sorted(counters.items()):
            for bucket, count in sorted(counts.items()):
                writer.writerow([dimension, bucket, count])


def clear_label_breakdown(rows: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    buckets = sorted({str(row[dimension]) for row in rows if row["clear"]})
    for bucket in buckets:
        selected = [row for row in rows if row["clear"] and str(row[dimension]) == bucket]
        stop = sum(row["label"] == "stop" for row in selected)
        motion = len(selected) - stop
        result[bucket] = {
            "recognizable_rows": len(selected),
            "stop_rows": stop,
            "motion_rows": motion,
            "motion_fraction_pct": round(100.0 * motion / len(selected), 4) if selected else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed-frame small-target Stop threshold evaluation.")
    parser.add_argument("--selections", type=Path, nargs="+", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selections = load_small_selections(args.selections)
    training, total_training_rows = load_training_rows(args.frames, set(selections))
    joined: list[dict[str, Any]] = []
    missing_training_rows = 0
    for episode_key, selection in selections.items():
        for assessment in selection.get("assessments", []):
            frame_idx = int(assessment["frame_idx"])
            training_row = training.get((episode_key, frame_idx))
            if training_row is None:
                missing_training_rows += 1
                continue
            pixels = int(assessment.get("pixel_count", 0))
            distance = assessment.get("distance_to_target")
            center = assessment.get("center_offset")
            clear = bool(assessment.get("clear"))
            joined.append(
                {
                    **training_row,
                    "size_bucket": "small",
                    "mask_pixels": pixels,
                    "mask_fraction": pixels / float(512 * 512),
                    "bbox_short_side": int(assessment.get("bbox_short_side", 0)),
                    "distance_to_target_m": float(distance) if distance is not None else None,
                    "center_offset": float(center) if center is not None else None,
                    "clear": clear,
                    "reasons": list(assessment.get("reasons", [])),
                    "clipping_score": (assessment.get("quality_components") or {}).get("clipping"),
                    "pixel_bin": pixel_bin(pixels),
                    "distance_bin": distance_bin(float(distance) if distance is not None else None),
                    "center_bin": center_bin(float(center) if center is not None else None),
                    "completeness_bin": completeness_bin(assessment, pixels),
                    "visibility_group": visibility_group(clear, pixels, training_row["label"]),
                }
            )

    counters: dict[str, Counter] = {
        "visibility_group": Counter(row["visibility_group"] for row in joined),
        "pixel_bin": Counter(row["pixel_bin"] for row in joined),
        "distance_bin": Counter(row["distance_bin"] for row in joined),
        "center_bin": Counter(row["center_bin"] for row in joined),
        "completeness_bin": Counter(row["completeness_bin"] for row in joined),
        "action": Counter(row["label"] for row in joined),
    }
    clear_rows = [row for row in joined if row["clear"]]
    clear_motion = [row for row in clear_rows if row["label"] != "stop"]
    samples = stratified_sample(joined, args.seed)
    attach_prompts(samples, args.sft)

    write_jsonl(args.output_dir / "fixed_frames.jsonl", samples)
    write_jsonl(args.output_dir / "fixed_frames_smoke.jsonl", samples[:12])
    write_jsonl(
        args.output_dir / "clear_target_motion_rows.jsonl",
        sorted(clear_motion, key=lambda row: (row["episode_key"], row["frame_idx"])),
    )
    write_bucket_counts(args.output_dir / "training_bucket_counts.csv", counters)

    summary = {
        "inputs": {
            "selections": [str(path) for path in args.selections],
            "frames": str(args.frames),
            "sft": str(args.sft),
        },
        "definitions": {
            "recognizable": "v4 assessment clear=true",
            "clear_motion": "recognizable small target frame retained with a non-Stop action label",
            "completeness": "complete when target is present without severe clipping and clipping score >= 0.7",
        },
        "selection_episodes": len(selections),
        "training_rows_total": total_training_rows,
        "joined_small_path_rows": len(joined),
        "assessment_rows_not_retained_in_training": missing_training_rows,
        "recognizable_rows": len(clear_rows),
        "recognizable_motion_rows": len(clear_motion),
        "recognizable_motion_fraction_pct": round(100.0 * len(clear_motion) / len(clear_rows), 4) if clear_rows else 0.0,
        "fixed_frame_samples": len(samples),
        "sample_counts": dict(Counter(row["visibility_group"] for row in samples)),
        "bucket_counts": {key: dict(value) for key, value in counters.items()},
        "recognizable_label_breakdown": {
            dimension: clear_label_breakdown(joined, dimension)
            for dimension in ("pixel_bin", "distance_bin", "center_bin", "completeness_bin")
        },
    }
    (args.output_dir / "training_visibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

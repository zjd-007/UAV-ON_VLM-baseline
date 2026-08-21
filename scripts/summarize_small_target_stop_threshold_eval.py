#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def relative_stop_probability(row: dict[str, Any]) -> float:
    scores = row.get("candidate_scores") or {}
    values = {label: float(stats["mean_logprob"]) for label, stats in scores.items()}
    if not values:
        return float(row.get("pred_command") == "stop")
    maximum = max(values.values())
    denominator = sum(math.exp(value - maximum) for value in values.values())
    return math.exp(values["stop"] - maximum) / denominator


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    stop_count = sum(row.get("pred_command") == "stop" for row in rows)
    stop_probabilities = [relative_stop_probability(row) for row in rows]
    return {
        "samples": count,
        "stop_predictions": stop_count,
        "stop_rate_pct": round(100.0 * stop_count / count, 4) if count else 0.0,
        "mean_relative_stop_probability": round(sum(stop_probabilities) / count, 6) if count else 0.0,
        "errors": sum(bool(row.get("error")) for row in rows),
    }


def grouped(rows: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(dimension, "unknown"))].append(row)
    return {bucket: summarize(bucket_rows) for bucket, bucket_rows in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired small-target Stop threshold predictions.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_rows = {int(row["sample_index"]): row for row in read_jsonl(args.old)}
    new_rows = {int(row["sample_index"]): row for row in read_jsonl(args.new)}
    common = sorted(set(old_rows) & set(new_rows))
    if not common:
        raise ValueError("No paired predictions found")

    dimensions = [
        "visibility_group",
        "pixel_bin",
        "distance_bin",
        "center_bin",
        "completeness_bin",
        "label",
        "source_run_label",
        "seen_group",
    ]
    report: dict[str, Any] = {
        "old": str(args.old),
        "new": str(args.new),
        "paired_samples": len(common),
        "overall": {
            "old": summarize([old_rows[index] for index in common]),
            "new": summarize([new_rows[index] for index in common]),
        },
        "by_dimension": {},
    }
    for dimension in dimensions:
        old_grouped = grouped([old_rows[index] for index in common], dimension)
        new_grouped = grouped([new_rows[index] for index in common], dimension)
        report["by_dimension"][dimension] = {
            bucket: {"old": old_grouped[bucket], "new": new_grouped[bucket]}
            for bucket in sorted(set(old_grouped) | set(new_grouped))
        }

    report["within_visibility_group"] = {}
    for visibility_group in sorted(
        {str(old_rows[index].get("visibility_group", "unknown")) for index in common}
    ):
        group_indices = [
            index
            for index in common
            if str(old_rows[index].get("visibility_group", "unknown")) == visibility_group
        ]
        report["within_visibility_group"][visibility_group] = {}
        for dimension in ("pixel_bin", "distance_bin", "center_bin"):
            old_grouped = grouped([old_rows[index] for index in group_indices], dimension)
            new_grouped = grouped([new_rows[index] for index in group_indices], dimension)
            report["within_visibility_group"][visibility_group][dimension] = {
                bucket: {"old": old_grouped[bucket], "new": new_grouped[bucket]}
                for bucket in sorted(set(old_grouped) | set(new_grouped))
            }

    report["visibility_by_context"] = {}
    for context_dimension in ("source_run_label", "seen_group"):
        report["visibility_by_context"][context_dimension] = {}
        context_values = sorted(
            {str(old_rows[index].get(context_dimension, "unknown")) for index in common}
        )
        for context_value in context_values:
            context_indices = [
                index
                for index in common
                if str(old_rows[index].get(context_dimension, "unknown")) == context_value
            ]
            old_grouped = grouped([old_rows[index] for index in context_indices], "visibility_group")
            new_grouped = grouped([new_rows[index] for index in context_indices], "visibility_group")
            report["visibility_by_context"][context_dimension][context_value] = {
                bucket: {"old": old_grouped[bucket], "new": new_grouped[bucket]}
                for bucket in sorted(set(old_grouped) | set(new_grouped))
            }

    paired_rows = []
    for index in common:
        old_row = old_rows[index]
        new_row = new_rows[index]
        paired_rows.append(
            {
                "sample_index": index,
                "episode_key": old_row["episode_key"],
                "frame_idx": old_row["frame_idx"],
                "true_name": old_row.get("true_name"),
                "label": old_row["label"],
                "visibility_group": old_row["visibility_group"],
                "mask_pixels": old_row["mask_pixels"],
                "mask_fraction": old_row["mask_fraction"],
                "distance_to_target_m": old_row.get("distance_to_target_m"),
                "center_offset": old_row.get("center_offset"),
                "completeness_bin": old_row["completeness_bin"],
                "old_action": old_row["pred_command"],
                "new_action": new_row["pred_command"],
                "old_stop_probability": relative_stop_probability(old_row),
                "new_stop_probability": relative_stop_probability(new_row),
                "new_minus_old_stop_probability": relative_stop_probability(new_row) - relative_stop_probability(old_row),
                "image": old_row["image"],
            }
        )

    with (args.output_dir / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

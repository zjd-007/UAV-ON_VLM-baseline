#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    sources = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for source in sources:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if key_field not in row:
                continue
            rows[str(row[key_field])] = row
    return rows


def frame_metrics(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("mask") or frame


def rank_value(value: Any) -> int:
    return int(value) if value is not None else 10**9


def build_review_record(
    key: str,
    cached: dict[str, Any],
    selection: dict[str, Any],
    min_pixel_growth: float,
    min_semantic_gain: float,
    max_semantic_drop: float,
    max_quality_drop: float,
) -> dict[str, Any] | None:
    selected_idx = selection.get("selected_frame_idx")
    assessments = {
        int(row["frame_idx"]): row for row in selection.get("assessments") or []
    }
    frames = {int(row["frame_idx"]): row for row in cached.get("frames") or []}
    if selected_idx is None:
        return {
            "trajectory_key": key,
            "target": cached.get("true_name"),
            "size": cached.get("size"),
            "selected_frame_idx": None,
            "suggested_frame_idx": None,
            "reason_codes": ["no_valid_stop_frame"],
            "selected_pixels": 0,
            "suggested_pixels": 0,
            "selected_semantic_score": None,
            "suggested_semantic_score": None,
            "selected_semantic_rank": None,
            "suggested_semantic_rank": None,
            "selected_quality": None,
            "suggested_quality": None,
        }

    selected_idx = int(selected_idx)
    selected_frame = frames[selected_idx]
    selected = assessments[selected_idx]
    selected_metrics = frame_metrics(selected_frame)
    selected_pixels = int(selected_metrics.get("pixel_count", 0))
    selected_semantic = selected.get("semantic_score")
    selected_rank = selected.get("semantic_rank")
    selected_quality = float(selected.get("quality_score", 0.0))
    selected_clipping = float(
        (selected.get("quality_components") or {}).get("clipping", 0.0)
    )

    later_clear = [
        row
        for idx, row in assessments.items()
        if idx > selected_idx and row.get("clear") and idx in frames
    ]
    candidates = []
    for assessment in later_clear:
        frame = frames[int(assessment["frame_idx"])]
        metrics = frame_metrics(frame)
        pixels = int(metrics.get("pixel_count", 0))
        semantic = assessment.get("semantic_score")
        semantic_drop = (
            0.0
            if selected_semantic is None or semantic is None
            else float(selected_semantic) - float(semantic)
        )
        quality = float(assessment.get("quality_score", 0.0))
        clipping = float(
            (assessment.get("quality_components") or {}).get("clipping", 0.0)
        )
        larger = pixels >= max(1, selected_pixels) * min_pixel_growth
        semantic_better = (
            selected_semantic is not None
            and semantic is not None
            and float(semantic) >= float(selected_semantic) + min_semantic_gain
        ) or rank_value(assessment.get("semantic_rank")) < rank_value(selected_rank)
        comparable = (
            semantic_drop <= max_semantic_drop
            and quality >= selected_quality - max_quality_drop
            and clipping >= selected_clipping - 0.15
        )
        if (larger and comparable) or semantic_better:
            candidates.append(
                {
                    "assessment": assessment,
                    "pixels": pixels,
                    "semantic": semantic,
                    "quality": quality,
                    "clipping": clipping,
                    "larger": larger,
                    "semantic_better": semantic_better,
                }
            )

    if not candidates:
        return None
    suggested = max(
        candidates,
        key=lambda row: (
            row["semantic_better"],
            rank_value(selected_rank) - rank_value(row["assessment"].get("semantic_rank")),
            row["pixels"] / max(1, selected_pixels),
            row["quality"],
        ),
    )
    reasons = []
    if suggested["larger"]:
        reasons.append("later_clear_frame_much_larger")
    if suggested["semantic_better"]:
        reasons.append("later_frame_semantically_stronger")
    return {
        "trajectory_key": key,
        "target": cached.get("true_name"),
        "size": cached.get("size"),
        "selected_frame_idx": selected_idx,
        "suggested_frame_idx": int(suggested["assessment"]["frame_idx"]),
        "reason_codes": reasons,
        "selected_pixels": selected_pixels,
        "suggested_pixels": suggested["pixels"],
        "selected_semantic_score": selected_semantic,
        "suggested_semantic_score": suggested["semantic"],
        "selected_semantic_rank": selected_rank,
        "suggested_semantic_rank": suggested["assessment"].get("semantic_rank"),
        "selected_quality": selected_quality,
        "suggested_quality": suggested["quality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag Stop selections that need targeted human review."
    )
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-pixel-growth", type=float, default=1.5)
    parser.add_argument("--min-semantic-gain", type=float, default=0.02)
    parser.add_argument("--max-semantic-drop", type=float, default=0.03)
    parser.add_argument("--max-quality-drop", type=float, default=0.08)
    args = parser.parse_args()

    cache = load_jsonl(args.visibility_cache, "trajectory_key")
    selections = load_jsonl(args.selections, "trajectory_key")
    records = []
    for key, selection in sorted(selections.items()):
        record = build_review_record(
            key,
            cache[key],
            selection,
            args.min_pixel_growth,
            args.min_semantic_gain,
            args.max_semantic_drop,
            args.max_quality_drop,
        )
        if record is not None:
            records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    jsonl_path = args.output_dir / "review_queue.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(records[0]) if records else [
        "trajectory_key",
        "target",
        "size",
        "selected_frame_idx",
        "suggested_frame_idx",
        "reason_codes",
    ]
    csv_path = args.output_dir / "review_queue.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **row,
                    "reason_codes": ";".join(row["reason_codes"]),
                }
            )

    overrides_path = args.output_dir / "manual_overrides_template.csv"
    with overrides_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "trajectory_key",
                "auto_selected_frame_idx",
                "suggested_frame_idx",
                "reviewed_frame_idx",
                "decision",
                "notes",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "trajectory_key": row["trajectory_key"],
                    "auto_selected_frame_idx": row["selected_frame_idx"],
                    "suggested_frame_idx": row["suggested_frame_idx"],
                    "reviewed_frame_idx": "",
                    "decision": "",
                    "notes": "",
                }
            )

    print(
        json.dumps(
            {
                "trajectory_count": len(selections),
                "review_count": len(records),
                "no_valid_stop_count": sum(
                    "no_valid_stop_frame" in row["reason_codes"] for row in records
                ),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

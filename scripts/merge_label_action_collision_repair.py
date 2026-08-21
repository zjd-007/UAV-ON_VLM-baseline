#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict[str, Any]) -> str | None:
    key = row.get("key")
    return str(key) if key else None


def count_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    counts["rows"] = len(rows)
    for row in rows:
        if row.get("error"):
            counts["errors"] += 1
        if row.get("new_collision_after_action"):
            counts["new_collisions"] += 1
        if row.get("initial_collided"):
            counts["initial_collided"] += 1
        if row.get("collided_after_action"):
            counts["collided_after_action"] += 1
    return dict(counts)


def merge_scene(original_dir: Path, repair_dir: Path, output_dir: Path, scene: str) -> dict[str, Any]:
    original_rows = read_jsonl(original_dir / f"{scene}.jsonl")
    repair_rows = read_jsonl(repair_dir / f"{scene}.jsonl")
    repair_by_key: dict[str, dict[str, Any]] = {}
    repair_duplicates = 0
    for row in repair_rows:
        key = row_key(row)
        if not key:
            continue
        if key in repair_by_key:
            repair_duplicates += 1
        repair_by_key[key] = row

    replaced = 0
    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in original_rows:
        key = row_key(row)
        if key and key in repair_by_key:
            merged_rows.append(repair_by_key[key])
            replaced += 1
            seen_keys.add(key)
        else:
            merged_rows.append(row)
            if key:
                seen_keys.add(key)

    extras = 0
    for key, row in repair_by_key.items():
        if key not in seen_keys:
            merged_rows.append(row)
            extras += 1

    write_jsonl(output_dir / f"{scene}.jsonl", merged_rows)
    return {
        "scene": scene,
        "original": count_rows(original_rows),
        "repair": count_rows(repair_rows),
        "merged": count_rows(merged_rows),
        "replaced_keys": replaced,
        "extra_repair_keys": extras,
        "repair_duplicate_keys": repair_duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge label-action collision repair rows by sample key.")
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", type=str, required=True, help="Comma-separated scenes to merge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = [scene for scene in args.scene_list.split(",") if scene]
    summaries = [merge_scene(args.original_dir, args.repair_dir, args.output_dir, scene) for scene in scenes]
    total = {
        "scene": "TOTAL",
        "original": Counter(),
        "repair": Counter(),
        "merged": Counter(),
        "replaced_keys": 0,
        "extra_repair_keys": 0,
        "repair_duplicate_keys": 0,
    }
    for summary in summaries:
        for section in ["original", "repair", "merged"]:
            total[section].update(summary[section])
        total["replaced_keys"] += summary["replaced_keys"]
        total["extra_repair_keys"] += summary["extra_repair_keys"]
        total["repair_duplicate_keys"] += summary["repair_duplicate_keys"]
    total["original"] = dict(total["original"])
    total["repair"] = dict(total["repair"])
    total["merged"] = dict(total["merged"])
    summaries.append(total)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "merge_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

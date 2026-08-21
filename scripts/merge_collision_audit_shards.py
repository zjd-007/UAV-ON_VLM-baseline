#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def source_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded label-action collision audits and verify source coverage."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        required=True,
        help="Collision shard root. Repeat to merge previous and resumed runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default="Neighborhood")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.scene}.jsonl"
    manifest_path = args.output_dir / "summary.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    source_keys = []
    with args.source.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["scene_id"]) == args.scene:
                source_keys.append(source_key(row))
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source contains duplicate sample keys")

    attempts: dict[str, list[dict[str, Any]]] = {}
    shard_paths = sorted(
        {
            path
            for root in args.shard_root
            for path in root.glob("lane*/Neighborhood.jsonl")
        }
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"no lane collision outputs in {[str(root) for root in args.shard_root]}"
        )
    for path in shard_paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("key") or "")
                if key:
                    attempts.setdefault(key, []).append(row)

    selected: dict[str, dict[str, Any]] = {}
    for key, rows in attempts.items():
        successful = [row for row in rows if not row.get("error")]
        selected[key] = successful[-1] if successful else rows[-1]
    missing = sorted(set(source_keys) - set(selected))
    unexpected = sorted(set(selected) - set(source_keys))
    if unexpected:
        raise ValueError(f"collision output has unexpected keys: {unexpected[:10]}")

    counters: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    action_new_collisions: Counter[str] = Counter()
    with output_path.open("x", encoding="utf-8") as output:
        for key in source_keys:
            row = selected.get(key)
            if row is None:
                continue
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            counters["rows"] += 1
            if row.get("error"):
                counters["errors"] += 1
                continue
            counters["checked"] += 1
            command = str(row.get("label_command") or "unknown")
            action_counts[command] += 1
            for field in (
                "initial_collided",
                "collided_after_action",
                "new_collision_after_action",
            ):
                if bool(row.get(field)):
                    counters[field] += 1
            if bool(row.get("new_collision_after_action")):
                action_new_collisions[command] += 1

    checked = counters["checked"]
    manifest = {
        "format": "merged_label_action_collision_audit_v1",
        "source": str(args.source.resolve()),
        "shard_roots": [str(root.resolve()) for root in args.shard_root],
        "shard_outputs": [str(path.resolve()) for path in shard_paths],
        "scene": args.scene,
        "source_rows": len(source_keys),
        "rows": counters["rows"],
        "checked": checked,
        "errors": counters["errors"],
        "missing": len(missing),
        "missing_examples": missing[:20],
        "initial_collided": counters["initial_collided"],
        "collided_after_action": counters["collided_after_action"],
        "new_collision_after_action": counters["new_collision_after_action"],
        "new_collision_after_action_rate": (
            counters["new_collision_after_action"] / checked if checked else None
        ),
        "action_counts": dict(action_counts),
        "action_new_collision_after_action": dict(action_new_collisions),
        "output": str(output_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps({"totals": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if missing or counters["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

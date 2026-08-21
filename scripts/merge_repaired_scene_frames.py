#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def sample_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge one repaired scene into an existing frame dataset."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-quarantine", type=Path)
    parser.add_argument("--repair-quarantine", type=Path)
    parser.add_argument("--quarantine-output", type=Path)
    args = parser.parse_args()
    quarantine_args = (
        args.base_quarantine,
        args.repair_quarantine,
        args.quarantine_output,
    )
    if any(quarantine_args) and not all(quarantine_args):
        raise ValueError(
            "--base-quarantine, --repair-quarantine, and --quarantine-output "
            "must be provided together"
        )
    for path in (args.output, args.manifest, args.quarantine_output):
        if path is None:
            continue
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.quarantine_output:
        args.quarantine_output.parent.mkdir(parents=True, exist_ok=True)

    seen_samples: set[str] = set()
    seen_episodes: set[str] = set()
    scene_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    def write_rows(output, rows: Iterable[dict[str, Any]], source_name: str) -> None:
        for row in rows:
            scene = str(row["scene_id"])
            if source_name == "base" and scene == args.scene:
                source_counts["base_replaced_scene_rows_skipped"] += 1
                continue
            if source_name == "repair" and scene != args.scene:
                raise ValueError(
                    f"repair input contains scene {scene}, expected only {args.scene}"
                )
            key = sample_key(row)
            if key in seen_samples:
                raise ValueError(f"duplicate merged sample key: {key}")
            seen_samples.add(key)
            seen_episodes.add(str(row["episode_key"]))
            scene_counts[scene] += 1
            action_counts[str(row["action_name"])] += 1
            source_counts[f"{source_name}_rows_written"] += 1
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.output.open("x", encoding="utf-8") as output:
        write_rows(output, read_jsonl(args.base), "base")
        write_rows(output, read_jsonl(args.repair), "repair")

    quarantine_counts: Counter[str] = Counter()
    if args.quarantine_output:
        quarantine_keys: set[str] = set()
        with args.quarantine_output.open("x", encoding="utf-8") as output:
            for row in read_jsonl(args.base_quarantine):
                if str(row.get("scene_id")) == args.scene:
                    quarantine_counts["base_replaced_scene_rows_skipped"] += 1
                    continue
                key = str(row["trajectory_key"])
                if key in quarantine_keys:
                    raise ValueError(f"duplicate merged quarantine key: {key}")
                quarantine_keys.add(key)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                quarantine_counts["base_rows_written"] += 1
            for row in read_jsonl(args.repair_quarantine):
                if str(row.get("scene_id")) != args.scene:
                    raise ValueError(
                        "repair quarantine contains unexpected scene: "
                        f"{row.get('scene_id')}"
                    )
                key = str(row["trajectory_key"])
                if key in quarantine_keys:
                    raise ValueError(f"duplicate merged quarantine key: {key}")
                quarantine_keys.add(key)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                quarantine_counts["repair_rows_written"] += 1

    if source_counts["repair_rows_written"] == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError(f"repair input contains no {args.scene} rows")
    manifest = {
        "format": "uavon_frames_with_repaired_scene_v1",
        "base": str(args.base.resolve()),
        "repair": str(args.repair.resolve()),
        "replaced_scene": args.scene,
        "output": str(args.output.resolve()),
        "rows": len(seen_samples),
        "episodes": len(seen_episodes),
        "source_counts": dict(source_counts),
        "quarantine": (
            {
                "base": str(args.base_quarantine.resolve()),
                "repair": str(args.repair_quarantine.resolve()),
                "output": str(args.quarantine_output.resolve()),
                "counts": dict(quarantine_counts),
            }
            if args.quarantine_output
            else None
        ),
        "scene_counts": dict(scene_counts),
        "action_counts": dict(action_counts),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

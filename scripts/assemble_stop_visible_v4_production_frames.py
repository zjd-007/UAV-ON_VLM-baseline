#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble repaired expert frames and the independent standoff Stop bank."
    )
    parser.add_argument("--base-frames", type=Path, required=True)
    parser.add_argument("--stop-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    episode_keys = set()
    sample_keys = set()
    with args.output.open("x", encoding="utf-8") as output:
        for source_name, source_path in (
            ("repaired_expert", args.base_frames),
            ("standoff_stop_bank", args.stop_bank),
        ):
            with source_path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_key = (
                        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
                        f"{int(row['frame_idx'])}"
                    )
                    if sample_key in sample_keys:
                        raise ValueError(f"duplicate assembled sample key: {sample_key}")
                    sample_keys.add(sample_key)
                    episode_keys.add(str(row["episode_key"]))
                    stats["rows"] += 1
                    stats[f"rows_{source_name}"] += 1
                    action_counts[str(row["action_name"])] += 1
                    scene_counts[str(row["scene_id"])] += 1
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "uavon_stop_visible_v4_production_frames",
        "base_frames": str(args.base_frames.resolve()),
        "stop_bank": str(args.stop_bank.resolve()),
        "output": str(args.output.resolve()),
        "rows": stats["rows"],
        "episodes": len(episode_keys),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "preserve_all_real_motion_before_selected_stop": True,
        "standoff_stops_are_independent_episodes": True,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

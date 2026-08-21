#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an action-balanced multimodal SFT smoke set.")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-action", type=int, default=8)
    args = parser.parse_args()

    selected = []
    counts: Counter[str] = Counter()
    stop_source_counts: Counter[str] = Counter()
    stop_source_limits = {
        "actor_stop_bank_appended_to_original_episode": 2,
        "target_facing_standoff_appended_to_original_episode": 2,
        "expert_path": max(1, args.per_action - 4),
    }

    for index, pair in enumerate(
        zip_longest(iter_jsonl(args.frames), iter_jsonl(args.sft)), start=1
    ):
        frame, sample = pair
        if frame is None or sample is None:
            raise ValueError("frame and SFT JSONL row counts differ")
        command = str(sample["conversations"][1]["value"])
        if counts[command] >= args.per_action:
            continue
        if command == "stop":
            source_type = str((frame.get("stop_visibility") or {}).get("source_type") or "expert_path")
            limit = stop_source_limits.get(source_type, 0)
            if stop_source_counts[source_type] >= limit:
                continue
            stop_source_counts[source_type] += 1
        counts[command] += 1
        selected.append((index, sample, frame))

    if len(counts) != 6 or any(value < args.per_action for value in counts.values()):
        raise ValueError(f"could not select {args.per_action} rows for every action: {dict(counts)}")

    selected.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for _, sample, _ in selected:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")

    manifest = {
        "format": "uavon_balanced_sft_smoke_v1",
        "source_frames": str(args.frames.resolve()),
        "source_sft": str(args.sft.resolve()),
        "output": str(args.output.resolve()),
        "rows": len(selected),
        "per_action": args.per_action,
        "action_counts": dict(sorted(counts.items())),
        "stop_source_counts": dict(sorted(stop_source_counts.items())),
        "sample_keys": [
            f"{frame['scene_id']}::{frame['episode_id']}::{frame['pose_idx']}::{int(frame['frame_idx'])}"
            for _, _, frame in selected
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

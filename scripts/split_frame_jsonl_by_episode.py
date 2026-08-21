#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def load_groups(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            groups.setdefault(str(row["episode_key"]), []).append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split frame JSONL by complete episodes with balanced frame counts."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f"lane{index}.jsonl" for index in range(args.shards)]
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in [*outputs, manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    groups = load_groups(args.source)
    shard_groups: list[list[tuple[str, list[dict[str, Any]]]]] = [
        [] for _ in range(args.shards)
    ]
    loads = [0 for _ in range(args.shards)]
    for episode_key, rows in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        shard = min(range(args.shards), key=lambda index: (loads[index], index))
        shard_groups[shard].append((episode_key, rows))
        loads[shard] += len(rows)

    summaries = []
    for index, (path, groups_for_shard) in enumerate(zip(outputs, shard_groups)):
        with path.open("x", encoding="utf-8") as output:
            for _, rows in sorted(groups_for_shard, key=lambda item: item[0]):
                for row in sorted(rows, key=lambda value: int(value["frame_idx"])):
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries.append(
            {
                "shard": index,
                "path": str(path.resolve()),
                "episodes": len(groups_for_shard),
                "frames": loads[index],
            }
        )

    manifest = {
        "format": "frame_jsonl_episode_balanced_shards_v1",
        "source": str(args.source.resolve()),
        "source_episodes": len(groups),
        "source_frames": sum(loads),
        "shards": summaries,
        "episode_integrity": "each episode_key appears in exactly one shard",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

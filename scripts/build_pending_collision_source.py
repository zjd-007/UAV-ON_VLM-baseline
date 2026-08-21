#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def source_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a collision-audit source containing only unresolved samples."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--audit-root",
        type=Path,
        action="append",
        required=True,
        help="Existing collision shard root. Repeat for multiple attempts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    existing = [path for path in (args.output, args.manifest) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    successful: set[str] = set()
    error_keys: set[str] = set()
    audit_paths = sorted(
        {
            path
            for root in args.audit_root
            for path in root.glob("lane*/Neighborhood.jsonl")
        }
    )
    for path in audit_paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("key") or "")
                if not key:
                    continue
                if row.get("error"):
                    error_keys.add(key)
                else:
                    successful.add(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    pending_rows = 0
    duplicate_keys: set[str] = set()
    seen: set[str] = set()
    with args.source.open(encoding="utf-8") as source, args.output.open(
        "x", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = source_key(row)
            source_rows += 1
            if key in seen:
                duplicate_keys.add(key)
            seen.add(key)
            if key in successful:
                continue
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            pending_rows += 1

    if duplicate_keys:
        raise ValueError(f"source contains duplicate keys: {sorted(duplicate_keys)[:10]}")
    unexpected = sorted(successful - seen)
    if unexpected:
        raise ValueError(f"audit contains keys outside source: {unexpected[:10]}")

    manifest = {
        "format": "pending_collision_source_v1",
        "source": str(args.source.resolve()),
        "audit_roots": [str(root.resolve()) for root in args.audit_root],
        "audit_outputs": [str(path.resolve()) for path in audit_paths],
        "source_rows": source_rows,
        "successful_existing_keys": len(successful),
        "error_keys_seen": len(error_keys),
        "pending_rows": pending_rows,
        "coverage_ok": source_rows == len(successful) + pending_rows,
        "output": str(args.output.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["coverage_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

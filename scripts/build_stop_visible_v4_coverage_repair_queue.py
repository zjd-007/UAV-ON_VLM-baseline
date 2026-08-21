#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standoff queue for navigation actors missing Stop coverage."
    )
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "keys": args.output_dir / "coverage_repair_actor_keys.txt",
        "manifest": args.output_dir / "coverage_repair_queue.jsonl",
        "summary": args.output_dir / "coverage_repair_queue_summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite queue outputs: {existing}")

    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    requested = set(report["navigation_only_actors_without_stop_coverage"])
    by_actor: dict[str, list[dict[str, Any]]] = {key: [] for key in requested}
    for path in sorted(args.audit_dir.glob("*.jsonl")):
        if path.name.endswith("_actors.jsonl"):
            continue
        for row in read_jsonl(path):
            actor = f"{row['scene_id']}::{row.get('object_name') or ''}"
            if actor in by_actor:
                by_actor[actor].append(row)

    missing = sorted(actor for actor, rows in by_actor.items() if not rows)
    if missing:
        raise KeyError(f"actors missing from visibility audit: {missing}")
    queue = []
    for actor, rows in sorted(by_actor.items()):
        representative = min(rows, key=lambda row: str(row["trajectory_key"]))
        queue.append(
            {
                "capture_group": "semantic_coverage_repair",
                "trajectory_key": representative["trajectory_key"],
                "scene_id": representative["scene_id"],
                "object_name": representative["object_name"],
                "true_name": representative["true_name"],
                "size": representative["size"],
                "represented_trajectory_count": len(rows),
                "represented_trajectory_keys": sorted(
                    str(row["trajectory_key"]) for row in rows
                ),
                "validation_causes": report[
                    "navigation_only_actors_without_stop_coverage"
                ][actor],
            }
        )

    paths["keys"].write_text(
        "".join(f"{row['trajectory_key']}\n" for row in queue), encoding="utf-8"
    )
    with paths["manifest"].open("x", encoding="utf-8") as output:
        for row in queue:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "validation_report": str(args.validation_report.resolve()),
        "audit_dir": str(args.audit_dir.resolve()),
        "actors": len(queue),
        "represented_trajectories": sum(
            int(row["represented_trajectory_count"]) for row in queue
        ),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

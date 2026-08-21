#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def has_metadata(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("true_name") or "").strip()
        and str(row.get("target_description") or "").strip()
        and str(row.get("size") or "").strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one representative trajectory per actor for production standoff capture."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--selections",
        type=Path,
        help=(
            "Optional production selection decisions. When provided, every retained "
            "trajectory without an eligible expert-path Stop is queued."
        ),
    )
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "repairable": args.output_dir / "repairable_actor_keys.txt",
        "rescue": args.output_dir / "below_threshold_actor_keys.txt",
        "manifest": args.output_dir / "standoff_queue.jsonl",
        "summary": args.output_dir / "standoff_queue_summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite queue outputs: {existing}")

    causes = {
        str(row["trajectory_key"]): str(row.get("refined_problem_cause") or "unknown")
        for row in read_jsonl(
            args.audit_dir / "summary_collision_filtered" / "problem_trajectories.jsonl"
        )
    }
    selection_by_key = (
        {
            str(row["trajectory_key"]): row
            for row in read_jsonl(args.selections)
        }
        if args.selections
        else {}
    )
    quarantined = (
        {
            str(row["trajectory_key"])
            for row in read_jsonl(args.quarantine)
        }
        if args.quarantine
        else set()
    )
    trajectories = []
    for path in sorted(args.audit_dir.glob("*.jsonl")):
        if path.name.endswith("_actors.jsonl"):
            continue
        trajectories.extend(read_jsonl(path))

    by_group: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        "repairable": defaultdict(list),
        "rescue": defaultdict(list),
    }
    for row in trajectories:
        if not has_metadata(row):
            continue
        key = str(row["trajectory_key"])
        if key in quarantined:
            continue
        decision = selection_by_key.get(key)
        if args.selections and (
            decision is None or decision.get("selected_frame_idx") is not None
        ):
            continue
        cause = causes.get(str(row["trajectory_key"]))
        if cause == "target_facing_standoff_repairable":
            group = "repairable"
        elif (
            cause == "target_pixels_exist_but_below_clear_threshold_after_standoff"
            or args.selections
        ):
            group = "rescue"
        else:
            continue
        actor_key = (str(row["scene_id"]), str(row["object_name"]))
        by_group[group][actor_key].append(row)

    queue_rows = []
    for group, actors in by_group.items():
        keys = []
        for actor_key, rows in sorted(actors.items()):
            representative = min(rows, key=lambda row: str(row["trajectory_key"]))
            keys.append(str(representative["trajectory_key"]))
            queue_rows.append(
                {
                    "capture_group": group,
                    "trajectory_key": representative["trajectory_key"],
                    "scene_id": representative["scene_id"],
                    "object_name": representative["object_name"],
                    "true_name": representative["true_name"],
                    "size": representative["size"],
                    "represented_trajectory_count": len(rows),
                    "represented_trajectory_keys": sorted(
                        str(row["trajectory_key"]) for row in rows
                    ),
                }
            )
        outputs[group].write_text("\n".join(keys) + "\n", encoding="utf-8")

    with outputs["manifest"].open("x", encoding="utf-8") as output:
        for row in sorted(
            queue_rows,
            key=lambda value: (value["capture_group"], value["scene_id"], value["object_name"]),
        ):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "audit_dir": str(args.audit_dir.resolve()),
        "actors_by_group": dict(Counter(row["capture_group"] for row in queue_rows)),
        "represented_trajectories_by_group": {
            group: sum(
                int(row["represented_trajectory_count"])
                for row in queue_rows
                if row["capture_group"] == group
            )
            for group in by_group
        },
        "scenes_by_group": {
            group: sorted(
                {row["scene_id"] for row in queue_rows if row["capture_group"] == group}
            )
            for group in by_group
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def sample_key(row: dict[str, Any]) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stop-visible v4 production frames.")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--allowed-uncovered-actors", type=Path)
    parser.add_argument(
        "--allowed-image-root",
        type=Path,
        action="append",
        default=[],
        help="Additional verified aligned/recaptured image root. Can be repeated.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {args.report}")

    quarantined = {
        str(row["trajectory_key"])
        for _, row in read_jsonl(args.quarantine)
    }
    allowed_uncovered = set()
    if args.allowed_uncovered_actors:
        allowed_uncovered = {
            str(row["actor_key"])
            for _, row in read_jsonl(args.allowed_uncovered_actors)
        }
    sample_keys: set[str] = set()
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    stop_actors: set[tuple[str, str]] = set()
    navigation_actor_causes: dict[tuple[str, str], set[str]] = defaultdict(set)
    errors: list[str] = []
    allowed_image_roots = [path.resolve() for path in args.allowed_image_root]

    def under_allowed_root(path: Path) -> bool:
        resolved = path.resolve()
        return any(
            resolved == root or root in resolved.parents for root in allowed_image_roots
        )

    for line_number, row in read_jsonl(args.frames):
        stats["rows"] += 1
        key = sample_key(row)
        if key in sample_keys:
            errors.append(f"duplicate sample key at line {line_number}: {key}")
        sample_keys.add(key)
        episode_key = str(row["episode_key"])
        if episode_key in quarantined:
            errors.append(f"quarantined trajectory present at line {line_number}: {episode_key}")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            errors.append(f"missing image at line {line_number}: {image_path}")
        if (
            "record_output/images/" in image_path.as_posix()
            and "record_output_transition_aligned/images/" not in image_path.as_posix()
            and not under_allowed_root(image_path)
        ):
            errors.append(f"old unaligned image path at line {line_number}: {image_path}")
        if not str(row.get("target_description") or "").strip():
            errors.append(f"empty target_description at line {line_number}: {key}")

        action = str(row.get("action_name") or "").strip().lower()
        action_counts[action] += 1
        actor_key = (str(row["scene_id"]), str(row.get("object_name") or ""))
        repair = row.get("trajectory_repair") or {}
        if action == "stop":
            stop_actors.add(actor_key)
        elif repair.get("mode") == "navigation_only_pending_independent_stop_bank":
            navigation_actor_causes[actor_key].add(
                str(repair.get("problem_cause") or "unknown")
            )
        stop_visibility = row.get("stop_visibility") or {}
        if stop_visibility.get("source_type") in {
            "independent_target_facing_standoff_stop_bank",
            "target_facing_standoff_appended_to_original_episode",
            "actor_stop_bank_appended_to_original_episode",
        }:
            stats["standoff_stop_rows"] += 1
            if action != "stop":
                errors.append(f"standoff bank row is not Stop: {key}")
            if row.get("depth_grid") is None:
                errors.append(f"standoff bank row has no inline depth grid: {key}")
        if (
            stop_visibility.get("source_type")
            in {
                "target_facing_standoff_appended_to_original_episode",
                "actor_stop_bank_appended_to_original_episode",
            }
        ):
            stats["appended_standoff_stop_rows"] += 1
            if not stop_visibility.get("appended_to_original_trajectory"):
                errors.append(f"appended standoff Stop lacks append marker: {key}")
        if (row.get("trajectory_repair") or {}).get(
            "earlier_stop_eligible_motion_preserved"
        ):
            stats["earlier_stop_eligible_motion_rows_preserved"] += 1
        episodes[episode_key].append(row)

    for episode_key, rows in episodes.items():
        ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
        stops = [row for row in ordered if str(row.get("action_name") or "").lower() == "stop"]
        if len(stops) > 1:
            errors.append(f"episode has multiple Stop rows: {episode_key}")
        if stops and stops[0] is not ordered[-1]:
            errors.append(f"Stop is not the final retained frame: {episode_key}")
        if stops:
            stats["episodes_with_stop"] += 1
        else:
            stats["navigation_only_episodes"] += 1

    actors_without_stop = {
        actor: sorted(causes)
        for actor, causes in navigation_actor_causes.items()
        if actor not in stop_actors
    }
    viewpoint_cause = "trajectory_viewpoint_distance_or_occlusion"
    for actor, causes in actors_without_stop.items():
        actor_text = f"{actor[0]}::{actor[1]}"
        if viewpoint_cause in causes and actor_text not in allowed_uncovered:
            errors.append(
                "viewpoint-only actor unexpectedly has no clear Stop coverage: "
                f"{actor_text}"
            )
    actual_uncovered = {
        f"{scene}::{object_name}" for scene, object_name in actors_without_stop
    }
    stale_allowlist = sorted(allowed_uncovered - actual_uncovered)
    if stale_allowlist:
        errors.append(
            "allowed uncovered actors are not actually uncovered: "
            + ", ".join(stale_allowlist)
        )

    report = {
        "format": "uavon_stop_visible_v4_validation",
        "frames": str(args.frames.resolve()),
        "quarantine": str(args.quarantine.resolve()),
        "valid": not errors,
        "rows": stats["rows"],
        "episodes": len(episodes),
        "quarantined_trajectories": len(quarantined),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "navigation_only_actors": len(navigation_actor_causes),
        "allowed_uncovered_actors": sorted(allowed_uncovered),
        "navigation_only_actors_without_stop_coverage": {
            f"{scene}::{object_name}": causes
            for (scene, object_name), causes in sorted(actors_without_stop.items())
        },
        "error_count": len(errors),
        "errors": errors[:100],
        "errors_truncated": max(0, len(errors) - 100),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

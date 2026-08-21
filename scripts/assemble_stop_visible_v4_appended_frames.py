#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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


def load_append_specs(
    stop_bank: Path,
    queue_manifests: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    queue_by_capture_key: dict[str, dict[str, Any]] = {}
    for path in queue_manifests:
        for row in read_jsonl(path):
            key = str(row["trajectory_key"])
            if key in queue_by_capture_key:
                raise ValueError(f"duplicate queue capture key: {key}")
            queue_by_capture_key[key] = row

    append_specs: dict[str, dict[str, Any]] = {}
    stop_rows = list(read_jsonl(stop_bank))
    for stop_row in stop_rows:
        visibility = stop_row.get("stop_visibility") or {}
        capture_key = str(visibility.get("source_trajectory_key") or "")
        queue_row = queue_by_capture_key.get(capture_key)
        if queue_row is None:
            raise KeyError(f"Stop bank row has no queue entry: {capture_key}")
        represented = [str(key) for key in queue_row["represented_trajectory_keys"]]
        expected_count = int(queue_row["represented_trajectory_count"])
        if len(represented) != expected_count:
            raise ValueError(
                f"represented trajectory count mismatch for {capture_key}: "
                f"{len(represented)} != {expected_count}"
            )
        for episode_key in represented:
            if episode_key in append_specs:
                raise ValueError(f"episode covered by multiple Stop rows: {episode_key}")
            append_specs[episode_key] = {
                "stop_row": stop_row,
                "queue_row": queue_row,
            }

    metadata = {
        "unique_standoff_stop_images": len(stop_rows),
        "represented_episode_count": len(append_specs),
        "queue_capture_keys": len(queue_by_capture_key),
    }
    return append_specs, metadata


def build_appended_stop(
    episode_rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(episode_rows, key=lambda row: int(row["frame_idx"]))
    if any(str(row.get("action_name") or "").lower() == "stop" for row in ordered):
        raise ValueError(
            f"refusing to append a second Stop to {ordered[0]['episode_key']}"
        )
    previous = ordered[-1]
    template = copy.deepcopy(spec["stop_row"])
    queue_row = spec["queue_row"]
    source_visibility = template.get("stop_visibility") or {}
    source_repair = previous.get("trajectory_repair") or {}
    next_frame_idx = max(int(row["frame_idx"]) for row in ordered) + 1

    template.update(
        {
            "episode_key": str(previous["episode_key"]),
            "scene_id": str(previous["scene_id"]),
            "episode_id": str(previous["episode_id"]),
            "pose_idx": str(previous["pose_idx"]),
            "frame_idx": next_frame_idx,
            "target_description": str(previous["target_description"]),
            "true_name": str(previous.get("true_name") or template.get("true_name") or ""),
            "object_name": str(
                previous.get("object_name") or template.get("object_name") or ""
            ),
            "size": str(previous.get("size") or template.get("size") or ""),
            "action_name": "Stop",
            "action_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "coordinate_repair": previous.get("coordinate_repair"),
            "coordinate_repair_start_recovery": previous.get(
                "coordinate_repair_start_recovery"
            ),
            "trajectory_repair": {
                "version": "stop_visible_v4_production_appended",
                "mode": "navigation_only_with_appended_standoff_stop",
                "removed_invalid_stop": bool(source_repair.get("removed_invalid_stop")),
                "problem_cause": source_repair.get("problem_cause"),
                "appended_after_frame_idx": int(previous["frame_idx"]),
            },
            "stop_visibility": {
                **source_visibility,
                "version": "v4_production_appended",
                "source_type": "target_facing_standoff_appended_to_original_episode",
                "capture_group": str(queue_row["capture_group"]),
                "source_capture_trajectory_key": str(queue_row["trajectory_key"]),
                "source_stop_bank_episode_key": str(spec["stop_row"]["episode_key"]),
                "represented_trajectory_count": int(
                    queue_row["represented_trajectory_count"]
                ),
                "appended_to_original_trajectory": True,
                "not_appended_to_original_trajectory": False,
                "recaptured_pose_is_not_physical_next_pose": True,
            },
        }
    )
    return template


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append each selected target-facing standoff Stop to every original "
            "episode represented by that actor-level capture."
        )
    )
    parser.add_argument("--base-frames", type=Path, required=True)
    parser.add_argument("--stop-bank", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--source-frames",
        type=Path,
        help="Original frame rows used to report the size of quarantined episodes.",
    )
    parser.add_argument("--base-quarantine", type=Path)
    parser.add_argument("--quarantine-output", type=Path)
    parser.add_argument(
        "--quarantine-missing-base-episodes",
        action="store_true",
        help=(
            "Quarantine represented episodes with no retained collision-safe base "
            "rows instead of failing assembly."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    quarantine_args = (args.base_quarantine, args.quarantine_output)
    if any(quarantine_args) and not all(quarantine_args):
        raise ValueError(
            "--base-quarantine and --quarantine-output must be provided together"
        )
    if args.quarantine_missing_base_episodes and args.quarantine_output is None:
        raise ValueError(
            "--quarantine-missing-base-episodes requires --quarantine-output"
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

    append_specs, append_metadata = load_append_specs(
        args.stop_bank,
        args.queue_manifest,
    )
    consumed_specs: set[str] = set()
    closed_episodes: set[str] = set()
    sample_keys: set[str] = set()
    episode_keys: set[str] = set()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    def write_row(output, row: dict[str, Any], source: str) -> None:
        key = sample_key(row)
        if key in sample_keys:
            raise ValueError(f"duplicate assembled sample key: {key}")
        sample_keys.add(key)
        episode_keys.add(str(row["episode_key"]))
        action_counts[str(row["action_name"])] += 1
        scene_counts[str(row["scene_id"])] += 1
        stats["rows"] += 1
        stats[f"rows_{source}"] += 1
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush_episode(output, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        episode_key = str(rows[0]["episode_key"])
        if episode_key in closed_episodes:
            raise ValueError(f"base frames are not grouped by episode: {episode_key}")
        closed_episodes.add(episode_key)
        ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
        for row in ordered:
            write_row(output, row, "repaired_expert")
        spec = append_specs.get(episode_key)
        if spec is not None:
            appended = build_appended_stop(ordered, spec)
            write_row(output, appended, "appended_standoff_stop")
            consumed_specs.add(episode_key)
            group = str(spec["queue_row"]["capture_group"])
            stats[f"appended_{group}"] += 1

    with args.output.open("x", encoding="utf-8") as output:
        current_key = None
        current_rows: list[dict[str, Any]] = []
        for row in read_jsonl(args.base_frames):
            episode_key = str(row["episode_key"])
            if current_key is not None and episode_key != current_key:
                flush_episode(output, current_rows)
                current_rows = []
            current_key = episode_key
            current_rows.append(row)
        flush_episode(output, current_rows)

    missing_specs = sorted(set(append_specs) - consumed_specs)
    if missing_specs and not args.quarantine_missing_base_episodes:
        args.output.unlink(missing_ok=True)
        raise ValueError(
            f"{len(missing_specs)} represented episodes had no retained base rows; "
            f"examples: {missing_specs[:10]}"
        )

    source_row_counts: Counter[str] = Counter()
    if args.source_frames:
        for row in read_jsonl(args.source_frames):
            source_row_counts[str(row["episode_key"])] += 1

    quarantine_stats: Counter[str] = Counter()
    if args.quarantine_output:
        quarantine_keys: set[str] = set()
        with args.quarantine_output.open("x", encoding="utf-8") as output:
            for row in read_jsonl(args.base_quarantine):
                key = str(row["trajectory_key"])
                if key in quarantine_keys:
                    raise ValueError(f"duplicate base quarantine key: {key}")
                quarantine_keys.add(key)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                quarantine_stats["base_rows_copied"] += 1
            for episode_key in missing_specs:
                if episode_key in quarantine_keys:
                    raise ValueError(
                        f"missing-base episode already quarantined: {episode_key}"
                    )
                queue_row = append_specs[episode_key]["queue_row"]
                parts = episode_key.split("::")
                if len(parts) != 3:
                    raise ValueError(f"invalid episode key: {episode_key}")
                row = {
                    "trajectory_key": episode_key,
                    "reason": "no_retained_collision_safe_prefix",
                    "row_count": int(source_row_counts[episode_key]),
                    "retained_row_count": 0,
                    "scene_id": parts[0],
                    "episode_id": parts[1],
                    "pose_idx": parts[2],
                    "object_name": str(queue_row.get("object_name") or ""),
                    "true_name": str(queue_row.get("true_name") or ""),
                    "size": str(queue_row.get("size") or ""),
                    "capture_group": str(queue_row.get("capture_group") or ""),
                    "repair_decision": "quarantine_entire_trajectory",
                    "append_policy": (
                        "standoff_stop_not_emitted_without_original_safe_prefix"
                    ),
                }
                quarantine_keys.add(episode_key)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                quarantine_stats["missing_safe_prefix_rows_added"] += 1

    manifest = {
        "format": "uavon_stop_visible_v4_production_frames_appended_standoff",
        "base_frames": str(args.base_frames.resolve()),
        "stop_bank": str(args.stop_bank.resolve()),
        "queue_manifests": [str(path.resolve()) for path in args.queue_manifest],
        "output": str(args.output.resolve()),
        "rows": stats["rows"],
        "episodes": len(episode_keys),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "append_metadata": append_metadata,
        "append_results": {
            "represented_episodes_requested": len(append_specs),
            "represented_episodes_appended": len(consumed_specs),
            "represented_episodes_quarantined_no_safe_prefix": len(missing_specs),
            "quarantined_episode_examples": missing_specs[:20],
        },
        "quarantine": (
            {
                "base": str(args.base_quarantine.resolve()),
                "output": str(args.quarantine_output.resolve()),
                "stats": dict(quarantine_stats),
            }
            if args.quarantine_output
            else None
        ),
        "preserve_all_real_motion_before_selected_stop": True,
        "standoff_stops_are_independent_episodes": False,
        "standoff_stop_append_policy": (
            "One actor-level recaptured Stop image is cloned into every represented "
            "original episode and assigned max(retained frame_idx)+1."
        ),
        "sequence_disclosure": (
            "The appended target-facing pose was recaptured independently and is not "
            "guaranteed to be the physical next pose after the preceding expert frame."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

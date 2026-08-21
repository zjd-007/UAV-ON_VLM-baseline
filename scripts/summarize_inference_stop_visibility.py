#!/usr/bin/env python3
"""Summarize replayed inference Stop visibility by seen split and target size."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUP_ORDER = [
    "seen_small",
    "seen_mid",
    "seen_big",
    "unseen_small",
    "unseen_mid",
    "unseen_big",
]

CAPTURE_COUNT_FIELDS = [
    "capture_errors",
    "camera_mismatches",
    "valid_camera_frames",
    "target_present",
    "target_clear",
    "segmentation_ambiguous",
    "target_absent",
    "canonical_present",
    "fallback_present",
    "valid_success_frames",
    "success_target_present",
    "success_target_clear",
    "success_target_absent",
    "success_segmentation_ambiguous",
    "success_camera_mismatch",
    "success_capture_error",
]


def percentage(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}%"


def load_captures(audit_dir: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for path in sorted((audit_dir / "captures").glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    rows_by_key[str(row["capture_key"])] = row
    return list(rows_by_key.values())


def group_key(row: dict[str, Any]) -> str:
    return f"{row['seen_group']}_{row['size_bucket']}"


def target_center_view_relation(row: dict[str, Any]) -> str | None:
    camera = row.get("saved_camera_position")
    quaternion = row.get("saved_camera_quat_xyzw")
    targets = row.get("target_positions") or []
    if not camera or not quaternion or not targets:
        return None
    target = min(
        targets,
        key=lambda point: sum((point[index] - camera[index]) ** 2 for index in range(3)),
    )
    world = [target[index] - camera[index] for index in range(3)]
    x, y, z, w = quaternion
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    local = [
        sum(rotation[world_index][local_index] * world[world_index] for world_index in range(3))
        for local_index in range(3)
    ]
    horizontal = math.degrees(math.atan2(local[1], local[0]))
    vertical = math.degrees(math.atan2(local[2], local[0]))
    if local[0] <= 0:
        return "behind_camera"
    if abs(horizontal) > 45:
        return "outside_horizontal_fov"
    if abs(vertical) > 45:
        return "outside_vertical_fov"
    return "center_in_nominal_fov"


def aggregate(
    rows: list[dict[str, Any]],
    totals: dict[str, dict[str, dict[str, int]]],
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row["run_label"]), group_key(row))].append(row)

    output = []
    for run_label in sorted(totals):
        for key in GROUP_ORDER:
            base = totals[run_label].get(key) or {}
            group_rows = by_group.get((run_label, key), [])
            counts = Counter()
            counts["episodes"] = int(base.get("episodes", 0))
            counts["nominal_successes"] = int(base.get("successes", 0))
            counts["oracle_successes"] = int(base.get("oracle_successes", 0))
            counts["stop_frames"] = int(base.get("stops", len(group_rows)))
            counts["successful_stops"] = int(
                base.get("successful_stops", sum(bool(row.get("success")) for row in group_rows))
            )
            for field in CAPTURE_COUNT_FIELDS:
                counts[field] += 0
            for row in group_rows:
                ok = row.get("status") == "ok"
                camera_match = ok and bool(row.get("camera_pose_match"))
                present = camera_match and bool(row.get("target_present"))
                ambiguous = (
                    camera_match
                    and not present
                    and bool(row.get("segmentation_ambiguous"))
                )
                absent = camera_match and not present and not ambiguous
                clear = present and bool(row.get("geometry_clear"))
                success = bool(row.get("success"))
                counts["capture_errors"] += int(not ok)
                counts["camera_mismatches"] += int(ok and not camera_match)
                counts["valid_camera_frames"] += int(camera_match)
                counts["target_present"] += int(present)
                counts["target_clear"] += int(clear)
                counts["segmentation_ambiguous"] += int(ambiguous)
                counts["target_absent"] += int(absent)
                counts["canonical_present"] += int(
                    present and row.get("mask_source") == "canonical_id42"
                )
                counts["fallback_present"] += int(
                    present
                    and row.get("mask_source")
                    == "dominant_changed_color_fallback"
                )
                if success:
                    counts["valid_success_frames"] += int(camera_match)
                    counts["success_target_present"] += int(present)
                    counts["success_target_clear"] += int(clear)
                    counts["success_target_absent"] += int(absent)
                    counts["success_segmentation_ambiguous"] += int(ambiguous)
                    counts["success_camera_mismatch"] += int(ok and not camera_match)
                    counts["success_capture_error"] += int(not ok)

            episodes = counts["episodes"]
            valid = counts["valid_camera_frames"]
            valid_definitive = valid - counts["segmentation_ambiguous"]
            valid_success = counts["valid_success_frames"]
            definitive_success = (
                valid_success - counts["success_segmentation_ambiguous"]
            )
            row = {
                "run_label": run_label,
                "group": key,
                **dict(counts),
                "nominal_sr_pct": percentage(counts["nominal_successes"], episodes),
                "osr_pct": percentage(counts["oracle_successes"], episodes),
                "visual_present_sr_pct": percentage(
                    counts["success_target_present"], episodes
                ),
                "visual_clear_sr_pct": percentage(
                    counts["success_target_clear"], episodes
                ),
                "stop_target_present_pct": percentage(
                    counts["target_present"], valid_definitive
                ),
                "success_target_present_pct": percentage(
                    counts["success_target_present"], definitive_success
                ),
                "success_target_clear_pct": percentage(
                    counts["success_target_clear"], definitive_success
                ),
                "metric_only_success_pct": percentage(
                    counts["success_target_absent"], definitive_success
                ),
                "stop_precision_pct": percentage(
                    counts["successful_stops"], counts["stop_frames"]
                ),
            }
            output.append(row)
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Inference Stop-frame target visibility",
        "",
        "`visual_present_sr` counts only metric-successful Stop frames whose replayed target mask is present.",
        "`visual_clear_sr` additionally requires the size-aware geometry threshold.",
        "Ambiguous segmentation and camera mismatches are reported separately and are not treated as absent.",
        "",
        "| Run | Group | N | Nominal success | Nominal SR | Visible success | Visual-present SR | Clear success | Visual-clear SR | Successful Stop target-present | Metric-only success | Ambiguous success | Camera mismatch success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_label} | {group} | {episodes} | {nominal_successes} | {nominal_sr} | "
            "{success_target_present} | {visual_present_sr} | {success_target_clear} | "
            "{visual_clear_sr} | {success_present_rate} | {metric_only_rate} | "
            "{ambiguous} | {camera_mismatch} |".format(
                **row,
                nominal_sr=format_rate(row["nominal_sr_pct"]),
                visual_present_sr=format_rate(row["visual_present_sr_pct"]),
                visual_clear_sr=format_rate(row["visual_clear_sr_pct"]),
                success_present_rate=format_rate(row["success_target_present_pct"]),
                metric_only_rate=format_rate(row["metric_only_success_pct"]),
                ambiguous=row["success_segmentation_ambiguous"],
                camera_mismatch=row["success_camera_mismatch"],
            )
        )
    lines.append("")
    lines.append("## Capture quality")
    lines.append("")
    lines.append(
        "| Run | Group | Stop frames | Valid camera | Present | Clear | Ambiguous | Absent | Canonical | Fallback | Errors | Camera mismatch |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {run_label} | {group} | {stop_frames} | {valid_camera_frames} | "
            "{target_present} | {target_clear} | {segmentation_ambiguous} | "
            "{target_absent} | {canonical_present} | {fallback_present} | "
            "{capture_errors} | {camera_mismatches} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_review_cases(rows: list[dict[str, Any]], audit_dir: Path) -> None:
    cases = []
    metric_only = []
    for row in rows:
        if not bool(row.get("success")):
            continue
        if row.get("status") != "ok":
            visibility = "capture_error"
        elif not bool(row.get("camera_pose_match")):
            visibility = "camera_mismatch"
        elif bool(row.get("target_present")):
            visibility = "clear" if bool(row.get("geometry_clear")) else "present_weak"
        elif bool(row.get("segmentation_ambiguous")):
            visibility = "segmentation_ambiguous"
        else:
            visibility = "target_absent"
        debug = row.get("debug") or {}
        case = {
            "run_label": row.get("run_label"),
            "scene_id": row.get("scene_id"),
            "episode_id": row.get("episode_id"),
            "seen_group": row.get("seen_group"),
            "size_bucket": row.get("size_bucket"),
            "target_name": row.get("true_name") or row.get("object_name"),
            "stop_step": row.get("stop_step"),
            "distance_to_target_m": row.get("distance_to_target_m"),
            "target_center_view_relation": target_center_view_relation(row),
            "visibility": visibility,
            "mask_pixels": (row.get("mask") or {}).get("pixel_count"),
            "mask_source": row.get("mask_source"),
            "source_image_path": row.get("source_image_path"),
            "source_box_path": debug.get("source_box_path"),
            "replay_box_path": debug.get("replay_box_path"),
            "mask_path": debug.get("mask_path"),
        }
        cases.append(case)
        if visibility == "target_absent":
            metric_only.append(row)

    case_path = audit_dir / "successful_stop_visibility_cases.csv"
    with case_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(cases[0]) if cases else [])
        if cases:
            writer.writeheader()
            writer.writerows(cases)
    with (audit_dir / "metric_only_successes.jsonl").open("w", encoding="utf-8") as output:
        for row in metric_only:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_manual_adjudication(
    rows: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    if not path.is_file():
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = {str(case["capture_key"]): case for case in payload.get("cases", [])}
    adjusted = []
    for source in rows:
        row = dict(source)
        override = overrides.get(str(row["capture_key"]))
        if override:
            row["target_present"] = bool(override["target_present"])
            row["geometry_clear"] = bool(override["target_clear"])
            row["segmentation_ambiguous"] = False
            row["manual_adjudicated"] = True
        adjusted.append(row)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    captures = load_captures(audit_dir)
    rows = aggregate(captures, manifest["group_totals"])
    write_csv(rows, audit_dir / "visibility_by_seen_size.csv")
    write_markdown(rows, audit_dir / "visibility_summary.md")
    write_success_review_cases(captures, audit_dir)
    (audit_dir / "visibility_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    adjusted_captures = apply_manual_adjudication(
        captures, audit_dir / "manual_adjudication.json"
    )
    if adjusted_captures is not captures:
        adjusted = aggregate(adjusted_captures, manifest["group_totals"])
        write_csv(adjusted, audit_dir / "visibility_by_seen_size_adjudicated.csv")
        write_markdown(adjusted, audit_dir / "visibility_summary_adjudicated.md")
        (audit_dir / "visibility_summary_adjudicated.json").write_text(
            json.dumps(adjusted, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

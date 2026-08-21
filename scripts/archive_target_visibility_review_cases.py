#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_AUDIT_DIR = (
    DATASET_ROOT
    / "processed"
    / "stop_visible_full_audit"
    / "full_canonical_geometry_v1_20260812_153000"
)
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs" / "stop_visible_v4_completeness_balance_policy.json"
)
DEFAULT_PRIOR_SMOKE_ROOT = DATASET_ROOT / "processed" / "stop_visible_v2_smoke"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_baseline.stop_visibility import (  # noqa: E402
    VisibilityPolicy,
    select_first_clear_frame,
)


CATEGORY_DESCRIPTIONS = {
    "01_valid_stop_control": (
        "The original Stop is clear and is also the policy-selected recognition view."
    ),
    "02_valid_stop_better_earlier": (
        "The original Stop passes the visibility threshold, but an earlier frame has "
        "better framing or recognition-view quality."
    ),
    "03_zero_pixel_stop_earlier_clear": (
        "The original Stop contains zero target pixels, while an earlier frame is clear."
    ),
    "04_clipped_stop_earlier_clear": (
        "The original Stop is too close or severely clipped, while an earlier frame is clear."
    ),
    "05_weak_stop_earlier_clear": (
        "The original Stop has target pixels but is too small or thin, while an earlier "
        "frame is clear."
    ),
    "06_path_viewpoint_or_occlusion": (
        "No expert-path frame is clear because of viewpoint, distance, or occlusion."
    ),
    "07_target_facing_standoff_repairable": (
        "No expert-path frame is clear, but a target-facing standoff capture is clear."
    ),
    "08_standoff_pixels_below_threshold": (
        "Target pixels exist in target-facing standoff captures, but none meets the clear "
        "threshold."
    ),
    "09_dataset_simulator_coordinate_mismatch": (
        "The dataset target coordinate and the current simulator actor coordinate differ "
        "by more than 20 meters."
    ),
    "10_actor_expected_but_no_pixels": (
        "The actor is near the expected target coordinate, but neither path replay nor "
        "standoff replay produces target-instance pixels."
    ),
}


CAUSE_TO_CATEGORY = {
    "trajectory_viewpoint_distance_or_occlusion": "06_path_viewpoint_or_occlusion",
    "target_facing_standoff_repairable": "07_target_facing_standoff_repairable",
    "target_pixels_exist_but_below_clear_threshold_after_standoff": (
        "08_standoff_pixels_below_threshold"
    ),
    "dataset_simulator_xy_coordinate_mismatch": (
        "09_dataset_simulator_coordinate_mismatch"
    ),
    "target_actor_at_expected_xy_but_no_pixels_in_path_or_standoff": (
        "10_actor_expected_but_no_pixels"
    ),
}


PRIORITY_KEYS = [
    "CabinLake::354::0",
    "BrushifyUrban::0::0",
    "CityPark::905::0",
    "CityPark::152::0",
    "Venice::87::0",
    "Venice::232::0",
    "WesternTown::328::0",
    "WinterTown::576::0",
    "WinterTown::715::0",
    "DownTown::89::0",
    "WesternTown::735::0",
    "Neighborhood::393::0",
    "Neighborhood::450::0",
    "Neighborhood::730::0",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_policy(path: Path) -> VisibilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = VisibilityPolicy(**payload.get("policy", payload))
    return replace(
        policy,
        semantic_score_field=None,
        min_semantic_score=None,
        semantic_rank_field=None,
        max_semantic_rank=None,
        require_semantic_for_weak_geometry=False,
    )


def size_bucket(size: str | None) -> str:
    normalized = str(size or "").strip().lower()
    for value in ("small", "mid", "big"):
        if normalized.startswith(value):
            return value
    return "unknown"


def safe_name(value: str) -> str:
    value = value.replace("::", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def load_audit_data(
    audit_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, str],
]:
    rows = []
    actors: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sorted(audit_dir.glob("*.jsonl")):
        source_rows = read_jsonl(source)
        if source.name.endswith("_actors.jsonl"):
            for row in source_rows:
                actors[(str(row.get("scene_id")), str(row.get("object_name")))] = row
        else:
            rows.extend(source_rows)

    alignments: dict[tuple[str, str], dict[str, Any]] = {}
    alignment_dir = audit_dir / "actor_pose_alignment"
    for source in sorted(alignment_dir.glob("*.jsonl")):
        for row in read_jsonl(source):
            alignments[(str(row.get("scene_id")), str(row.get("object_name")))] = row

    causes = {}
    problem_path = audit_dir / "summary_collision_filtered" / "problem_trajectories.jsonl"
    for row in read_jsonl(problem_path):
        causes[str(row["trajectory_key"])] = str(row.get("refined_problem_cause") or "")
    return rows, actors, alignments, causes


def classify_case(
    row: dict[str, Any],
    selection: dict[str, Any],
    refined_cause: str,
) -> str:
    stop_idx = int(row["original_stop_frame_idx"])
    assessment_by_idx = {
        int(value["frame_idx"]): value for value in selection["assessments"]
    }
    stop = assessment_by_idx.get(stop_idx) or {}
    selected_idx = selection.get("selected_frame_idx")
    if bool(row.get("original_stop_clear_geometry")):
        if selected_idx == stop_idx:
            return "01_valid_stop_control"
        return "02_valid_stop_better_earlier"
    if int(row.get("clear_geometry_frame_count", 0)) > 0:
        if int(stop.get("pixel_count", 0)) == 0:
            return "03_zero_pixel_stop_earlier_clear"
        if "severely_clipped_or_too_close" in (stop.get("reasons") or []):
            return "04_clipped_stop_earlier_clear"
        return "05_weak_stop_earlier_clear"
    return CAUSE_TO_CATEGORY.get(refined_cause, "06_path_viewpoint_or_occlusion")


def representative_value(row: dict[str, Any]) -> float:
    return math.log1p(max(0, int(row.get("peak_target_pixels", 0))))


def select_representatives(
    candidates: list[dict[str, Any]],
    count: int,
    priority_keys: list[str],
) -> list[dict[str, Any]]:
    if len(candidates) <= count:
        return sorted(candidates, key=lambda row: str(row["trajectory_key"]))
    by_key = {str(row["trajectory_key"]): row for row in candidates}
    selected = [by_key[key] for key in priority_keys if key in by_key][:count]
    selected_keys = {str(row["trajectory_key"]) for row in selected}
    used_scenes = {str(row.get("scene_id")) for row in selected}
    used_targets = {str(row.get("true_name")) for row in selected}
    used_sizes = {size_bucket(row.get("size")) for row in selected}
    center = median(representative_value(row) for row in candidates)
    spread = max(
        1.0,
        max(abs(representative_value(row) - center) for row in candidates),
    )
    while len(selected) < count:
        best = None
        best_score = None
        for row in candidates:
            key = str(row["trajectory_key"])
            if key in selected_keys:
                continue
            scene = str(row.get("scene_id"))
            target = str(row.get("true_name"))
            bucket = size_bucket(row.get("size"))
            typicality = 1.0 - abs(representative_value(row) - center) / spread
            score = (
                5.0 * float(scene not in used_scenes)
                + 3.0 * float(target not in used_targets)
                + 2.0 * float(bucket not in used_sizes)
                + typicality
            )
            tie = (score, -int(row.get("total_frame_count", 0)), key)
            if best_score is None or tie > best_score:
                best = row
                best_score = tie
        if best is None:
            break
        selected.append(best)
        selected_keys.add(str(best["trajectory_key"]))
        used_scenes.add(str(best.get("scene_id")))
        used_targets.add(str(best.get("true_name")))
        used_sizes.add(size_bucket(best.get("size")))
    return selected


def source_frames(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_path = Path(str(row["source_record"]))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    camera_rows = payload.get("image_dict", {}).get("uav_on_0", [])
    actions = payload.get("action_type") or []
    poses = payload.get("record_list") or []
    if len(camera_rows) != len(poses):
        raise ValueError(
            f"record/image mismatch for {source_path}: {len(poses)} != {len(camera_rows)}"
        )
    aligned_root = source_path.parents[3]
    result = []
    for frame_idx, image_row in enumerate(camera_rows):
        image_path = aligned_root / "images" / str(row["scene_id"]) / str(image_row["rgb"])
        result.append(
            {
                "frame_idx": frame_idx,
                "image_path": image_path,
                "pose": poses[frame_idx],
                "action": actions[frame_idx] if frame_idx < len(actions) else "",
            }
        )
    return payload, result


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_tile(
    frame: dict[str, Any],
    audit_frame: dict[str, Any] | None,
    assessment: dict[str, Any] | None,
    stop_idx: int,
    selected_idx: int | None,
    peak_idx: int | None,
    tile_size: int,
) -> Image.Image:
    image_path = Path(frame["image_path"])
    if image_path.is_file():
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
    else:
        source = Image.new("RGB", (tile_size, tile_size), (215, 215, 215))
        ImageDraw.Draw(source).text((8, 8), "missing image", fill="black", font=font())
    source_width, source_height = source.size
    image = source.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    metrics = (audit_frame or {}).get("mask") or {}
    bbox = metrics.get("bbox")
    if bbox:
        mask_width = max(1, int(metrics.get("width", source_width)))
        mask_height = max(1, int(metrics.get("height", source_height)))
        x0, y0, x1, y1 = bbox
        scaled = [
            int(x0 / mask_width * tile_size),
            int(y0 / mask_height * tile_size),
            int((x1 + 1) / mask_width * tile_size),
            int((y1 + 1) / mask_height * tile_size),
        ]
        draw.rectangle(scaled, outline=(255, 220, 0), width=3)

    frame_idx = int(frame["frame_idx"])
    markers = []
    if assessment and assessment.get("clear"):
        markers.append((0, 190, 210))
    if frame_idx == peak_idx:
        markers.append((40, 110, 255))
    if frame_idx == selected_idx:
        markers.append((20, 190, 80))
    if frame_idx == stop_idx:
        markers.append((230, 40, 40))
    for offset, color in enumerate(markers):
        inset = offset * 4
        draw.rectangle(
            [inset, inset, tile_size - 1 - inset, tile_size - 1 - inset],
            outline=color,
            width=4,
        )

    caption_height = 62
    tile = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    tile.paste(image, (0, 0))
    caption = ImageDraw.Draw(tile)
    labels = []
    if frame_idx == stop_idx:
        labels.append("STOP")
    if frame_idx == selected_idx:
        labels.append("BEST")
    if frame_idx == peak_idx:
        labels.append("PEAK")
    if assessment and assessment.get("clear"):
        labels.append("CLEAR")
    marker_text = "/".join(labels) or "-"
    if audit_frame:
        distance = f"{float(audit_frame['distance_to_target']):.1f}m"
        pixels = str(int(metrics.get("pixel_count", 0)))
        short_side = min(
            int(metrics.get("bbox_width", 0)), int(metrics.get("bbox_height", 0))
        )
        quality = f"{float((assessment or {}).get('quality_score', 0.0)):.2f}"
        verdict = "clear" if assessment and assessment.get("clear") else "reject"
    else:
        distance, pixels, short_side, quality, verdict = "out>20m", "NA", 0, "NA", "not-audited"
    caption.text(
        (4, tile_size + 3),
        f"f{frame_idx:03d} {marker_text} d={distance}",
        fill="black",
        font=font(),
    )
    caption.text(
        (4, tile_size + 20),
        f"px={pixels} short={short_side} q={quality} {verdict}",
        fill="black",
        font=font(),
    )
    caption.text(
        (4, tile_size + 39),
        f"label={str(frame.get('action') or '-')[:28]}",
        fill="black",
        font=font(),
    )
    return tile


def build_sheet(
    row: dict[str, Any],
    full_frames: list[dict[str, Any]],
    selection: dict[str, Any],
    output_path: Path,
    tile_size: int,
    columns: int,
) -> None:
    audit_by_idx = {int(frame["frame_idx"]): frame for frame in row.get("frames") or []}
    assessment_by_idx = {
        int(value["frame_idx"]): value for value in selection.get("assessments") or []
    }
    peak_idx = None
    if audit_by_idx:
        peak_idx = max(
            audit_by_idx,
            key=lambda idx: int((audit_by_idx[idx].get("mask") or {}).get("pixel_count", 0)),
        )
    tiles = [
        draw_tile(
            frame,
            audit_by_idx.get(int(frame["frame_idx"])),
            assessment_by_idx.get(int(frame["frame_idx"])),
            int(row["original_stop_frame_idx"]),
            selection.get("selected_frame_idx"),
            peak_idx,
            tile_size,
        )
        for frame in full_frames
    ]
    rows = max(1, (len(tiles) + columns - 1) // columns)
    title_height = 62
    tile_height = tile_size + 62
    sheet = Image.new(
        "RGB",
        (columns * tile_size, title_height + rows * tile_height),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{row['trajectory_key']} | target={row.get('true_name')} | "
        f"size={size_bucket(row.get('size'))} | stop={row['original_stop_frame_idx']} | "
        f"best={selection.get('selected_frame_idx')}"
    )
    draw.text((8, 8), title, fill="black", font=font())
    draw.text(
        (8, 28),
        "red=original Stop, green=best view, cyan=clear, blue=peak pixels, yellow=mask bbox",
        fill="black",
        font=font(),
    )
    draw.text(
        (8, 45),
        "Frames marked out>20m were not replay-audited; inspect their RGB manually.",
        fill="black",
        font=font(),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_size
        y = title_height + (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_frame_csv(
    output_path: Path,
    row: dict[str, Any],
    full_frames: list[dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    audit_by_idx = {int(frame["frame_idx"]): frame for frame in row.get("frames") or []}
    assessment_by_idx = {
        int(value["frame_idx"]): value for value in selection.get("assessments") or []
    }
    fields = [
        "frame_idx",
        "action_label",
        "pose",
        "source_image",
        "within_20m_audit",
        "distance_to_target_m",
        "target_pixels",
        "bbox",
        "bbox_short_side",
        "clear_geometry",
        "reject_reasons",
        "quality_score",
        "is_original_stop",
        "is_selected_best_view",
        "review_target_visible",
        "review_target_recognizable",
        "review_preferred_stop",
        "review_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for frame in full_frames:
            idx = int(frame["frame_idx"])
            audit = audit_by_idx.get(idx)
            assessment = assessment_by_idx.get(idx)
            metrics = (audit or {}).get("mask") or {}
            writer.writerow(
                {
                    "frame_idx": idx,
                    "action_label": frame.get("action", ""),
                    "pose": json.dumps(frame.get("pose"), ensure_ascii=False),
                    "source_image": str(Path(frame["image_path"]).resolve()),
                    "within_20m_audit": int(audit is not None),
                    "distance_to_target_m": (
                        f"{float(audit['distance_to_target']):.6f}" if audit else ""
                    ),
                    "target_pixels": metrics.get("pixel_count", ""),
                    "bbox": json.dumps(metrics.get("bbox")),
                    "bbox_short_side": (
                        min(
                            int(metrics.get("bbox_width", 0)),
                            int(metrics.get("bbox_height", 0)),
                        )
                        if audit
                        else ""
                    ),
                    "clear_geometry": (
                        int(bool(assessment.get("clear"))) if assessment else ""
                    ),
                    "reject_reasons": (
                        "|".join(assessment.get("reasons") or []) if assessment else ""
                    ),
                    "quality_score": (
                        f"{float(assessment.get('quality_score', 0.0)):.6f}"
                        if assessment
                        else ""
                    ),
                    "is_original_stop": int(idx == int(row["original_stop_frame_idx"])),
                    "is_selected_best_view": int(
                        idx == selection.get("selected_frame_idx")
                    ),
                }
            )


def prior_sheet_map(root: Path | None) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    if root is None or not root.is_dir():
        return result
    for path in sorted(root.glob("**/sheets/*.jpg")):
        key = path.stem.replace("__", "::")
        result[key].append(path)
    return result


def archive_case(
    row: dict[str, Any],
    category: str,
    selection: dict[str, Any],
    actor: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
    prior_sheets: list[Path],
    output_dir: Path,
    tile_size: int,
    columns: int,
) -> dict[str, Any]:
    target_name = safe_name(str(row.get("true_name") or "target"))
    case_name = f"{safe_name(str(row['trajectory_key']))}__{target_name}"
    case_dir = output_dir / "cases" / category / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    source_payload, full_frames = source_frames(row)
    frame_links = case_dir / "all_frames"
    for frame in full_frames:
        source = Path(frame["image_path"])
        destination = frame_links / f"frame_{int(frame['frame_idx']):05d}{source.suffix}"
        relative_symlink(source, destination)

    sheet_path = case_dir / "trajectory_sheet.jpg"
    build_sheet(row, full_frames, selection, sheet_path, tile_size, columns)
    write_frame_csv(case_dir / "frames.csv", row, full_frames, selection)

    prior_links = []
    for index, source in enumerate(prior_sheets, start=1):
        parent_name = safe_name(source.parent.parent.name)
        destination = case_dir / "prior_smoke_sheets" / f"{index:02d}_{parent_name}.jpg"
        relative_symlink(source, destination)
        prior_links.append(str(destination.relative_to(case_dir)))

    stop_idx = int(row["original_stop_frame_idx"])
    assessment_by_idx = {
        int(value["frame_idx"]): value for value in selection.get("assessments") or []
    }
    stop_assessment = assessment_by_idx.get(stop_idx) or {}
    metadata = {
        "category": category,
        "category_description": CATEGORY_DESCRIPTIONS[category],
        "trajectory_key": row["trajectory_key"],
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "pose_idx": row.get("pose_idx"),
        "true_name": row.get("true_name"),
        "object_name": row.get("object_name"),
        "target_description": row.get("target_description"),
        "size": row.get("size"),
        "source_record": row.get("source_record"),
        "target_positions": row.get("target_positions"),
        "total_frame_count": row.get("total_frame_count"),
        "original_stop_frame_idx": stop_idx,
        "original_stop_clear_geometry": row.get("original_stop_clear_geometry"),
        "original_stop_assessment": stop_assessment,
        "selected_best_view_frame_idx": selection.get("selected_frame_idx"),
        "selected_best_view_quality": selection.get("selected_quality_score"),
        "peak_target_pixels": row.get("peak_target_pixels"),
        "path_status": row.get("path_status"),
        "no_target_cause": row.get("no_target_cause"),
        "actor_audit": actor,
        "actor_pose_alignment": alignment,
        "prior_smoke_sheets": prior_links,
        "source_record_payload": source_payload,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "category": category,
        "trajectory_key": str(row["trajectory_key"]),
        "scene_id": str(row.get("scene_id")),
        "target": str(row.get("true_name")),
        "object_name": str(row.get("object_name")),
        "size": str(row.get("size")),
        "size_bucket": size_bucket(row.get("size")),
        "total_frames": int(row.get("total_frame_count", 0)),
        "original_stop_frame": stop_idx,
        "original_stop_pixels": int(stop_assessment.get("pixel_count", 0)),
        "original_stop_clear": int(bool(row.get("original_stop_clear_geometry"))),
        "selected_best_frame": selection.get("selected_frame_idx"),
        "peak_target_pixels": int(row.get("peak_target_pixels", 0)),
        "path_status": str(row.get("path_status")),
        "case_dir": str(case_dir.resolve()),
        "sheet": str(sheet_path.resolve()),
        "prior_smoke_sheet_count": len(prior_links),
    }


def write_reports(
    output_dir: Path,
    report_rows: list[dict[str, Any]],
    category_counts: Counter[str],
    audit_dir: Path,
    policy_path: Path,
) -> None:
    index_fields = list(report_rows[0])
    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=index_fields)
        writer.writeheader()
        writer.writerows(report_rows)

    review_fields = [
        "category",
        "trajectory_key",
        "scene_id",
        "target",
        "review_category_correct",
        "review_original_stop_contains_target",
        "review_original_stop_recognizable",
        "review_best_frame_recognizable",
        "review_preferred_stop_frame",
        "review_keep_relabel_recollect_delete",
        "review_notes",
    ]
    with (output_dir / "manual_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=review_fields)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({name: row.get(name, "") for name in review_fields})

    summary = {
        "audit_dir": str(audit_dir.resolve()),
        "policy": str(policy_path.resolve()),
        "selected_case_count": len(report_rows),
        "selected_by_category": dict(Counter(row["category"] for row in report_rows)),
        "full_population_by_category": dict(category_counts),
        "categories": CATEGORY_DESCRIPTIONS,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme_lines = [
        "# Target visibility manual review cases",
        "",
        "This package is a stratified sample from the completed full AirSim audit.",
        "It does not modify any training record or source image.",
        "",
        "## How to review",
        "",
        "1. Open `index.html` for a visual overview.",
        "2. Open a case's `trajectory_sheet.jpg` and inspect the full trajectory.",
        "3. Use `all_frames/` for the original-resolution RGB images.",
        "4. Fill `manual_review.csv`; use each case's `frames.csv` for frame-level notes.",
        "5. Check `prior_smoke_sheets/` when present for earlier synchronized/standoff tests.",
        "",
        "## Colors",
        "",
        "- Red: original Stop frame",
        "- Green: policy-selected best recognition view",
        "- Cyan: frame passes the geometry-only clear threshold",
        "- Blue: maximum target-pixel frame",
        "- Yellow: replay-derived target-instance bounding box",
        "",
        "Frames farther than 20m are included as RGB but were not instance-mask audited.",
        "A replay-derived box may differ slightly from source RGB for dynamic actors; verify RGB manually.",
        "",
        "## Categories",
        "",
    ]
    for category, description in CATEGORY_DESCRIPTIONS.items():
        selected = sum(row["category"] == category for row in report_rows)
        readme_lines.append(
            f"- `{category}`: {description} Population={category_counts[category]}, "
            f"archived={selected}."
        )
    (output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )

    category_html = []
    for category, description in CATEGORY_DESCRIPTIONS.items():
        rows = [row for row in report_rows if row["category"] == category]
        cases_html = []
        for row in rows:
            sheet = Path(row["sheet"]).relative_to(output_dir.resolve())
            case_dir = Path(row["case_dir"]).relative_to(output_dir.resolve())
            cases_html.append(
                "<article>"
                f"<h3>{html.escape(row['trajectory_key'])} | "
                f"{html.escape(row['target'])} | {html.escape(row['size_bucket'])}</h3>"
                f"<p>Stop f{row['original_stop_frame']}, pixels={row['original_stop_pixels']}; "
                f"best={row['selected_best_frame']}; peak={row['peak_target_pixels']}. "
                f"<a href=\"{html.escape(str(case_dir / 'metadata.json'))}\">metadata</a> | "
                f"<a href=\"{html.escape(str(case_dir / 'frames.csv'))}\">frames.csv</a> | "
                f"<a href=\"{html.escape(str(case_dir / 'all_frames'))}/\">all frames</a></p>"
                f"<a href=\"{html.escape(str(sheet))}\"><img src=\"{html.escape(str(sheet))}\" loading=\"lazy\"></a>"
                "</article>"
            )
        category_html.append(
            f"<section><h2>{html.escape(category)}</h2>"
            f"<p>{html.escape(description)} Population={category_counts[category]}, "
            f"archived={len(rows)}.</p>{''.join(cases_html)}</section>"
        )
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Target visibility review</title>"
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa;color:#222}"
        "section{margin:0 0 64px}article{margin:24px 0 40px;padding-top:8px;border-top:1px solid #bbb}"
        "img{max-width:100%;height:auto;border:1px solid #888}h2{position:sticky;top:0;background:#fafafa;padding:8px 0}"
        "h3{font-size:17px;margin-bottom:4px}p{color:#444}a{color:#075ea8}</style>"
        "<h1>Target visibility manual review</h1>"
        "<p>Red=Stop, green=best view, cyan=clear, blue=peak pixels, yellow=target bbox.</p>"
        + "".join(category_html),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive representative target-visibility cases for human review."
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-category", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=200)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--prior-smoke-root", type=Path, default=DEFAULT_PRIOR_SMOKE_ROOT)
    args = parser.parse_args()
    if args.cases_per_category <= 0:
        raise ValueError("--cases-per-category must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.policy)
    rows, actors, alignments, causes = load_audit_data(args.audit_dir)
    enriched = []
    category_counts: Counter[str] = Counter()
    for row in rows:
        selection = select_first_clear_frame(row.get("frames") or [], row.get("size"), policy)
        category = classify_case(row, selection, causes.get(str(row["trajectory_key"]), ""))
        if category not in CATEGORY_DESCRIPTIONS:
            continue
        value = {**row, "_selection": selection, "_category": category}
        enriched.append(value)
        category_counts[category] += 1

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_category[str(row["_category"])].append(row)
    selected = []
    for category in CATEGORY_DESCRIPTIONS:
        selected.extend(
            select_representatives(
                by_category[category],
                args.cases_per_category,
                PRIORITY_KEYS,
            )
        )

    prior_map = prior_sheet_map(args.prior_smoke_root)
    report_rows = []
    for row in selected:
        actor_key = (str(row.get("scene_id")), str(row.get("object_name")))
        report_rows.append(
            archive_case(
                row,
                str(row["_category"]),
                row["_selection"],
                actors.get(actor_key),
                alignments.get(actor_key),
                prior_map.get(str(row["trajectory_key"]), []),
                args.output_dir,
                args.tile_size,
                args.columns,
            )
        )
    report_rows.sort(key=lambda row: (row["category"], row["trajectory_key"]))
    write_reports(args.output_dir, report_rows, category_counts, args.audit_dir, args.policy)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "index_html": str((args.output_dir / "index.html").resolve()),
                "manual_review_csv": str((args.output_dir / "manual_review.csv").resolve()),
                "selected_case_count": len(report_rows),
                "selected_by_category": dict(
                    Counter(row["category"] for row in report_rows)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

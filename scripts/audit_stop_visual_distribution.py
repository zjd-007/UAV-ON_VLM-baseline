#!/usr/bin/env python3
"""Archive small-target Stop examples and summarize Stop visual statistics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT

CATEGORY_GROUPS = {
    "unchanged_existing_stop": [
        "clear_candidate_stop_unchanged",
        "neighborhood_coordinate_repaired__clear_candidate_stop_unchanged",
    ],
    "reselected_existing_path": [
        "clear_candidate_stop_reselected",
        "neighborhood_coordinate_repaired__clear_candidate_stop_reselected",
    ],
    "direct_standoff_recaptured": [
        "target_facing_repairable_appended",
        "below_threshold_rescue_appended",
        "neighborhood_coordinate_repaired__below_threshold_rescue_appended",
    ],
    "actor_bank_reused_appended": ["actor_stop_bank_appended"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def size_bucket(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    for bucket in ("small", "mid", "big"):
        if normalized.startswith(bucket):
            return bucket
    return "unknown"


def safe_name(value: Any) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in str(value)
    )
    cleaned = cleaned.strip("_.")
    return cleaned or "unknown"


def load_font(size: int):
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def valid_mask_annotation(record: dict[str, Any], role: str) -> bool:
    annotation = record.get(f"{role}_mask_annotation") or {}
    return bool(annotation.get("mask_bbox")) and int(
        annotation.get("mask_pixel_count") or 0
    ) > 0


def choose_diverse_examples(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if size_bucket(row.get("size")) == "small"
        and valid_mask_annotation(row, "previous")
        and valid_mask_annotation(row, "stop")
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("true_name") or ""),
            str(row.get("scene_id") or ""),
            str(row.get("episode_id") or ""),
            str(row.get("pose_idx") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    used_episode_keys: set[str] = set()
    used_stop_images: set[str] = set()
    used_targets: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        episode_key = str(row["episode_key"])
        stop_image = str(row.get("stop_image") or "")
        if episode_key in used_episode_keys or stop_image in used_stop_images:
            return False
        selected.append(row)
        used_episode_keys.add(episode_key)
        used_stop_images.add(stop_image)
        used_targets.add(str(row.get("true_name") or ""))
        return True

    for row in candidates:
        if str(row.get("true_name") or "") in used_targets:
            continue
        add(row)
        if len(selected) == limit:
            return selected
    for row in candidates:
        add(row)
        if len(selected) == limit:
            return selected
    return selected


def make_pair_sheet(previous_path: Path, stop_path: Path, output_path: Path) -> None:
    previous = Image.open(previous_path).convert("RGB")
    stop = Image.open(stop_path).convert("RGB")
    height = max(previous.height, stop.height)
    header_height = 34
    canvas = Image.new("RGB", (previous.width + stop.width, height + header_height), "black")
    canvas.paste(previous, (0, header_height))
    canvas.paste(stop, (previous.width, header_height))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    draw.text((10, 7), "PREVIOUS", fill="white", font=font)
    draw.text((previous.width + 10, 7), "STOP", fill="white", font=font)
    canvas.save(output_path, format="JPEG", quality=92, optimize=True)


def archive_examples(
    audit_dir: Path, output_dir: Path, examples_per_group: int
) -> dict[str, Any]:
    archive_dir = output_dir / "small_target_examples"
    archive_dir.mkdir()
    all_manifest_rows: list[dict[str, Any]] = []
    group_summaries: dict[str, Any] = {}

    for group, category_names in CATEGORY_GROUPS.items():
        source_rows = [
            row
            for category in category_names
            for row in read_jsonl(audit_dir / f"{category}.jsonl")
        ]
        selected = choose_diverse_examples(source_rows, examples_per_group)
        if len(selected) < examples_per_group:
            raise RuntimeError(
                f"{group}: only {len(selected)} valid small-target examples"
            )
        group_dir = archive_dir / group
        group_dir.mkdir()
        manifest_rows: list[dict[str, Any]] = []
        for index, source_row in enumerate(selected, start=1):
            target = safe_name(source_row.get("true_name"))
            scene = safe_name(source_row.get("scene_id"))
            episode = safe_name(source_row.get("episode_id"))
            pose = safe_name(source_row.get("pose_idx"))
            prefix = f"{index:02d}__{target}__{scene}_{episode}_{pose}"
            previous_source = audit_dir / str(source_row["archived_previous_image"])
            stop_source = audit_dir / str(source_row["archived_stop_image"])
            previous_output = group_dir / f"{prefix}__previous.jpg"
            stop_output = group_dir / f"{prefix}__stop.jpg"
            pair_output = group_dir / f"{prefix}__pair.jpg"
            shutil.copy2(previous_source, previous_output)
            shutil.copy2(stop_source, stop_output)
            make_pair_sheet(previous_output, stop_output, pair_output)
            row = dict(source_row)
            row["processing_group"] = group
            row["archived_previous"] = previous_output.relative_to(output_dir).as_posix()
            row["archived_stop"] = stop_output.relative_to(output_dir).as_posix()
            row["archived_pair"] = pair_output.relative_to(output_dir).as_posix()
            manifest_rows.append(row)
            all_manifest_rows.append(row)
        write_jsonl(group_dir / "manifest.jsonl", manifest_rows)
        group_summaries[group] = {
            "examples": len(manifest_rows),
            "targets": dict(Counter(str(row["true_name"]) for row in manifest_rows)),
            "scenes": dict(Counter(str(row["scene_id"]) for row in manifest_rows)),
            "source_categories": dict(
                Counter(str(row["repair_category"]) for row in manifest_rows)
            ),
        }

    write_jsonl(archive_dir / "manifest.jsonl", all_manifest_rows)
    cards = []
    for row in all_manifest_rows:
        pair = html.escape(row["archived_pair"])
        cards.append(
            "<article><h3>"
            + html.escape(
                f"{row['processing_group']} | {row['true_name']} | {row['episode_key']}"
            )
            + f'</h3><a href="../{pair}"><img src="../{pair}"></a>'
            + "<p>"
            + html.escape(
                f"size={row['size']} | source={row['repair_category']} | "
                f"previous_distance={row['previous_mask_annotation'].get('distance_to_target')} | "
                f"stop_distance={row['stop_mask_annotation'].get('distance_to_target')}"
            )
            + "</p></article>"
        )
    index = (
        "<!doctype html><meta charset='utf-8'><title>Small target Stop examples</title>"
        "<style>body{font-family:sans-serif;background:#181818;color:#eee;margin:24px}"
        "article{margin:0 0 28px}img{max-width:100%;height:auto}p{color:#bbb}</style>"
        "<h1>Small-target Stop processing examples</h1>"
        "<p>Green rectangles are projected target instance-mask bounds. Red translucent "
        "pixels are shown where a saved pixel mask is available.</p>"
        + "".join(cards)
    )
    (archive_dir / "index.html").write_text(index, encoding="utf-8")
    return {
        "archive_dir": str(archive_dir.resolve()),
        "total_examples": len(all_manifest_rows),
        "examples_per_group": examples_per_group,
        "groups": group_summaries,
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "p25": quantile(values, 0.25),
        "median": median(values) if values else None,
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
    }


def visual_metrics(row: dict[str, Any]) -> dict[str, Any]:
    annotation = row.get("stop_mask_annotation") or {}
    bbox = annotation.get("mask_bbox")
    pixels = int(annotation.get("mask_pixel_count") or 0)
    distance = annotation.get("distance_to_target")
    center_offset = None
    centered = None
    touches_border = None
    bbox_fraction = None
    if bbox:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        dx = (center_x - (IMAGE_WIDTH - 1) / 2.0) / ((IMAGE_WIDTH - 1) / 2.0)
        dy = (center_y - (IMAGE_HEIGHT - 1) / 2.0) / ((IMAGE_HEIGHT - 1) / 2.0)
        center_offset = math.hypot(dx, dy) / math.sqrt(2.0)
        centered = abs(dx) <= 0.25 and abs(dy) <= 0.25
        touches_border = x0 <= 0 or y0 <= 0 or x1 >= 511 or y1 >= 511
        bbox_fraction = ((x1 - x0 + 1) * (y1 - y0 + 1)) / IMAGE_AREA
    return {
        "episode_key": row.get("episode_key"),
        "size_bucket": size_bucket(row.get("size")),
        "repair_category": row.get("repair_category"),
        "mask_pixel_fraction": pixels / IMAGE_AREA,
        "bbox_area_fraction": bbox_fraction,
        "distance_to_target_m": float(distance) if distance is not None else None,
        "center_offset_normalized": center_offset,
        "centered_in_middle_half": centered,
        "touches_image_border": touches_border,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric(field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) is not None]

    centered = [row["centered_in_middle_half"] for row in rows if row.get("centered_in_middle_half") is not None]
    border = [row["touches_image_border"] for row in rows if row.get("touches_image_border") is not None]
    return {
        "stops": len(rows),
        "mask_pixel_fraction": summarize_values(numeric("mask_pixel_fraction")),
        "bbox_area_fraction": summarize_values(numeric("bbox_area_fraction")),
        "distance_to_target_m": summarize_values(numeric("distance_to_target_m")),
        "center_offset_normalized": summarize_values(numeric("center_offset_normalized")),
        "centered_in_middle_half_pct": 100.0 * sum(centered) / len(centered) if centered else None,
        "touches_image_border_pct": 100.0 * sum(border) / len(border) if border else None,
    }


def write_size_csv(path: Path, by_size: dict[str, dict[str, Any]]) -> None:
    fields = [
        "size",
        "stops",
        "mask_fraction_mean_pct",
        "mask_fraction_median_pct",
        "mask_fraction_p75_pct",
        "distance_mean_m",
        "distance_median_m",
        "distance_p75_m",
        "center_offset_mean",
        "center_offset_median",
        "centered_middle_half_pct",
        "touches_border_pct",
    ]
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for bucket in ("small", "mid", "big", "unknown"):
            summary = by_size.get(bucket)
            if not summary:
                continue
            mask = summary["mask_pixel_fraction"]
            distance = summary["distance_to_target_m"]
            center = summary["center_offset_normalized"]
            writer.writerow(
                {
                    "size": bucket,
                    "stops": summary["stops"],
                    "mask_fraction_mean_pct": 100.0 * mask["mean"],
                    "mask_fraction_median_pct": 100.0 * mask["median"],
                    "mask_fraction_p75_pct": 100.0 * mask["p75"],
                    "distance_mean_m": distance["mean"],
                    "distance_median_m": distance["median"],
                    "distance_p75_m": distance["p75"],
                    "center_offset_mean": center["mean"],
                    "center_offset_median": center["median"],
                    "centered_middle_half_pct": summary["centered_in_middle_half_pct"],
                    "touches_border_pct": summary["touches_image_border_pct"],
                }
            )


def write_markdown(path: Path, by_size: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Stop visual distribution by target size",
        "",
        "Mask area is target instance-mask pixels divided by 512x512 image pixels. ",
        "Center offset is 0 at image center and 1 at a corner, using the mask-bbox center.",
        "",
        "| Size | Stops | Mask median | Mask mean | Distance median | Distance mean | Center offset median | Centered middle half | Touches border |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in ("small", "mid", "big", "unknown"):
        summary = by_size.get(bucket)
        if not summary:
            continue
        mask = summary["mask_pixel_fraction"]
        distance = summary["distance_to_target_m"]
        center = summary["center_offset_normalized"]
        lines.append(
            f"| {bucket} | {summary['stops']} | {100 * mask['median']:.3f}% | "
            f"{100 * mask['mean']:.3f}% | {distance['median']:.2f}m | "
            f"{distance['mean']:.2f}m | {center['median']:.3f} | "
            f"{summary['centered_in_middle_half_pct']:.2f}% | "
            f"{summary['touches_image_border_pct']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_distribution(audit_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = read_jsonl(audit_dir / "stop_pairs.jsonl")
    rows = [visual_metrics(row) for row in source_rows]
    by_size_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group_size_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    category_to_group = {
        category: group
        for group, categories in CATEGORY_GROUPS.items()
        for category in categories
    }
    for source_row, metric_row in zip(source_rows, rows):
        bucket = str(metric_row["size_bucket"])
        by_size_rows[bucket].append(metric_row)
        group = category_to_group.get(str(source_row.get("repair_category")), "unknown")
        by_group_size_rows[group][bucket].append(metric_row)

    by_size = {
        bucket: summarize_metric_rows(bucket_rows)
        for bucket, bucket_rows in sorted(by_size_rows.items())
    }
    by_processing_group = {
        group: {
            bucket: summarize_metric_rows(bucket_rows)
            for bucket, bucket_rows in sorted(group_rows.items())
        }
        for group, group_rows in sorted(by_group_size_rows.items())
    }
    report = {
        "format": "stop_visual_distribution_v1",
        "source": str((audit_dir / "stop_pairs.jsonl").resolve()),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "definitions": {
            "mask_pixel_fraction": "target instance-mask pixels / 512x512 pixels",
            "center_offset_normalized": "bbox-center distance from image center; 0=center, 1=corner",
            "centered_in_middle_half": "bbox center lies within central 50% width and height",
            "touches_image_border": "mask bbox touches any image edge",
        },
        "stops": len(rows),
        "size_counts": dict(Counter(row["size_bucket"] for row in rows)),
        "by_size": by_size,
        "by_processing_group_and_size": by_processing_group,
    }
    (output_dir / "stop_visual_distribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output_dir / "stop_visual_metrics.jsonl", rows)
    write_size_csv(output_dir / "stop_visual_distribution_by_size.csv", by_size)
    write_markdown(output_dir / "stop_visual_distribution.md", by_size)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-group", type=int, default=10)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    audit_dir = dataset_dir / "stop_pair_audit_maskboxed"
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)

    archive_report = archive_examples(
        audit_dir, output_dir, args.examples_per_group
    )
    distribution_report = analyze_distribution(audit_dir, output_dir)
    manifest = {
        "format": "small_target_examples_and_stop_visual_distribution_v1",
        "dataset_dir": str(dataset_dir),
        "audit_dir": str(audit_dir.resolve()),
        "output_dir": str(output_dir),
        "archive": archive_report,
        "distribution": {
            "stops": distribution_report["stops"],
            "size_counts": distribution_report["size_counts"],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

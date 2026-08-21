#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from archive_stop_visible_pairs import read_jsonl, safe_name, write_html_pages


def load_expert_mask_lookup(
    audit_dir: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], Counter[str]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for path in sorted(audit_dir.glob("*.jsonl")):
        if path.stem.endswith("_actors"):
            continue
        for trajectory in read_jsonl(path):
            trajectory_key = str(trajectory["trajectory_key"])
            for frame in trajectory.get("frames") or []:
                key = (trajectory_key, int(frame["frame_idx"]))
                if key in lookup:
                    raise ValueError(f"duplicate expert mask frame: {key}")
                lookup[key] = {
                    "mask": frame.get("mask") or {},
                    "mask_path": frame.get("mask_path"),
                    "mask_source": "expert_replay_instance_mask_bbox",
                    "distance_to_target": frame.get("distance_to_target"),
                }
                stats["expert_mask_records"] += 1
    return lookup, stats


def load_standoff_mask_lookup(
    capture_dirs: list[Path],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    lookup: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for capture_dir in capture_dirs:
        for path in sorted(capture_dir.glob("*.jsonl")):
            for trajectory in read_jsonl(path):
                for frame in trajectory.get("frames") or []:
                    image_path = Path(str(frame["replay_image_path"]))
                    key = str(image_path.resolve())
                    lookup[key] = {
                        "mask": frame.get("mask") or {},
                        "mask_path": frame.get("mask_path"),
                        "mask_source": "standoff_instance_mask",
                        "distance_to_target": frame.get("distance_to_target"),
                    }
                    stats["standoff_mask_records"] += 1
    return lookup, stats


def load_font(size: int):
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def bounded_stem(value: str, max_length: int = 190) -> str:
    cleaned = safe_name(value) or "unknown"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: max_length - 14]}__{digest}"


def scale_bbox(mask: dict[str, Any], image_size: tuple[int, int]):
    bbox = mask.get("bbox")
    if not bbox:
        return None
    source_width = int(mask.get("width") or image_size[0])
    source_height = int(mask.get("height") or image_size[1])
    scale_x = image_size[0] / source_width
    scale_y = image_size[1] / source_height
    x0, y0, x1, y1 = bbox
    return (
        round(float(x0) * scale_x),
        round(float(y0) * scale_y),
        round(float(x1) * scale_x),
        round(float(y1) * scale_y),
    )


def annotate_image(
    source: Path,
    destination: Path,
    mask_record: dict[str, Any] | None,
    label: str,
    episode_key: str,
    frame_idx: int,
    role: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image.open(source).convert("RGBA")
    mask = (mask_record or {}).get("mask") or {}
    bbox = scale_bbox(mask, image.size)
    pixel_count = int(mask.get("pixel_count") or 0)
    mask_path_value = (mask_record or {}).get("mask_path")
    mask_path = Path(str(mask_path_value)) if mask_path_value else None
    actual_mask_overlay = False

    if mask_path is not None and mask_path.is_file() and pixel_count > 0:
        target_mask = Image.open(mask_path).convert("L")
        if target_mask.size != image.size:
            target_mask = target_mask.resize(image.size, Image.Resampling.NEAREST)
        alpha = target_mask.point(lambda value: 88 if value > 0 else 0)
        overlay = Image.new("RGBA", image.size, (255, 32, 32, 0))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
        actual_mask_overlay = True

    draw = ImageDraw.Draw(image)
    if bbox is not None:
        width = max(3, image.width // 128)
        draw.rectangle(bbox, outline=(40, 255, 70, 255), width=width)

    if mask_record is None:
        mask_status = "MASK RECORD UNAVAILABLE"
    elif bbox is None or pixel_count == 0:
        mask_status = "NO TARGET MASK"
    elif actual_mask_overlay:
        mask_status = f"MASK + BBOX, pixels={pixel_count}"
    else:
        mask_status = f"MASK BBOX, pixels={pixel_count}"

    caption_height = 68
    canvas = Image.new("RGB", (image.width, image.height + caption_height), "black")
    canvas.paste(image.convert("RGB"), (0, caption_height))
    caption = ImageDraw.Draw(canvas)
    font_main = load_font(16)
    font_small = load_font(13)
    caption.text((8, 6), f"{role.upper()} | {label}", fill="white", font=font_main)
    status_color = "#66ff7a" if bbox is not None else "#ffb347"
    caption.text(
        (8, 34),
        f"{episode_key} | f{frame_idx:05d} | {mask_status}",
        fill=status_color,
        font=font_small,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=jpeg_quality, optimize=True)
    return {
        "mask_source": (mask_record or {}).get("mask_source"),
        "mask_bbox": list(bbox) if bbox is not None else None,
        "mask_pixel_count": pixel_count,
        "actual_mask_overlay": actual_mask_overlay,
        "mask_status": mask_status,
        "distance_to_target": (mask_record or {}).get("distance_to_target"),
    }


def count_annotation(stats: Counter[str], role: str, annotation: dict[str, Any]) -> None:
    stats[f"{role}_images_written"] += 1
    if annotation.get("mask_bbox") is not None:
        stats[f"{role}_with_mask_bbox"] += 1
    elif not annotation.get("mask_source"):
        stats[f"{role}_mask_record_unavailable"] += 1
    else:
        stats[f"{role}_without_target_pixels"] += 1
    if annotation.get("actual_mask_overlay"):
        stats[f"{role}_with_pixel_mask_overlay"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render Stop/previous audit images with target instance-mask boxes and "
            "target names in filenames."
        )
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expert-mask-audit-dir", type=Path, required=True)
    parser.add_argument("--standoff-capture-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite annotated archive: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    expert_lookup, expert_stats = load_expert_mask_lookup(args.expert_mask_audit_dir)
    standoff_lookup, standoff_stats = load_standoff_mask_lookup(
        args.standoff_capture_dir
    )
    source_records = list(read_jsonl(args.source_archive / "stop_pairs.jsonl"))
    category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter(expert_stats)
    stats.update(standoff_stats)
    all_index_path = args.output_dir / "stop_pairs.jsonl"

    with all_index_path.open("x", encoding="utf-8") as all_index:
        for source_record in source_records:
            record = dict(source_record)
            episode_key = str(record["episode_key"])
            target = safe_name(str(record.get("true_name") or "unknown_target"))
            actor = safe_name(str(record.get("object_name") or "unknown_actor"))
            name_prefix = bounded_stem(f"{target}__{actor}__{safe_name(episode_key)}")
            category = str(record["repair_category"])
            pair_dir = args.output_dir / "pairs" / category
            label = f"target={record.get('true_name') or 'unknown'} | actor={record.get('object_name') or 'unknown'}"

            previous_image = record.get("previous_image")
            previous_frame_idx = record.get("previous_frame_idx")
            previous_annotation = None
            previous_destination = None
            if previous_image and previous_frame_idx is not None:
                previous_source = Path(str(previous_image))
                expert_key = (episode_key, int(previous_frame_idx))
                previous_mask = expert_lookup.get(expert_key)
                previous_destination = pair_dir / (
                    f"{name_prefix}__f{int(previous_frame_idx):05d}__previous.jpg"
                )
                previous_annotation = annotate_image(
                    previous_source,
                    previous_destination,
                    previous_mask,
                    label,
                    episode_key,
                    int(previous_frame_idx),
                    "previous",
                    args.jpeg_quality,
                )
                count_annotation(stats, "previous", previous_annotation)

            stop_source = Path(str(record["stop_image"]))
            stop_frame_idx = int(record["stop_frame_idx"])
            stop_mask = standoff_lookup.get(str(stop_source.resolve()))
            stop_mask_frame_idx = stop_frame_idx
            stop_mask_episode_key = episode_key
            if stop_mask is None:
                source_stop_episode_key = record.get("source_stop_episode_key")
                source_stop_mask_frame_idx = record.get("source_stop_mask_frame_idx")
                source_original_frame_idx = record.get("source_original_frame_idx")
                if source_stop_episode_key is not None:
                    stop_mask_episode_key = str(source_stop_episode_key)
                if source_stop_mask_frame_idx is not None:
                    stop_mask_frame_idx = int(source_stop_mask_frame_idx)
                elif source_original_frame_idx is not None:
                    stop_mask_frame_idx = int(source_original_frame_idx)
                stop_mask = expert_lookup.get(
                    (stop_mask_episode_key, stop_mask_frame_idx)
                )
            stop_destination = pair_dir / (
                f"{name_prefix}__f{stop_frame_idx:05d}__stop.jpg"
            )
            stop_annotation = annotate_image(
                stop_source,
                stop_destination,
                stop_mask,
                label,
                episode_key,
                stop_frame_idx,
                "stop",
                args.jpeg_quality,
            )
            count_annotation(stats, "stop", stop_annotation)

            record["archived_previous_image"] = (
                previous_destination.relative_to(args.output_dir).as_posix()
                if previous_destination is not None
                else None
            )
            record["archived_stop_image"] = stop_destination.relative_to(
                args.output_dir
            ).as_posix()
            record["previous_mask_annotation"] = previous_annotation
            record["stop_mask_annotation"] = stop_annotation
            record["stop_mask_frame_idx"] = stop_mask_frame_idx
            record["stop_mask_episode_key"] = stop_mask_episode_key
            all_index.write(json.dumps(record, ensure_ascii=False) + "\n")
            category_records[category].append(record)

    category_indexes = {}
    for category, records in sorted(category_records.items()):
        path = args.output_dir / f"{category}.jsonl"
        with path.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        category_indexes[category] = str(path.resolve())

    pages = write_html_pages(
        args.output_dir,
        category_records,
        args.page_size,
        image_note=(
            "Every image is a rendered audit copy. Green rectangles are target "
            "instance-mask bounding boxes. Target-facing recaptures also include a "
            "translucent red pixel-mask overlay. Filenames contain target and actor names."
        ),
    )
    summary = {
        "format": "stop_visible_v4_mask_box_annotated_pair_audit",
        "source_archive": str(args.source_archive.resolve()),
        "expert_mask_audit_dir": str(args.expert_mask_audit_dir.resolve()),
        "standoff_capture_dirs": [
            str(path.resolve()) for path in args.standoff_capture_dir
        ],
        "output_dir": str(args.output_dir.resolve()),
        "image_format": "JPEG",
        "jpeg_quality": args.jpeg_quality,
        "filename_format": (
            "{true_name}__{object_name}__{scene_episode_pose}__f{frame_idx}__"
            "{previous|stop}.jpg"
        ),
        "records": len(source_records),
        "category_counts": {
            category: len(records)
            for category, records in sorted(category_records.items())
        },
        "stats": dict(stats),
        "category_indexes": category_indexes,
        "html_pages": pages,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

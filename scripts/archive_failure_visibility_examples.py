#!/usr/bin/env python3
"""Archive representative inference-failure visibility cases for review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=4)
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_captures(capture_dir: Path) -> dict[str, dict]:
    captures: dict[str, dict] = {}
    for path in sorted(capture_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                captures[row["episode_key"]] = row
    return captures


def representative_order(rows: list[dict]) -> list[dict]:
    """Prefer size/scene diversity, then cases near the class median."""
    peaks = sorted(as_int(row.get("peak_target_pixels"), 0) for row in rows)
    median_peak = peaks[len(peaks) // 2] if peaks else 0

    def rank(row: dict) -> tuple:
        peak = as_int(row.get("peak_target_pixels"), 0)
        return (abs(peak - median_peak), -as_int(row.get("audited_frame_count"), 0), row["episode_key"])

    remaining = sorted(rows, key=rank)
    ordered: list[dict] = []
    used_keys: set[str] = set()
    for size in ("small", "mid", "big"):
        match = next((row for row in remaining if row.get("size_bucket") == size), None)
        if match:
            ordered.append(match)
            used_keys.add(match["episode_key"])

    used_scenes = {row.get("scene_id") for row in ordered}
    for row in remaining:
        if row["episode_key"] in used_keys or row.get("scene_id") in used_scenes:
            continue
        ordered.append(row)
        used_keys.add(row["episode_key"])
        used_scenes.add(row.get("scene_id"))
    ordered.extend(row for row in remaining if row["episode_key"] not in used_keys)
    return ordered


def choose_frames(capture: dict, max_frames: int) -> list[dict]:
    frames = capture.get("frames", [])
    by_step = {as_int(frame.get("step")): frame for frame in frames}
    priorities = [
        capture.get("first_clear_within_20m_step"),
        capture.get("first_visible_within_20m_step"),
        capture.get("best_visible_step"),
        capture.get("first_clear_step"),
        capture.get("first_visible_step"),
    ]
    stop_steps = [frame.get("step") for frame in frames if as_int(frame.get("action_id")) == 0]
    priorities.extend(stop_steps)
    if frames:
        closest = min(frames, key=lambda frame: as_float(frame.get("camera_target_distance_m"), math.inf))
        priorities.extend([closest.get("step"), frames[-1].get("step")])

    selected: list[dict] = []
    seen: set[int] = set()
    for raw_step in priorities:
        step = as_int(raw_step)
        if step < 0 or step in seen or step not in by_step:
            continue
        selected.append(by_step[step])
        seen.add(step)
        if len(selected) >= max_frames:
            break
    if not selected:
        selected = frames[:max_frames]
    return sorted(selected, key=lambda frame: as_int(frame.get("step")))


def find_lane_dir(result_dir: Path, scene: str, episode_id: str) -> Path:
    matches = list(result_dir.glob(f"lane*/temp/{scene}-{episode_id}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one result JSON for {scene}::{episode_id}, got {matches}")
    return matches[0].parents[1]


def annotate_frame(source: Path, frame: dict, target: str, destination: Path) -> Image.Image:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    mask = frame.get("mask") or {}
    bbox = mask.get("bbox")
    if bbox:
        draw.rectangle(tuple(bbox), outline=(40, 255, 70), width=4)
        x0, y0, _, _ = bbox
        label = f"TARGET: {target}"
        label_box = draw.textbbox((x0, max(0, y0 - 18)), label)
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.text((x0, max(0, y0 - 18)), label, fill=(40, 255, 70))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)
    return image


def make_sheet(title: str, panels: list[tuple[Image.Image, list[str]]], destination: Path) -> None:
    tile_w, image_h, caption_h = 512, 512, 86
    tile_h = image_h + caption_h
    cols = 2
    rows = max(1, math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * tile_w, 52 + rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 12), title, fill="black", font=font)
    for index, (image, lines) in enumerate(panels):
        x = (index % cols) * tile_w
        y = 52 + (index // cols) * tile_h
        sheet.paste(image.resize((tile_w, image_h)), (x, y))
        for line_index, line in enumerate(lines[:4]):
            draw.text((x + 8, y + image_h + 5 + line_index * 18), line, fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=94)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(csv.DictReader((args.audit_dir / "episodes.csv").open(encoding="utf-8")))
    captures = load_captures(args.audit_dir / "captures")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        category = row.get("distance_aware_failure_class") or "audit_error_actor_mapping"
        grouped[category].append(row)

    manifest: list[dict] = []
    summary: dict[str, dict] = {}
    for category, rows in sorted(grouped.items()):
        selected = representative_order(rows)[: args.per_class]
        summary[category] = {"total": len(rows), "archived": len(selected)}
        class_dir = args.output_dir / category
        for row in selected:
            key = row["episode_key"]
            capture = captures.get(key)
            if not capture:
                manifest.append({**row, "archive_status": "missing_capture", "sheet": ""})
                continue
            scene = row["scene_id"]
            episode_id = row["episode_id"]
            target = row["true_name"]
            lane_dir = find_lane_dir(args.result_dir, scene, episode_id)
            chosen = choose_frames({**capture, **row}, args.max_frames)
            stem = f"{safe_name(scene)}__{safe_name(episode_id)}__{safe_name(target)}"
            panels: list[tuple[Image.Image, list[str]]] = []
            frame_names: list[str] = []
            for frame in chosen:
                step = as_int(frame.get("step"))
                source = lane_dir / frame["image_path"]
                frame_name = f"{stem}__step_{step:03d}.jpg"
                destination = class_dir / "frames" / frame_name
                annotated = annotate_frame(source, frame, target, destination)
                mask = frame.get("mask") or {}
                distance = as_float(frame.get("camera_target_distance_m"))
                within = distance <= 20.0 if math.isfinite(distance) else False
                panels.append(
                    (
                        annotated,
                        [
                            f"step={step} action={frame.get('parsed_command')} distance={distance:.2f}m within20={within}",
                            f"mask_pixels={mask.get('pixel_count', 0)} bbox={mask.get('bbox')} clear={frame.get('geometry_clear', False)}",
                            f"target_present={frame.get('target_present', False)} size={row.get('size_bucket')}",
                            f"source={frame.get('image_path')}",
                        ],
                    )
                )
                frame_names.append(str(destination.relative_to(args.output_dir)))
            sheet = class_dir / "sheets" / f"{stem}.jpg"
            make_sheet(
                f"class={category} | episode={key} | target={target} | size={row.get('size_bucket')}",
                panels,
                sheet,
            )
            manifest.append(
                {
                    **row,
                    "archive_status": "ok",
                    "sheet": str(sheet.relative_to(args.output_dir)),
                    "archived_frames": ";".join(frame_names),
                }
            )

    fieldnames = sorted({key for row in manifest for key in row})
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = """# Failure visibility examples

Each non-empty distance-aware failure class contains up to five representative episodes.

- Green rectangle: exact bounding box of the AirSim instance-segmentation target mask.
- `mask_pixels=0`: the target actor did not project to the replayed RGB observation.
- `clear`: geometry-based recognizability threshold used by the visibility audit.
- `within20`: camera-to-target distance is no greater than 20 meters.
- Frames are replay-aligned saved inference RGB frames; no model inference is rerun here.
- Selection favors target-size and scene diversity, then cases near the class median mask area.

Open each class's `sheets/` directory for review. Per-frame annotated JPEGs are in `frames/`,
and `manifest.csv` records the complete metrics and source mapping.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "classes": summary}, indent=2))


if __name__ == "__main__":
    main()

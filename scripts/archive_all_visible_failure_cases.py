#!/usr/bin/env python3
"""Archive every target-visible failed episode into four review classes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ["outside20_weak", "outside20_clear", "within20_weak", "within20_clear"]
CLASS_DESCRIPTION = {
    "outside20_weak": "Target pixels appear only beyond 20m, but never satisfy the clear threshold.",
    "outside20_clear": "Target is clear beyond 20m, but is never visible within 20m.",
    "within20_weak": "Target pixels appear within 20m, but never satisfy the clear threshold within 20m.",
    "within20_clear": "Target satisfies the clear threshold within 20m.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sheet-frames", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-sheets-only", action="store_true")
    return parser.parse_args()


def flag(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "unknown"


def classify(row: dict) -> str | None:
    if flag(row.get("clear_target_ever_visible_within_20m")):
        return "within20_clear"
    if flag(row.get("target_ever_visible_within_20m")):
        return "within20_weak"
    if flag(row.get("clear_target_ever_visible")):
        return "outside20_clear"
    if flag(row.get("target_ever_visible")):
        return "outside20_weak"
    return None


def load_captures(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for jsonl in sorted(path.glob("*.jsonl")):
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["episode_key"]] = row
    return rows


def find_result_json(result_dir: Path, scene: str, episode_id: str) -> Path:
    matches = list(result_dir.glob(f"lane*/temp/{scene}-{episode_id}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one result for {scene}::{episode_id}, got {matches}")
    return matches[0]


def fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), 19), ImageFont.truetype(str(path), 15)
    default = ImageFont.load_default()
    return default, default


def corner_box(draw: ImageDraw.ImageDraw, bbox: list[int], image_size: tuple[int, int]) -> None:
    """Draw thin brackets outside the mask bbox so tiny targets stay visible."""
    width, height = image_size
    pad = 4
    x0 = max(0, int(bbox[0]) - pad)
    y0 = max(0, int(bbox[1]) - pad)
    x1 = min(width - 1, int(bbox[2]) + pad)
    y1 = min(height - 1, int(bbox[3]) + pad)
    arm = max(6, min(18, max(x1 - x0, y1 - y0) // 3))
    color = (20, 255, 50)
    stroke = 2
    segments = [
        ((x0, y0), (x0 + arm, y0)), ((x0, y0), (x0, y0 + arm)),
        ((x1, y0), (x1 - arm, y0)), ((x1, y0), (x1, y0 + arm)),
        ((x0, y1), (x0 + arm, y1)), ((x0, y1), (x0, y1 - arm)),
        ((x1, y1), (x1 - arm, y1)), ((x1, y1), (x1, y1 - arm)),
    ]
    for start, end in segments:
        draw.line([start, end], fill=color, width=stroke)


def annotate_visible_frame(source: Path, destination: Path, audit_frame: dict) -> None:
    image = Image.open(source).convert("RGB")
    bbox = (audit_frame.get("mask") or {}).get("bbox")
    if bbox:
        corner_box(ImageDraw.Draw(image), bbox, image.size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def choose_sheet_frames(capture: dict, row: dict, max_frames: int) -> list[dict]:
    frames = capture.get("frames", [])
    by_step = {as_int(frame.get("step")): frame for frame in frames}
    priorities = [
        row.get("first_visible_step"),
        row.get("first_clear_step"),
        row.get("first_visible_within_20m_step"),
        row.get("first_clear_within_20m_step"),
        row.get("best_visible_step"),
    ]
    priorities.extend(frame.get("step") for frame in frames if as_int(frame.get("action_id")) == 0)
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
    return sorted(selected, key=lambda frame: as_int(frame.get("step")))


def make_sheet(
    destination: Path,
    category: str,
    episode_key: str,
    target: str,
    size: str,
    description: str,
    panels: list[tuple[Path, dict]],
) -> None:
    title_font, text_font = fonts()
    tile_w, image_h, caption_h = 512, 512, 130
    header_h = 142
    cols = 2
    rows = max(1, math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * tile_w, header_h + rows * (image_h + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"class={category} | episode={episode_key} | target={target} | size={size}", fill="black", font=title_font)
    wrapped = textwrap.wrap(f"Target description: {description}", width=120)[:4]
    for index, line in enumerate(wrapped):
        draw.text((10, 38 + index * 21), line, fill="black", font=text_font)
    for index, (frame_path, frame) in enumerate(panels):
        x = (index % cols) * tile_w
        y = header_h + (index // cols) * (image_h + caption_h)
        image = Image.open(frame_path).convert("RGB").resize((tile_w, image_h))
        sheet.paste(image, (x, y))
        mask = frame.get("mask") or {}
        distance = as_float(frame.get("camera_target_distance_m"))
        lines = [
            f"step={as_int(frame.get('step'))} | action={frame.get('parsed_command')}",
            f"distance={distance:.2f}m | within20={distance <= 20.0} | clear={frame.get('geometry_clear', False)}",
            f"mask_pixels={mask.get('pixel_count', 0)} | bbox={mask.get('bbox')}",
            f"target_present={frame.get('target_present', False)}",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x + 7, y + image_h + 6 + line_index * 24), line, fill="black", font=text_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.refresh_sheets_only:
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(csv.DictReader((args.audit_dir / "episodes.csv").open(encoding="utf-8")))
    captures = load_captures(args.audit_dir / "captures")
    selected = [(classify(row), row) for row in episodes]
    selected = [(category, row) for category, row in selected if category is not None]

    manifest: list[dict] = []
    counts = {category: {"episodes": 0, "trajectory_frames": 0, "boxed_frames": 0} for category in CLASS_ORDER}
    for index, (category, row) in enumerate(selected, 1):
        key = row["episode_key"]
        capture = captures[key]
        scene = row["scene_id"]
        episode_id = row["episode_id"]
        target = row["true_name"]
        description = capture.get("target_description", "")
        result_json_path = find_result_json(args.result_dir, scene, episode_id)
        result_row = json.loads(result_json_path.read_text(encoding="utf-8"))
        lane_dir = result_json_path.parents[1]
        source_dir = lane_dir / "images" / scene / episode_id
        task_name = f"{safe_name(scene)}__{safe_name(episode_id)}__{safe_name(target)}"
        task_frames_dir = args.output_dir / category / "frames" / task_name
        audit_by_step = {as_int(frame.get("step")): frame for frame in capture.get("frames", [])}
        boxed = 0
        trajectory_frames = 0
        for source in sorted(source_dir.glob("step_*.jpg")):
            match = re.search(r"step_(\d+)", source.stem)
            if not match:
                continue
            step = int(match.group(1))
            destination = task_frames_dir / source.name
            audit_frame = audit_by_step.get(step)
            has_box = bool(
                audit_frame
                and audit_frame.get("target_present")
                and (audit_frame.get("mask") or {}).get("bbox")
            )
            if has_box:
                boxed += 1
            if not args.refresh_sheets_only:
                if has_box:
                    annotate_visible_frame(source, destination, audit_frame)
                else:
                    hardlink_or_copy(source, destination)
            trajectory_frames += 1

        sheet_frames = choose_sheet_frames(capture, row, args.max_sheet_frames)
        panels: list[tuple[Path, dict]] = []
        for frame in sheet_frames:
            step = as_int(frame.get("step"))
            frame_path = task_frames_dir / f"step_{step:03d}.jpg"
            if frame_path.exists():
                panels.append((frame_path, frame))
        sheet_path = args.output_dir / category / "sheets" / f"{task_name}.jpg"
        make_sheet(sheet_path, category, key, target, row.get("size_bucket", ""), description, panels)

        counts[category]["episodes"] += 1
        counts[category]["trajectory_frames"] += trajectory_frames
        counts[category]["boxed_frames"] += boxed
        manifest.append(
            {
                "review_class": category,
                "episode_key": key,
                "scene_id": scene,
                "episode_id": episode_id,
                "true_name": target,
                "size_bucket": row.get("size_bucket", ""),
                "target_description": description,
                "termination_reason": row.get("termination_reason", ""),
                "official_osr": row.get("official_osr", ""),
                "ever_stop_command": row.get("ever_stop_command", ""),
                "peak_target_pixels": row.get("peak_target_pixels", ""),
                "best_visible_distance_m": row.get("best_visible_distance_m", ""),
                "trajectory_frames": trajectory_frames,
                "boxed_frames": boxed,
                "frames_dir": str(task_frames_dir.relative_to(args.output_dir)),
                "sheet": str(sheet_path.relative_to(args.output_dir)),
                "source_result_json": str(result_json_path),
                "source_image_count": len(result_row.get("image_paths", [])),
            }
        )
        if index % 50 == 0:
            print(f"archived {index}/{len(selected)} episodes", flush=True)

    fields = list(manifest[0])
    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    (args.output_dir / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    readme = [
        "# Visible failure cases",
        "",
        "All target-visible failed episodes are assigned to exactly one of four classes:",
        "",
    ]
    for category in CLASS_ORDER:
        readme.append(f"- `{category}`: {CLASS_DESCRIPTION[category]}")
    readme.extend(
        [
            "",
            "Layout:",
            "",
            "- `<class>/frames/<scene>__<episode>__<target>/`: complete saved inference trajectory.",
            "- `<class>/sheets/`: selected visibility, distance, Stop, and final-context frames.",
            "- Thin green corner brackets are expanded four pixels outside the exact instance-mask bbox.",
            "  They avoid covering tiny targets; all text is outside the RGB image in the sheet.",
            "- Frames without a verified target mask are hard-linked from the original result to avoid duplication.",
            "- `manifest.csv` contains target descriptions and source paths.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

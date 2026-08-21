#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"


def iter_jsonl_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(path)


def load_cache(path: Path) -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source in iter_jsonl_files(path):
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" not in row:
                continue
            rows[str(row["trajectory_key"])] = row
    return rows


def load_selections(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["trajectory_key"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_legacy_stops(path: Path) -> dict[str, int]:
    last_indices = {}
    stops = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["episode_key"])
        frame_idx = int(row["frame_idx"])
        last_indices[key] = max(last_indices.get(key, -1), frame_idx)
        if str(row.get("action_name")) == "Stop":
            stops[key] = frame_idx
    return {key: stops.get(key, frame_idx) for key, frame_idx in last_indices.items()}


def draw_frame_tile(
    frame: dict[str, Any],
    assessment: dict[str, Any] | None,
    selected_idx: int | None,
    peak_idx: int | None,
    legacy_idx: int | None,
    tile_size: int,
) -> Image.Image:
    source = Image.open(
        frame.get("replay_image_path") or frame["image_path"]
    ).convert("RGB")
    source_width, source_height = source.size
    image = source.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    metrics = frame.get("mask") or {}
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
    marker_colors = []
    if frame_idx == legacy_idx:
        marker_colors.append((230, 40, 40))
    if frame_idx == peak_idx:
        marker_colors.append((40, 120, 255))
    if frame_idx == selected_idx:
        marker_colors.append((20, 190, 80))
    for offset, color in enumerate(marker_colors):
        draw.rectangle(
            [offset * 4, offset * 4, tile_size - 1 - offset * 4, tile_size - 1 - offset * 4],
            outline=color,
            width=4,
        )

    caption_height = 48
    tile = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    tile.paste(image, (0, 0))
    caption = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    labels = []
    if frame_idx == selected_idx:
        labels.append("AUTO")
    if frame_idx == peak_idx:
        labels.append("PEAK")
    if frame_idx == legacy_idx:
        labels.append("OLD")
    status = "/".join(labels) or "-"
    pixels = int(metrics.get("pixel_count", 0))
    short = min(int(metrics.get("bbox_width", 0)), int(metrics.get("bbox_height", 0)))
    distance = float(frame.get("distance_to_target", float("nan")))
    clear = "clear" if assessment and assessment.get("clear") else "reject"
    quality = float((assessment or {}).get("quality_score", 0.0))
    caption.text((3, tile_size + 3), f"f={frame_idx} d={distance:.1f}m {status}", fill="black", font=font)
    caption.text(
        (3, tile_size + 20),
        f"px={pixels} short={short} q={quality:.2f} {clear}",
        fill="black",
        font=font,
    )
    return tile


def build_sheet(
    cached: dict[str, Any],
    selection: dict[str, Any],
    legacy_idx: int | None,
    output_path: Path,
    tile_size: int,
    columns: int,
) -> dict[str, Any]:
    selected_idx = selection.get("selected_frame_idx")
    assessments = {
        int(row["frame_idx"]): row for row in selection.get("assessments") or []
    }
    visible_frames = [
        frame
        for frame in cached.get("frames") or []
        if int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
        or int(frame["frame_idx"]) in {selected_idx, legacy_idx}
    ]
    if not visible_frames:
        visible_frames = list(cached.get("frames") or [])[-1:]
    peak_frame = max(
        visible_frames,
        key=lambda frame: int((frame.get("mask") or {}).get("pixel_count", 0)),
    )
    peak_idx = int(peak_frame["frame_idx"])
    tiles = [
        draw_frame_tile(
            frame,
            assessments.get(int(frame["frame_idx"])),
            selected_idx,
            peak_idx,
            legacy_idx,
            tile_size,
        )
        for frame in visible_frames
    ]
    rows = (len(tiles) + columns - 1) // columns
    title_height = 44
    sheet = Image.new(
        "RGB",
        (columns * tile_size, title_height + rows * (tile_size + 48)),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{cached['trajectory_key']} | {cached.get('true_name')} | {cached.get('size')} | "
        f"auto={selected_idx}, old={legacy_idx}, peak={peak_idx}"
    )
    draw.text((8, 8), title, fill="black", font=ImageFont.load_default())
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_size
        y = title_height + (index // columns) * (tile_size + 48)
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "peak_frame_idx": peak_idx,
        "visible_frame_count": len(visible_frames),
        "sheet": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-review report for Stop visibility.")
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=180)
    parser.add_argument("--columns", type=int, default=7)
    args = parser.parse_args()

    cache = load_cache(args.visibility_cache)
    selections = load_selections(args.selections)
    legacy_stops = load_legacy_stops(args.source)
    sheets_dir = args.output_dir / "sheets"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for key, selection in sorted(selections.items()):
        cached = cache[key]
        legacy_idx = legacy_stops.get(key)
        safe_name = key.replace("::", "__")
        sheet_path = sheets_dir / f"{safe_name}.jpg"
        sheet_info = build_sheet(
            cached,
            selection,
            legacy_idx,
            sheet_path,
            args.tile_size,
            args.columns,
        )
        report_rows.append(
            {
                "trajectory_key": key,
                "target": cached.get("true_name"),
                "size": cached.get("size"),
                "auto_selected_frame": selection.get("selected_frame_idx"),
                "legacy_stop_frame": legacy_idx,
                **sheet_info,
            }
        )

    review_path = args.output_dir / "manual_review.csv"
    fieldnames = [
        "trajectory_key",
        "target",
        "size",
        "auto_selected_frame",
        "legacy_stop_frame",
        "peak_frame_idx",
        "visible_frame_count",
        "human_first_clear_frame",
        "accept_auto",
        "notes",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trajectory_count": len(report_rows),
                "visibility_cache": str(args.visibility_cache.resolve()),
                "selections": str(args.selections.resolve()),
                "manual_review": str(review_path.resolve()),
                "rows": report_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    html_rows = []
    for row in report_rows:
        relative_sheet = Path(row["sheet"]).relative_to(args.output_dir.resolve())
        html_rows.append(
            "<section>"
            f"<h2>{html.escape(row['trajectory_key'])} | {html.escape(str(row['target']))}</h2>"
            f"<p>size={html.escape(str(row['size']))}; auto={row['auto_selected_frame']}; "
            f"old={row['legacy_stop_frame']}; peak={row['peak_frame_idx']}</p>"
            f"<img src=\"{html.escape(str(relative_sheet))}\" loading=\"lazy\">"
            "</section>"
        )
    index_path = args.output_dir / "index.html"
    index_path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Stop visibility audit</title>"
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa}"
        "section{margin:0 0 36px}img{max-width:100%;height:auto;border:1px solid #aaa}"
        "h2{font-size:18px;margin-bottom:4px}p{margin-top:0;color:#444}</style>"
        "<h1>Stop visibility smoke audit</h1>"
        "<p>Green=AUTO, blue=PEAK, red=legacy Stop, yellow=target-instance bbox.</p>"
        + "".join(html_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "index": str(index_path.resolve()),
                "manual_review": str(review_path.resolve()),
                "summary": str(summary_path.resolve()),
                "trajectory_count": len(report_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

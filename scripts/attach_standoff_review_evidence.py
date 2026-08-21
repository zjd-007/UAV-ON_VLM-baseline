#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_baseline.stop_visibility import (  # noqa: E402
    VisibilityPolicy,
    select_first_clear_frame,
)


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
        reject_collided=True,
    )


def relative_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def select_sheet_frames(
    frames: list[dict[str, Any]],
    selection: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    assessments = {
        int(row["frame_idx"]): row for row in selection.get("assessments") or []
    }
    collision_free = [
        frame
        for frame in frames
        if not bool((frame.get("collision_info") or {}).get("has_collided"))
    ]
    visible = [
        frame
        for frame in collision_free
        if int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
    ]
    if visible:
        ranked = sorted(
            visible,
            key=lambda frame: (
                float(assessments.get(int(frame["frame_idx"]), {}).get("quality_score", 0.0)),
                int((frame.get("mask") or {}).get("pixel_count", 0)),
            ),
            reverse=True,
        )
        chosen = ranked[:limit]
        important_indices = {
            selection.get("selected_frame_idx"),
            max(
                visible,
                key=lambda frame: int((frame.get("mask") or {}).get("pixel_count", 0)),
            )["frame_idx"],
        }
        chosen_by_idx = {int(frame["frame_idx"]): frame for frame in chosen}
        all_by_idx = {int(frame["frame_idx"]): frame for frame in collision_free}
        for frame_idx in important_indices:
            if frame_idx is not None and int(frame_idx) in all_by_idx:
                chosen_by_idx[int(frame_idx)] = all_by_idx[int(frame_idx)]
        chosen = list(chosen_by_idx.values())
        if len(chosen) > limit:
            protected = {int(value) for value in important_indices if value is not None}
            optional = [
                frame for frame in chosen if int(frame["frame_idx"]) not in protected
            ]
            chosen = [
                frame for frame in chosen if int(frame["frame_idx"]) in protected
            ] + optional[: max(0, limit - len(protected))]
    else:
        if len(collision_free) <= limit:
            chosen = collision_free
        else:
            indices = [
                round(index * (len(collision_free) - 1) / (limit - 1))
                for index in range(limit)
            ]
            chosen = [collision_free[index] for index in indices]
    return sorted(chosen, key=lambda frame: int(frame["frame_idx"]))


def draw_tile(
    frame: dict[str, Any],
    assessment: dict[str, Any] | None,
    selected_idx: int | None,
    peak_idx: int | None,
    tile_size: int,
) -> Image.Image:
    image_path = Path(frame["replay_image_path"])
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    width, height = source.size
    image = source.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    metrics = frame.get("mask") or {}
    bbox = metrics.get("bbox")
    if bbox:
        mask_width = max(1, int(metrics.get("width", width)))
        mask_height = max(1, int(metrics.get("height", height)))
        x0, y0, x1, y1 = bbox
        draw.rectangle(
            [
                int(x0 / mask_width * tile_size),
                int(y0 / mask_height * tile_size),
                int((x1 + 1) / mask_width * tile_size),
                int((y1 + 1) / mask_height * tile_size),
            ],
            outline=(255, 220, 0),
            width=3,
        )
    frame_idx = int(frame["frame_idx"])
    markers = []
    if assessment and assessment.get("clear"):
        markers.append((0, 190, 210))
    if frame_idx == peak_idx:
        markers.append((40, 110, 255))
    if frame_idx == selected_idx:
        markers.append((20, 190, 80))
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
    font = ImageFont.load_default()
    labels = []
    if frame_idx == selected_idx:
        labels.append("BEST")
    if frame_idx == peak_idx:
        labels.append("PEAK")
    if assessment and assessment.get("clear"):
        labels.append("CLEAR")
    caption.text(
        (4, tile_size + 3),
        f"{frame.get('candidate_id')} {'/'.join(labels) or '-'} "
        f"r={float(frame.get('radius_m', 0)):.0f}m h={float(frame.get('height_offset_m', 0)):.1f}m",
        fill="black",
        font=font,
    )
    caption.text(
        (4, tile_size + 20),
        f"az={float(frame.get('azimuth_deg', 0)):.0f} px={int(metrics.get('pixel_count', 0))} "
        f"short={min(int(metrics.get('bbox_width', 0)), int(metrics.get('bbox_height', 0)))}",
        fill="black",
        font=font,
    )
    caption.text(
        (4, tile_size + 39),
        f"q={float((assessment or {}).get('quality_score', 0.0)):.2f} "
        f"{'clear' if assessment and assessment.get('clear') else 'reject'}",
        fill="black",
        font=font,
    )
    return tile


def build_sheet(
    row: dict[str, Any],
    selection: dict[str, Any],
    frames: list[dict[str, Any]],
    output_path: Path,
    tile_size: int,
    columns: int,
) -> None:
    assessments = {
        int(value["frame_idx"]): value for value in selection.get("assessments") or []
    }
    collision_free = [
        frame
        for frame in row.get("frames") or []
        if not bool((frame.get("collision_info") or {}).get("has_collided"))
    ]
    peak_idx = None
    if collision_free:
        peak_idx = int(
            max(
                collision_free,
                key=lambda frame: int((frame.get("mask") or {}).get("pixel_count", 0)),
            )["frame_idx"]
        )
    tiles = [
        draw_tile(
            frame,
            assessments.get(int(frame["frame_idx"])),
            selection.get("selected_frame_idx"),
            peak_idx,
            tile_size,
        )
        for frame in frames
    ]
    rows = max(1, (len(tiles) + columns - 1) // columns)
    title_height = 58
    tile_height = tile_size + 62
    sheet = Image.new(
        "RGB",
        (columns * tile_size, title_height + rows * tile_height),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"{row['trajectory_key']} | target={row.get('true_name')} | "
        f"standoff candidates={row.get('candidate_count')} | best={selection.get('selected_frame_idx')}",
        fill="black",
        font=ImageFont.load_default(),
    )
    draw.text(
        (8, 28),
        "green=best, cyan=clear, blue=peak pixels, yellow=target bbox",
        fill="black",
        font=ImageFont.load_default(),
    )
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((index % columns) * tile_size, title_height + (index // columns) * tile_height),
        )
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach target-facing standoff evidence to a manual-review archive."
    )
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--standoff-cache", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--sheet-frame-limit", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=200)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    policy = load_policy(args.policy)

    archive_rows = list(csv.DictReader((args.archive_dir / "index.csv").open()))
    case_by_key = {
        str(row["trajectory_key"]): Path(str(row["case_dir"])) for row in archive_rows
    }
    standoff_rows = []
    for source in sorted(args.standoff_cache.glob("*.jsonl")):
        standoff_rows.extend(read_jsonl(source))

    report = []
    html_rows = []
    for row in standoff_rows:
        key = str(row["trajectory_key"])
        case_dir = case_by_key.get(key)
        if case_dir is None:
            continue
        frames = list(row.get("frames") or [])
        selection = select_first_clear_frame(frames, row.get("size"), policy)
        sheet_frames = select_sheet_frames(frames, selection, args.sheet_frame_limit)
        sheet_path = case_dir / "standoff_sheet.jpg"
        build_sheet(
            row,
            selection,
            sheet_frames,
            sheet_path,
            args.tile_size,
            args.columns,
        )
        candidate_source = args.standoff_cache / "candidates" / key.replace("::", "__")
        relative_symlink(candidate_source, case_dir / "standoff_candidates")
        payload = {**row, "selection": selection}
        (case_dir / "standoff_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        collision_free = [
            frame
            for frame in frames
            if not bool((frame.get("collision_info") or {}).get("has_collided"))
        ]
        clear_count = sum(
            bool(value.get("clear")) for value in selection.get("assessments") or []
        )
        visible_count = sum(
            int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
            for frame in collision_free
        )
        result = {
            "trajectory_key": key,
            "target": row.get("true_name"),
            "candidate_count": len(frames),
            "collision_free_count": len(collision_free),
            "visible_count": visible_count,
            "clear_count": clear_count,
            "selected_best_frame": selection.get("selected_frame_idx"),
            "case_dir": str(case_dir.resolve()),
            "standoff_sheet": str(sheet_path.resolve()),
        }
        report.append(result)
        relative_sheet = sheet_path.resolve().relative_to(args.archive_dir.resolve())
        html_rows.append(
            "<section>"
            f"<h2>{html.escape(key)} | {html.escape(str(row.get('true_name')))}</h2>"
            f"<p>collision-free={len(collision_free)}, visible={visible_count}, "
            f"clear={clear_count}, best={selection.get('selected_frame_idx')}</p>"
            f"<a href=\"{html.escape(str(relative_sheet))}\"><img "
            f"src=\"{html.escape(str(relative_sheet))}\" loading=\"lazy\"></a>"
            "</section>"
        )

    fields = list(report[0]) if report else []
    with (args.archive_dir / "standoff_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(report)
    (args.archive_dir / "standoff_index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Standoff review</title>"
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa}"
        "section{margin:0 0 40px}img{max-width:100%;height:auto;border:1px solid #888}"
        "h2{font-size:18px;margin-bottom:4px}p{color:#444}</style>"
        "<h1>Target-facing standoff review</h1>"
        "<p>Green=best view, cyan=clear, blue=peak pixels, yellow=target bbox.</p>"
        + "".join(html_rows),
        encoding="utf-8",
    )
    summary = {
        "attached_case_count": len(report),
        "status_counts": dict(
            Counter(
                "clear_available" if int(row["clear_count"]) > 0 else "no_clear_view"
                for row in report
            )
        ),
        "rows": report,
    }
    (args.archive_dir / "standoff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

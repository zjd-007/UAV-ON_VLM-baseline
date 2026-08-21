#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_baseline.stop_visibility import (  # noqa: E402
    VisibilityPolicy,
    assess_visibility_frame,
    parse_size_bucket,
)


DEFAULT_POLICY = PROJECT_ROOT / "configs" / "stop_visible_v4_completeness_balance_policy.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def add_mask_box(image: Image.Image, metrics: dict[str, Any]) -> Image.Image:
    output = image.convert("RGB")
    bbox = metrics.get("bbox")
    if bbox and len(bbox) == 4:
        draw = ImageDraw.Draw(output)
        draw.rectangle(tuple(float(value) for value in bbox), outline=(255, 0, 0), width=4)
    return output


def make_sheet(row: dict[str, Any], destination: Path) -> None:
    visible = [
        frame
        for frame in row.get("frames") or []
        if int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
        and frame.get("replay_image_path")
    ]
    ranked = sorted(
        visible,
        key=lambda frame: int((frame.get("mask") or {}).get("pixel_count", 0)),
        reverse=True,
    )[:4]
    if not ranked:
        return
    cells = []
    for frame in ranked:
        image = Image.open(frame["replay_image_path"]).convert("RGB")
        image = add_mask_box(image, frame.get("mask") or {})
        image.thumbnail((480, 360))
        canvas = Image.new("RGB", (500, 420), "white")
        canvas.paste(image, ((500 - image.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        label = (
            f"f{frame['frame_idx']}  d={frame['distance_to_target']:.1f}m  "
            f"pixels={(frame.get('mask') or {}).get('pixel_count', 0)}"
        )
        draw.text((10, 380), label, fill="black", font=ImageFont.load_default())
        cells.append(canvas)
    sheet = Image.new("RGB", (500 * len(cells), 470), "white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, (index * 500, 50))
    ImageDraw.Draw(sheet).text(
        (10, 10),
        f"{row.get('trajectory_key')} | {row.get('true_name')} | {row.get('object_name')}",
        fill="black",
        font=ImageFont.load_default(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Neighborhood coordinate smoke.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()

    policy_payload = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = VisibilityPolicy(**policy_payload.get("policy", policy_payload))
    policy = replace(
        policy,
        semantic_score_field=None,
        min_semantic_score=None,
        semantic_rank_field=None,
        max_semantic_rank=None,
        require_semantic_for_weak_geometry=False,
    )

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = read_jsonl(args.run_dir / "sample_manifest.jsonl")
    sample_by_key = {row["trajectory_key"]: row for row in samples}
    visibility = []
    collisions_raw = []
    for lane in range(2):
        visibility.extend(read_jsonl(args.run_dir / "visibility" / f"lane{lane}" / "Neighborhood.jsonl"))
        collisions_raw.extend(
            read_jsonl(args.run_dir / "collision" / f"lane{lane}" / "Neighborhood.jsonl")
        )
    collision_by_key = {}
    for row in collisions_raw:
        collision_by_key[str(row.get("key"))] = row
    collisions = list(collision_by_key.values())
    alignment = read_jsonl(args.run_dir / "actor_alignment_after" / "Neighborhood.jsonl")

    alignment_rows = []
    for row in alignment:
        residual = row.get("actor_to_target_error_xy_m")
        alignment_rows.append({**row, "shifted_xy_residual_m": residual})

    visibility_by_status = Counter(str(row.get("status")) for row in visibility)
    visible_trajectory_count = 0
    clear_trajectory_count = 0
    total_visible_frames = 0
    for row in visibility:
        sample = sample_by_key.get(str(row.get("trajectory_key")), {})
        row.update(
            {
                "true_name": sample.get("true_name"),
                "size": sample.get("size"),
            }
        )
        frames = row.get("frames") or []
        visible = [
            frame for frame in frames if int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
        ]
        peak_pixels = max(
            (int((frame.get("mask") or {}).get("pixel_count", 0)) for frame in frames),
            default=0,
        )
        total_visible_frames += len(visible)
        visible_trajectory_count += bool(visible)
        clear_trajectory_count += any(
            assess_visibility_frame(
                frame,
                parse_size_bucket(str(sample.get("size") or "")),
                peak_pixels,
                policy,
            )["clear"]
            for frame in frames
        )
        safe_name = (
            f"{sample.get('episode_id', 'unknown')}_"
            f"{str(sample.get('true_name') or 'unknown').replace('/', '_')}.jpg"
        )
        make_sheet(row, args.run_dir / "review_sheets" / safe_name)

    checked = [row for row in collisions if not row.get("error")]
    collision_summary = {
        "rows": len(collisions),
        "raw_rows_including_retries": len(collisions_raw),
        "checked": len(checked),
        "errors": sum(bool(row.get("error")) for row in collisions),
        "initial_collisions": sum(bool(row.get("initial_collided")) for row in checked),
        "new_collisions": sum(bool(row.get("new_collision_after_action")) for row in checked),
    }
    collision_summary["initial_collision_rate"] = (
        collision_summary["initial_collisions"] / len(checked) if checked else None
    )
    collision_summary["new_collision_rate"] = (
        collision_summary["new_collisions"] / len(checked) if checked else None
    )
    residuals = [
        float(row["shifted_xy_residual_m"])
        for row in alignment_rows
        if row.get("shifted_xy_residual_m") is not None
    ]
    summary = {
        "format": "neighborhood_coordinate_xy_shift_smoke_summary_v1",
        "transform": manifest["transform"],
        "trajectory_count": len(samples),
        "actor_count": len(alignment_rows),
        "alignment": {
            "actors": len(residuals),
            "xy_residual_min_m": min(residuals) if residuals else None,
            "xy_residual_median_m": float(np.median(residuals)) if residuals else None,
            "xy_residual_max_m": max(residuals) if residuals else None,
            "within_0_1m": sum(value <= 0.1 for value in residuals),
            "within_1m": sum(value <= 1.0 for value in residuals),
        },
        "visibility": {
            "rows": len(visibility),
            "status_counts": dict(visibility_by_status),
            "trajectories_with_target_pixels": visible_trajectory_count,
            "trajectories_with_basic_clear_geometry": clear_trajectory_count,
            "visible_frames": total_visible_frames,
        },
        "collision": collision_summary,
        "review_sheets": str((args.run_dir / "review_sheets").resolve()),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.run_dir / "actor_alignment_residuals.jsonl").open("w", encoding="utf-8") as output:
        for row in alignment_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    links = []
    for sheet in sorted((args.run_dir / "review_sheets").glob("*.jpg")):
        links.append(f'<li><a href="review_sheets/{html.escape(sheet.name)}">{html.escape(sheet.name)}</a></li>')
    (args.run_dir / "index.html").write_text(
        "<html><head><meta charset=\"utf-8\"><title>Neighborhood smoke</title></head>"
        "<body><h1>Neighborhood coordinate smoke</h1><pre>"
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre><ul>"
        + "".join(links)
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

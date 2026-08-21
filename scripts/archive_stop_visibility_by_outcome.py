#!/usr/bin/env python3
"""Archive Stop-frame visibility review artifacts by metric outcome."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


DEBUG_FIELDS = (
    "source_box_path",
    "replay_box_path",
    "replay_path",
    "mask_path",
    "dominant_changed_mask_path",
    "dominant_changed_box_path",
)


def safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown").strip())
    return cleaned.strip("_") or "unknown"


def load_captures(audit_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in sorted((audit_dir / "captures").glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    by_key[str(row["capture_key"])] = row
    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row.get("run_label")),
            str(row.get("scene_id")),
            int(row.get("episode_id", 0)),
        ),
    )


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def make_sheet(
    cases: list[dict[str, Any]],
    output_path: Path,
    columns: int = 5,
) -> None:
    tile_width = 220
    image_size = 200
    caption_height = 58
    rows = (len(cases) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_size + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(cases):
        image = Image.open(case["source_box_path"]).convert("RGB")
        image = ImageOps.contain(image, (image_size, image_size))
        column = index % columns
        row = index // columns
        x = column * tile_width + (tile_width - image.width) // 2
        y = row * (image_size + caption_height)
        sheet.paste(image, (x, y))
        caption = (
            f"{case['scene_id']} / {case['episode_id']}\n"
            f"{case['target_name']} | d={case['distance_to_target_m']:.1f}m\n"
            f"present={int(case['target_present'])} clear={int(case['geometry_clear'])}"
        )
        draw.multiline_text((column * tile_width + 5, y + image_size + 2), caption, fill="black", spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sheet-page-size", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    output_dir = (args.output_dir or audit_dir / "review_by_outcome").resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rows = load_captures(audit_dir)
    adjudication_path = audit_dir / "manual_adjudication.json"
    manual_overrides: dict[str, dict[str, Any]] = {}
    if adjudication_path.is_file():
        payload = json.loads(adjudication_path.read_text(encoding="utf-8"))
        manual_overrides = {
            str(case["capture_key"]): case for case in payload.get("cases", [])
        }
    indexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sheets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()

    for row in rows:
        override = manual_overrides.get(str(row["capture_key"]))
        if override:
            row["target_present"] = bool(override["target_present"])
            row["geometry_clear"] = bool(override["target_clear"])
            row["segmentation_ambiguous"] = False
            row["manual_adjudicated"] = True
        outcome = "success" if bool(row.get("success")) else "failure"
        run_label = safe_name(row.get("run_label"))
        seen_group = safe_name(row.get("seen_group"))
        size_bucket = safe_name(row.get("size_bucket"))
        target_name = safe_name(row.get("true_name") or row.get("object_name"))
        stem = safe_name(
            f"{run_label}__{row.get('scene_id')}__{row.get('episode_id')}__"
            f"{target_name}__step{int(row.get('stop_step', 0)):03d}"
        )
        case_dir = output_dir / outcome / run_label / seen_group / size_bucket
        debug = row.get("debug") or {}
        archived: dict[str, str] = {}
        for field in DEBUG_FIELDS:
            raw_source = debug.get(field)
            if not raw_source:
                continue
            source = Path(raw_source).resolve()
            if not source.is_file():
                counts["missing_artifacts"] += 1
                continue
            suffix = source.suffix
            artifact = field[:-5] if field.endswith("_path") else field
            destination = case_dir / f"{stem}__{artifact}{suffix}"
            relative_symlink(source, destination)
            archived[field] = str(destination)

        source_box = archived.get("source_box_path")
        index_row = {
            "outcome": outcome,
            "run_label": row.get("run_label"),
            "scene_id": row.get("scene_id"),
            "episode_id": row.get("episode_id"),
            "target_name": row.get("true_name") or row.get("object_name"),
            "seen_group": row.get("seen_group"),
            "size_bucket": row.get("size_bucket"),
            "distance_to_target_m": row.get("distance_to_target_m"),
            "target_present": bool(row.get("target_present")),
            "geometry_clear": bool(row.get("geometry_clear")),
            "segmentation_ambiguous": bool(row.get("segmentation_ambiguous")),
            "manual_adjudicated": bool(row.get("manual_adjudicated")),
            "termination_reason": "stop",
            "source_image_path": row.get("source_image_path"),
            **archived,
        }
        indexes[outcome].append(index_row)
        if source_box:
            sheets[(outcome, run_label, seen_group, size_bucket)].append(
                {**index_row, "source_box_path": source_box}
            )
        counts[outcome] += 1

    for outcome, index_rows in indexes.items():
        path = output_dir / f"{outcome}_index.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)

    for key, cases in sheets.items():
        outcome, run_label, seen_group, size_bucket = key
        for page_index, offset in enumerate(range(0, len(cases), args.sheet_page_size), start=1):
            make_sheet(
                cases[offset : offset + args.sheet_page_size],
                output_dir
                / "sheets"
                / outcome
                / f"{run_label}__{seen_group}__{size_bucket}__page{page_index:03d}.jpg",
            )

    readme = (
        "# Stop visibility review by outcome\n\n"
        "`success` contains metric-successful Stop episodes (distance <= 20 m).\n"
        "`failure` contains metric-failed episodes that nevertheless terminated with Stop.\n"
        "Collision and step-limit episodes without a Stop frame are outside this audit.\n\n"
        f"Success Stop episodes: {counts['success']}\n\n"
        f"Failed Stop episodes: {counts['failure']}\n\n"
        f"Missing linked artifacts: {counts['missing_artifacts']}\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "archive_summary.json").write_text(
        json.dumps(dict(counts), indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **counts}, indent=2))


if __name__ == "__main__":
    main()

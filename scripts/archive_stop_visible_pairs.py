#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def stop_category(stop: dict[str, Any]) -> str:
    visibility = stop.get("stop_visibility") or {}
    source_type = str(visibility.get("source_type") or "")
    if source_type == "actor_stop_bank_appended_to_original_episode":
        return "actor_stop_bank_appended"
    if source_type == "target_facing_standoff_appended_to_original_episode":
        capture_group = str(visibility.get("capture_group") or "unknown")
        if capture_group == "repairable":
            category = "target_facing_repairable_appended"
            return (
                f"neighborhood_coordinate_repaired__{category}"
                if stop.get("coordinate_repair")
                else category
            )
        if capture_group == "rescue":
            category = "below_threshold_rescue_appended"
            return (
                f"neighborhood_coordinate_repaired__{category}"
                if stop.get("coordinate_repair")
                else category
            )
        category = f"standoff_{safe_name(capture_group)}_appended"
        return (
            f"neighborhood_coordinate_repaired__{category}"
            if stop.get("coordinate_repair")
            else category
        )
    original_action = str(stop.get("original_action_name") or "").lower()
    if source_type == "expert_path" and original_action == "stop":
        category = "clear_candidate_stop_unchanged"
        return (
            f"neighborhood_coordinate_repaired__{category}"
            if stop.get("coordinate_repair")
            else category
        )
    if source_type == "expert_path":
        category = "clear_candidate_stop_reselected"
        return (
            f"neighborhood_coordinate_repaired__{category}"
            if stop.get("coordinate_repair")
            else category
        )
    category = "other_stop_source"
    return (
        f"neighborhood_coordinate_repaired__{category}"
        if stop.get("coordinate_repair")
        else category
    )


def link_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination)


def write_html_pages(
    output_dir: Path,
    category_records: dict[str, list[dict[str, Any]]],
    page_size: int,
    image_note: str = (
        "Images are symbolic links to source files. Each row compares the final "
        "retained frame before Stop with the repaired Stop frame."
    ),
) -> dict[str, list[str]]:
    css = """
body { font-family: sans-serif; margin: 20px; color: #202124; }
a { color: #0757a6; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td { border: 1px solid #bbb; padding: 8px; vertical-align: top; }
th { background: #f3f4f5; position: sticky; top: 0; }
.meta { width: 24%; font-size: 13px; overflow-wrap: anywhere; }
.image { width: 38%; }
img { display: block; width: 100%; height: auto; max-height: 360px; object-fit: contain; background: #111; }
.missing { padding: 40px; background: #eee; text-align: center; }
""".strip()
    pages_by_category: dict[str, list[str]] = {}
    for category, records in sorted(category_records.items()):
        category_pages = []
        page_dir = output_dir / "pages" / category
        page_dir.mkdir(parents=True, exist_ok=True)
        for page_index, start in enumerate(range(0, len(records), page_size), start=1):
            page_records = records[start : start + page_size]
            page_path = page_dir / f"page_{page_index:04d}.html"
            rows = []
            for record in page_records:
                prev_rel = record.get("archived_previous_image")
                stop_rel = record["archived_stop_image"]
                prev_src = (
                    f'<img loading="lazy" src="../../{html.escape(prev_rel)}">'
                    if prev_rel
                    else '<div class="missing">No retained previous frame</div>'
                )
                stop_src = f'<img loading="lazy" src="../../{html.escape(stop_rel)}">'
                target = html.escape(str(record.get("true_name") or ""))
                actor = html.escape(str(record.get("object_name") or ""))
                episode = html.escape(record["episode_key"])
                original_action = html.escape(
                    str(record.get("stop_original_action") or "n/a")
                )
                previous_status = html.escape(
                    str(record.get("previous_frame_source") or "none")
                )
                rows.append(
                    "<tr>"
                    f'<td class="meta"><b>{episode}</b><br>Target: {target}<br>'
                    f'Actor: {actor}<br>Previous: f{record.get("previous_frame_idx")} '
                    f'({previous_status})<br>'
                    f'Stop: f{record["stop_frame_idx"]}<br>'
                    f'Original Stop-frame label: {original_action}</td>'
                    f'<td class="image">{prev_src}</td>'
                    f'<td class="image">{stop_src}</td>'
                    "</tr>"
                )
            previous_link = (
                f'<a href="page_{page_index - 1:04d}.html">Previous</a>'
                if page_index > 1
                else "Previous"
            )
            next_link = (
                f'<a href="page_{page_index + 1:04d}.html">Next</a>'
                if start + page_size < len(records)
                else "Next"
            )
            page_path.write_text(
                "<!doctype html><html><head><meta charset=\"utf-8\">"
                f"<title>{html.escape(category)} page {page_index}</title>"
                f"<style>{css}</style></head><body>"
                f"<h1>{html.escape(category)}</h1>"
                f"<p>{previous_link} | {next_link} | "
                '<a href="../../index.html">Summary</a></p>'
                "<table><thead><tr><th>Episode</th><th>Previous retained frame</th>"
                "<th>Stop frame</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></body></html>",
                encoding="utf-8",
            )
            relative_page = page_path.relative_to(output_dir).as_posix()
            category_pages.append(relative_page)
        pages_by_category[category] = category_pages

    summary_rows = []
    for category, records in sorted(category_records.items()):
        first_page = pages_by_category[category][0]
        summary_rows.append(
            f"<tr><td>{html.escape(category)}</td><td>{len(records)}</td>"
            f'<td><a href="{html.escape(first_page)}">Open audit pages</a></td></tr>'
        )
    (output_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Stop pair audit</title><style>{css}</style></head><body>"
        "<h1>Stop / previous-frame audit</h1>"
        f"<p>{html.escape(image_note)}</p>"
        "<table><thead><tr><th>Repair category</th><th>Episodes</th><th>Pages</th>"
        "</tr></thead><tbody>"
        + "".join(summary_rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return pages_by_category


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive the Stop and previous retained frame for every Stop episode."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit archive: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_indexes = {}
    all_index_path = args.output_dir / "stop_pairs.jsonl"
    stats: Counter[str] = Counter()
    closed_episodes: set[str] = set()

    def archive_episode(rows: list[dict[str, Any]], all_index) -> None:
        if not rows:
            return
        episode_key = str(rows[0]["episode_key"])
        if episode_key in closed_episodes:
            raise ValueError(f"frames are not grouped by episode: {episode_key}")
        closed_episodes.add(episode_key)
        ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
        stops = [
            row
            for row in ordered
            if str(row.get("action_name") or "").lower() == "stop"
        ]
        if not stops:
            stats["episodes_without_stop"] += 1
            return
        if len(stops) != 1 or stops[0] is not ordered[-1]:
            raise ValueError(f"invalid Stop placement in {episode_key}")
        stop = stops[0]
        previous = ordered[-2] if len(ordered) >= 2 else None
        category = stop_category(stop)
        stem = safe_name(episode_key)
        pair_dir = args.output_dir / "pairs" / category
        stop_source = Path(str(stop["image_path"]))
        stop_link = pair_dir / f"{stem}__stop{stop_source.suffix.lower()}"
        link_image(stop_source, stop_link)
        previous_link = None
        previous_frame_idx = int(previous["frame_idx"]) if previous else None
        previous_action = previous.get("action_name") if previous else None
        previous_image = previous.get("image_path") if previous else None
        previous_frame_source = "retained_training_row" if previous else "none"
        if previous is not None:
            previous_source = Path(str(previous["image_path"]))
            previous_link_path = (
                pair_dir / f"{stem}__previous{previous_source.suffix.lower()}"
            )
            link_image(previous_source, previous_link_path)
            previous_link = previous_link_path.relative_to(args.output_dir).as_posix()
        else:
            stop_frame_idx = int(stop["frame_idx"])
            stop_suffix = stop_source.suffix.lower()
            source_previous = stop_source.parent / f"{stop_frame_idx - 1:05d}{stop_suffix}"
            if stop_frame_idx > 0 and source_previous.is_file():
                previous_link_path = (
                    pair_dir / f"{stem}__source_previous_excluded{stop_suffix}"
                )
                link_image(source_previous, previous_link_path)
                previous_link = previous_link_path.relative_to(args.output_dir).as_posix()
                previous_frame_idx = stop_frame_idx - 1
                previous_image = str(source_previous.resolve())
                previous_frame_source = "source_frame_excluded_from_training"
                stats["source_previous_frames_added_for_audit"] += 1
            else:
                stats["stop_episodes_without_any_previous_frame"] += 1

        visibility = stop.get("stop_visibility") or {}
        record = {
            "repair_category": category,
            "episode_key": episode_key,
            "scene_id": stop["scene_id"],
            "episode_id": stop["episode_id"],
            "pose_idx": stop["pose_idx"],
            "true_name": stop.get("true_name"),
            "object_name": stop.get("object_name"),
            "size": stop.get("size"),
            "target_description": stop.get("target_description"),
            "previous_frame_idx": previous_frame_idx,
            "previous_action": previous_action,
            "previous_image": previous_image,
            "previous_frame_source": previous_frame_source,
            "stop_frame_idx": int(stop["frame_idx"]),
            "stop_image": stop["image_path"],
            "stop_original_action": stop.get("original_action_name"),
            "stop_source_type": visibility.get("source_type"),
            "capture_group": visibility.get("capture_group"),
            "quality_score": visibility.get("quality_score"),
            "source_candidate_frame_idx": visibility.get(
                "source_candidate_frame_idx"
            ),
            "source_original_frame_idx": visibility.get(
                "source_original_frame_idx", stop.get("original_frame_idx")
            ),
            "source_stop_episode_key": visibility.get("source_stop_episode_key"),
            "source_stop_frame_idx": visibility.get("source_stop_frame_idx"),
            "source_stop_mask_frame_idx": visibility.get(
                "source_stop_mask_frame_idx"
            ),
            "source_stop_source_type": visibility.get("source_stop_source_type"),
            "actor_stop_bank_id": visibility.get("actor_stop_bank_id"),
            "actor_stop_bank_candidate_index": visibility.get(
                "actor_stop_bank_candidate_index"
            ),
            "recaptured_pose_is_not_physical_next_pose": visibility.get(
                "recaptured_pose_is_not_physical_next_pose", False
            ),
            "archived_previous_image": previous_link,
            "archived_stop_image": stop_link.relative_to(args.output_dir).as_posix(),
        }
        all_index.write(json.dumps(record, ensure_ascii=False) + "\n")
        category_records[category].append(record)
        stats["episodes_with_stop"] += 1
        stats[f"category_{category}"] += 1

    with all_index_path.open("x", encoding="utf-8") as all_index:
        current_key = None
        current_rows: list[dict[str, Any]] = []
        for row in read_jsonl(args.frames):
            episode_key = str(row["episode_key"])
            if current_key is not None and episode_key != current_key:
                archive_episode(current_rows, all_index)
                current_rows = []
            current_key = episode_key
            current_rows.append(row)
        archive_episode(current_rows, all_index)

    for category, records in sorted(category_records.items()):
        path = args.output_dir / f"{category}.jsonl"
        with path.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        category_indexes[category] = str(path.resolve())

    pages = write_html_pages(args.output_dir, category_records, args.page_size)
    summary = {
        "format": "stop_visible_v4_stop_previous_pair_audit",
        "frames": str(args.frames.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "link_mode": "symbolic_link",
        "previous_frame_definition": (
            "Previous retained training row in frame_idx order; if none exists and "
            "Stop frame_idx > 0, source frame_idx-1 is linked for audit only."
        ),
        "stats": dict(stats),
        "category_counts": {
            category: len(records)
            for category, records in sorted(category_records.items())
        },
        "category_indexes": category_indexes,
        "html_pages": pages,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

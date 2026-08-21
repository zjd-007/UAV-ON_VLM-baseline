#!/usr/bin/env python3
"""Archive successful UAV-ON episodes for visual and trajectory review."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from archive_osr_only_tasks import (
    first_position,
    make_keyframe_sheet,
    make_trajectory_plot,
    relative_symlink,
    safe_name,
    trajectory_stats,
    write_task_markdown,
    write_trajectory_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive successful UAV-ON episodes for manual analysis."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-keyframes", type=int, default=12)
    return parser.parse_args()


def episode_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    episode = str(row.get("episode_id", ""))
    try:
        numeric = int(episode)
    except ValueError:
        numeric = 10**12
    return str(row.get("map_name", "")), numeric, episode


def load_success_rows(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("lane*/temp/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("acc", 0)) != 1:
            continue
        key = str(row.get("map_name")), str(row.get("episode_id"))
        if key in rows:
            raise ValueError(f"Duplicate successful episode {key}: {rows[key][0]} and {path}")
        rows[key] = path, row
    return sorted(rows.values(), key=lambda item: episode_sort_key(item[1]))


def write_index_html(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        task_dir = html.escape(str(row["task_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['scene']))}</td>"
            f"<td>{html.escape(str(row['episode_id']))}</td>"
            f"<td>{html.escape(str(row['true_name']))}</td>"
            f"<td>{html.escape(str(row['description']))}</td>"
            f"<td>{html.escape(str(row['size']))}</td>"
            f"<td>{row['used_in_train']}</td>"
            f"<td>{row['steps']}</td>"
            f"<td>{row['first_reach_step']}</td>"
            f"<td>{row['minimum_distance']:.2f}</td>"
            f"<td>{row['final_distance']:.2f}</td>"
            f'<td><a href="{task_dir}/task.md">summary</a> | '
            f'<a href="{task_dir}/trajectory_xy_distance.png">path</a> | '
            f'<a href="{task_dir}/keyframes.jpg">frames</a> | '
            f'<a href="{task_dir}/all_frames/">all</a></td>'
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Successful task archive</title>
<style>
body {{ font-family: sans-serif; margin: 24px; color: #1d242b; }}
input {{ width: min(680px, 90vw); padding: 8px; margin: 8px 0 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ccd2d8; padding: 6px; vertical-align: top; }}
th {{ background: #eef2f5; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f8fafb; }}
</style>
</head>
<body>
<h1>Successful task archive</h1>
<p>{len(rows)} tasks with <code>acc=1</code>.</p>
<input id="filter" placeholder="Filter scene, episode, target, or description">
<table id="tasks">
<thead><tr><th>Scene</th><th>Episode</th><th>Target</th><th>Description</th><th>Size</th><th>Seen</th><th>Steps</th><th>First <=20m</th><th>Min m</th><th>Final m</th><th>Files</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<script>
const input = document.getElementById('filter');
input.addEventListener('input', () => {{
  const query = input.value.toLowerCase();
  document.querySelectorAll('#tasks tbody tr').forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(query) ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    entries = load_success_rows(run_dir)
    output_dir = (args.output_dir or run_dir / f"success_tasks_{len(entries)}").resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output_dir}")
    output_dir.mkdir(parents=True)

    index_rows: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    seen_counts: Counter[str] = Counter()

    for source_json, row in entries:
        scene = str(row.get("map_name"))
        episode = str(row.get("episode_id"))
        target_name = str(row.get("true_name") or row.get("object_name") or "unknown").strip()
        task_rel = Path("tasks") / safe_name(scene) / f"{safe_name(episode)}_{safe_name(target_name)}"
        task_dir = output_dir / task_rel
        task_dir.mkdir(parents=True)
        stats = trajectory_stats(row)

        compact_summary = {
            "criterion": {"acc": 1},
            "scene": scene,
            "episode_id": episode,
            "source_lane": source_json.parent.parent.name,
            "target": {
                "true_name": target_name,
                "object_name": row.get("object_name"),
                "description": str(row.get("description") or "").strip(),
                "category": row.get("category"),
                "size": row.get("size"),
                "position": first_position(row.get("pose")),
                "used_in_train": row.get("used-in-train"),
            },
            "start_pose": row.get("start_pose"),
            "info": row.get("info"),
            "outcome": {
                "termination_reason": row.get("termination_reason"),
                "acc": row.get("acc"),
                "osr": row.get("osr"),
                **stats,
            },
            "source_result": str(source_json.relative_to(run_dir)),
        }
        (task_dir / "summary.json").write_text(
            json.dumps(compact_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_trajectory_csv(task_dir / "trajectory.csv", row)
        make_trajectory_plot(row, stats, task_dir / "trajectory_xy_distance.png")
        keyframes = make_keyframe_sheet(
            source_json, row, stats, task_dir / "keyframes.jpg", args.max_keyframes
        )
        write_task_markdown(
            task_dir / "task.md", row, stats, keyframes, source_json.parent.parent.name
        )
        relative_symlink(source_json, task_dir / "source_result.json")
        image_dir = source_json.parent.parent / "images" / safe_name(scene) / safe_name(episode)
        if image_dir.is_dir():
            relative_symlink(image_dir, task_dir / "all_frames", directory=True)

        index_rows.append(
            {
                "scene": scene,
                "episode_id": episode,
                "source_lane": source_json.parent.parent.name,
                "true_name": target_name,
                "object_name": row.get("object_name"),
                "description": str(row.get("description") or "").strip(),
                "size": str(row.get("size") or ""),
                "used_in_train": row.get("used-in-train"),
                "steps": stats["step_count"],
                "first_reach_step": stats["first_reach_step"],
                "closest_step": stats["closest_step"],
                "minimum_distance": stats["minimum_distance"],
                "final_distance": stats["final_distance"],
                "task_dir": task_rel.as_posix(),
                "source_result": str(source_json.relative_to(run_dir)),
            }
        )
        scene_counts[scene] += 1
        target_counts[target_name] += 1
        size_counts[str(row.get("size") or "unknown").split("(", 1)[0].strip()] += 1
        seen_counts[str(row.get("used-in-train"))] += 1

    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]) if index_rows else [])
        writer.writeheader()
        writer.writerows(index_rows)
    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_index_html(output_dir, index_rows)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": str(run_dir),
        "criterion": "acc == 1",
        "task_count": len(index_rows),
        "scene_counts": dict(sorted(scene_counts.items())),
        "size_counts": dict(sorted(size_counts.items())),
        "seen_counts": dict(sorted(seen_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# Successful Task Archive

- Source run: `{run_dir}`
- Selection: `acc == 1`
- Tasks: **{len(index_rows)}**
- Scene counts: `{dict(scene_counts)}`
- Size counts: `{dict(size_counts)}`
- Seen counts: `{dict(seen_counts)}`

Open `index.html` for a searchable table. Each task contains target metadata,
trajectory CSV, top-down path/distance plot, selected keyframes, and links to the
original result JSON and all captured frames.

Benchmark success is distance-based. A successful stop does not by itself prove
that the target is clearly visible in the final RGB frame.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"archive={output_dir}")


if __name__ == "__main__":
    main()

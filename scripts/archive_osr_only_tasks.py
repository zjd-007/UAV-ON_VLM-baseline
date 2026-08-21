#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont, ImageOps


SUCCESS_RADIUS_METERS = 20.0
ACTION_ORDER = (
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive OSR-only UAV-ON episodes for manual trajectory analysis."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-keyframes", type=int, default=12)
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def episode_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    episode = str(row.get("episode_id", ""))
    try:
        numeric = int(episode)
    except ValueError:
        numeric = 10**12
    return str(row.get("map_name", "")), numeric, episode


def first_position(value: Any) -> list[float] | None:
    if isinstance(value, list) and value:
        if isinstance(value[0], list):
            value = value[0]
        if len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    return None


def vector(values: Any, length: int) -> list[float | None]:
    if not isinstance(values, (list, tuple)):
        return [None] * length
    result: list[float | None] = []
    for index in range(length):
        result.append(float(values[index]) if index < len(values) else None)
    return result


def load_osr_only_rows(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("lane*/temp/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("osr", 0)) != 1 or int(row.get("acc", 0)) != 0:
            continue
        key = (str(row.get("map_name")), str(row.get("episode_id")))
        if key in rows:
            raise ValueError(f"Duplicate completed episode {key}: {rows[key][0]} and {path}")
        rows[key] = (path, row)
    return sorted(rows.values(), key=lambda item: episode_sort_key(item[1]))


def trajectory_stats(row: dict[str, Any]) -> dict[str, Any]:
    records = row.get("step_records") or []
    distances = [float(record["distance_after"]) for record in records]
    first_reach_index = next(
        (index for index, distance in enumerate(distances) if distance <= SUCCESS_RADIUS_METERS),
        None,
    )
    closest_index = min(range(len(distances)), key=distances.__getitem__) if distances else None
    action_counts = Counter(str(record.get("parsed_command", "")) for record in records)
    return {
        "step_count": len(records),
        "first_reach_step": first_reach_index,
        "closest_step": closest_index,
        "minimum_distance": distances[closest_index] if closest_index is not None else row.get("ne"),
        "final_distance": float(row.get("ne", math.nan)),
        "final_within_20m": bool(float(row.get("ne", math.inf)) <= SUCCESS_RADIUS_METERS),
        "steps_after_first_reach": (
            len(records) - first_reach_index - 1 if first_reach_index is not None else None
        ),
        "action_counts": {action: action_counts.get(action, 0) for action in ACTION_ORDER},
    }


def select_keyframe_indices(
    record_count: int,
    first_reach: int | None,
    closest: int | None,
    maximum: int,
) -> list[int]:
    if record_count <= 0 or maximum <= 0:
        return []
    required = {0, record_count - 1}
    for index in (first_reach, closest):
        if index is not None:
            required.add(index)
            if index > 0:
                required.add(index - 1)
            if index + 1 < record_count:
                required.add(index + 1)
    if len(required) < maximum:
        slots = maximum - len(required)
        for offset in range(1, slots + 1):
            required.add(round(offset * (record_count - 1) / (slots + 1)))
    ordered = sorted(required)
    if len(ordered) <= maximum:
        return ordered
    priority = {0, record_count - 1, first_reach, closest}
    kept = [index for index in ordered if index in priority]
    for index in ordered:
        if index not in kept and len(kept) < maximum:
            kept.append(index)
    return sorted(kept)


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve_step_image(source_json: Path, record: dict[str, Any]) -> Path | None:
    image_path = record.get("image_path")
    if not image_path:
        return None
    lane_dir = source_json.parent.parent
    path = lane_dir / str(image_path)
    return path if path.is_file() else None


def make_keyframe_sheet(
    source_json: Path,
    row: dict[str, Any],
    stats: dict[str, Any],
    output: Path,
    maximum: int,
) -> list[int]:
    records = row.get("step_records") or []
    indices = select_keyframe_indices(
        len(records), stats["first_reach_step"], stats["closest_step"], maximum
    )
    tile_width, image_height, label_height = 360, 203, 55
    tile_height = image_height + label_height
    columns = 4
    rows_count = max(1, math.ceil(len(indices) / columns))
    title_height = 70
    sheet = Image.new(
        "RGB", (columns * tile_width, title_height + rows_count * tile_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    target = str(row.get("true_name") or row.get("object_name") or "unknown").strip()
    title = f"{row.get('map_name')} / episode {row.get('episode_id')} | Target: {target}"
    draw.text((16, 12), title, fill="black", font=font(22))
    draw.text(
        (16, 42),
        "Frames: start, uniformly sampled, first reach, closest, final",
        fill="#444444",
        font=font(15),
    )

    for position, index in enumerate(indices):
        record = records[index]
        x = (position % columns) * tile_width
        y = title_height + (position // columns) * tile_height
        image_path = resolve_step_image(source_json, record)
        if image_path is None:
            tile = Image.new("RGB", (tile_width, image_height), "#dddddd")
            ImageDraw.Draw(tile).text((12, 12), "missing image", fill="black", font=font(18))
        else:
            with Image.open(image_path) as image:
                tile = ImageOps.fit(image.convert("RGB"), (tile_width, image_height))
        sheet.paste(tile, (x, y))
        label_draw = ImageDraw.Draw(sheet)
        action = str(record.get("parsed_command", ""))
        distance = float(record.get("distance_after", math.nan))
        flags = []
        if index == stats["first_reach_step"]:
            flags.append("FIRST<=20")
        if index == stats["closest_step"]:
            flags.append("CLOSEST")
        flag_text = f" [{' | '.join(flags)}]" if flags else ""
        label_draw.rectangle((x, y + image_height, x + tile_width, y + tile_height), fill="black")
        label_draw.text(
            (x + 8, y + image_height + 5),
            f"step {index:03d}  d={distance:.2f}m{flag_text}",
            fill="white",
            font=font(14),
        )
        label_draw.text(
            (x + 8, y + image_height + 29), action, fill="#f0d86b", font=font(14)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return indices


def make_trajectory_plot(row: dict[str, Any], stats: dict[str, Any], output: Path) -> None:
    records = row.get("step_records") or []
    if not records:
        return
    initial = vector(records[0].get("pose_before"), 4)
    poses = [initial] + [vector(record.get("pose_after"), 4) for record in records]
    xs = [float(pose[0]) for pose in poses if pose[0] is not None]
    ys = [float(pose[1]) for pose in poses if pose[1] is not None]
    distances = [float(records[0].get("distance_before", math.nan))] + [
        float(record.get("distance_after", math.nan)) for record in records
    ]
    target = first_position(row.get("pose"))

    fig, (axis_xy, axis_distance) = plt.subplots(1, 2, figsize=(13, 5.5))
    axis_xy.plot(xs, ys, color="#2f6f9f", linewidth=1.5, alpha=0.8)
    scatter = axis_xy.scatter(
        xs, ys, c=range(len(xs)), cmap="viridis", s=13, zorder=3, label="UAV poses"
    )
    axis_xy.scatter(xs[0], ys[0], marker="o", s=85, color="#2ca02c", label="start", zorder=5)
    axis_xy.scatter(xs[-1], ys[-1], marker="s", s=75, color="#d62728", label="final", zorder=5)
    if target is not None:
        axis_xy.scatter(target[0], target[1], marker="*", s=180, color="#ff7f0e", label="target")
        axis_xy.add_patch(
            Circle(
                (target[0], target[1]),
                SUCCESS_RADIUS_METERS,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="#ff7f0e",
                alpha=0.7,
                label="20m XY reference",
            )
        )
    for step_key, marker, color, label in (
        ("first_reach_step", "D", "#9467bd", "first 3D distance <=20m"),
        ("closest_step", "X", "#17becf", "closest"),
    ):
        step = stats.get(step_key)
        if step is not None and step + 1 < len(xs):
            axis_xy.scatter(xs[step + 1], ys[step + 1], marker=marker, s=90, color=color, label=label)
    axis_xy.set_xlabel("x (m)")
    axis_xy.set_ylabel("y (m)")
    axis_xy.set_title("Top-down navigation path")
    axis_xy.axis("equal")
    axis_xy.grid(alpha=0.25)
    axis_xy.legend(fontsize=8, loc="best")
    fig.colorbar(scatter, ax=axis_xy, label="pose index", fraction=0.045)

    axis_distance.plot(range(len(distances)), distances, color="#3a6ea5", linewidth=1.6)
    axis_distance.axhline(SUCCESS_RADIUS_METERS, color="#d62728", linestyle="--", label="20m")
    if stats.get("first_reach_step") is not None:
        axis_distance.axvline(
            stats["first_reach_step"] + 1,
            color="#9467bd",
            linestyle=":",
            label="first reach",
        )
    if stats.get("closest_step") is not None:
        axis_distance.scatter(
            [stats["closest_step"] + 1],
            [stats["minimum_distance"]],
            marker="X",
            s=80,
            color="#17becf",
            label="closest",
        )
    axis_distance.set_xlabel("pose index (initial pose = 0)")
    axis_distance.set_ylabel("3D distance to target (m)")
    axis_distance.set_title("Target distance over trajectory")
    axis_distance.grid(alpha=0.25)
    axis_distance.legend(fontsize=8)

    target_name = str(row.get("true_name") or row.get("object_name") or "unknown").strip()
    fig.suptitle(
        f"{row.get('map_name')} / episode {row.get('episode_id')} / {target_name}",
        fontsize=13,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_trajectory_csv(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "step",
        "action",
        "raw_action_text",
        "image_path",
        "x_before",
        "y_before",
        "z_before",
        "yaw_before_rad",
        "yaw_before_deg",
        "x_after",
        "y_after",
        "z_after",
        "yaw_after_rad",
        "yaw_after_deg",
        "distance_before",
        "distance_after",
        "inside_20m_after",
        "collided",
        "depth_grid",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in row.get("step_records") or []:
            before = vector(record.get("pose_before"), 4)
            after = vector(record.get("pose_after"), 4)
            distance_after = float(record.get("distance_after", math.nan))
            writer.writerow(
                {
                    "step": record.get("step"),
                    "action": record.get("parsed_command"),
                    "raw_action_text": record.get("raw_action_text"),
                    "image_path": record.get("image_path"),
                    "x_before": before[0],
                    "y_before": before[1],
                    "z_before": before[2],
                    "yaw_before_rad": before[3],
                    "yaw_before_deg": math.degrees(before[3]) if before[3] is not None else None,
                    "x_after": after[0],
                    "y_after": after[1],
                    "z_after": after[2],
                    "yaw_after_rad": after[3],
                    "yaw_after_deg": math.degrees(after[3]) if after[3] is not None else None,
                    "distance_before": record.get("distance_before"),
                    "distance_after": distance_after,
                    "inside_20m_after": int(distance_after <= SUCCESS_RADIUS_METERS),
                    "collided": int(bool(record.get("collided"))),
                    "depth_grid": json.dumps(
                        (record.get("depth_avoidance") or {}).get("depth_grid"),
                        ensure_ascii=False,
                    ),
                }
            )


def relative_symlink(source: Path, destination: Path, directory: bool = False) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=directory)


def write_task_markdown(
    path: Path,
    row: dict[str, Any],
    stats: dict[str, Any],
    keyframes: list[int],
    source_lane: str,
) -> None:
    target = str(row.get("true_name") or "").strip()
    description = str(row.get("description") or "").strip()
    target_position = first_position(row.get("pose"))
    info = row.get("info") or {}
    actions = ", ".join(
        f"{action}: {count}" for action, count in stats["action_counts"].items() if count
    )
    content = f"""# {row.get('map_name')} / Episode {row.get('episode_id')}

## Target

- **True name:** {target}
- **Object asset:** {row.get('object_name')}
- **Description:** {description}
- **Size:** {row.get('size')}
- **Target position:** {target_position}
- **Used in train:** {row.get('used-in-train')}

## Episode

- **Source lane:** {source_lane}
- **Termination:** {row.get('termination_reason')}
- **Steps:** {stats['step_count']}
- **Initial geodesic distance:** {info.get('geodesic_distance')}
- **Initial Euclidean distance:** {info.get('euclidean_distance')}
- **First step within 20 m:** {stats['first_reach_step']}
- **Closest step / distance:** {stats['closest_step']} / {stats['minimum_distance']:.3f} m
- **Final distance:** {stats['final_distance']:.3f} m
- **Final pose still within 20 m:** {stats['final_within_20m']}
- **Steps after first reach:** {stats['steps_after_first_reach']}
- **Actions:** {actions}
- **Keyframe steps:** {keyframes}

## Files

- [Trajectory plot](trajectory_xy_distance.png)
- [Keyframes](keyframes.jpg)
- [Step-by-step trajectory table](trajectory.csv)
- [Compact task summary](summary.json)
- [Original result JSON](source_result.json)
- [All captured frames](all_frames/)
"""
    path.write_text(content, encoding="utf-8")


def write_index_html(output_dir: Path, index_rows: list[dict[str, Any]]) -> None:
    body_rows = []
    for row in index_rows:
        task_dir = row["task_dir"]
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['scene']))}</td>"
            f"<td>{html.escape(str(row['episode_id']))}</td>"
            f"<td>{html.escape(str(row['true_name']))}</td>"
            f"<td>{html.escape(str(row['object_name']))}</td>"
            f"<td>{html.escape(str(row['description']))}</td>"
            f"<td>{html.escape(str(row['termination_reason']))}</td>"
            f"<td>{row['steps']}</td>"
            f"<td>{row['first_reach_step']}</td>"
            f"<td>{row['minimum_distance']:.2f}</td>"
            f"<td>{row['final_distance']:.2f}</td>"
            f"<td>{row['final_within_20m']}</td>"
            f"<td><a href=\"{html.escape(task_dir)}/task.md\">summary</a> | "
            f"<a href=\"{html.escape(task_dir)}/trajectory_xy_distance.png\">path</a> | "
            f"<a href=\"{html.escape(task_dir)}/keyframes.jpg\">frames</a> | "
            f"<a href=\"{html.escape(task_dir)}/all_frames/\">all</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OSR-only task archive</title>
<style>
body {{ font-family: sans-serif; margin: 24px; color: #1d242b; }}
input {{ width: min(680px, 90vw); padding: 8px; margin: 8px 0 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ccd2d8; padding: 6px; vertical-align: top; }}
th {{ background: #eef2f5; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f8fafb; }}
.description {{ max-width: 420px; }}
</style>
</head>
<body>
<h1>OSR-only task archive</h1>
<p>{len(index_rows)} tasks with <code>osr=1</code> and <code>acc=0</code>.</p>
<input id="filter" placeholder="Filter scene, episode, target, asset, or description">
<table id="tasks">
<thead><tr><th>Scene</th><th>Episode</th><th>Target</th><th>Asset</th><th>Description</th><th>End</th><th>Steps</th><th>First ≤20m</th><th>Min m</th><th>Final m</th><th>Final ≤20m</th><th>Files</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody>
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
    entries = load_osr_only_rows(run_dir)
    output_dir = (args.output_dir or run_dir / f"osr_only_tasks_{len(entries)}").resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output_dir}")
    output_dir.mkdir(parents=True)

    index_rows: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for source_json, row in entries:
        scene = str(row.get("map_name"))
        episode = str(row.get("episode_id"))
        target_name = str(row.get("true_name") or row.get("object_name") or "unknown").strip()
        task_rel = Path("tasks") / safe_name(scene) / f"{safe_name(episode)}_{safe_name(target_name)}"
        task_dir = output_dir / task_rel
        task_dir.mkdir(parents=True)
        stats = trajectory_stats(row)
        target_position = first_position(row.get("pose"))
        summary = {
            "criterion": {"osr": 1, "acc": 0},
            "scene": scene,
            "episode_id": episode,
            "source_lane": source_json.parent.parent.name,
            "target": {
                "true_name": target_name,
                "object_name": row.get("object_name"),
                "description": str(row.get("description") or "").strip(),
                "category": row.get("category"),
                "size": row.get("size"),
                "position": target_position,
                "used_in_train": row.get("used-in-train"),
            },
            "start_pose": row.get("start_pose"),
            "info": row.get("info"),
            "outcome": {
                "termination_reason": row.get("termination_reason"),
                "acc": row.get("acc"),
                "osr": row.get("osr"),
                "oracle_success": row.get("oracle_success"),
                **stats,
            },
            "source_result": str(source_json.relative_to(run_dir)),
        }
        (task_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
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

        index_row = {
            "scene": scene,
            "episode_id": episode,
            "source_lane": source_json.parent.parent.name,
            "used_in_train": row.get("used-in-train"),
            "true_name": target_name,
            "object_name": row.get("object_name"),
            "description": str(row.get("description") or "").strip(),
            "size": row.get("size"),
            "target_position": json.dumps(target_position),
            "termination_reason": row.get("termination_reason"),
            "steps": stats["step_count"],
            "first_reach_step": stats["first_reach_step"],
            "closest_step": stats["closest_step"],
            "minimum_distance": stats["minimum_distance"],
            "final_distance": stats["final_distance"],
            "final_within_20m": stats["final_within_20m"],
            "steps_after_first_reach": stats["steps_after_first_reach"],
            "task_dir": task_rel.as_posix(),
            "source_result": str(source_json.relative_to(run_dir)),
        }
        index_rows.append(index_row)
        scene_counts[scene] += 1
        termination_counts[str(row.get("termination_reason"))] += 1
        target_counts[target_name] += 1

    index_fields = list(index_rows[0]) if index_rows else []
    with (output_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        writer.writerows(index_rows)
    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_index_html(output_dir, index_rows)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": str(run_dir),
        "criterion": "osr == 1 and acc == 0",
        "success_radius_meters": SUCCESS_RADIUS_METERS,
        "task_count": len(index_rows),
        "scene_counts": dict(sorted(scene_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "final_within_20m": sum(bool(row["final_within_20m"]) for row in index_rows),
        "target_counts": dict(sorted(target_counts.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# OSR-only Task Archive

- Source run: `{run_dir}`
- Selection: `osr == 1 and acc == 0`
- Tasks: **{len(index_rows)}**
- Termination counts: `{dict(termination_counts)}`
- Final pose still within 20 m: **{manifest['final_within_20m']}**

Open `index.html` for a searchable table, or use `index.csv` for analysis. Each task directory contains the target metadata, trajectory CSV, top-down path/distance plot, selected keyframes, and links to the original JSON and all captured frames.

The 20 m circle in the XY plot is a visual reference. UAV-ON success distance is computed in 3D.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"archive={output_dir}")


if __name__ == "__main__":
    main()

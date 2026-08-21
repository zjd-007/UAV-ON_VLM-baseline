#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt  # noqa: E402
from vlm_baseline.prompting import build_prompt  # noqa: E402


ACTION_IDS = {
    "stop": 0,
    "forward 3m": 1,
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pixel_bin(pixel_count: int) -> str:
    if pixel_count <= 0:
        return "p00000"
    if pixel_count < 128:
        return "p00001_00127"
    if pixel_count < 512:
        return "p00128_00511"
    if pixel_count < 2048:
        return "p00512_02047"
    if pixel_count < 8192:
        return "p02048_08191"
    return "p08192_plus"


def distance_bin(distance: float | None) -> str:
    if distance is None:
        return "distance_unknown"
    if distance < 8.0:
        return "d00_08"
    if distance < 12.0:
        return "d08_12"
    if distance < 16.0:
        return "d12_16"
    if distance <= 20.0:
        return "d16_20"
    return "d20_plus"


def center_offset(mask: dict[str, Any]) -> float | None:
    centroid = mask.get("centroid")
    width = int(mask.get("width", 512))
    height = int(mask.get("height", 512))
    if not centroid:
        return None
    x, y = (float(centroid[0]), float(centroid[1]))
    dx = (x - (width - 1) / 2.0) / max(width / 2.0, 1.0)
    dy = (y - (height - 1) / 2.0) / max(height / 2.0, 1.0)
    return math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)


def center_bin(value: float | None) -> str:
    if value is None:
        return "center_unknown"
    if value <= 0.15:
        return "centered"
    if value <= 0.35:
        return "off_center"
    return "edge"


def completeness_bin(mask: dict[str, Any], present: bool) -> str:
    if not present:
        return "absent"
    if bool(mask.get("touches_opposite_borders")) or int(mask.get("clipped_sides_count", 0)) >= 2:
        return "severely_clipped"
    if int(mask.get("clipped_sides_count", 0)) == 1:
        return "partial"
    return "complete"


def load_episode_indexes(run_files: set[Path]) -> dict[Path, dict[tuple[str, str], dict[str, Any]]]:
    indexes: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    for run_file in sorted(run_files):
        index = {}
        for row in read_jsonl(run_file):
            index[(str(row.get("map_name")), str(row.get("episode_id")))] = row
        indexes[run_file] = index
    return indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out paired fixed frames from inference Stop visibility captures.")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    captures = []
    for path in sorted((args.capture_dir / "captures").glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("status") != "ok" or row.get("size_bucket") != "small":
                continue
            image = Path(str(row.get("source_image_path")))
            if not image.is_file():
                continue
            captures.append(row)

    run_files = {Path(str(row["run_file"])) for row in captures}
    episode_indexes = load_episode_indexes(run_files)
    depth_module = UAVONSingleViewDepthPrompt()
    samples = []
    missing_episode = 0
    missing_step = 0
    for capture in captures:
        run_file = Path(str(capture["run_file"]))
        key = (str(capture["scene_id"]), str(capture["episode_id"]))
        episode = episode_indexes[run_file].get(key)
        if episode is None:
            missing_episode += 1
            continue
        stop_step = int(capture["stop_step"])
        step_records = episode.get("step_records") or []
        if stop_step >= len(step_records):
            missing_step += 1
            continue
        step = step_records[stop_step]
        depth_record = step.get("depth_avoidance") or {}
        depth_grid = depth_record.get("depth_grid")
        if depth_grid is None:
            depth_prompt = "CurrentViewDepth is unavailable. Use the RGB image and choose cautiously. Avoid moving forward if the image shows nearby obstacles."
        else:
            depth_prompt = depth_module.format_prompt(np.asarray(depth_grid, dtype=np.float32))
        memory_prompt = str((step.get("memory_context") or {}).get("prompt_text") or "")
        description = str(capture.get("target_description") or episode.get("description") or "")
        prompt = build_prompt(description, depth_context=depth_prompt, memory_context=memory_prompt)
        mask = capture.get("mask") or {}
        pixels = int(mask.get("pixel_count", 0))
        present = bool(capture.get("target_present"))
        clear = bool(capture.get("geometry_clear"))
        center = center_offset(mask)
        if clear:
            visibility_group = "inference_clear"
        elif present:
            visibility_group = "inference_visible_weak"
        else:
            visibility_group = "inference_absent"
        label = "stop" if clear else "forward 3m"
        samples.append(
            {
                "source_index": len(samples),
                "sample_index": len(samples),
                "capture_key": str(capture.get("capture_key")),
                "source_run_label": str(capture.get("run_label")),
                "episode_key": f"{capture['scene_id']}::{capture['episode_id']}",
                "scene_id": str(capture["scene_id"]),
                "frame_idx": stop_step,
                "image": str(capture["source_image_path"]),
                "prompt": prompt,
                "target_description": description,
                "true_name": str(capture.get("true_name") or ""),
                "object_name": str(capture.get("object_name") or ""),
                "label": label,
                "label_id": ACTION_IDS[label],
                "size_bucket": "small",
                "mask_pixels": pixels,
                "mask_fraction": float(mask.get("pixel_fraction", 0.0)),
                "bbox_short_side": min(int(mask.get("bbox_width", 0)), int(mask.get("bbox_height", 0))),
                "distance_to_target_m": float(capture.get("distance_to_target_m")),
                "center_offset": center,
                "clear": clear,
                "pixel_bin": pixel_bin(pixels),
                "distance_bin": distance_bin(float(capture.get("distance_to_target_m"))),
                "center_bin": center_bin(center),
                "completeness_bin": completeness_bin(mask, present),
                "visibility_group": visibility_group,
                "nominal_success": bool(capture.get("success")),
                "oracle_success": bool(capture.get("oracle_success")),
                "seen_group": str(capture.get("seen_group") or "unknown"),
            }
        )

    write_jsonl(args.output_dir / "fixed_frames.jsonl", samples)
    write_jsonl(args.output_dir / "fixed_frames_smoke.jsonl", samples[:12])
    counts: dict[str, dict[str, int]] = {}
    for dimension in ("source_run_label", "visibility_group", "pixel_bin", "distance_bin", "center_bin", "seen_group"):
        values: dict[str, int] = {}
        for row in samples:
            value = str(row[dimension])
            values[value] = values.get(value, 0) + 1
        counts[dimension] = values
    summary = {
        "capture_dir": str(args.capture_dir),
        "samples": len(samples),
        "missing_episode": missing_episode,
        "missing_step": missing_step,
        "counts": counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

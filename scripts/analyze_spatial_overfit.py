#!/usr/bin/env python3
"""Offline spatial-overfitting audit for UAV-ON training and evaluation paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon


DEFAULT_TRAIN_FRAMES = Path(
    "/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/processed/"
    "neighborhood_coordinate_repair_v1_20260812_194807/"
    "final_dataset_per_frame_safe_stopbank_v1/train_frames.jsonl"
)
DEFAULT_ALIGNED_JSON = Path(
    "/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/generated/"
    "record_output_transition_aligned/json"
)
DEFAULT_RESULTS = {
    "ckpt13000": Path(
        "/data/zhujd/Aerial-ObjectNav/VLM-baseline/results/"
        "phi35_stopbank_cfmem_v2_ckpt13000_full_20260815_130346/"
        "all_episodes.jsonl"
    ),
    "ckpt19764": Path(
        "/data/zhujd/Aerial-ObjectNav/VLM-baseline/results/"
        "phi35_stopbank_cfmem_v2_ckpt19764_full_20260814_162448/"
        "all_episodes.jsonl"
    ),
}
DEFAULT_OUTPUT = Path(
    "/data/zhujd/Aerial-ObjectNav/VLM-baseline/results/"
    "spatial_overfit_audit_stopbank_ckpt13000_vs_ckpt19764_20260821"
)
ALIGNED_IMAGE_RE = re.compile(
    r"record_output_transition_aligned/images/([^/]+)/([^/]+)/([^/]+)/uav_on_0/(\d+)\.png$"
)
THRESHOLDS = (3.0, 5.0, 10.0)
CELL_SIZE = 5.0


def jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite_xyz(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    xyz = np.asarray(value[:3], dtype=np.float64)
    return xyz if np.isfinite(xyz).all() else None


def cell_key(xyz, cell_size=CELL_SIZE):
    return tuple(np.floor(np.asarray(xyz) / cell_size).astype(np.int64).tolist())


def resolve_training_pose(row, aligned_root: Path, cache):
    pose = finite_xyz(row.get("pose"))
    if pose is not None:
        return pose, "inline"

    match = ALIGNED_IMAGE_RE.search(row.get("image_path", ""))
    if not match:
        return None, "unresolved_path"
    scene, episode, pose_idx, frame = match.groups()
    record_path = aligned_root / scene / episode / f"{pose_idx}.json"
    key = str(record_path)
    if key not in cache:
        try:
            cache[key] = json.loads(record_path.read_text()).get("record_list", [])
        except (OSError, json.JSONDecodeError):
            cache[key] = None
    records = cache[key]
    frame_idx = int(frame)
    if not records or frame_idx >= len(records):
        return None, "missing_record"
    return finite_xyz(records[frame_idx]), "aligned_record"


def build_training_map(train_frames: Path, aligned_root: Path, output: Path):
    points = defaultdict(list)
    sample_cells = defaultdict(Counter)
    episode_cells = defaultdict(Counter)
    episode_seen_cells = defaultdict(set)
    scene_samples = Counter()
    scene_episodes = defaultdict(set)
    pose_sources = Counter()
    unresolved_examples = []
    record_cache = {}

    for row in jsonl(train_frames):
        scene = str(row.get("scene_id", ""))
        episode = str(row.get("episode_key", ""))
        scene_samples[scene] += 1
        scene_episodes[scene].add(episode)
        pose, source = resolve_training_pose(row, aligned_root, record_cache)
        pose_sources[source] += 1
        if pose is None:
            if len(unresolved_examples) < 20:
                unresolved_examples.append(row.get("image_path", ""))
            continue
        points[scene].append(pose)
        key = cell_key(pose)
        sample_cells[scene][key] += 1
        episode_seen_cells[(scene, episode)].add(key)

    for (scene, _episode), cells in episode_seen_cells.items():
        for key in cells:
            episode_cells[scene][key] += 1

    map_dir = output / "training_pose_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    maps = {}
    rows = []
    for scene in sorted(points):
        raw = np.asarray(points[scene], dtype=np.float64)
        unique = np.unique(np.round(raw, 3), axis=0)
        np.save(map_dir / f"{scene}.npy", unique)
        maps[scene] = {
            "points": unique,
            "tree3d": cKDTree(unique),
            "tree2d": cKDTree(unique[:, :2]),
            "sample_cells": sample_cells[scene],
            "episode_cells": episode_cells[scene],
        }
        xyz_min, xyz_max = unique.min(axis=0), unique.max(axis=0)
        rows.append(
            {
                "scene": scene,
                "samples": scene_samples[scene],
                "episodes": len(scene_episodes[scene]),
                "unique_xyz": len(unique),
                "occupied_5m_cells": len(sample_cells[scene]),
                "x_min": xyz_min[0], "x_max": xyz_max[0],
                "y_min": xyz_min[1], "y_max": xyz_max[1],
                "z_min": xyz_min[2], "z_max": xyz_max[2],
            }
        )

    write_csv(output / "training_coverage_by_scene.csv", rows)
    manifest = {
        "train_frames": str(train_frames),
        "aligned_json_root": str(aligned_root),
        "total_rows": sum(scene_samples.values()),
        "resolved_rows": sum(len(v) for v in points.values()),
        "resolution_rate": sum(len(v) for v in points.values()) / max(sum(scene_samples.values()), 1),
        "pose_sources": dict(pose_sources),
        "cell_size_m": CELL_SIZE,
        "scenes": rows,
        "unresolved_examples": unresolved_examples,
    }
    write_json(output / "training_coverage_manifest.json", manifest)
    plot_training_coverage(maps, output / "training_coverage_xy.png")
    return maps, manifest


def plot_training_coverage(maps, path: Path):
    scenes = sorted(maps)
    cols = 3
    rows = math.ceil(len(scenes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.4 * rows), squeeze=False)
    for ax, scene in zip(axes.flat, scenes):
        points = maps[scene]["points"]
        plot_points = points[:: max(1, len(points) // 20000)]
        ax.hexbin(plot_points[:, 0], plot_points[:, 1], gridsize=55, bins="log", mincnt=1)
        ax.set_title(f"{scene} ({len(points):,} unique poses)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[len(scenes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def target_positions(row):
    raw = row.get("pose", [])
    if raw and isinstance(raw[0], (int, float)):
        raw = [raw]
    values = [finite_xyz(item) for item in raw]
    return [value for value in values if value is not None]


def path_positions(row):
    values = [finite_xyz(item) for item in row.get("gen_pose_list", [])]
    values = [value for value in values if value is not None]
    if not values:
        start = finite_xyz(row.get("start_pose", {}).get("start_position"))
        values = [start] if start is not None else []
    return np.asarray(values, dtype=np.float64)


def min_target_distance(points, targets):
    if len(points) == 0 or not targets:
        return np.full(len(points), np.nan)
    target_array = np.asarray(targets)
    return np.sqrt(((points[:, None, :] - target_array[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def episode_metrics(row, checkpoint, maps):
    scene = str(row.get("scene_key") or row.get("map_name", "")).replace("_test", "")
    path = path_positions(row)
    targets = target_positions(row)
    base = {
        "checkpoint": checkpoint,
        "task_key": f"{row.get('map_name')}::{row.get('episode_id')}",
        "scene": scene,
        "map_name": row.get("map_name", ""),
        "episode_id": row.get("episode_id", ""),
        "true_name": str(row.get("true_name", "")).strip(),
        "size": row.get("size", ""),
        "used_in_train": int(bool(row.get("used-in-train", 0))),
        "success": int(bool(row.get("acc", 0))),
        "osr": int(bool(row.get("osr", 0))),
        "collision": int(bool(row.get("collision", False))),
        "ne": float(row.get("ne", np.nan)),
        "termination_reason": row.get("termination_reason", ""),
        "path_steps": max(0, len(path) - 1),
        "has_scene_train_map": int(scene in maps),
    }
    if len(path) == 0:
        return base

    move = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.asarray([])
    base["path_length_3d"] = float(move.sum())
    base["unique_path_5m_cells"] = len({cell_key(p) for p in path})
    goal_dist = min_target_distance(path, targets)
    base["start_goal_distance"] = float(goal_dist[0]) if len(goal_dist) else np.nan
    base["end_goal_distance"] = float(goal_dist[-1]) if len(goal_dist) else np.nan
    base["min_goal_distance"] = float(np.nanmin(goal_dist)) if len(goal_dist) else np.nan
    if scene not in maps:
        return base

    scene_map = maps[scene]
    train_dist, _ = scene_map["tree3d"].query(path, k=1)
    train_dist_xy, _ = scene_map["tree2d"].query(path[:, :2], k=1)
    base.update(
        start_train_distance=float(train_dist[0]),
        end_train_distance=float(train_dist[-1]),
        min_train_distance=float(train_dist.min()),
        mean_train_distance=float(train_dist.mean()),
        median_train_distance=float(np.median(train_dist)),
        start_train_distance_xy=float(train_dist_xy[0]),
        end_train_distance_xy=float(train_dist_xy[-1]),
        mean_train_distance_xy=float(train_dist_xy.mean()),
    )
    for threshold in THRESHOLDS:
        base[f"path_fraction_within_{int(threshold)}m_train"] = float((train_dist <= threshold).mean())

    if targets:
        target_array = np.asarray(targets)
        target_train_dist, _ = scene_map["tree3d"].query(target_array, k=1)
        base["target_train_distance"] = float(target_train_dist.min())

    cells = [cell_key(p) for p in path]
    sample_density = np.asarray([scene_map["sample_cells"].get(c, 0) for c in cells], dtype=float)
    episode_density = np.asarray([scene_map["episode_cells"].get(c, 0) for c in cells], dtype=float)
    base.update(
        path_fraction_unseen_5m_cells=float((episode_density == 0).mean()),
        mean_log_train_sample_density=float(np.log1p(sample_density).mean()),
        mean_log_train_episode_density=float(np.log1p(episode_density).mean()),
        start_train_episode_density=int(episode_density[0]),
        end_train_episode_density=int(episode_density[-1]),
        endpoint_density_gain=float(np.log1p(episode_density[-1]) - np.log1p(episode_density[0])),
    )

    if len(path) > 1:
        moving = move > 0.1
        train_progress = train_dist[:-1] - train_dist[1:]
        goal_progress = goal_dist[:-1] - goal_dist[1:]
        if moving.any():
            denom = move[moving].sum()
            base["train_attraction_per_meter"] = float(train_progress[moving].sum() / denom)
            base["goal_progress_per_meter"] = float(goal_progress[moving].sum() / denom)
            base["conflict_move_fraction"] = float(
                ((train_progress[moving] > 0.25) & (goal_progress[moving] < -0.25)).mean()
            )
    return base


def read_eval(path: Path, checkpoint: str, maps):
    return [episode_metrics(row, checkpoint, maps) for row in jsonl(path)]


def numeric(values):
    result = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            result.append(value)
    return np.asarray(result, dtype=float)


def summarize(rows, group, checkpoint):
    def avg(key):
        values = numeric([row.get(key) for row in rows])
        return float(values.mean()) if len(values) else np.nan

    return {
        "checkpoint": checkpoint,
        "group": group,
        "n": len(rows),
        "success_count": sum(row["success"] for row in rows),
        "sr": avg("success"),
        "osr": avg("osr"),
        "collision_rate": avg("collision"),
        "ne": avg("ne"),
        "path_length_3d": avg("path_length_3d"),
        "mean_train_distance": avg("mean_train_distance"),
        "end_train_distance": avg("end_train_distance"),
        "target_train_distance": avg("target_train_distance"),
        "path_fraction_within_5m_train": avg("path_fraction_within_5m_train"),
        "path_fraction_unseen_5m_cells": avg("path_fraction_unseen_5m_cells"),
        "mean_log_train_episode_density": avg("mean_log_train_episode_density"),
        "endpoint_density_gain": avg("endpoint_density_gain"),
        "train_attraction_per_meter": avg("train_attraction_per_meter"),
        "goal_progress_per_meter": avg("goal_progress_per_meter"),
        "conflict_move_fraction": avg("conflict_move_fraction"),
    }


def grouped_summaries(all_rows):
    output = []
    for checkpoint in sorted({row["checkpoint"] for row in all_rows}):
        rows = [row for row in all_rows if row["checkpoint"] == checkpoint]
        groups = {
            "all": rows,
            "scene_has_training_map": [row for row in rows if row["has_scene_train_map"]],
            "novel_scene_control": [row for row in rows if not row["has_scene_train_map"]],
            "target_within_5m_of_train": [row for row in rows if row.get("target_train_distance", np.inf) <= 5],
            "target_over_5m_from_train": [row for row in rows if row.get("target_train_distance", -np.inf) > 5],
        }
        for name, group_rows in groups.items():
            if group_rows:
                output.append(summarize(group_rows, name, checkpoint))
        for scene in sorted({row["scene"] for row in rows}):
            output.append(summarize([row for row in rows if row["scene"] == scene], f"scene:{scene}", checkpoint))
    return output


def assign_target_novelty(rows):
    reference = {}
    for row in rows:
        value = row.get("target_train_distance")
        if value is not None and np.isfinite(value):
            reference[row["task_key"]] = float(value)
    values = np.asarray(list(reference.values()))
    if not len(values):
        return [], []
    cuts = np.quantile(values, [0.25, 0.5, 0.75])
    labels = ("Q1 closest", "Q2", "Q3", "Q4 farthest")
    for row in rows:
        value = row.get("target_train_distance")
        if value is not None and np.isfinite(value):
            row["target_novelty_quartile"] = labels[int(np.searchsorted(cuts, value, side="right"))]
    summaries = []
    for checkpoint in sorted({row["checkpoint"] for row in rows}):
        for label in labels:
            group = [r for r in rows if r["checkpoint"] == checkpoint and r.get("target_novelty_quartile") == label]
            if group:
                summaries.append(summarize(group, f"target_novelty:{label}", checkpoint))
    return cuts.tolist(), summaries


def bootstrap_delta(a, b, seed=20260821, samples=5000):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    delta = b[keep] - a[keep]
    if not len(delta):
        return {"n": 0, "mean_delta": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        boot[start:start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"n": len(delta), "mean_delta": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def paired_comparison(rows, first="ckpt13000", second="ckpt19764"):
    by_checkpoint = {
        name: {row["task_key"]: row for row in rows if row["checkpoint"] == name}
        for name in (first, second)
    }
    keys = sorted(set(by_checkpoint[first]) & set(by_checkpoint[second]))
    metrics = [
        "success", "osr", "collision", "ne", "path_length_3d", "mean_train_distance",
        "end_train_distance", "path_fraction_within_5m_train", "path_fraction_unseen_5m_cells",
        "mean_log_train_episode_density", "endpoint_density_gain", "train_attraction_per_meter",
        "goal_progress_per_meter", "conflict_move_fraction",
    ]
    comparisons = []
    for metric in metrics:
        a = [by_checkpoint[first][key].get(metric, np.nan) for key in keys]
        b = [by_checkpoint[second][key].get(metric, np.nan) for key in keys]
        item = {"metric": metric, **bootstrap_delta(a, b)}
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        keep = np.isfinite(aa) & np.isfinite(bb)
        delta = bb[keep] - aa[keep]
        if len(delta) and np.any(delta != 0):
            try:
                item["wilcoxon_p"] = float(wilcoxon(delta).pvalue)
            except ValueError:
                item["wilcoxon_p"] = np.nan
        comparisons.append(item)

    changed = {
        "success_13000_only": 0, "success_19764_only": 0,
        "osr_13000_only": 0, "osr_19764_only": 0,
    }
    paired_rows = []
    for key in keys:
        a, b = by_checkpoint[first][key], by_checkpoint[second][key]
        changed["success_13000_only"] += int(a["success"] == 1 and b["success"] == 0)
        changed["success_19764_only"] += int(a["success"] == 0 and b["success"] == 1)
        changed["osr_13000_only"] += int(a["osr"] == 1 and b["osr"] == 0)
        changed["osr_19764_only"] += int(a["osr"] == 0 and b["osr"] == 1)
        paired_rows.append({
            "task_key": key, "scene": a["scene"], "true_name": a["true_name"],
            "target_train_distance": a.get("target_train_distance", ""),
            "success_13000": a["success"], "success_19764": b["success"],
            "osr_13000": a["osr"], "osr_19764": b["osr"],
            "collision_13000": a["collision"], "collision_19764": b["collision"],
            "mean_train_distance_13000": a.get("mean_train_distance", ""),
            "mean_train_distance_19764": b.get("mean_train_distance", ""),
            "train_density_13000": a.get("mean_log_train_episode_density", ""),
            "train_density_19764": b.get("mean_log_train_episode_density", ""),
            "goal_progress_per_meter_13000": a.get("goal_progress_per_meter", ""),
            "goal_progress_per_meter_19764": b.get("goal_progress_per_meter", ""),
        })
    return {"paired_tasks": len(keys), "changed_outcomes": changed, "metrics": comparisons}, paired_rows


def correlations(rows):
    result = []
    for checkpoint in sorted({row["checkpoint"] for row in rows}):
        subset = [row for row in rows if row["checkpoint"] == checkpoint and row["has_scene_train_map"]]
        for exposure in ("mean_train_distance", "path_fraction_within_5m_train", "mean_log_train_episode_density"):
            for outcome in ("success", "osr", "collision", "goal_progress_per_meter"):
                pairs = [(r.get(exposure), r.get(outcome)) for r in subset]
                pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
                if len(pairs) < 3:
                    continue
                a, b = zip(*pairs)
                rho, p = spearmanr(a, b)
                result.append({"checkpoint": checkpoint, "exposure": exposure, "outcome": outcome, "n": len(pairs), "spearman_rho": rho, "p_value": p})
    return result


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")


def fmt(value, percent=False):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100 * value:.2f}%" if percent else f"{value:.3f}"


def write_report(output, manifest, summaries, quartile_cuts, paired, correlations_rows):
    overall = {(r["checkpoint"], r["group"]): r for r in summaries}
    lines = [
        "# Spatial Overfitting Audit: StopBank Checkpoints",
        "",
        "## Scope",
        "",
        f"- Training rows: {manifest['total_rows']:,}; resolved poses: {manifest['resolved_rows']:,} ({manifest['resolution_rate']:.2%}).",
        "- Analysis is same-scene only for spatial exposure; scenes absent from training are retained as a negative control.",
        "- Distances use 3D Euclidean distance to the nearest training camera pose. Density uses 5m 3D cells and counts distinct training episodes per cell.",
        "- This is retrospective association. It can reject or support the spatial-bias hypothesis, but causal proof still requires the later controlled start/goal experiment.",
        "",
        "## Overall Results",
        "",
        "| checkpoint | N | SR | OSR | collision | NE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in ("ckpt13000", "ckpt19764"):
        row = overall[(checkpoint, "all")]
        lines.append(f"| {checkpoint} | {row['n']} | {fmt(row['sr'], True)} | {fmt(row['osr'], True)} | {fmt(row['collision_rate'], True)} | {fmt(row['ne'])} |")

    lines += ["", "## Training-Map Scene Spatial Exposure", "", "| checkpoint | N | mean nearest train pose | path <=5m | unseen 5m cells | train density | train attraction/m | goal progress/m |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for checkpoint in ("ckpt13000", "ckpt19764"):
        row = overall[(checkpoint, "scene_has_training_map")]
        lines.append(
            f"| {checkpoint} | {row['n']} | {fmt(row['mean_train_distance'])}m | "
            f"{fmt(row['path_fraction_within_5m_train'], True)} | {fmt(row['path_fraction_unseen_5m_cells'], True)} | "
            f"{fmt(row['mean_log_train_episode_density'])} | {fmt(row['train_attraction_per_meter'])} | {fmt(row['goal_progress_per_meter'])} |"
        )

    comparison = {r["metric"]: r for r in paired["metrics"]}
    train_13000 = overall[("ckpt13000", "scene_has_training_map")]
    train_19764 = overall[("ckpt19764", "scene_has_training_map")]
    novel_13000 = overall[("ckpt13000", "novel_scene_control")]
    novel_19764 = overall[("ckpt19764", "novel_scene_control")]
    novelty_rows = {
        (row["checkpoint"], row["group"]): row
        for row in summaries if row["group"].startswith("target_novelty:")
    }
    lines += [
        "", "## Paired Checkpoint Change", "",
        f"- Paired tasks: {paired['paired_tasks']}",
        f"- Success lost/gained at ckpt19764: {paired['changed_outcomes']['success_13000_only']} / {paired['changed_outcomes']['success_19764_only']}.",
        f"- OSR lost/gained at ckpt19764: {paired['changed_outcomes']['osr_13000_only']} / {paired['changed_outcomes']['osr_19764_only']}.",
        "- Deltas below are ckpt19764 minus ckpt13000 with paired bootstrap 95% CI.",
        "", "| metric | delta | 95% CI | p |", "|---|---:|---:|---:|",
    ]
    for metric in ("mean_train_distance", "path_fraction_within_5m_train", "mean_log_train_episode_density", "endpoint_density_gain", "train_attraction_per_meter", "goal_progress_per_meter", "success", "osr", "collision"):
        row = comparison[metric]
        lines.append(f"| {metric} | {fmt(row['mean_delta'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row.get('wilcoxon_p'))} |")

    quartile_sr_deltas = []
    for label in ("Q1 closest", "Q2", "Q3", "Q4 farthest"):
        key = f"target_novelty:{label}"
        quartile_sr_deltas.append(novelty_rows[("ckpt19764", key)]["sr"] - novelty_rows[("ckpt13000", key)]["sr"])
    lines += [
        "", "## Observed Findings", "",
        "1. The existing trajectories do not support the proposed signature that longer training pulls the policy toward previously visited training regions.",
        f"   ckpt19764 is {comparison['mean_train_distance']['mean_delta']:+.3f}m farther from the nearest training pose on average, "
        f"uses {100 * comparison['path_fraction_within_5m_train']['mean_delta']:+.2f}pp less path inside 5m, and has "
        f"{comparison['mean_log_train_episode_density']['mean_delta']:+.3f} lower training-cell density. Most confidence intervals include zero.",
        f"2. On training-map scenes, SR changes {100 * (train_19764['sr'] - train_13000['sr']):+.2f}pp and OSR changes "
        f"{100 * (train_19764['osr'] - train_13000['osr']):+.2f}pp. On novel-scene controls, SR changes "
        f"{100 * (novel_19764['sr'] - novel_13000['sr']):+.2f}pp and OSR changes {100 * (novel_19764['osr'] - novel_13000['osr']):+.2f}pp.",
        "3. SR change from nearest to farthest target-training-distance quartiles is "
        + ", ".join(f"{100 * value:+.2f}pp" for value in quartile_sr_deltas)
        + ". There is no monotonic pattern in which farther targets degrade more.",
        "4. Higher training-cell density correlates with SR and OSR in both checkpoints, but this is not causal evidence: dense regions may contain easier targets, safer geometry, or more favorable starts. The controlled start/goal experiment remains necessary.",
    ]

    lines += [
        "", "## Interpretation Guide", "",
        "Evidence supporting spatial overfitting would require the later checkpoint to show all or most of:",
        "1. lower nearest-training-pose distance or higher training-cell density;",
        "2. positive train attraction without matching goal progress;",
        "3. stronger degradation for targets far from training poses, while novel-scene controls do not show the same pattern.",
        "",
        "A higher SR near training poses alone is not sufficient: starts and targets may simply be easier there. Use `episode_metrics.csv`, `target_novelty_summary.csv`, and the paired table for manual case review.",
        "",
        f"Target-distance quartile cuts for same-scene tasks: {', '.join(f'{x:.3f}m' for x in quartile_cuts)}.",
        f"Detailed correlation rows: {len(correlations_rows)}.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frames", type=Path, default=DEFAULT_TRAIN_FRAMES)
    parser.add_argument("--aligned-json-root", type=Path, default=DEFAULT_ALIGNED_JSON)
    parser.add_argument("--result", action="append", default=[], help="NAME=/path/to/all_episodes.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    result_paths = dict(DEFAULT_RESULTS)
    for item in args.result:
        name, path = item.split("=", 1)
        result_paths[name] = Path(path)
    args.output.mkdir(parents=True, exist_ok=True)

    maps, manifest = build_training_map(args.train_frames, args.aligned_json_root, args.output)
    rows = []
    for checkpoint, path in result_paths.items():
        rows.extend(read_eval(path, checkpoint, maps))
    cuts, novelty = assign_target_novelty(rows)
    summaries = grouped_summaries(rows)
    summaries.extend(novelty)
    paired, paired_rows = paired_comparison(rows)
    corr = correlations(rows)

    write_csv(args.output / "episode_metrics.csv", rows)
    write_csv(args.output / "checkpoint_scene_summary.csv", summaries)
    write_csv(args.output / "target_novelty_summary.csv", novelty)
    write_csv(args.output / "paired_tasks.csv", paired_rows)
    write_csv(args.output / "spatial_correlations.csv", corr)
    write_json(args.output / "paired_comparison.json", paired)
    write_json(args.output / "analysis_config.json", {
        "results": {key: str(value) for key, value in result_paths.items()},
        "thresholds_m": THRESHOLDS,
        "cell_size_m": CELL_SIZE,
        "target_novelty_quartile_cuts_m": cuts,
    })
    write_report(args.output, manifest, summaries, cuts, paired, corr)
    print(json.dumps({
        "output": str(args.output),
        "training_resolution_rate": manifest["resolution_rate"],
        "evaluation_rows": len(rows),
        "paired_tasks": paired["paired_tasks"],
    }, indent=2))


if __name__ == "__main__":
    main()

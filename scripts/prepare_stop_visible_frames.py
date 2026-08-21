#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

from vlm_baseline.stop_visibility import VisibilityPolicy, select_first_clear_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_SOURCE = DATASET_ROOT / "processed" / "nomemory_baseline" / "train_frames.jsonl"
DEFAULT_CACHE = DATASET_ROOT / "processed" / "stop_visible_v1" / "visibility_cache"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed" / "stop_visible_v1"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
STOP_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def iter_cache_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(path)


def load_visibility_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    for cache_file in iter_cache_files(path):
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" not in row or "trajectory_key" not in row:
                continue
            cache[str(row["trajectory_key"])] = row
    return cache


def load_semantic_scores(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    scores = {}
    for score_file in iter_cache_files(path):
        for line in score_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["trajectory_key"]), int(row["frame_idx"]))
            scores[key] = {
                name: value
                for name, value in row.items()
                if name not in {"trajectory_key", "frame_idx"}
            }
    return scores


def attach_semantic_scores(
    visibility: dict[str, dict[str, Any]],
    semantic_scores: dict[tuple[str, int], dict[str, Any]],
) -> tuple[int, int]:
    attached = 0
    visible_without_score = 0
    for trajectory_key, trajectory in visibility.items():
        for frame in trajectory.get("frames") or []:
            score = semantic_scores.get((trajectory_key, int(frame["frame_idx"])))
            if score is not None:
                frame.update(score)
                attached += 1
            elif int((frame.get("mask") or {}).get("pixel_count", 0)) > 0:
                visible_without_score += 1
    return attached, visible_without_score


def load_source_groups(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        groups.setdefault(str(row["episode_key"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["frame_idx"]))
    return groups


def rewrite_trajectory(
    rows: list[dict[str, Any]],
    selected_frame_idx: int,
    selection: dict[str, Any],
    aligned_root: Path,
) -> list[dict[str, Any]]:
    rewritten = []
    for source_row in rows:
        frame_idx = int(source_row["frame_idx"])
        if frame_idx > selected_frame_idx:
            break
        row = dict(source_row)
        normalized = str(row["image_path"]).replace("\\", "/")
        marker = "record_output/images/"
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            aligned_image = aligned_root / "images" / suffix
            if not aligned_image.is_file():
                raise FileNotFoundError(aligned_image)
            row["image_path"] = str(aligned_image.resolve())
        if frame_idx == selected_frame_idx:
            selected_image_path = selection.get("selected_replay_image_path")
            if selected_image_path:
                selected_image = Path(selected_image_path)
                if not selected_image.is_file():
                    raise FileNotFoundError(selected_image)
                row["image_path"] = str(selected_image.resolve())
            row["original_action_name"] = row.get("action_name")
            row["original_action_vector"] = row.get("action_vector")
            row["action_name"] = "Stop"
            row["action_vector"] = list(STOP_VECTOR)
            stop_visibility = {
                "selected": True,
                "peak_pixels": selection["peak_pixels"],
                "size_bucket": selection["size_bucket"],
                "selection_mode": selection["selection_mode"],
                "quality_score": selection.get("selected_quality_score"),
                "image_source": (
                    "synchronized_replay"
                    if selected_image_path
                    else "original_recording"
                ),
            }
            row["stop_visibility"] = stop_visibility
            if selection["selection_mode"] == "first_clear":
                row["stop_visible_v1"] = stop_visibility
        rewritten.append(row)
    if not rewritten or int(rewritten[-1]["frame_idx"]) != selected_frame_idx:
        raise KeyError(f"selected frame {selected_frame_idx} is missing from source trajectory")
    return rewritten


def build_policy(args: argparse.Namespace) -> VisibilityPolicy:
    if args.policy_config:
        payload = json.loads(args.policy_config.read_text(encoding="utf-8"))
        return VisibilityPolicy(**payload.get("policy", payload))
    return VisibilityPolicy(
        min_pixels={
            "small": args.min_pixels_small,
            "mid": args.min_pixels_mid,
            "big": args.min_pixels_big,
            "unknown": args.min_pixels_unknown,
        },
        min_bbox_short_side={
            "small": args.min_bbox_small,
            "mid": args.min_bbox_mid,
            "big": args.min_bbox_big,
            "unknown": args.min_bbox_unknown,
        },
        strong_pixels={
            "small": args.strong_pixels_small,
            "mid": args.strong_pixels_mid,
            "big": args.strong_pixels_big,
            "unknown": args.strong_pixels_unknown,
        },
        min_relative_to_peak=args.min_relative_to_peak,
        max_pixel_fraction=args.max_pixel_fraction,
        max_edge_contact_fraction=args.max_edge_contact_fraction,
        max_clipped_sides=args.max_clipped_sides,
        reject_opposite_borders=not args.allow_opposite_borders,
        semantic_score_field=args.semantic_score_field or None,
        min_semantic_score=args.min_semantic_score,
        semantic_rank_field=args.semantic_rank_field or None,
        max_semantic_rank=args.max_semantic_rank,
        require_semantic_for_weak_geometry=args.require_semantic_for_weak_geometry,
    )


def convert(
    source: Path,
    cache_path: Path,
    semantic_scores_path: Path | None,
    aligned_root: Path,
    policy_config_path: Path | None,
    output_dir: Path,
    policy: VisibilityPolicy,
    only_cached: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_frames.jsonl"
    selections_path = output_dir / "selections.jsonl"
    rejected_path = output_dir / "rejected_trajectories.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in (output_path, selections_path, rejected_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")

    visibility = load_visibility_cache(cache_path)
    semantic_scores = load_semantic_scores(semantic_scores_path)
    attached_scores, visible_without_score = attach_semantic_scores(
        visibility,
        semantic_scores,
    )
    groups = load_source_groups(source)
    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    stop_shift_counts: Counter[str] = Counter()

    with ExitStack() as stack:
        output = stack.enter_context(output_path.open("w", encoding="utf-8"))
        selections = stack.enter_context(selections_path.open("w", encoding="utf-8"))
        rejected = stack.enter_context(rejected_path.open("w", encoding="utf-8"))
        for key, rows in groups.items():
            cached = visibility.get(key)
            if cached is None:
                stats["uncached_trajectories"] += 1
                if only_cached:
                    continue
                raise KeyError(f"visibility cache missing trajectory: {key}")
            stats["cached_trajectories"] += 1
            if cached.get("status") != "ok":
                stats["rejected_trajectories"] += 1
                rejected.write(
                    json.dumps(
                        {
                            "trajectory_key": key,
                            "reason": cached.get("status"),
                            "error": cached.get("error"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            selection = select_first_clear_frame(
                cached.get("frames") or [],
                cached.get("size"),
                policy,
            )
            selected_idx = selection["selected_frame_idx"]
            selected_cached_frame = next(
                (
                    frame
                    for frame in cached.get("frames") or []
                    if int(frame["frame_idx"]) == selected_idx
                ),
                None,
            )
            decision = {
                "trajectory_key": key,
                "scene_id": cached.get("scene_id"),
                "episode_id": cached.get("episode_id"),
                "pose_idx": cached.get("pose_idx"),
                "true_name": cached.get("true_name"),
                "object_name": cached.get("object_name"),
                "size": cached.get("size"),
                **selection,
                "selected_replay_image_path": (
                    selected_cached_frame.get("replay_image_path")
                    if selected_cached_frame
                    else None
                ),
            }
            selections.write(json.dumps(decision, ensure_ascii=False) + "\n")
            if selected_idx is None:
                stats["rejected_trajectories"] += 1
                rejected.write(
                    json.dumps(
                        {
                            "trajectory_key": key,
                            "reason": "no_clear_visible_frame_within_20m",
                            "selection": selection,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            rewritten = rewrite_trajectory(
                rows,
                int(selected_idx),
                decision,
                aligned_root,
            )
            original_stop_indices = [
                int(row["frame_idx"])
                for row in rows
                if str(row.get("action_name")) == "Stop"
            ]
            original_stop_idx = original_stop_indices[-1] if original_stop_indices else max(
                int(row["frame_idx"]) for row in rows
            )
            shift = original_stop_idx - int(selected_idx)
            stop_shift_counts[str(shift)] += 1
            stats["retained_trajectories"] += 1
            stats["rows_before_retained"] += len(rows)
            stats["rows_written"] += len(rewritten)
            stats["rows_removed_after_new_stop"] += len(rows) - len(rewritten)
            if shift > 0:
                stats["stop_moved_earlier"] += 1
            elif shift == 0:
                stats["stop_unchanged"] += 1
            else:
                stats["stop_moved_later"] += 1
            for row in rewritten:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                action_counts[str(row["action_name"])] += 1
                scene_counts[str(row["scene_id"])] += 1

    is_quality_v2 = policy.selection_mode == "best_recognition_view"
    manifest = {
        "format": (
            "uavon_train_frames_stop_visible_v2"
            if is_quality_v2
            else "uavon_train_frames_stop_visible_v1"
        ),
        "source": str(source),
        "aligned_root": str(aligned_root),
        "visibility_cache": str(cache_path),
        "semantic_scores": str(semantic_scores_path) if semantic_scores_path else None,
        "policy_config": str(policy_config_path) if policy_config_path else None,
        "semantic_scores_attached": attached_scores,
        "visible_frames_without_semantic_score": visible_without_score,
        "output": str(output_path),
        "selections": str(selections_path),
        "rejected": str(rejected_path),
        "only_cached": only_cached,
        "policy": policy.to_dict(),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "stop_shift_histogram": dict(
            sorted(stop_shift_counts.items(), key=lambda item: int(item[0]))
        ),
        "notes": [
            (
                "Select the best recognizable target-instance view within 20m "
                "using occupancy, centering, clipping, distance, and semantic quality."
                if is_quality_v2
                else "Select the first clear target-instance frame within 20m."
            ),
            "Relabel the selected frame as Stop and remove all later frames.",
            (
                "Use synchronized replay RGB for the selected Stop frame when available; "
                "other RGB rows and depth-grid keys are reused."
            ),
            "Original data is not modified.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    defaults = VisibilityPolicy()
    parser = argparse.ArgumentParser(
        description="Build a non-destructive training-frame copy with visible Stop labels."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--visibility-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--semantic-scores", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-cached", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-pixels-small", type=int, default=defaults.min_pixels["small"])
    parser.add_argument("--min-pixels-mid", type=int, default=defaults.min_pixels["mid"])
    parser.add_argument("--min-pixels-big", type=int, default=defaults.min_pixels["big"])
    parser.add_argument("--min-pixels-unknown", type=int, default=defaults.min_pixels["unknown"])
    parser.add_argument("--min-bbox-small", type=int, default=defaults.min_bbox_short_side["small"])
    parser.add_argument("--min-bbox-mid", type=int, default=defaults.min_bbox_short_side["mid"])
    parser.add_argument("--min-bbox-big", type=int, default=defaults.min_bbox_short_side["big"])
    parser.add_argument("--min-bbox-unknown", type=int, default=defaults.min_bbox_short_side["unknown"])
    parser.add_argument("--strong-pixels-small", type=int, default=defaults.strong_pixels["small"])
    parser.add_argument("--strong-pixels-mid", type=int, default=defaults.strong_pixels["mid"])
    parser.add_argument("--strong-pixels-big", type=int, default=defaults.strong_pixels["big"])
    parser.add_argument(
        "--strong-pixels-unknown",
        type=int,
        default=defaults.strong_pixels["unknown"],
    )
    parser.add_argument("--min-relative-to-peak", type=float, default=defaults.min_relative_to_peak)
    parser.add_argument("--max-pixel-fraction", type=float, default=defaults.max_pixel_fraction)
    parser.add_argument(
        "--max-edge-contact-fraction",
        type=float,
        default=defaults.max_edge_contact_fraction,
    )
    parser.add_argument("--max-clipped-sides", type=int, default=defaults.max_clipped_sides)
    parser.add_argument("--allow-opposite-borders", action="store_true")
    parser.add_argument("--semantic-score-field", default="")
    parser.add_argument("--min-semantic-score", type=float)
    parser.add_argument("--semantic-rank-field", default="")
    parser.add_argument("--max-semantic-rank", type=int)
    parser.add_argument("--require-semantic-for-weak-geometry", action="store_true")
    args = parser.parse_args()

    manifest = convert(
        source=args.source,
        cache_path=args.visibility_cache,
        semantic_scores_path=args.semantic_scores,
        aligned_root=args.aligned_root,
        policy_config_path=args.policy_config,
        output_dir=args.output_dir,
        policy=build_policy(args),
        only_cached=args.only_cached,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

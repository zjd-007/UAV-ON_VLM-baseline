#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


STOP_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def sample_key(row: dict[str, Any], frame_idx: int | None = None) -> str:
    return (
        f"{row['scene_id']}::{row['episode_id']}::{row['pose_idx']}::"
        f"{int(row['frame_idx'] if frame_idx is None else frame_idx)}"
    )


def actor_key(row: dict[str, Any]) -> str:
    return f"{row['scene_id']}::{row.get('object_name') or ''}"


def quality_score(row: dict[str, Any]) -> float:
    value = (row.get("stop_visibility") or {}).get("quality_score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.int16,
        )
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def select_diverse_candidates(
    rows: list[dict[str, Any]],
    bank_size: int,
) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = Path(str(row["image_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = str(path)
        current = by_image.get(key)
        if current is None or quality_score(row) > quality_score(current):
            by_image[key] = row

    candidates = []
    for row in by_image.values():
        item = copy.deepcopy(row)
        item["_image_hash"] = difference_hash(Path(str(row["image_path"])))
        candidates.append(item)
    candidates.sort(key=lambda row: (-quality_score(row), str(row["image_path"])))
    if len(candidates) <= bank_size:
        return candidates

    selected = [candidates.pop(0)]
    while candidates and len(selected) < bank_size:
        def selection_score(row: dict[str, Any]) -> tuple[float, float, str]:
            diversity = min(
                hamming_distance(row["_image_hash"], chosen["_image_hash"])
                for chosen in selected
            ) / 64.0
            return (
                quality_score(row) + 0.25 * diversity,
                diversity,
                str(row["image_path"]),
            )

        chosen = max(candidates, key=selection_score)
        candidates.remove(chosen)
        selected.append(chosen)
    return selected


def load_depth_grids(
    cache_dir: Path,
    required_keys: set[str],
) -> dict[str, list[list[float]]]:
    result = {}
    for path in sorted(cache_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            key = str(row.get("key") or sample_key(row))
            if key in required_keys:
                result[key] = row["depth_grid"]
    return result


def iter_episode_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
    current_key = None
    rows: list[dict[str, Any]] = []
    closed: set[str] = set()
    for row in read_jsonl(path):
        key = str(row["episode_key"])
        if current_key is not None and key != current_key:
            if current_key in closed:
                raise ValueError(f"source frames are not grouped: {current_key}")
            closed.add(current_key)
            yield rows
            rows = []
        current_key = key
        rows.append(row)
    if rows:
        if str(rows[0]["episode_key"]) in closed:
            raise ValueError(f"source frames are not grouped: {rows[0]['episode_key']}")
        yield rows


def build_appended_stop(
    episode_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    bank_id: str,
    candidate_index: int,
    candidate_count: int,
    reuse_index: int,
    depth_grid: list[list[float]],
) -> dict[str, Any]:
    previous = max(episode_rows, key=lambda row: int(row["frame_idx"]))
    next_frame_idx = int(previous["frame_idx"]) + 1
    row = copy.deepcopy(candidate)
    row.pop("_image_hash", None)
    source_visibility = row.get("stop_visibility") or {}
    source_episode_key = str(row["episode_key"])
    source_frame_idx = int(row["frame_idx"])
    source_mask_frame_idx = int(
        source_visibility.get("source_original_frame_idx", source_frame_idx)
    )
    source_image_path = str(row["image_path"])
    source_record = row.get("source_record")

    for field in ("target_description", "true_name", "object_name", "size"):
        row[field] = previous.get(field)
    row.update(
        {
            "episode_key": str(previous["episode_key"]),
            "scene_id": str(previous["scene_id"]),
            "episode_id": str(previous["episode_id"]),
            "pose_idx": str(previous["pose_idx"]),
            "frame_idx": next_frame_idx,
            "step_id": next_frame_idx,
            "map_name": previous.get("map_name"),
            "source_record": previous.get("source_record"),
            "action_name": "Stop",
            "action_id": 0,
            "uavon_action": "stop",
            "action_vector": list(STOP_VECTOR),
            "depth_grid": depth_grid,
            "coordinate_repair": previous.get("coordinate_repair"),
            "coordinate_repair_start_recovery": previous.get(
                "coordinate_repair_start_recovery"
            ),
            "trajectory_repair": {
                "version": "actor_stop_bank_v1",
                "mode": "navigation_with_actor_stop_bank_appended",
                "problem_cause": (
                    previous.get("trajectory_repair") or {}
                ).get("problem_cause"),
                "appended_after_frame_idx": int(previous["frame_idx"]),
                "per_frame_training_only": True,
                "cross_row_continuity_required": False,
            },
            "stop_visibility": {
                **source_visibility,
                "selected": True,
                "version": "actor_stop_bank_v1",
                "source_type": "actor_stop_bank_appended_to_original_episode",
                "source_actor_key": actor_key(previous),
                "source_stop_episode_key": source_episode_key,
                "source_stop_frame_idx": source_frame_idx,
                "source_stop_mask_frame_idx": source_mask_frame_idx,
                "source_stop_source_type": source_visibility.get("source_type"),
                "source_stop_record": source_record,
                "source_image_path": source_image_path,
                "actor_stop_bank_id": bank_id,
                "actor_stop_bank_candidate_index": candidate_index,
                "actor_stop_bank_candidate_count": candidate_count,
                "actor_stop_bank_reuse_index": reuse_index,
                "appended_to_original_trajectory": True,
                "not_appended_to_original_trajectory": False,
                "recaptured_pose_is_not_physical_next_pose": True,
                "per_frame_training_only": True,
            },
        }
    )
    if row.get("pose") is not None:
        row["next_pose"] = copy.deepcopy(row["pose"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append a diverse, actor-matched Stop bank view to navigation-only "
            "episodes in a single-frame SFT dataset."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-quarantine", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True)
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--assignments-output", type=Path, required=True)
    parser.add_argument("--unresolved-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=5)
    args = parser.parse_args()
    if args.bank_size < 1:
        raise ValueError("--bank-size must be positive")
    output_paths = (
        args.output,
        args.quarantine_output,
        args.bank_output,
        args.assignments_output,
        args.unresolved_output,
        args.manifest,
    )
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    episode_info: dict[str, dict[str, Any]] = {}
    stop_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    for row in read_jsonl(args.source):
        source_rows += 1
        episode_key = str(row["episode_key"])
        info = episode_info.setdefault(
            episode_key,
            {
                "actor_key": actor_key(row),
                "scene_id": str(row["scene_id"]),
                "object_name": str(row.get("object_name") or ""),
                "true_name": str(row.get("true_name") or ""),
                "has_stop": False,
                "rows": 0,
            },
        )
        info["rows"] += 1
        if str(row.get("action_name") or "").strip().lower() == "stop":
            info["has_stop"] = True
            stop_candidates[actor_key(row)].append(copy.deepcopy(row))

    navigation_only_by_actor: dict[str, list[str]] = defaultdict(list)
    for episode_key, info in episode_info.items():
        if not info["has_stop"]:
            navigation_only_by_actor[info["actor_key"]].append(episode_key)
    for keys in navigation_only_by_actor.values():
        keys.sort()

    selected_banks: dict[str, list[dict[str, Any]]] = {}
    unresolved_actors = []
    for key in sorted(navigation_only_by_actor):
        candidates = stop_candidates.get(key) or []
        if not candidates:
            example = episode_info[navigation_only_by_actor[key][0]]
            unresolved_actors.append(
                {
                    "actor_key": key,
                    "scene_id": example["scene_id"],
                    "object_name": example["object_name"],
                    "true_name": example["true_name"],
                    "navigation_only_episode_count": len(
                        navigation_only_by_actor[key]
                    ),
                    "decision": "retain_navigation_only",
                    "reason": "no_recognizable_actor_stop_bank_candidate",
                }
            )
            continue
        selected_banks[key] = select_diverse_candidates(candidates, args.bank_size)

    required_depth_keys = {
        sample_key(row)
        for bank in selected_banks.values()
        for row in bank
        if row.get("depth_grid") is None
    }
    depth_grids = load_depth_grids(args.depth_cache, required_depth_keys)
    missing_depth = sorted(required_depth_keys - set(depth_grids))
    if missing_depth:
        raise KeyError(
            f"depth cache missing {len(missing_depth)} Stop bank rows: "
            f"{missing_depth[:10]}"
        )

    bank_rows = []
    for key, bank in sorted(selected_banks.items()):
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        bank_id = f"actor_stop_bank_{digest}"
        for index, row in enumerate(bank):
            bank_rows.append(
                {
                    "actor_key": key,
                    "bank_id": bank_id,
                    "candidate_index": index,
                    "candidate_count": len(bank),
                    "source_episode_key": row["episode_key"],
                    "source_frame_idx": int(row["frame_idx"]),
                    "source_mask_frame_idx": int(
                        (row.get("stop_visibility") or {}).get(
                            "source_original_frame_idx", row["frame_idx"]
                        )
                    ),
                    "source_type": (row.get("stop_visibility") or {}).get(
                        "source_type"
                    ),
                    "quality_score": quality_score(row),
                    "image_path": row["image_path"],
                    "perceptual_hash": f"{row['_image_hash']:016x}",
                }
            )

    assignments = []
    assignment_by_episode = {}
    reuse_counts: Counter[tuple[str, int]] = Counter()
    for key, episode_keys in sorted(navigation_only_by_actor.items()):
        bank = selected_banks.get(key)
        if not bank:
            continue
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        bank_id = f"actor_stop_bank_{digest}"
        for assignment_index, episode_key in enumerate(episode_keys):
            candidate_index = assignment_index % len(bank)
            reuse_counts[(key, candidate_index)] += 1
            assignment = {
                "episode_key": episode_key,
                "actor_key": key,
                "bank_id": bank_id,
                "candidate_index": candidate_index,
                "candidate_count": len(bank),
                "reuse_index": reuse_counts[(key, candidate_index)],
                "source_stop_episode_key": bank[candidate_index]["episode_key"],
                "source_stop_frame_idx": int(bank[candidate_index]["frame_idx"]),
                "source_image_path": bank[candidate_index]["image_path"],
            }
            assignments.append(assignment)
            assignment_by_episode[episode_key] = assignment

    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    output_episodes: set[str] = set()
    output_samples: set[str] = set()
    navigation_only_output: set[str] = set()

    def write_row(output, row: dict[str, Any], source: str) -> None:
        key = sample_key(row)
        if key in output_samples:
            raise ValueError(f"duplicate output sample key: {key}")
        output_samples.add(key)
        output_episodes.add(str(row["episode_key"]))
        action_counts[str(row["action_name"])] += 1
        scene_counts[str(row["scene_id"])] += 1
        stats["rows"] += 1
        stats[f"rows_{source}"] += 1
        output.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.output.open("x", encoding="utf-8") as output:
        for episode_rows in iter_episode_groups(args.source):
            episode_key = str(episode_rows[0]["episode_key"])
            for row in episode_rows:
                write_row(output, row, "source")
            assignment = assignment_by_episode.get(episode_key)
            if assignment is None:
                if not any(
                    str(row.get("action_name") or "").strip().lower() == "stop"
                    for row in episode_rows
                ):
                    navigation_only_output.add(episode_key)
                continue
            candidate = selected_banks[assignment["actor_key"]][
                assignment["candidate_index"]
            ]
            depth_grid = candidate.get("depth_grid") or depth_grids[
                sample_key(candidate)
            ]
            stop = build_appended_stop(
                episode_rows,
                candidate,
                assignment["bank_id"],
                assignment["candidate_index"],
                assignment["candidate_count"],
                assignment["reuse_index"],
                depth_grid,
            )
            write_row(output, stop, "actor_stop_bank")
            stats["episodes_repaired_with_actor_stop_bank"] += 1

    shutil.copyfile(args.source_quarantine, args.quarantine_output)
    with args.bank_output.open("x", encoding="utf-8") as output:
        for row in bank_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.assignments_output.open("x", encoding="utf-8") as output:
        for row in assignments:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.unresolved_output.open("x", encoding="utf-8") as output:
        for row in unresolved_actors:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(output_episodes) != len(episode_info):
        raise AssertionError(
            f"episode count changed: {len(output_episodes)} != {len(episode_info)}"
        )
    expected_rows = source_rows + len(assignments)
    if len(output_samples) != expected_rows:
        raise AssertionError(
            f"row count mismatch: {len(output_samples)} != {expected_rows}"
        )
    manifest = {
        "format": "uavon_actor_stop_bank_appended_v1",
        "training_mode": "single_frame_sft",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "source_rows": source_rows,
        "source_episodes": len(episode_info),
        "source_navigation_only_episodes": sum(
            len(keys) for keys in navigation_only_by_actor.values()
        ),
        "source_navigation_only_actors": len(navigation_only_by_actor),
        "covered_actor_banks": len(selected_banks),
        "unresolved_actors": len(unresolved_actors),
        "unresolved_navigation_only_episodes": len(navigation_only_output),
        "bank_size_limit": args.bank_size,
        "bank_candidates": len(bank_rows),
        "assignments": len(assignments),
        "maximum_single_candidate_reuse": max(reuse_counts.values(), default=0),
        "output_rows": len(output_samples),
        "output_episodes": len(output_episodes),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "scene_counts": dict(scene_counts),
        "outputs": {
            "quarantine": str(args.quarantine_output.resolve()),
            "actor_stop_bank": str(args.bank_output.resolve()),
            "assignments": str(args.assignments_output.resolve()),
            "unresolved_actors": str(args.unresolved_output.resolve()),
        },
        "policy": {
            "actor_identity": "exact (scene_id, object_name) match",
            "candidate_source": "previously validated clear Stop rows",
            "candidate_deduplication": "resolved image path",
            "candidate_selection": (
                "greedy quality plus perceptual dHash diversity, at most five views"
            ),
            "assignment": "deterministic round-robin within each actor bank",
            "unresolved": "retain original navigation-only episode",
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_stop_visible_frames import (  # noqa: E402
    attach_semantic_scores,
    load_semantic_scores,
    load_visibility_cache,
)
from vlm_baseline.stop_visibility import VisibilityPolicy, select_first_clear_frame  # noqa: E402


STOP_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_capture_rows(
    paths: list[Path], allow_later_override: bool = False
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = {}
    overridden = []
    for path in paths:
        cache = load_visibility_cache(path)
        overlap = set(rows) & set(cache)
        if overlap and not allow_later_override:
            raise ValueError(f"duplicate standoff trajectory keys: {sorted(overlap)}")
        overridden.extend(sorted(overlap))
        rows.update(cache)
    return rows, overridden


def load_all_semantic_scores(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    scores = {}
    for path in paths:
        scores.update(load_semantic_scores(path))
    return scores


def load_policy(path: Path) -> VisibilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return replace(
        VisibilityPolicy(**payload.get("policy", payload)),
        reject_collided=True,
    )


def synthetic_episode_id(scene: str, object_name: str) -> str:
    digest = hashlib.sha1(f"{scene}::{object_name}".encode("utf-8")).hexdigest()[:12]
    return f"standoff_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated independent Stop bank from target-facing captures."
    )
    parser.add_argument("--capture-cache", type=Path, action="append", required=True)
    parser.add_argument("--semantic-scores", type=Path, action="append", required=True)
    parser.add_argument("--queue-manifest", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-later-capture-override", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_frames": args.output_dir / "train_frames_standoff_stop_bank.jsonl",
        "selections": args.output_dir / "standoff_selections.jsonl",
        "rejected": args.output_dir / "standoff_rejected.jsonl",
        "manifest": args.output_dir / "standoff_stop_bank_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite Stop-bank outputs: {existing}")

    captures, overridden_capture_keys = load_capture_rows(
        args.capture_cache,
        allow_later_override=args.allow_later_capture_override,
    )
    semantic = load_all_semantic_scores(args.semantic_scores)
    attached, missing = attach_semantic_scores(captures, semantic)
    queue = {}
    for queue_manifest in args.queue_manifest:
        queue.update(
            {
                str(row["trajectory_key"]): row
                for row in read_jsonl(queue_manifest)
            }
        )
    policy = load_policy(args.policy)
    stats: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    with (
        paths["train_frames"].open("x", encoding="utf-8") as output,
        paths["selections"].open("x", encoding="utf-8") as selections,
        paths["rejected"].open("x", encoding="utf-8") as rejected,
    ):
        for key, row in sorted(captures.items()):
            queue_row = queue.get(key)
            if queue_row is None:
                raise KeyError(f"capture not present in queue manifest: {key}")
            selection = select_first_clear_frame(
                row.get("frames") or [], row.get("size"), policy
            )
            selected_idx = selection.get("selected_frame_idx")
            selection_record = {
                "trajectory_key": key,
                "capture_group": queue_row["capture_group"],
                "scene_id": row.get("scene_id"),
                "object_name": row.get("object_name"),
                "true_name": row.get("true_name"),
                "represented_trajectory_count": queue_row[
                    "represented_trajectory_count"
                ],
                **selection,
            }
            selections.write(json.dumps(selection_record, ensure_ascii=False) + "\n")
            stats["actors_processed"] += 1
            stats[f"actors_{queue_row['capture_group']}"] += 1
            if selected_idx is None:
                stats["actors_without_eligible_stop"] += 1
                stats[f"rejected_{queue_row['capture_group']}"] += 1
                rejected.write(
                    json.dumps(selection_record, ensure_ascii=False) + "\n"
                )
                continue
            selected = next(
                frame
                for frame in row.get("frames") or []
                if int(frame["frame_idx"]) == int(selected_idx)
            )
            depth_grid = selected.get("depth_grid")
            if depth_grid is None:
                raise KeyError(f"selected standoff frame has no depth grid: {key}::{selected_idx}")
            image_path = Path(str(selected["replay_image_path"]))
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            scene = str(row["scene_id"])
            episode_id = synthetic_episode_id(scene, str(row["object_name"]))
            sample = {
                "episode_key": f"{scene}::{episode_id}::0",
                "scene_id": scene,
                "episode_id": episode_id,
                "pose_idx": "0",
                "frame_idx": 0,
                "image_path": str(image_path.resolve()),
                "target_description": str(row["target_description"]),
                "true_name": str(row["true_name"]),
                "object_name": str(row["object_name"]),
                "size": str(row["size"]),
                "action_name": "Stop",
                "action_vector": list(STOP_VECTOR),
                "depth_grid": depth_grid,
                "pose": selected.get("pose"),
                "stop_visibility": {
                    "selected": True,
                    "version": "v4_production",
                    "source_type": "independent_target_facing_standoff_stop_bank",
                    "source_trajectory_key": key,
                    "source_candidate_frame_idx": int(selected_idx),
                    "capture_group": queue_row["capture_group"],
                    "quality_score": selection.get("selected_quality_score"),
                    "represented_trajectory_count": queue_row[
                        "represented_trajectory_count"
                    ],
                    "not_appended_to_original_trajectory": True,
                },
            }
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["stop_samples_written"] += 1
            stats[f"written_{queue_row['capture_group']}"] += 1
            stats["represented_trajectories_with_new_stop_coverage"] += int(
                queue_row["represented_trajectory_count"]
            )
            action_counts["Stop"] += 1

    manifest = {
        "format": "uavon_independent_standoff_stop_bank_v4",
        "capture_caches": [str(path.resolve()) for path in args.capture_cache],
        "semantic_scores": [str(path.resolve()) for path in args.semantic_scores],
        "semantic_scores_attached": attached,
        "visible_frames_without_semantic_score": missing,
        "queue_manifests": [str(path.resolve()) for path in args.queue_manifest],
        "policy": str(args.policy.resolve()),
        "deduplication": "one selected Stop per unique scene/object actor",
        "allow_later_capture_override": args.allow_later_capture_override,
        "overridden_capture_keys": overridden_capture_keys,
        "sequence_semantics": (
            "Each Stop is an independent single-view SFT episode and is never appended "
            "as a fake transition to an expert trajectory."
        ),
        "stats": dict(stats),
        "action_counts": dict(action_counts),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

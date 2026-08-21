#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_stop_visible_frames import (  # noqa: E402
    attach_semantic_scores,
    load_semantic_scores,
    load_visibility_cache,
)
from vlm_baseline.stop_visibility import VisibilityPolicy, select_first_clear_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Stop candidates from a visibility cache without rewriting data."
    )
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--semantic-scores", type=Path)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = load_visibility_cache(args.visibility_cache)
    semantic_scores = load_semantic_scores(args.semantic_scores)
    attached, missing = attach_semantic_scores(cache, semantic_scores)
    payload = json.loads(args.policy_config.read_text(encoding="utf-8"))
    policy = VisibilityPolicy(**payload.get("policy", payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    with args.output.open("x", encoding="utf-8") as output:
        for key, trajectory in sorted(cache.items()):
            selection = select_first_clear_frame(
                trajectory.get("frames") or [],
                trajectory.get("size"),
                policy,
            )
            if selection["selected_frame_idx"] is not None:
                selected_count += 1
            record = {
                "trajectory_key": key,
                "scene_id": trajectory.get("scene_id"),
                "episode_id": trajectory.get("episode_id"),
                "pose_idx": trajectory.get("pose_idx"),
                "true_name": trajectory.get("true_name"),
                "object_name": trajectory.get("object_name"),
                "size": trajectory.get("size"),
                **selection,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "trajectory_count": len(cache),
                "selected_count": selected_count,
                "semantic_scores_attached": attached,
                "visible_frames_without_semantic_score": missing,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

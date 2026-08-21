#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_METADATA = DATASET_ROOT / "splits" / "uavon_raw_json" / "train.json"
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "stop_visible_v4_completeness_balance_policy.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from capture_stop_visibility_cache import (  # noqa: E402
    SEGMENTATION_COLOR_BY_ID,
    SegmentationAirsimTrajRecorder,
    capture_scene_rgb,
    capture_segmentation,
    load_requested_keys,
    nearest_target_distance,
    normalize_scene,
    set_target_segmentation_id,
    trajectory_key,
)
from capture_target_standoff_candidates import standoff_poses  # noqa: E402
from vlm_baseline.stop_visibility import (  # noqa: E402
    VisibilityPolicy,
    assess_visibility_frame,
    canonical_stencil_difference_mask,
    mask_metrics,
    parse_size_bucket,
)


def load_policy(path: Path) -> VisibilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("policy", payload)
    policy = VisibilityPolicy(**values)
    return replace(
        policy,
        semantic_score_field=None,
        min_semantic_score=None,
        semantic_rank_field=None,
        max_semantic_rank=None,
        require_semantic_for_weak_geometry=False,
    )


def load_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        scene = normalize_scene(
            str(row.get("scene_key") or row.get("map_name", ""))
            .replace("_TrainSets", "")
            .replace("_train", "")
        )
        lookup[(scene, str(row["episode_id"]))] = row
    return lookup


def pose_key(pose: list[float]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in pose)  # type: ignore[return-value]


def load_scene_trajectories(
    aligned_root: Path,
    metadata_lookup: dict[tuple[str, str], dict[str, Any]],
    scene: str,
    distance_threshold: float,
    camera_name: str,
    requested_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    json_root = aligned_root / "json" / scene
    trajectories = []
    for record_path in sorted(json_root.glob("*/*.json")):
        episode_id = record_path.parent.name
        pose_idx = record_path.stem
        key = trajectory_key(scene, episode_id, pose_idx)
        if requested_keys is not None and key not in requested_keys:
            continue
        metadata = metadata_lookup.get((scene, episode_id))
        if metadata is None:
            raise KeyError(f"metadata not found: {scene}::{episode_id}::{pose_idx}")
        data = json.loads(record_path.read_text(encoding="utf-8"))
        poses = data.get("record_list") or []
        camera_rows = data.get("image_dict", {}).get(camera_name, [])
        if len(poses) != len(camera_rows):
            raise ValueError(
                f"record/image mismatch for {record_path}: "
                f"{len(poses)} != {len(camera_rows)}"
            )
        target_positions = metadata.get("pose") or [data.get("goal_pos")]
        if not target_positions or target_positions == [None]:
            raise ValueError(f"missing target positions: {record_path}")
        object_name = str(metadata.get("object_name") or "").strip()
        if not object_name:
            raise ValueError(f"missing object_name: {record_path}")

        candidates = []
        for frame_idx, (pose, image_row) in enumerate(zip(poses, camera_rows)):
            distance = nearest_target_distance(pose, target_positions)
            if distance > distance_threshold:
                continue
            image_rel = str(image_row.get("rgb") or "")
            image_path = aligned_root / "images" / scene / image_rel
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "pose": [float(value) for value in pose],
                    "pose_key": pose_key(pose),
                    "distance_to_target": float(distance),
                    "image_path": str(image_path.resolve()),
                }
            )
        trajectories.append(
            {
                "trajectory_key": key,
                "scene_id": scene,
                "episode_id": episode_id,
                "pose_idx": pose_idx,
                "source_record": str(record_path.resolve()),
                "object_name": object_name,
                "true_name": str(metadata.get("true_name") or "").strip(),
                "target_description": str(metadata.get("description") or "").strip(),
                "size": str(metadata.get("size") or ""),
                "target_positions": [
                    [float(value) for value in position]
                    for position in target_positions
                ],
                "distance_threshold_m": float(distance_threshold),
                "total_frame_count": len(poses),
                "original_stop_frame_idx": len(poses) - 1,
                "candidates": candidates,
            }
        )
    return trajectories


def capture_mask(
    env: SegmentationAirsimTrajRecorder,
    resolved_object_name: str,
    pose: list[float],
    target_object_id: int,
    settle_frames: int,
    segmentation_settle_frames: int,
    camera_name: str,
    evidence_dir: Path | None = None,
    evidence_stem: str | None = None,
) -> dict[str, Any]:
    client = env._client
    if not client.simSetSegmentationObjectID(resolved_object_name, 0, False):
        raise RuntimeError(f"failed to clear segmentation ID: {resolved_object_name}")
    if segmentation_settle_frames > 0:
        client.simPause(False)
        client.simContinueForFrames(int(segmentation_settle_frames))
        client.simPause(True)
    env._set_camera_pose(*pose, 0.0, 0.0, settle_frames=settle_frames)
    baseline = capture_segmentation(client, camera_name)
    if not client.simSetSegmentationObjectID(
        resolved_object_name,
        int(target_object_id),
        False,
    ):
        raise RuntimeError(f"failed to mark segmentation ID: {resolved_object_name}")
    marked = capture_segmentation(client, camera_name)
    canonical_color = SEGMENTATION_COLOR_BY_ID[int(target_object_id)]
    mask = canonical_stencil_difference_mask(baseline, marked, canonical_color)
    metrics = mask_metrics(mask)
    if evidence_dir is not None:
        if not evidence_stem:
            raise ValueError("evidence_stem is required when evidence_dir is set")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        replay_rgb = capture_scene_rgb(client, camera_name)
        replay_path = evidence_dir / f"{evidence_stem}_replay.jpg"
        mask_path = evidence_dir / f"{evidence_stem}_mask.png"
        Image.fromarray(replay_rgb, mode="RGB").save(replay_path, quality=95)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        metrics["_replay_image_path"] = str(replay_path.resolve())
        metrics["_mask_path"] = str(mask_path.resolve())
    return metrics


def public_mask_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "actor"


def geometry_is_clear(
    frame_idx: int,
    metrics: dict[str, Any],
    size: str,
    peak_pixels: int,
    policy: VisibilityPolicy,
) -> bool:
    assessment = assess_visibility_frame(
        {"frame_idx": frame_idx, "mask": metrics},
        parse_size_bucket(size),
        peak_pixels,
        policy,
    )
    return bool(assessment["clear"])


def classify_path(
    trajectory: dict[str, Any],
    metrics_by_pose: dict[tuple[float, float, float, float], dict[str, Any]],
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    frames = []
    for candidate in trajectory["candidates"]:
        metrics = metrics_by_pose[candidate["pose_key"]]
        frame = {
            "frame_idx": candidate["frame_idx"],
            "pose": candidate["pose"],
            "distance_to_target": candidate["distance_to_target"],
            "image_path": candidate["image_path"],
            "mask": public_mask_metrics(metrics),
        }
        if metrics.get("_replay_image_path"):
            frame["replay_image_path"] = metrics["_replay_image_path"]
        if metrics.get("_mask_path"):
            frame["mask_path"] = metrics["_mask_path"]
        frames.append(frame)
    peak_pixels = max(
        (int(frame["mask"]["pixel_count"]) for frame in frames),
        default=0,
    )
    visible = [frame for frame in frames if int(frame["mask"]["pixel_count"]) > 0]
    clear = [
        frame
        for frame in frames
        if geometry_is_clear(
            frame["frame_idx"],
            frame["mask"],
            trajectory["size"],
            peak_pixels,
            policy,
        )
    ]
    stop = next(
        (
            frame
            for frame in frames
            if int(frame["frame_idx"]) == trajectory["original_stop_frame_idx"]
        ),
        None,
    )
    if not visible:
        path_status = "no_detectable_target_pixels"
    elif not clear:
        path_status = "target_visible_but_no_clear_geometry"
    else:
        path_status = "clear_geometry_available"
    return {
        **{key: value for key, value in trajectory.items() if key != "candidates"},
        "status": "ok",
        "candidate_frame_count": len(frames),
        "visible_frame_count": len(visible),
        "clear_geometry_frame_count": len(clear),
        "peak_target_pixels": peak_pixels,
        "original_stop_has_target_pixels": bool(
            stop and int(stop["mask"]["pixel_count"]) > 0
        ),
        "original_stop_clear_geometry": bool(
            stop
            and geometry_is_clear(
                stop["frame_idx"],
                stop["mask"],
                trajectory["size"],
                peak_pixels,
                policy,
            )
        ),
        "path_status": path_status,
        "frames": frames,
    }


def capture_standoff_until_clear(
    env: SegmentationAirsimTrajRecorder,
    resolved_object_name: str,
    target_positions: list[list[float]],
    size: str,
    target_object_id: int,
    azimuth_count: int,
    height_offsets: tuple[float, ...],
    settle_frames: int,
    segmentation_settle_frames: int,
    camera_name: str,
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    attempted = 0
    best_pixels = 0
    any_target_pixels = False
    clear_spec = None
    for target_index, target_position in enumerate(target_positions):
        for spec in standoff_poses(
            target_position,
            parse_size_bucket(size),
            azimuth_count,
            height_offsets,
        ):
            attempted += 1
            metrics = capture_mask(
                env,
                resolved_object_name,
                spec["pose"],
                target_object_id,
                settle_frames,
                segmentation_settle_frames,
                camera_name,
            )
            pixels = int(metrics["pixel_count"])
            best_pixels = max(best_pixels, pixels)
            any_target_pixels = any_target_pixels or pixels > 0
            if geometry_is_clear(
                attempted,
                metrics,
                size,
                pixels,
                policy,
            ):
                clear_spec = {
                    "target_position_index": target_index,
                    **spec,
                    "mask": public_mask_metrics(metrics),
                }
                break
        if clear_spec is not None:
            break
    return {
        "attempted_candidate_count": attempted,
        "any_target_pixels": any_target_pixels,
        "clear_geometry_available": clear_spec is not None,
        "best_pixel_count": best_pixels,
        "first_clear_candidate": clear_spec,
    }


def unique_target_positions(trajectories: list[dict[str, Any]]) -> list[list[float]]:
    unique = {}
    for trajectory in trajectories:
        for position in trajectory["target_positions"]:
            key = tuple(round(float(value), 5) for value in position)
            unique[key] = [float(value) for value in position]
    return list(unique.values())


def process_scene(
    scene: str,
    trajectories: list[dict[str, Any]],
    args: argparse.Namespace,
    policy: VisibilityPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_actor[trajectory["object_name"]].append(trajectory)

    env = SegmentationAirsimTrajRecorder(
        scene,
        airsim_port=args.base_port + args.gpu,
        device_id=args.gpu,
        segmentation_width=args.segmentation_width,
        segmentation_height=args.segmentation_height,
    )
    results = []
    actor_results = []
    try:
        client = env._client
        if not client.simSetSegmentationObjectID(".*", 0, True):
            raise RuntimeError("failed to initialize all scene objects to ID 0")
        processed = 0
        for actor_index, (object_name, actor_trajectories) in enumerate(
            sorted(by_actor.items()),
            start=1,
        ):
            target_positions = unique_target_positions(actor_trajectories)
            actor_base = {
                "scene_id": scene,
                "object_name": object_name,
                "true_names": sorted(
                    {row["true_name"] for row in actor_trajectories if row["true_name"]}
                ),
                "trajectory_count": len(actor_trajectories),
                "target_positions": target_positions,
            }
            try:
                resolved_name, match_mode, pose_error = set_target_segmentation_id(
                    client,
                    object_name,
                    args.target_object_id,
                    target_positions,
                )
                metrics_by_pose = {}
                poses = {
                    candidate["pose_key"]: candidate["pose"]
                    for trajectory in actor_trajectories
                    for candidate in trajectory["candidates"]
                }
                for index, (key, pose) in enumerate(poses.items(), start=1):
                    evidence_dir = None
                    evidence_stem = None
                    if args.save_replay_evidence:
                        evidence_dir = (
                            args.output_dir
                            / "replay_evidence"
                            / scene
                            / f"a{actor_index:03d}_{safe_name(object_name)}"
                        )
                        evidence_stem = f"pose_{index:05d}"
                    metrics_by_pose[key] = capture_mask(
                        env,
                        resolved_name,
                        pose,
                        args.target_object_id,
                        args.settle_frames,
                        args.segmentation_settle_frames,
                        args.camera_name,
                        evidence_dir=evidence_dir,
                        evidence_stem=evidence_stem,
                    )
                    if index % args.progress_every_poses == 0:
                        print(
                            json.dumps(
                                {
                                    "scene": scene,
                                    "actor": object_name,
                                    "actor_progress": f"{actor_index}/{len(by_actor)}",
                                    "pose_progress": f"{index}/{len(poses)}",
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                actor_paths = [
                    classify_path(trajectory, metrics_by_pose, policy)
                    for trajectory in actor_trajectories
                ]
                any_path_clear = any(
                    int(row["clear_geometry_frame_count"]) > 0 for row in actor_paths
                )
                standoff = None
                if not any_path_clear:
                    standoff = capture_standoff_until_clear(
                        env,
                        resolved_name,
                        target_positions,
                        actor_trajectories[0]["size"],
                        args.target_object_id,
                        args.standoff_azimuth_count,
                        args.standoff_height_offsets,
                        args.settle_frames,
                        args.segmentation_settle_frames,
                        args.camera_name,
                        policy,
                    )
                if any_path_clear:
                    actor_status = "clear_on_at_least_one_expert_path"
                elif standoff and standoff["clear_geometry_available"]:
                    actor_status = "target_facing_standoff_repairable"
                else:
                    actor_status = "unresolved_after_target_facing_standoff"
                for row in actor_paths:
                    if row["path_status"] == "clear_geometry_available":
                        row["no_target_cause"] = None
                    elif any_path_clear:
                        row["no_target_cause"] = (
                            "trajectory_viewpoint_distance_or_occlusion"
                        )
                    elif standoff and standoff["clear_geometry_available"]:
                        row["no_target_cause"] = "target_facing_standoff_repairable"
                    else:
                        row["no_target_cause"] = (
                            "target_mapping_coordinate_or_scene_asset_unresolved"
                        )
                    row["actor_audit_status"] = actor_status
                results.extend(actor_paths)
                actor_results.append(
                    {
                        **actor_base,
                        "status": "ok",
                        "resolved_object_name": resolved_name,
                        "object_name_match": match_mode,
                        "object_name_pose_error_m": pose_error,
                        "unique_expert_pose_count": len(poses),
                        "expert_path_clear_geometry": any_path_clear,
                        "actor_audit_status": actor_status,
                        "standoff": standoff,
                    }
                )
            except Exception as exc:
                error = repr(exc)
                actor_results.append(
                    {
                        **actor_base,
                        "status": "error",
                        "actor_audit_status": "object_mapping_or_capture_error",
                        "error": error,
                    }
                )
                for trajectory in actor_trajectories:
                    results.append(
                        {
                            **{
                                key: value
                                for key, value in trajectory.items()
                                if key != "candidates"
                            },
                            "status": "error",
                            "path_status": "error",
                            "no_target_cause": "object_mapping_or_capture_error",
                            "actor_audit_status": "object_mapping_or_capture_error",
                            "error": error,
                            "frames": [],
                        }
                    )
            processed += len(actor_trajectories)
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "actor_progress": f"{actor_index}/{len(by_actor)}",
                        "trajectory_progress": f"{processed}/{len(trajectories)}",
                        "object_name": object_name,
                        "actor_status": actor_results[-1]["actor_audit_status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        env.cleanup()
    return results, actor_results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit target-instance visibility for every training trajectory with "
            "deduplicated AirSim pose replay."
        )
    )
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--trajectory-keys", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--base-port", type=int, default=39500)
    parser.add_argument("--distance-threshold", type=float, default=20.0)
    parser.add_argument("--target-object-id", type=int, default=42)
    parser.add_argument("--segmentation-width", type=int, default=512)
    parser.add_argument("--segmentation-height", type=int, default=512)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--segmentation-settle-frames", type=int, default=0)
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--standoff-azimuth-count", type=int, default=8)
    parser.add_argument("--standoff-height-offsets", default="1.5,3.0")
    parser.add_argument("--progress-every-poses", type=int, default=50)
    parser.add_argument(
        "--save-replay-evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save the synchronized replay RGB and instance mask for each unique "
            "expert pose."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.standoff_height_offsets = tuple(
        float(value) for value in args.standoff_height_offsets.split(",")
    )
    scenes = [item for item in args.scene_list.split(",") if item]
    metadata_lookup = load_metadata(args.metadata)
    requested_keys = load_requested_keys(args.trajectory_keys)
    policy = load_policy(args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        trajectory_path = args.output_dir / f"{scene}.jsonl"
        actor_path = args.output_dir / f"{scene}_actors.jsonl"
        if args.resume and trajectory_path.exists() and actor_path.exists():
            print(
                json.dumps(
                    {"scene": scene, "event": "scene_skip_completed"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        if (trajectory_path.exists() or actor_path.exists()) and not args.overwrite:
            raise FileExistsError(
                f"output exists for {scene}; pass --overwrite to replace it"
            )
        trajectories = load_scene_trajectories(
            args.aligned_root,
            metadata_lookup,
            scene,
            args.distance_threshold,
            args.camera_name,
            requested_keys,
        )
        print(
            json.dumps(
                {
                    "scene": scene,
                    "event": "scene_start",
                    "trajectories": len(trajectories),
                    "actors": len({row["object_name"] for row in trajectories}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        results, actor_results = process_scene(scene, trajectories, args, policy)
        write_jsonl(trajectory_path, results)
        write_jsonl(actor_path, actor_results)
        print(
            json.dumps(
                {
                    "scene": scene,
                    "event": "scene_complete",
                    "trajectories": len(results),
                    "errors": sum(row["status"] == "error" for row in results),
                    "no_target_paths": sum(
                        row["path_status"] == "no_detectable_target_pixels"
                        for row in results
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

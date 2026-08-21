#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import airsim
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_ALIGNED_ROOT = DATASET_ROOT / "generated" / "record_output_transition_aligned"
DEFAULT_METADATA = DATASET_ROOT / "splits" / "uavon_raw_json" / "train.json"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed" / "stop_visible_v1" / "visibility_cache"

sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import eval_utils  # noqa: E402
from eval_utils import AirsimTrajRecorder  # noqa: E402
from vlm_baseline.stop_visibility import (  # noqa: E402
    canonical_stencil_difference_mask,
    mask_metrics,
)


SEGMENTATION_COLOR_BY_ID = {
    42: (106, 31, 92),
}


def normalize_scene(scene: str) -> str:
    aliases = {
        "NeighborhoodTrain": "Neighborhood",
        "ModularNeighborhood": "Neighborhood",
    }
    return aliases.get(scene, scene)


def trajectory_key(scene: str, episode_id: str, pose_idx: str) -> str:
    return f"{scene}::{episode_id}::{pose_idx}"


def _normalized_actor_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
    return re.sub(r"c\d+$", "", normalized)


def set_target_segmentation_id(
    client: airsim.MultirotorClient,
    requested_name: str,
    object_id: int,
    target_positions: list[list[float]],
    max_pose_distance: float = 10.0,
) -> tuple[str, str, float | None]:
    if client.simSetSegmentationObjectID(requested_name, int(object_id), False):
        return requested_name, "exact", None

    scene_objects = client.simListSceneObjects(".*")
    requested_lower = requested_name.lower()
    requested_normalized = _normalized_actor_name(requested_name)
    candidates = [
        name
        for name in scene_objects
        if name.lower() == requested_lower
        or name.lower().startswith(requested_lower + "_")
        or _normalized_actor_name(name) == requested_normalized
    ]
    candidates = sorted(set(candidates))
    if len(candidates) == 1 and client.simSetSegmentationObjectID(
        candidates[0],
        int(object_id),
        False,
    ):
        return candidates[0], "unique_alias", None

    requested_stem = re.sub(r"_\d+$", "", requested_lower)
    family_candidates = sorted(
        {
            name
            for name in scene_objects
            if name.lower().startswith(requested_stem + "_")
        }
    )
    targets = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    pose_candidates = []
    for candidate in family_candidates:
        pose = client.simGetObjectPose(candidate)
        position = np.asarray(
            [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            dtype=np.float64,
        )
        if not np.isfinite(position).all():
            continue
        distance = float(np.linalg.norm(targets - position[None, :], axis=1).min())
        pose_candidates.append((distance, candidate, position.tolist()))
    pose_candidates.sort()
    if pose_candidates:
        nearest_distance, nearest_name, _ = pose_candidates[0]
        second_distance = pose_candidates[1][0] if len(pose_candidates) > 1 else math.inf
        if (
            nearest_distance <= max_pose_distance
            and second_distance - nearest_distance > 0.25
            and client.simSetSegmentationObjectID(nearest_name, int(object_id), False)
        ):
            return nearest_name, "nearest_pose_alias", nearest_distance

    close = difflib.get_close_matches(requested_name, scene_objects, n=8, cutoff=0.45)
    raise RuntimeError(
        "target actor not found uniquely for segmentation: "
        f"requested={requested_name!r}, alias_candidates={candidates!r}, "
        f"pose_candidates={pose_candidates[:8]!r}, close_matches={close!r}"
    )


def load_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for row in rows:
        scene = normalize_scene(
            str(row.get("scene_key") or row.get("map_name", "")).replace(
                "_TrainSets", ""
            )
        )
        lookup[(scene, str(row["episode_id"]))] = row
    return lookup


def load_requested_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            row = json.loads(line)
            line = trajectory_key(
                str(row["scene_id"]),
                str(row["episode_id"]),
                str(row.get("pose_idx", "0")),
            )
        if len(line.split("::")) != 3:
            raise ValueError(f"invalid trajectory key: {line}")
        keys.add(line)
    return keys


def iter_record_files(
    aligned_root: Path,
    scenes: set[str] | None,
    requested_keys: set[str] | None,
) -> Iterable[tuple[str, str, str, Path]]:
    json_root = aligned_root / "json"
    for path in sorted(json_root.glob("*/*/*.json")):
        rel = path.relative_to(json_root)
        scene, episode_id, filename = rel.parts
        pose_idx = Path(filename).stem
        key = trajectory_key(scene, episode_id, pose_idx)
        if scenes and scene not in scenes:
            continue
        if requested_keys is not None and key not in requested_keys:
            continue
        yield scene, episode_id, pose_idx, path


def load_completed(path: Path, retry_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if retry_errors and row.get("status") == "error":
            continue
        completed.add(str(row["trajectory_key"]))
    return completed


def nearest_target_distance(pose: list[float], target_positions: list[list[float]]) -> float:
    position = np.asarray(pose[:3], dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(targets - position[None, :], axis=1).min())


class SegmentationAirsimTrajRecorder(AirsimTrajRecorder):
    def __init__(
        self,
        env_name: str,
        airsim_port: int,
        device_id: int,
        segmentation_width: int,
        segmentation_height: int,
    ) -> None:
        self.segmentation_width = int(segmentation_width)
        self.segmentation_height = int(segmentation_height)
        super().__init__(
            env_name,
            airsim_port=airsim_port,
            device_id=device_id,
        )

    def change_and_save_settings(self, air_port: int) -> str:
        settings = json.loads(Path(eval_utils.BASE_SETTINGS).read_text(encoding="utf-8"))
        settings["ApiServerPort"] = int(air_port)
        camera = settings["Vehicles"]["drone_1"]["Cameras"]["uav_on_0"]
        capture_settings = [
            row for row in camera.get("CaptureSettings", []) if int(row["ImageType"]) != 5
        ]
        capture_settings.append(
            {
                "ImageType": 5,
                "Width": self.segmentation_width,
                "Height": self.segmentation_height,
                "FOV_Degrees": 90,
            }
        )
        camera["CaptureSettings"] = capture_settings
        path = Path(eval_utils.SETTINGS_ROOT) / f"settings_stop_visible_{air_port}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return str(path)


def capture_segmentation(client: airsim.MultirotorClient, camera_name: str) -> np.ndarray:
    response = client.simGetImages(
        [
            airsim.ImageRequest(
                camera_name,
                airsim.ImageType.Segmentation,
                pixels_as_float=False,
                compress=False,
            )
        ]
    )[0]
    if response.width <= 0 or response.height <= 0 or not response.image_data_uint8:
        raise RuntimeError("AirSim returned an empty segmentation image")
    return np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
        response.height,
        response.width,
        3,
    )


def capture_scene_rgb(client: airsim.MultirotorClient, camera_name: str) -> np.ndarray:
    response = client.simGetImages(
        [
            airsim.ImageRequest(
                camera_name,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=True,
            )
        ]
    )[0]
    if response.width <= 0 or response.height <= 0 or not response.image_data_uint8:
        raise RuntimeError("AirSim returned an empty scene image")
    return np.asarray(
        Image.open(io.BytesIO(response.image_data_uint8)).convert("RGB")
    )


def segmentation_color_mask(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.all(image == color.reshape(1, 1, 3), axis=2)


def identify_changed_segmentation_color(
    baseline: np.ndarray,
    marked: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    changed = np.any(marked != baseline, axis=2)
    if not np.any(changed):
        return None, 0
    colors, counts = np.unique(marked[changed].reshape(-1, 3), axis=0, return_counts=True)
    best_color = colors[int(np.argmax(counts))]
    baseline_count = int(segmentation_color_mask(baseline, best_color).sum())
    return best_color, baseline_count


def advance_simulation(client: airsim.MultirotorClient, frames: int) -> None:
    if frames <= 0:
        return
    client.simPause(False)
    client.simContinueForFrames(int(frames))
    client.simPause(True)


def wait_for_zero_segmentation(
    client: airsim.MultirotorClient,
    camera_name: str,
    settle_frames: int,
    max_attempts: int = 8,
) -> None:
    for _ in range(max_attempts):
        advance_simulation(client, settle_frames)
        if not np.any(capture_segmentation(client, camera_name)):
            return
    raise RuntimeError("segmentation background did not settle to object ID 0")


def save_debug_images(
    mask: np.ndarray,
    source_rgb: Path,
    replay_rgb: np.ndarray,
    output_dir: Path,
    frame_idx: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_path = output_dir / f"{frame_idx:05d}_mask.png"
    mask_image.save(mask_path)

    replay = Image.fromarray(replay_rgb, mode="RGB")
    replay_path = output_dir / f"{frame_idx:05d}_replay.jpg"
    replay.save(replay_path, quality=95)

    def save_overlay(rgb: Image.Image, suffix: str) -> Path:
        local_mask = mask_image
        if local_mask.size != rgb.size:
            local_mask = local_mask.resize(rgb.size, resample=Image.Resampling.NEAREST)
        red = Image.new("RGB", rgb.size, (255, 0, 0))
        alpha = local_mask.point(lambda value: 120 if value else 0)
        overlay = Image.composite(red, rgb, alpha).convert("RGB")
        path = output_dir / f"{frame_idx:05d}_{suffix}.jpg"
        overlay.save(path, quality=92)
        return path

    source_overlay = save_overlay(Image.open(source_rgb).convert("RGB"), "source_overlay")
    replay_overlay = save_overlay(replay, "replay_overlay")
    return {
        "mask_path": str(mask_path.resolve()),
        "source_overlay_path": str(source_overlay.resolve()),
        "replay_rgb_path": str(replay_path.resolve()),
        "replay_overlay_path": str(replay_overlay.resolve()),
    }


def process_trajectory(
    env: SegmentationAirsimTrajRecorder,
    aligned_root: Path,
    metadata: dict[str, Any],
    scene: str,
    episode_id: str,
    pose_idx: str,
    record_path: Path,
    distance_threshold: float,
    target_object_id: int,
    settle_frames: int,
    segmentation_settle_frames: int,
    camera_name: str,
    save_debug: bool,
    debug_root: Path,
) -> dict[str, Any]:
    data = json.loads(record_path.read_text(encoding="utf-8"))
    poses = data.get("record_list") or []
    camera_rows = data.get("image_dict", {}).get(camera_name, [])
    if len(poses) != len(camera_rows):
        raise ValueError(
            f"record/image mismatch for {record_path}: {len(poses)} != {len(camera_rows)}"
        )

    target_positions = metadata.get("pose") or [data.get("goal_pos")]
    if not target_positions or target_positions == [None]:
        raise ValueError(f"missing target pose for {scene}/{episode_id}")
    object_name = str(metadata.get("object_name") or "").strip()
    if not object_name:
        raise ValueError(f"missing object_name for {scene}/{episode_id}")

    client = env._client
    resolved_object_name, object_name_match, object_name_pose_error = set_target_segmentation_id(
        client,
        object_name,
        int(target_object_id),
        target_positions,
    )

    frames = []
    observed_target_colors: list[list[int]] = []
    try:
        if not client.simSetSegmentationObjectID(resolved_object_name, 0, False):
            raise RuntimeError(
                f"failed to initialize target segmentation ID: {resolved_object_name}"
            )
        advance_simulation(client, segmentation_settle_frames)
        for frame_idx, pose in enumerate(poses):
            distance = nearest_target_distance(pose, target_positions)
            if distance > distance_threshold:
                continue
            env.zero_kinematics_at_pose(pose, settle_frames=settle_frames)
            if not client.simSetSegmentationObjectID(resolved_object_name, 0, False):
                raise RuntimeError(
                    f"failed to clear target segmentation ID: {resolved_object_name}"
                )
            # Keep both captures on the same paused simulation frame. Segmentation
            # shades vary with distance and antialiasing, so carrying one RGB color
            # across the trajectory drops valid target pixels at another scale.
            baseline = capture_segmentation(client, camera_name)
            if not client.simSetSegmentationObjectID(
                resolved_object_name,
                int(target_object_id),
                False,
            ):
                raise RuntimeError(
                    f"failed to set target segmentation ID: {resolved_object_name}"
                )
            segmentation = capture_segmentation(client, camera_name)
            canonical_color = SEGMENTATION_COLOR_BY_ID.get(int(target_object_id))
            if canonical_color is None:
                raise ValueError(
                    f"missing canonical segmentation color for object ID {target_object_id}"
                )
            mask = canonical_stencil_difference_mask(
                baseline,
                segmentation,
                canonical_color,
            )
            replay_rgb = capture_scene_rgb(client, camera_name)
            discovered_color, _ = identify_changed_segmentation_color(
                baseline,
                segmentation,
            )
            if discovered_color is not None:
                observed_target_colors.append(discovered_color.astype(int).tolist())
            image_rel = str(camera_rows[frame_idx].get("rgb") or "")
            source_rgb = aligned_root / "images" / scene / image_rel
            if not source_rgb.is_file():
                raise FileNotFoundError(source_rgb)
            frame = {
                "frame_idx": frame_idx,
                "pose": [float(value) for value in pose],
                "distance_to_target": distance,
                "image_path": str(source_rgb.resolve()),
                "mask": mask_metrics(mask),
            }
            if save_debug:
                debug_dir = debug_root / scene / episode_id / pose_idx
                frame["debug"] = save_debug_images(
                    mask,
                    source_rgb,
                    replay_rgb,
                    debug_dir,
                    frame_idx,
                )
                frame["replay_image_path"] = frame["debug"]["replay_rgb_path"]
            frames.append(frame)
    finally:
        if not client.simSetSegmentationObjectID(resolved_object_name, 0, False):
            raise RuntimeError(
                f"failed to clear target segmentation ID: {resolved_object_name}"
            )
    return {
        "trajectory_key": trajectory_key(scene, episode_id, pose_idx),
        "status": "ok" if frames else "no_candidates_within_distance",
        "scene_id": scene,
        "episode_id": episode_id,
        "pose_idx": pose_idx,
        "source_record": str(record_path.resolve()),
        "object_name": object_name,
        "resolved_object_name": resolved_object_name,
        "object_name_match": object_name_match,
        "object_name_pose_error_m": object_name_pose_error,
        "true_name": str(metadata.get("true_name") or "").strip(),
        "target_description": str(metadata.get("description") or "").strip(),
        "size": str(metadata.get("size") or ""),
        "target_positions": target_positions,
        "distance_threshold_m": float(distance_threshold),
        "segmentation_object_id": int(target_object_id),
        "segmentation_mask_mode": "per_frame_paused_canonical_stencil_difference",
        "observed_target_segmentation_colors_rgb": observed_target_colors,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay expert poses and cache exact target-instance visibility metrics."
    )
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-keys", type=Path)
    parser.add_argument("--scene-list", default="")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=37400)
    parser.add_argument("--distance-threshold", type=float, default=20.0)
    parser.add_argument("--target-object-id", type=int, default=42)
    parser.add_argument("--segmentation-width", type=int, default=512)
    parser.add_argument("--segmentation-height", type=int, default=512)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--segmentation-settle-frames", type=int, default=4)
    parser.add_argument("--camera-name", default="uav_on_0")
    parser.add_argument("--max-trajectories-per-scene", type=int, default=0)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    scenes = {item for item in args.scene_list.split(",") if item} or None
    requested_keys = load_requested_keys(args.trajectory_keys)
    metadata_lookup = load_metadata(args.metadata)
    records_by_scene: dict[str, list[tuple[str, str, Path]]] = {}
    for scene, episode_id, pose_idx, path in iter_record_files(
        args.aligned_root,
        scenes,
        requested_keys,
    ):
        records_by_scene.setdefault(scene, []).append((episode_id, pose_idx, path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_root = args.output_dir / "debug"
    summaries = []
    for scene, records in sorted(records_by_scene.items()):
        if args.max_trajectories_per_scene:
            records = records[: args.max_trajectories_per_scene]
        output_path = args.output_dir / f"{scene}.jsonl"
        completed = load_completed(output_path, retry_errors=args.retry_errors)
        env = SegmentationAirsimTrajRecorder(
            scene,
            airsim_port=args.base_port + args.gpu,
            device_id=args.gpu,
            segmentation_width=args.segmentation_width,
            segmentation_height=args.segmentation_height,
        )
        counts = {"requested": len(records), "written": 0, "skipped": 0, "errors": 0}
        try:
            all_zero_ok = env._client.simSetSegmentationObjectID(".*", 0, True)
            if not all_zero_ok:
                raise RuntimeError("failed to initialize all scene objects to segmentation ID 0")
            advance_simulation(env._client, args.segmentation_settle_frames)
            with output_path.open("a", encoding="utf-8") as output:
                for episode_id, pose_idx, record_path in records:
                    key = trajectory_key(scene, episode_id, pose_idx)
                    if key in completed:
                        counts["skipped"] += 1
                        continue
                    metadata = metadata_lookup.get((scene, episode_id))
                    if metadata is None:
                        result = {
                            "trajectory_key": key,
                            "status": "error",
                            "error": "metadata_not_found",
                        }
                    else:
                        try:
                            result = process_trajectory(
                                env=env,
                                aligned_root=args.aligned_root,
                                metadata=metadata,
                                scene=scene,
                                episode_id=episode_id,
                                pose_idx=pose_idx,
                                record_path=record_path,
                                distance_threshold=args.distance_threshold,
                                target_object_id=args.target_object_id,
                                settle_frames=args.settle_frames,
                                segmentation_settle_frames=args.segmentation_settle_frames,
                                camera_name=args.camera_name,
                                save_debug=args.save_debug,
                                debug_root=debug_root,
                            )
                        except Exception as exc:
                            result = {
                                "trajectory_key": key,
                                "status": "error",
                                "scene_id": scene,
                                "episode_id": episode_id,
                                "pose_idx": pose_idx,
                                "source_record": str(record_path.resolve()),
                                "error": repr(exc),
                            }
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    counts["written"] += 1
                    if result["status"] == "error":
                        counts["errors"] += 1
                    frame_rows = result.get("frames") or []
                    print(
                        json.dumps(
                            {
                                "trajectory_key": key,
                                "status": result["status"],
                                "progress": counts["written"] + counts["skipped"],
                                "total": len(records),
                                "candidate_frames": len(frame_rows),
                                "visible_frames": sum(
                                    int((frame.get("mask") or {}).get("pixel_count", 0)) > 0
                                    for frame in frame_rows
                                ),
                                "error": result.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        finally:
            env.cleanup()
        summary = {"scene": scene, **counts, "output": str(output_path)}
        summaries.append(summary)
        print(json.dumps({"scene_summary": summary}, ensure_ascii=False), flush=True)

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from vlm_baseline.stop_visibility import (
    VisibilityPolicy,
    canonical_stencil_difference_mask,
    mask_metrics,
    parse_size_bucket,
    select_first_clear_frame,
)


def _frame(frame_idx: int, pixels: tuple[slice, slice] | None) -> dict:
    mask = np.zeros((100, 100), dtype=bool)
    if pixels is not None:
        mask[pixels] = True
    return {"frame_idx": frame_idx, "mask": mask_metrics(mask)}


def test_mask_metrics_tracks_bbox_and_clipping() -> None:
    metrics = _frame(0, (slice(10, 20), slice(0, 8)))["mask"]
    assert metrics["pixel_count"] == 80
    assert metrics["bbox"] == [0, 10, 7, 19]
    assert metrics["border_sides"] == ["left"]
    assert metrics["touches_opposite_borders"] is False


def test_selects_first_clear_frame_not_largest_or_final() -> None:
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in ("small", "mid", "big", "unknown")},
        min_bbox_short_side={bucket: 5 for bucket in ("small", "mid", "big", "unknown")},
        min_relative_to_peak=0.10,
    )
    frames = [
        _frame(3, None),
        _frame(4, (slice(20, 28), slice(20, 28))),
        _frame(5, (slice(20, 40), slice(20, 40))),
        _frame(6, None),
    ]
    selection = select_first_clear_frame(frames, "small(1*1=1 square)", policy)
    assert selection["selected_frame_idx"] == 4
    assert selection["peak_pixels"] == 400


def test_rejects_severely_clipped_closeup() -> None:
    policy = VisibilityPolicy(
        min_pixels={bucket: 20 for bucket in ("small", "mid", "big", "unknown")},
        min_bbox_short_side={bucket: 3 for bucket in ("small", "mid", "big", "unknown")},
        min_relative_to_peak=0.0,
    )
    frames = [_frame(0, (slice(0, 100), slice(0, 100)))]
    selection = select_first_clear_frame(frames, "big(5*5=25 squares)", policy)
    assert selection["selected_frame_idx"] is None
    assert "severely_clipped_or_too_close" in selection["assessments"][0]["reasons"]


def test_size_bucket_parsing_is_stable() -> None:
    assert parse_size_bucket(" small(1*1=1 square)") == "small"
    assert parse_size_bucket("mid(2*2=4 squares)") == "mid"
    assert parse_size_bucket(" big(5*5=25 squares)") == "big"
    assert parse_size_bucket("") == "unknown"


def test_weak_geometry_requires_semantic_support_when_enabled() -> None:
    buckets = ("small", "mid", "big", "unknown")
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        strong_pixels={bucket: 100 for bucket in buckets},
        min_relative_to_peak=0.0,
        semantic_rank_field="clip_rank",
        max_semantic_rank=5,
        require_semantic_for_weak_geometry=True,
    )
    unsupported = _frame(0, (slice(20, 28), slice(20, 28)))
    supported = _frame(1, (slice(20, 28), slice(20, 28)))
    supported["clip_rank"] = 3
    selection = select_first_clear_frame([unsupported, supported], "small", policy)
    assert selection["selected_frame_idx"] == 1
    assert "weak_geometry_without_semantic_support" in selection["assessments"][0]["reasons"]


def test_strong_geometry_does_not_require_clip_support() -> None:
    buckets = ("small", "mid", "big", "unknown")
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        strong_pixels={bucket: 60 for bucket in buckets},
        min_relative_to_peak=0.0,
        semantic_rank_field="clip_rank",
        max_semantic_rank=5,
        require_semantic_for_weak_geometry=True,
    )
    frame = _frame(0, (slice(20, 28), slice(20, 28)))
    frame["clip_rank"] = 100
    selection = select_first_clear_frame([frame], "small", policy)
    assert selection["selected_frame_idx"] == 0


def test_best_recognition_view_prefers_centered_complete_frame() -> None:
    buckets = ("small", "mid", "big", "unknown")
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        strong_pixels={bucket: 40 for bucket in buckets},
        min_relative_to_peak=0.0,
        selection_mode="best_recognition_view",
        preferred_pixel_fraction={bucket: 0.20 for bucket in buckets},
    )
    edge_frame = _frame(0, (slice(20, 50), slice(85, 100)))
    edge_frame["distance_to_target"] = 16.0
    centered_frame = _frame(1, (slice(30, 70), slice(30, 70)))
    centered_frame["distance_to_target"] = 10.0
    selection = select_first_clear_frame([edge_frame, centered_frame], "big", policy)
    assert selection["selected_frame_idx"] == 1
    assessments = {row["frame_idx"]: row for row in selection["assessments"]}
    assert assessments[1]["quality_score"] > assessments[0]["quality_score"]


def test_saturating_occupancy_does_not_penalize_larger_clear_target() -> None:
    buckets = ("small", "mid", "big", "unknown")
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        strong_pixels={bucket: 40 for bucket in buckets},
        min_relative_to_peak=0.0,
        selection_mode="best_recognition_view",
        preferred_pixel_fraction={bucket: 0.05 for bucket in buckets},
        occupancy_quality_mode="saturating",
        quality_weights={
            "occupancy": 1.0,
            "center": 0.0,
            "clipping": 0.0,
            "distance": 0.0,
            "semantic": 0.0,
        },
        earliest_within_best_score=0.0,
    )
    preferred_size = _frame(0, (slice(30, 55), slice(30, 50)))
    larger_clear = _frame(1, (slice(20, 70), slice(20, 70)))

    selection = select_first_clear_frame(
        [preferred_size, larger_clear],
        "mid",
        policy,
    )

    assessments = {row["frame_idx"]: row for row in selection["assessments"]}
    assert assessments[0]["quality_components"]["occupancy"] == 1.0
    assert assessments[1]["quality_components"]["occupancy"] == 1.0
    assert selection["selected_frame_idx"] == 0


def test_relative_mask_density_can_prefer_more_complete_view() -> None:
    buckets = ("small", "mid", "big", "unknown")
    dense = _frame(0, (slice(20, 50), slice(20, 50)))
    sparse_mask = np.zeros((100, 100), dtype=bool)
    sparse_mask[20:50:3, 20:50] = True
    sparse = {"frame_idx": 1, "mask": mask_metrics(sparse_mask)}
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        strong_pixels={bucket: 40 for bucket in buckets},
        min_relative_to_peak=0.0,
        selection_mode="best_recognition_view",
        quality_weights={
            "occupancy": 0.0,
            "center": 0.0,
            "clipping": 0.0,
            "distance": 0.0,
            "semantic": 0.0,
            "mask_density": 1.0,
        },
        earliest_within_best_score=0.0,
    )

    selection = select_first_clear_frame([sparse, dense], "mid", policy)

    assessments = {row["frame_idx"]: row for row in selection["assessments"]}
    assert assessments[0]["relative_bbox_fill_fraction"] == 1.0
    assert assessments[1]["relative_bbox_fill_fraction"] < 0.5
    assert selection["selected_frame_idx"] == 0


def test_rewrite_trajectory_uses_synchronized_replay_for_stop() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_stop_visible_frames.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_stop_visible_frames", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        replay_image = root / "selected_replay.jpg"
        replay_image.write_bytes(b"replay")
        rows = [
            {
                "frame_idx": 4,
                "image_path": str(root / "legacy.jpg"),
                "action_name": "Move Forward",
                "action_vector": [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ]
        selection = {
            "selected_replay_image_path": str(replay_image),
            "peak_pixels": 1000,
            "size_bucket": "mid",
            "selection_mode": "best_recognition_view",
            "selected_quality_score": 0.8,
        }

        rewritten = module.rewrite_trajectory(rows, 4, selection, root)

    assert rewritten[0]["image_path"] == str(replay_image.resolve())
    assert rewritten[0]["action_name"] == "Stop"
    assert rewritten[0]["stop_visibility"]["image_source"] == "synchronized_replay"


def test_standoff_policy_rejects_collided_candidate() -> None:
    buckets = ("small", "mid", "big", "unknown")
    policy = VisibilityPolicy(
        min_pixels={bucket: 40 for bucket in buckets},
        min_bbox_short_side={bucket: 5 for bucket in buckets},
        min_relative_to_peak=0.0,
        reject_collided=True,
    )
    collided = _frame(0, (slice(20, 40), slice(20, 40)))
    collided["collision_info"] = {"has_collided": True}
    safe = _frame(1, (slice(20, 35), slice(20, 35)))
    safe["collision_info"] = {"has_collided": False}

    selection = select_first_clear_frame([collided, safe], "mid", policy)

    assert selection["selected_frame_idx"] == 1
    assert "candidate_pose_collided" in selection["assessments"][0]["reasons"]


def test_canonical_stencil_mask_rejects_remote_render_noise() -> None:
    baseline = np.zeros((20, 20, 3), dtype=np.uint8)
    marked = baseline.copy()
    marked[8:12, 8:12] = (106, 31, 92)
    marked[7:13, 7] = (120, 45, 106)
    marked[1:4, 1:4] = (55, 71, 76)

    mask = canonical_stencil_difference_mask(
        baseline,
        marked,
        (106, 31, 92),
        halo_pixels=2,
    )

    assert mask[8:12, 8:12].all()
    assert mask[7:13, 7].all()
    assert not mask[1:4, 1:4].any()


def _load_v4_production_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_stop_visible_v4_production_frames.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_stop_visible_v4_production_frames",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_production_preserves_real_motion_before_selected_stop() -> None:
    module = _load_v4_production_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        aligned = root / "record_output_transition_aligned"
        images = aligned / "images" / "Scene" / "1" / "0" / "uav_on_0"
        images.mkdir(parents=True)
        for frame_idx in range(4):
            (images / f"{frame_idx:05d}.png").write_bytes(b"rgb")
        rows = [
            {
                "episode_key": "Scene::1::0",
                "scene_id": "Scene",
                "episode_id": "1",
                "pose_idx": "0",
                "frame_idx": frame_idx,
                "image_path": str(
                    root
                    / "record_output"
                    / "images"
                    / "Scene"
                    / "1"
                    / "0"
                    / "uav_on_0"
                    / f"{frame_idx:05d}.png"
                ),
                "action_name": "Move Forward" if frame_idx < 3 else "Stop",
                "action_vector": [0.0] * 8,
            }
            for frame_idx in range(4)
        ]
        audit = {
            "trajectory_key": "Scene::1::0",
            "frames": [
                {
                    "frame_idx": 2,
                    "image_path": str((images / "00002.png").resolve()),
                }
            ],
        }
        selection = {
            "selected_frame_idx": 2,
            "selection_mode": "best_recognition_view",
            "selected_quality_score": 0.8,
            "size_bucket": "mid",
            "peak_pixels": 1000,
            "assessments": [
                {"frame_idx": 0, "clear": True, "quality_score": 0.70},
                {"frame_idx": 1, "clear": True, "quality_score": 0.75},
                {"frame_idx": 2, "clear": True, "quality_score": 0.8},
            ],
        }
        rewritten, removed_collision = module.prepare_selected_trajectory(
            rows,
            audit,
            selection,
            {"Scene::1::0::1"},
            aligned,
        )

    assert [row["frame_idx"] for row in rewritten] == [0, 2]
    assert rewritten[0]["action_name"] == "Move Forward"
    assert rewritten[0]["trajectory_repair"].get(
        "earlier_stop_eligible_motion_preserved"
    ) is True
    assert rewritten[-1]["action_name"] == "Stop"
    assert rewritten[-1]["original_action_name"] == "Move Forward"
    assert rewritten[-1]["stop_visibility"]["preserved_earlier_motion_labels"] is True
    assert removed_collision == 1


def test_v4_navigation_only_removes_invalid_stop_without_fake_transition() -> None:
    module = _load_v4_production_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        aligned = root / "record_output_transition_aligned"
        images = aligned / "images" / "Scene" / "2" / "0" / "uav_on_0"
        images.mkdir(parents=True)
        for frame_idx in range(3):
            (images / f"{frame_idx:05d}.png").write_bytes(b"rgb")
        rows = [
            {
                "episode_key": "Scene::2::0",
                "scene_id": "Scene",
                "episode_id": "2",
                "pose_idx": "0",
                "frame_idx": frame_idx,
                "image_path": str(
                    root
                    / "record_output"
                    / "images"
                    / "Scene"
                    / "2"
                    / "0"
                    / "uav_on_0"
                    / f"{frame_idx:05d}.png"
                ),
                "action_name": "Move Forward" if frame_idx < 2 else "Stop",
                "action_vector": [0.0] * 8,
            }
            for frame_idx in range(3)
        ]
        rewritten, removed_collision, removed_stop = (
            module.prepare_navigation_only_trajectory(
                rows,
                "trajectory_viewpoint_distance_or_occlusion",
                {"Scene::2::0::0"},
                aligned,
            )
        )

    assert [row["frame_idx"] for row in rewritten] == [1]
    assert rewritten[0]["action_name"] == "Move Forward"
    assert rewritten[0]["trajectory_repair"]["removed_invalid_stop"] is True
    assert removed_collision == 1
    assert removed_stop == 1


def _load_script_module(filename: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_standoff_stop_can_be_appended_to_original_episode() -> None:
    module = _load_script_module("assemble_stop_visible_v4_appended_frames.py")
    rows = [
        {
            "episode_key": "Scene::2::0",
            "scene_id": "Scene",
            "episode_id": "2",
            "pose_idx": "0",
            "frame_idx": 7,
            "target_description": "target",
            "true_name": "Chair",
            "object_name": "Chair_1",
            "size": "mid",
            "action_name": "Move Forward",
            "trajectory_repair": {
                "removed_invalid_stop": True,
                "problem_cause": "target_facing_standoff_repairable",
            },
        }
    ]
    spec = {
        "stop_row": {
            "episode_key": "Scene::standoff::0",
            "scene_id": "Scene",
            "episode_id": "standoff",
            "pose_idx": "0",
            "frame_idx": 0,
            "image_path": "/tmp/stop.png",
            "depth_grid": [[1.0] * 3 for _ in range(3)],
            "action_name": "Stop",
            "stop_visibility": {
                "source_trajectory_key": "Scene::2::0",
                "source_candidate_frame_idx": 3,
            },
        },
        "queue_row": {
            "trajectory_key": "Scene::2::0",
            "capture_group": "repairable",
            "represented_trajectory_count": 1,
        },
    }

    appended = module.build_appended_stop(rows, spec)

    assert appended["episode_key"] == "Scene::2::0"
    assert appended["frame_idx"] == 8
    assert appended["action_name"] == "Stop"
    assert appended["target_description"] == "target"
    assert appended["stop_visibility"]["appended_to_original_trajectory"] is True
    assert (
        appended["stop_visibility"]["source_type"]
        == "target_facing_standoff_appended_to_original_episode"
    )


def test_stop_pair_archive_classifies_appended_repair_groups() -> None:
    module = _load_script_module("archive_stop_visible_pairs.py")
    repairable = {
        "stop_visibility": {
            "source_type": "target_facing_standoff_appended_to_original_episode",
            "capture_group": "repairable",
        }
    }
    rescue = {
        "stop_visibility": {
            "source_type": "target_facing_standoff_appended_to_original_episode",
            "capture_group": "rescue",
        }
    }

    assert module.stop_category(repairable) == "target_facing_repairable_appended"
    assert module.stop_category(rescue) == "below_threshold_rescue_appended"

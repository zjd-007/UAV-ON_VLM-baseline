from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


SIZE_BUCKETS = ("small", "mid", "big", "unknown")


def canonical_stencil_difference_mask(
    baseline: np.ndarray,
    marked: np.ndarray,
    canonical_color: tuple[int, int, int],
    halo_pixels: int = 2,
) -> np.ndarray:
    baseline_array = np.asarray(baseline, dtype=np.uint8)
    marked_array = np.asarray(marked, dtype=np.uint8)
    if baseline_array.shape != marked_array.shape or baseline_array.ndim != 3:
        raise ValueError(
            "baseline and marked segmentation images must have the same HxWxC shape"
        )
    changed = np.any(marked_array != baseline_array, axis=2)
    color = np.asarray(canonical_color, dtype=np.uint8).reshape(1, 1, 3)
    seed = changed & np.all(marked_array == color, axis=2)
    if not np.any(seed) or halo_pixels <= 0:
        return seed

    expanded = seed.copy()
    for _ in range(int(halo_pixels)):
        padded = np.pad(expanded, 1, mode="constant", constant_values=False)
        expanded = np.logical_or.reduce(
            [
                padded[y : y + seed.shape[0], x : x + seed.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return seed | (changed & expanded)


def parse_size_bucket(size_text: str | None) -> str:
    normalized = str(size_text or "").strip().lower()
    for bucket in SIZE_BUCKETS[:-1]:
        if normalized.startswith(bucket) or f" {bucket}" in normalized:
            return bucket
    return "unknown"


def mask_metrics(mask: np.ndarray, edge_band: int = 2) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError(f"expected a non-empty 2-D mask, got {binary.shape}")

    height, width = binary.shape
    pixels = int(binary.sum())
    result: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "pixel_count": pixels,
        "pixel_fraction": float(pixels / binary.size),
        "bbox": None,
        "bbox_width": 0,
        "bbox_height": 0,
        "bbox_area": 0,
        "bbox_fill_fraction": 0.0,
        "centroid": None,
        "border_sides": [],
        "clipped_sides_count": 0,
        "touches_opposite_borders": False,
        "edge_contact_pixels": 0,
        "edge_contact_fraction": 0.0,
    }
    if pixels == 0:
        return result

    ys, xs = np.where(binary)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    bbox_area = bbox_width * bbox_height

    band = max(1, int(edge_band))
    sides = []
    if bool(binary[:, :band].any()):
        sides.append("left")
    if bool(binary[:, max(0, width - band) :].any()):
        sides.append("right")
    if bool(binary[:band, :].any()):
        sides.append("top")
    if bool(binary[max(0, height - band) :, :].any()):
        sides.append("bottom")

    edge_mask = np.zeros_like(binary)
    edge_mask[:, :band] = True
    edge_mask[:, max(0, width - band) :] = True
    edge_mask[:band, :] = True
    edge_mask[max(0, height - band) :, :] = True
    edge_contact_pixels = int(np.logical_and(binary, edge_mask).sum())

    result.update(
        {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_width": int(bbox_width),
            "bbox_height": int(bbox_height),
            "bbox_area": int(bbox_area),
            "bbox_fill_fraction": float(pixels / bbox_area),
            "centroid": [float(xs.mean()), float(ys.mean())],
            "border_sides": sides,
            "clipped_sides_count": len(sides),
            "touches_opposite_borders": bool(
                ({"left", "right"}.issubset(sides))
                or ({"top", "bottom"}.issubset(sides))
            ),
            "edge_contact_pixels": edge_contact_pixels,
            "edge_contact_fraction": float(edge_contact_pixels / pixels),
        }
    )
    return result


@dataclass(frozen=True)
class VisibilityPolicy:
    min_pixels: dict[str, int] = field(
        default_factory=lambda: {
            "small": 96,
            "mid": 160,
            "big": 256,
            "unknown": 160,
        }
    )
    min_bbox_short_side: dict[str, int] = field(
        default_factory=lambda: {
            "small": 6,
            "mid": 8,
            "big": 10,
            "unknown": 8,
        }
    )
    strong_pixels: dict[str, int] = field(
        default_factory=lambda: {
            "small": 96,
            "mid": 160,
            "big": 256,
            "unknown": 160,
        }
    )
    min_relative_to_peak: float = 0.12
    max_pixel_fraction: float = 0.75
    max_edge_contact_fraction: float = 0.25
    max_clipped_sides: int = 2
    reject_opposite_borders: bool = True
    reject_collided: bool = False
    semantic_score_field: str | None = None
    min_semantic_score: float | None = None
    semantic_rank_field: str | None = None
    max_semantic_rank: int | None = None
    require_semantic_for_weak_geometry: bool = False
    selection_mode: str = "first_clear"
    preferred_pixel_fraction: dict[str, float] = field(
        default_factory=lambda: {
            "small": 0.02,
            "mid": 0.10,
            "big": 0.25,
            "unknown": 0.10,
        }
    )
    occupancy_quality_mode: str = "preferred_peak"
    quality_weights: dict[str, float] = field(
        default_factory=lambda: {
            "occupancy": 0.30,
            "center": 0.25,
            "clipping": 0.15,
            "distance": 0.20,
            "semantic": 0.10,
            "mask_density": 0.0,
        }
    )
    quality_aggregation: str = "weighted_arithmetic"
    quality_near_distance_m: float = 8.0
    quality_far_distance_m: float = 20.0
    quality_center_max_offset: float = 0.55
    semantic_score_low: float = 0.18
    semantic_score_high: float = 0.32
    earliest_within_best_score: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severe_clipping(metrics: dict[str, Any], policy: VisibilityPolicy) -> bool:
    if policy.reject_opposite_borders and metrics.get("touches_opposite_borders"):
        return True
    if int(metrics.get("clipped_sides_count", 0)) > policy.max_clipped_sides:
        return True
    if float(metrics.get("edge_contact_fraction", 0.0)) > policy.max_edge_contact_fraction:
        return True
    if float(metrics.get("pixel_fraction", 0.0)) > policy.max_pixel_fraction:
        return True
    return False


def assess_visibility_frame(
    frame: dict[str, Any],
    size_bucket: str,
    peak_pixels: int,
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    metrics = frame.get("mask") or frame
    pixels = int(metrics.get("pixel_count", 0))
    bbox_short_side = min(
        int(metrics.get("bbox_width", 0)),
        int(metrics.get("bbox_height", 0)),
    )
    min_pixels = int(policy.min_pixels.get(size_bucket, policy.min_pixels["unknown"]))
    min_short_side = int(
        policy.min_bbox_short_side.get(
            size_bucket,
            policy.min_bbox_short_side["unknown"],
        )
    )
    strong_pixels = int(
        policy.strong_pixels.get(size_bucket, policy.strong_pixels["unknown"])
    )
    relative_threshold = int(np.ceil(max(0, peak_pixels) * policy.min_relative_to_peak))
    reasons = []
    if pixels < min_pixels:
        reasons.append("too_few_pixels")
    if bbox_short_side < min_short_side:
        reasons.append("bbox_too_thin")
    if pixels < relative_threshold:
        reasons.append("too_small_relative_to_peak")
    if _severe_clipping(metrics, policy):
        reasons.append("severely_clipped_or_too_close")
    if policy.reject_collided and bool(
        (frame.get("collision_info") or {}).get("has_collided")
    ):
        reasons.append("candidate_pose_collided")

    semantic_score = None
    semantic_rank = None
    semantic_supported = False
    if policy.semantic_score_field:
        semantic_score = frame.get(policy.semantic_score_field)
        if semantic_score is not None and (
            policy.min_semantic_score is not None
            and float(semantic_score) >= policy.min_semantic_score
        ):
            semantic_supported = True
    if policy.semantic_rank_field:
        semantic_rank = frame.get(policy.semantic_rank_field)
        if semantic_rank is not None and (
            policy.max_semantic_rank is not None
            and int(semantic_rank) <= policy.max_semantic_rank
        ):
            semantic_supported = True
    weak_geometry = pixels < strong_pixels
    if (
        policy.require_semantic_for_weak_geometry
        and weak_geometry
        and not semantic_supported
    ):
        reasons.append("weak_geometry_without_semantic_support")

    return {
        "frame_idx": int(frame["frame_idx"]),
        "clear": not reasons,
        "reasons": reasons,
        "size_bucket": size_bucket,
        "pixel_count": pixels,
        "bbox_short_side": bbox_short_side,
        "peak_pixels": int(peak_pixels),
        "relative_pixel_fraction": float(pixels / peak_pixels) if peak_pixels else 0.0,
        "strong_pixels": strong_pixels,
        "weak_geometry": weak_geometry,
        "semantic_score": semantic_score,
        "semantic_rank": semantic_rank,
        "semantic_supported": semantic_supported,
    }


def _semantic_quality(assessment: dict[str, Any], policy: VisibilityPolicy) -> float:
    components = []
    semantic_score = assessment.get("semantic_score")
    if semantic_score is not None:
        denominator = max(1e-6, policy.semantic_score_high - policy.semantic_score_low)
        components.append(
            float(
                np.clip(
                    (float(semantic_score) - policy.semantic_score_low) / denominator,
                    0.0,
                    1.0,
                )
            )
        )
    semantic_rank = assessment.get("semantic_rank")
    if semantic_rank is not None:
        rank = int(semantic_rank)
        if rank <= 1:
            components.append(1.0)
        elif rank <= 3:
            components.append(0.85)
        elif rank <= 5:
            components.append(0.70)
        elif rank <= 10:
            components.append(0.45)
        else:
            components.append(0.20)
    return float(np.mean(components)) if components else 0.5


def recognition_view_quality(
    frame: dict[str, Any],
    assessment: dict[str, Any],
    size_bucket: str,
    policy: VisibilityPolicy,
) -> dict[str, Any]:
    metrics = frame.get("mask") or frame
    pixel_fraction = float(metrics.get("pixel_fraction", 0.0))
    preferred_fraction = float(
        policy.preferred_pixel_fraction.get(
            size_bucket,
            policy.preferred_pixel_fraction["unknown"],
        )
    )
    if preferred_fraction <= 0:
        occupancy_score = 0.0
    elif policy.occupancy_quality_mode == "preferred_peak":
        if pixel_fraction <= preferred_fraction:
            occupancy_score = pixel_fraction / preferred_fraction
        else:
            upper = max(preferred_fraction + 1e-6, policy.max_pixel_fraction)
            occupancy_score = 1.0 - (
                (pixel_fraction - preferred_fraction) / (upper - preferred_fraction)
            )
    elif policy.occupancy_quality_mode == "saturating":
        occupancy_score = pixel_fraction / preferred_fraction
    else:
        raise ValueError(
            f"unsupported occupancy_quality_mode: {policy.occupancy_quality_mode}"
        )
    occupancy_score = float(np.clip(occupancy_score, 0.0, 1.0))

    centroid = metrics.get("centroid")
    width = max(1, int(metrics.get("width", 1)))
    height = max(1, int(metrics.get("height", 1)))
    if centroid:
        normalized_x = float(centroid[0]) / width
        normalized_y = float(centroid[1]) / height
        center_offset = float(np.hypot(normalized_x - 0.5, normalized_y - 0.5))
        center_score = 1.0 - center_offset / max(
            1e-6,
            policy.quality_center_max_offset,
        )
    else:
        center_offset = None
        center_score = 0.0
    center_score = float(np.clip(center_score, 0.0, 1.0))

    clipped_sides = int(metrics.get("clipped_sides_count", 0))
    clipping_score = float(np.clip(1.0 - 0.15 * clipped_sides, 0.0, 1.0))
    if metrics.get("touches_opposite_borders"):
        clipping_score *= 0.5

    distance = frame.get("distance_to_target")
    if distance is None:
        distance_score = 0.5
    else:
        span = max(1e-6, policy.quality_far_distance_m - policy.quality_near_distance_m)
        distance_score = float(
            np.clip(
                (policy.quality_far_distance_m - float(distance)) / span,
                0.0,
                1.0,
            )
        )
    semantic_score = _semantic_quality(assessment, policy)
    mask_density_score = float(
        np.clip(assessment.get("relative_bbox_fill_fraction", 0.0), 0.0, 1.0)
    )
    components = {
        "occupancy": occupancy_score,
        "center": center_score,
        "clipping": clipping_score,
        "distance": distance_score,
        "semantic": semantic_score,
        "mask_density": mask_density_score,
    }
    weights = policy.quality_weights
    weight_sum = sum(max(0.0, float(weights.get(name, 0.0))) for name in components)
    quality_score = 0.0
    if weight_sum > 0:
        if policy.quality_aggregation == "weighted_arithmetic":
            quality_score = sum(
                components[name] * max(0.0, float(weights.get(name, 0.0)))
                for name in components
            ) / weight_sum
        elif policy.quality_aggregation == "weighted_geometric":
            quality_score = float(
                np.exp(
                    sum(
                        max(0.0, float(weights.get(name, 0.0)))
                        * np.log(max(0.02, components[name]))
                        for name in components
                    )
                    / weight_sum
                )
            )
        else:
            raise ValueError(
                f"unsupported quality_aggregation: {policy.quality_aggregation}"
            )
    return {
        "quality_score": float(quality_score),
        "quality_components": components,
        "center_offset": center_offset,
        "preferred_pixel_fraction": preferred_fraction,
        "distance_to_target": float(distance) if distance is not None else None,
    }


def select_first_clear_frame(
    frames: Iterable[dict[str, Any]],
    size_text: str | None,
    policy: VisibilityPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or VisibilityPolicy()
    ordered = sorted(frames, key=lambda row: int(row["frame_idx"]))
    size_bucket = parse_size_bucket(size_text)
    peak_candidates = []
    for frame in ordered:
        metrics = frame.get("mask") or frame
        if int(metrics.get("pixel_count", 0)) <= 0:
            continue
        if not _severe_clipping(metrics, policy):
            peak_candidates.append(int(metrics["pixel_count"]))
    peak_pixels = max(peak_candidates, default=0)
    assessments = [
        assess_visibility_frame(frame, size_bucket, peak_pixels, policy)
        for frame in ordered
    ]
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in ordered}
    peak_bbox_fill_fraction = max(
        (
            float(
                (frame_by_idx[assessment["frame_idx"]].get("mask") or {}).get(
                    "bbox_fill_fraction",
                    0.0,
                )
            )
            for assessment in assessments
            if assessment["clear"]
        ),
        default=0.0,
    )
    for assessment in assessments:
        metrics = frame_by_idx[assessment["frame_idx"]].get("mask") or {}
        bbox_fill_fraction = float(metrics.get("bbox_fill_fraction", 0.0))
        assessment["bbox_fill_fraction"] = bbox_fill_fraction
        assessment["peak_bbox_fill_fraction"] = peak_bbox_fill_fraction
        assessment["relative_bbox_fill_fraction"] = (
            bbox_fill_fraction / peak_bbox_fill_fraction
            if peak_bbox_fill_fraction > 0
            else 0.0
        )
        assessment.update(
            recognition_view_quality(
                frame_by_idx[assessment["frame_idx"]],
                assessment,
                size_bucket,
                policy,
            )
        )
    clear_assessments = [row for row in assessments if row["clear"]]
    if policy.selection_mode == "first_clear":
        selected = clear_assessments[0] if clear_assessments else None
    elif policy.selection_mode == "best_recognition_view":
        if clear_assessments:
            best_score = max(float(row["quality_score"]) for row in clear_assessments)
            near_best = [
                row
                for row in clear_assessments
                if float(row["quality_score"])
                >= best_score - policy.earliest_within_best_score
            ]
            selected = min(near_best, key=lambda row: int(row["frame_idx"]))
        else:
            selected = None
    else:
        raise ValueError(f"unsupported visibility selection_mode: {policy.selection_mode}")
    return {
        "selected_frame_idx": selected["frame_idx"] if selected else None,
        "selected_quality_score": selected["quality_score"] if selected else None,
        "selection_mode": policy.selection_mode,
        "size_bucket": size_bucket,
        "peak_pixels": peak_pixels,
        "policy": policy.to_dict(),
        "assessments": assessments,
    }

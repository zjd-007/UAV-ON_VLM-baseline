from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vlm_baseline.actions import ACTION_IDS


@dataclass(frozen=True)
class ActionRedirectResult:
    enabled: bool
    module: str
    original_action_id: int
    final_action_id: int
    original_command: str
    final_command: str
    changed: bool
    reason: str | None = None
    details: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "module": self.module,
            "original_action_id": self.original_action_id,
            "final_action_id": self.final_action_id,
            "original_command": self.original_command,
            "final_command": self.final_command,
            "changed": self.changed,
            "reason": self.reason,
            "details": self.details or {},
        }


COMMAND_BY_ID = {value: key for key, value in ACTION_IDS.items()}


class NoActionRedirect:
    name = "none"
    enabled = False

    def redirect(
        self,
        action_id: int,
        current_pose: list[float],
        start_position: list[float],
        depth_grid: list[list[float]] | None,
        get_pose_after_action: Callable[[list[float], int], list[float]],
    ) -> ActionRedirectResult:
        command = COMMAND_BY_ID.get(action_id, str(action_id))
        return ActionRedirectResult(
            enabled=False,
            module=self.name,
            original_action_id=action_id,
            final_action_id=action_id,
            original_command=command,
            final_command=command,
            changed=False,
        )


class UAVONBoundsDepthRedirect:
    """Small UAV-ON-style execution guard for the single-view Phi baseline."""

    name = "uavon_bounds_depth"
    enabled = True

    def __init__(
        self,
        search_radius: float = 50.0,
        near_obstacle_threshold: float = 2.0,
        forward_action_id: int = ACTION_IDS["forward 3m"],
        turn_left_action_id: int = ACTION_IDS["turn left 30 degree"],
        turn_right_action_id: int = ACTION_IDS["turn right 30 degree"],
    ) -> None:
        self.search_radius = float(search_radius)
        self.near_obstacle_threshold = float(near_obstacle_threshold)
        self.forward_action_id = int(forward_action_id)
        self.turn_left_action_id = int(turn_left_action_id)
        self.turn_right_action_id = int(turn_right_action_id)

    def redirect(
        self,
        action_id: int,
        current_pose: list[float],
        start_position: list[float],
        depth_grid: list[list[float]] | None,
        get_pose_after_action: Callable[[list[float], int], list[float]],
    ) -> ActionRedirectResult:
        original_command = COMMAND_BY_ID.get(action_id, str(action_id))
        final_action_id = action_id
        reason = None
        details: dict[str, Any] = {}

        if action_id == self.forward_action_id:
            forward_pose = get_pose_after_action(current_pose, self.forward_action_id)
            bounds = self._bounds(start_position)
            out_of_bounds = not self._within_bounds(forward_pose, bounds)
            if out_of_bounds:
                final_action_id = self._turn_toward_center(current_pose, start_position, get_pose_after_action)
                reason = "forward_out_of_search_bounds"
                details.update(
                    {
                        "bounds": bounds,
                        "current_xy": [round(float(current_pose[0]), 4), round(float(current_pose[1]), 4)],
                        "forward_xy": [round(float(forward_pose[0]), 4), round(float(forward_pose[1]), 4)],
                        "center_xy": [round(float(start_position[0]), 4), round(float(start_position[1]), 4)],
                    }
                )
            elif self._front_is_too_close(depth_grid):
                final_action_id = self._turn_toward_safer_side(depth_grid)
                reason = "forward_near_obstacle"
                details.update({"near_obstacle_threshold": self.near_obstacle_threshold})

        final_command = COMMAND_BY_ID.get(final_action_id, str(final_action_id))
        return ActionRedirectResult(
            enabled=True,
            module=self.name,
            original_action_id=action_id,
            final_action_id=final_action_id,
            original_command=original_command,
            final_command=final_command,
            changed=final_action_id != action_id,
            reason=reason,
            details=details,
        )

    def _bounds(self, start_position: list[float]) -> dict[str, float]:
        x, y = float(start_position[0]), float(start_position[1])
        return {
            "x_min": x - self.search_radius,
            "x_max": x + self.search_radius,
            "y_min": y - self.search_radius,
            "y_max": y + self.search_radius,
        }

    @staticmethod
    def _within_bounds(pose: list[float], bounds: dict[str, float]) -> bool:
        return (
            bounds["x_min"] <= float(pose[0]) <= bounds["x_max"]
            and bounds["y_min"] <= float(pose[1]) <= bounds["y_max"]
        )

    def _turn_toward_center(
        self,
        current_pose: list[float],
        start_position: list[float],
        get_pose_after_action: Callable[[list[float], int], list[float]],
    ) -> int:
        center_vec = np.asarray(
            [float(start_position[0]) - float(current_pose[0]), float(start_position[1]) - float(current_pose[1])],
            dtype=np.float32,
        )
        if np.linalg.norm(center_vec) < 1e-6:
            return self.turn_left_action_id

        left_pose = get_pose_after_action(current_pose, self.turn_left_action_id)
        right_pose = get_pose_after_action(current_pose, self.turn_right_action_id)
        left_score = self._heading_alignment(left_pose[3], center_vec)
        right_score = self._heading_alignment(right_pose[3], center_vec)
        return self.turn_left_action_id if left_score >= right_score else self.turn_right_action_id

    @staticmethod
    def _heading_alignment(yaw: float, target_vec: np.ndarray) -> float:
        heading = np.asarray([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
        denom = float(np.linalg.norm(heading) * np.linalg.norm(target_vec))
        if denom < 1e-6:
            return -1.0
        return float(np.dot(heading, target_vec) / denom)

    def _front_is_too_close(self, depth_grid: list[list[float]] | None) -> bool:
        arr = self._grid_array(depth_grid)
        if arr is None:
            return False
        center = arr.shape[1] // 2
        center_start = arr.shape[0] // 2
        front_center_min = float(np.min(arr[center_start:, center]))
        return front_center_min <= self.near_obstacle_threshold

    def _turn_toward_safer_side(self, depth_grid: list[list[float]] | None) -> int:
        arr = self._grid_array(depth_grid)
        if arr is None:
            return self.turn_left_action_id
        left_min = float(np.min(arr[:, 0]))
        right_min = float(np.min(arr[:, -1]))
        return self.turn_left_action_id if left_min >= right_min else self.turn_right_action_id

    @staticmethod
    def _grid_array(depth_grid: list[list[float]] | None) -> np.ndarray | None:
        if not depth_grid:
            return None
        arr = np.asarray(depth_grid, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).any():
            return None
        return np.where(np.isfinite(arr), arr, 0.0)


class UAVONDepthRedirect(UAVONBoundsDepthRedirect):
    """Depth-only guard: redirect unsafe forward moves, without bounds checks."""

    name = "uavon_depth"

    def redirect(
        self,
        action_id: int,
        current_pose: list[float],
        start_position: list[float],
        depth_grid: list[list[float]] | None,
        get_pose_after_action: Callable[[list[float], int], list[float]],
    ) -> ActionRedirectResult:
        original_command = COMMAND_BY_ID.get(action_id, str(action_id))
        final_action_id = action_id
        reason = None
        details: dict[str, Any] = {}

        if action_id == self.forward_action_id and self._front_is_too_close(depth_grid):
            final_action_id = self._turn_toward_safer_side(depth_grid)
            reason = "forward_near_obstacle"
            details.update({"near_obstacle_threshold": self.near_obstacle_threshold})

        final_command = COMMAND_BY_ID.get(final_action_id, str(final_action_id))
        return ActionRedirectResult(
            enabled=True,
            module=self.name,
            original_action_id=action_id,
            final_action_id=final_action_id,
            original_command=original_command,
            final_command=final_command,
            changed=final_action_id != action_id,
            reason=reason,
            details=details,
        )


def build_action_redirect(name: str, **kwargs: Any):
    if name == "none":
        return NoActionRedirect()
    if name == "uavon_depth":
        return UAVONDepthRedirect(**kwargs)
    if name == "uavon_bounds_depth":
        return UAVONBoundsDepthRedirect(**kwargs)
    raise ValueError(f"Unknown action redirect module: {name!r}")

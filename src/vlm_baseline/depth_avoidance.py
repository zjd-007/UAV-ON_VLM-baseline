from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthAvoidanceContext:
    enabled: bool
    module: str
    prompt_text: str
    depth_grid: list[list[float]] | None = None
    depth_summary: dict[str, Any] | None = None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "module": self.module,
            "depth_grid": self.depth_grid,
            "depth_summary": self.depth_summary,
            "error": self.error,
        }


class NoDepthAvoidance:
    name = "none"
    enabled = False

    def build_context(self, depth: np.ndarray | None) -> DepthAvoidanceContext:
        return DepthAvoidanceContext(enabled=False, module=self.name, prompt_text="")


class UAVONSingleViewDepthPrompt:
    """UAV-ON-style depth prompt for the current single forward view."""

    name = "uavon_single_view_prompt"
    enabled = True

    def __init__(
        self,
        grid_size: int = 3,
        max_meters: float = 100.0,
        forward_threshold: float = 4.0,
        turn_threshold: float = 1.5,
        descend_threshold: float = 6.0,
        ascend_top_threshold: float = 8.0,
    ) -> None:
        if grid_size < 2:
            raise ValueError(f"depth grid size must be >= 2, got {grid_size}")
        self.grid_size = int(grid_size)
        self.max_meters = float(max_meters)
        self.forward_threshold = float(forward_threshold)
        self.turn_threshold = float(turn_threshold)
        self.descend_threshold = float(descend_threshold)
        self.ascend_top_threshold = float(ascend_top_threshold)

    def build_context(self, depth: np.ndarray | None) -> DepthAvoidanceContext:
        if depth is None:
            return DepthAvoidanceContext(
                enabled=True,
                module=self.name,
                prompt_text=self._missing_depth_prompt(),
                error="depth image is missing",
            )
        try:
            grid = self.depth_to_grid(depth)
            prompt_text = self.format_prompt(grid)
            return DepthAvoidanceContext(
                enabled=True,
                module=self.name,
                prompt_text=prompt_text,
                depth_grid=grid.tolist(),
                depth_summary=None,
            )
        except Exception as exc:
            return DepthAvoidanceContext(
                enabled=True,
                module=self.name,
                prompt_text=self._missing_depth_prompt(),
                error=str(exc),
            )

    def depth_to_grid(self, depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"expected a non-empty 2-D depth image, got shape {arr.shape}")
        finite = np.isfinite(arr)
        if not finite.any():
            raise ValueError("depth image has no finite values")

        arr = np.where(finite, arr, self.max_meters)
        arr = np.clip(arr, 0.0, self.max_meters)

        rows = np.array_split(arr, self.grid_size, axis=0)
        grid_rows = []
        for row in rows:
            cols = np.array_split(row, self.grid_size, axis=1)
            grid_rows.append([float(np.min(col)) for col in cols])
        return np.round(np.asarray(grid_rows, dtype=np.float32), 1)

    def format_prompt(self, grid: np.ndarray) -> str:
        grid_text = self._format_grid(grid)
        return (
            "CurrentViewDepth in meters:\n"
            "Rows are top, middle, bottom; columns are left, center, right. "
            "Smaller values mean closer obstacles in the current forward camera view.\n"
            f"DepthGrid:\n{grid_text}\n\n"
            "Depth rules:\n"
            "- Use the RGB image to find the target and use CurrentViewDepth to avoid collisions.\n"
            "- Before moving forward, compare the intended 3m distance with the center and lower-center depth values.\n"
            "- If depth in the movement direction is less than the intended movement distance, do not move that way.\n"
            "- If the center or lower-center cells are shallow, consider turning or ascending instead of moving forward.\n"
            "- If the bottom row is shallow, descending is risky.\n"
            "- Turning can help inspect a safer direction when forward is unsafe.\n"
            "- Use \"stop\" only when the target is visually confirmed, not only because depth is small."
        )

    def _missing_depth_prompt(self) -> str:
        return (
            "CurrentViewDepth is unavailable. Use the RGB image and choose cautiously. "
            "Avoid moving forward if the image shows nearby obstacles."
        )

    @staticmethod
    def _format_grid(grid: np.ndarray) -> str:
        rows = []
        for row in grid.tolist():
            rows.append("[" + ", ".join(f"{value:g}" for value in row) + "]")
        return "[" + ",\n ".join(rows) + "]"


def build_depth_avoidance(name: str, **kwargs: Any):
    if name == "none":
        return NoDepthAvoidance()
    if name == "uavon_single_view_prompt":
        return UAVONSingleViewDepthPrompt(**kwargs)
    raise ValueError(f"Unknown depth avoidance module: {name!r}")

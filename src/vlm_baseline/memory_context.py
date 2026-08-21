from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_DIRECTED_V1_POLICY = """Target-directed memory policy:
- Navigation objective: maximize the chance of observing and reaching the target. Changing the recent action pattern is not itself a navigation goal.
- Silently assess current target evidence as ABSENT, POSSIBLE, or CONFIRMED_NEAR.
- CONFIRMED_NEAR requires a match in object category or shape, at least one distinctive target attribute, and an object large and clear enough to inspect. If CONFIRMED_NEAR, choose stop.
- If evidence is POSSIBLE, do not stop and do not scan away from it. Turn toward the candidate or approach it only when the complete movement path is depth-safe.
- If evidence is ABSENT, use the target description and visible scene context to choose the visible region most likely to contain the target.
- Prefer an action that reveals or approaches a promising target-consistent region while remaining depth-safe.
- If no visible region is more promising, choose the safest action that reveals unseen space from a new viewpoint.
- A full_rotation_loop means the current viewpoint is exhausted; it does not by itself determine the next action.
- Never turn only to make RecentActions look different. After a full scan, move toward a promising region when safe.
- If no promising region exists, move to a safe new viewpoint with high visual novelty. If translation is unsafe, inspect one safer unobserved direction and reassess.
- Current RGB and CurrentViewDepth describe the present state. Depth rules are hard constraints.
- Never choose stop merely because of a loop, stagnation, repeated actions, or high step count."""


TARGET_DIRECTED_V1_1_POLICY = """Memory rules:
- Memory contains executed history only. Current RGB and CurrentViewDepth describe the present state.
- Decision priority is: confirm the target, obey depth safety, follow visible target evidence, then recover from ineffective trajectory history.
- Choose stop only when the current RGB clearly shows the target category or shape and at least one distinctive described attribute. Do not stop for a tiny, partial, edge-clipped, heavily occluded, or merely similar object.
- If a target-like candidate is visible but not clear enough, keep it in view. Turn toward it, or approach only when it is near the image center and the complete 3m path is depth-safe.
- If no target-like candidate is visible, use the target description and visible scene context to inspect the most plausible direction. Changing the recent action pattern is not itself progress.
- Short same-direction turns are normal visual scanning. Avoid repeated left-right alternation.
- A full_rotation_loop or revisiting state means the current viewpoint is exhausted, but never move only to break the loop. Prefer forward only when it reveals a useful open region and the complete 3m path is safe.
- If the previous translation moved less than 1m, do not immediately repeat it. Never use descend only for loop recovery; use ascend only when the upper depth cells are clear.
- Never stop because of step count, stagnation, or repeated actions."""


@dataclass(frozen=True)
class MemoryPromptContext:
    enabled: bool
    module: str
    prompt_text: str
    summary: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "module": self.module,
            "prompt_text": self.prompt_text,
            "summary": self.summary,
        }


class NoEpisodicMemory:
    name = "none"
    enabled = False

    def build_context(self) -> MemoryPromptContext:
        return MemoryPromptContext(enabled=False, module=self.name, prompt_text="")

    def update(self, pose_before: list[float], pose_after: list[float], action: str) -> None:
        return None


class UAVONPoseHistoryMemory:
    """UAV_ON-style prompt-time episode memory for the single-view baseline."""

    name = "uavon_pose_history"
    enabled = True

    def __init__(
        self,
        start_pose: list[float],
        search_center: list[float] | None = None,
        history_size: int = 5,
        search_radius: float = 50.0,
        pose_yaw_unit: str = "radians",
        include_search_bounds: bool = False,
    ) -> None:
        if len(start_pose) < 4:
            raise ValueError(f"start_pose must be [x, y, z, yaw], got {start_pose}")
        if pose_yaw_unit not in {"radians", "legacy"}:
            raise ValueError(f"pose_yaw_unit must be 'radians' or 'legacy', got {pose_yaw_unit!r}")
        self.history_size = int(history_size)
        self.search_radius = float(search_radius)
        self.pose_yaw_unit = pose_yaw_unit
        self.include_search_bounds = bool(include_search_bounds)
        self.search_center = list(search_center[:3] if search_center else start_pose[:3])
        self.poses: list[list[float]] = [self._clean_pose(start_pose)]
        self.actions: list[str] = []
        self.distance_traveled = 0.0
        self.heading_changes: list[float] = []

    def build_context(self) -> MemoryPromptContext:
        recent_poses = self._recent_poses()
        pose_lines = []
        for pose in recent_poses:
            pose_lines.append(
                f"  [({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}), {pose[3]:.2f}]"
            )
        x0, y0, _ = self.search_center
        x_min = x0 - self.search_radius
        x_max = x0 + self.search_radius
        y_min = y0 - self.search_radius
        y_max = y0 + self.search_radius
        avg_heading = float(np.mean(self.heading_changes)) if self.heading_changes else 0.0
        recent_actions = self.actions[-self.history_size :]
        recent_actions_text = ", ".join(recent_actions) if recent_actions else "none"
        search_bounds_text = ""
        forward_rule = ""
        if self.include_search_bounds:
            search_bounds_text = (
                "SearchBounds are centered on the episode start position. "
                "Keep the next horizontal pose inside them.\n"
                f"SearchBounds: x=[{x_min:.1f}, {x_max:.1f}], "
                f"y=[{y_min:.1f}, {y_max:.1f}]\n"
            )
            forward_rule = (
                "- Before choosing forward 3m, consider whether the next pose "
                "would remain inside SearchBounds.\n"
            )
        prompt = (
            "EpisodeMemory:\n"
            f"{search_bounds_text}"
            f"{self._pose_history_title()}\n"
            "Format: [(x, y, z), yaw_degrees]\n"
            "[\n"
            + ",\n".join(pose_lines)
            + "\n]\n"
            "TrajectorySummary:\n"
            f"StepsSoFar = {len(self.actions)}\n"
            f"DistanceTraveled = {self.distance_traveled:.2f}m\n"
            f"AvgHeadingChange = {avg_heading:.2f}deg\n"
            f"RecentActions = {recent_actions_text}\n\n"
            "Memory rules:\n"
            "- Use EpisodeMemory to avoid revisiting the same place and avoid repeated rotations.\n"
            "- If RecentActions show repeated turns, prefer moving forward only when CurrentViewDepth is safe.\n"
            f"{forward_rule}"
            "- Do not stop only because memory shows many steps; stop only when the target is visually confirmed."
        )
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=prompt,
            summary={
                "history_size": self.history_size,
                "search_radius": self.search_radius,
                "include_search_bounds": self.include_search_bounds,
                "search_center": [round(v, 3) for v in self.search_center],
                "pose_yaw_input_unit": self.pose_yaw_unit,
                "pose_yaw_output_unit": "degrees",
                "steps_so_far": len(self.actions),
                "distance_traveled": round(self.distance_traveled, 3),
                "avg_heading_change": round(avg_heading, 3),
                "recent_actions": list(recent_actions),
                "recent_poses": [self._pose_to_record(pose) for pose in recent_poses],
            },
        )

    def update(self, pose_before: list[float], pose_after: list[float], action: str) -> None:
        before = self._clean_pose(pose_before)
        after = self._clean_pose(pose_after)
        self.actions.append(str(action))
        self.distance_traveled += float(np.linalg.norm(np.asarray(after[:3]) - np.asarray(before[:3])))
        self.heading_changes.append(abs(_angle_diff(after[3], before[3])))
        self.poses.append(after)

    def _recent_poses(self) -> list[list[float]]:
        return [list(pose) for pose in self.poses[-self.history_size :]]

    def _pose_history_title(self) -> str:
        return (
            f"Previous UAV Poses (up to last {self.history_size} actual poses, "
            "oldest to newest):"
        )

    def _clean_pose(self, pose: list[float]) -> list[float]:
        yaw = float(pose[3])
        if self.pose_yaw_unit == "radians":
            yaw = math.degrees(yaw)
        return [float(pose[0]), float(pose[1]), float(pose[2]), yaw]

    @staticmethod
    def _pose_to_record(pose: list[float]) -> dict[str, float]:
        return {
            "x": round(float(pose[0]), 3),
            "y": round(float(pose[1]), 3),
            "z": round(float(pose[2]), 3),
            "yaw": round(float(pose[3]), 3),
        }


class UAVONPoseHistoryV1Memory(UAVONPoseHistoryMemory):
    """Versioned alias that freezes the real5/action5 legacy prompt."""

    name = "uavon_pose_history_v1"


class UAVONPoseHistoryV2Memory(UAVONPoseHistoryMemory):
    """Compact, precomputed trajectory-state memory with the legacy 5/5 history."""

    name = "uavon_pose_history_v2"

    _TURN_DIRECTIONS = {
        "turn left 30 degree": "left",
        "turn right 30 degree": "right",
    }
    _TRANSLATION_ACTIONS = {"forward 3m", "ascend 3m", "descend 3m"}
    _STAGNANT_DISTANCE_METERS = 1.0
    _REVISIT_DISTANCE_METERS = 1.0
    _FULL_LOOP_POSITION_METERS = 1.0
    _FULL_LOOP_YAW_DEGREES = 15.0
    _FULL_LOOP_HEADING_CHANGE_DEGREES = 300.0

    def __init__(
        self,
        start_pose: list[float],
        search_center: list[float] | None = None,
        history_size: int = 5,
        search_radius: float = 50.0,
        pose_yaw_unit: str = "radians",
        include_search_bounds: bool = False,
        max_steps: int = 100,
    ) -> None:
        super().__init__(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
        )
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.max_steps = int(max_steps)
        self.action_effects: list[dict[str, float]] = []

    def update(self, pose_before: list[float], pose_after: list[float], action: str) -> None:
        before = self._clean_pose(pose_before)
        after = self._clean_pose(pose_after)
        self.action_effects.append(
            {
                "translation": float(
                    np.linalg.norm(np.asarray(after[:3]) - np.asarray(before[:3]))
                ),
                "heading_change": abs(_angle_diff(after[3], before[3])),
            }
        )
        super().update(pose_before, pose_after, action)

    def build_context(self) -> MemoryPromptContext:
        recent_poses = self._recent_poses()
        recent_actions = self.actions[-self.history_size :]
        pose_lines = [
            f"  [({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}), {pose[3]:.2f}]"
            for pose in recent_poses
        ]
        recent_actions_text = ", ".join(recent_actions) if recent_actions else "none"

        recent_net_displacement = self._recent_net_displacement()
        revisiting = self._is_revisiting()
        recent_motion_state = self._recent_motion_state(
            recent_actions,
            recent_net_displacement,
            revisiting,
        )
        recent_turn_state = self._recent_turn_state(recent_net_displacement)
        last_effect = self.action_effects[-1] if self.action_effects else None
        if last_effect is None:
            last_effect_text = "none"
        else:
            last_effect_text = (
                f"translation {last_effect['translation']:.2f}m, "
                f"heading_change {last_effect['heading_change']:.1f}deg"
            )

        search_bounds_text, search_bounds_rule = self._search_bounds_context()
        prompt = (
            "EpisodeMemory:\n"
            f"{search_bounds_text}"
            f"{self._pose_history_title()}\n"
            "Format: [(x, y, z), yaw_degrees]\n"
            "[\n"
            + ",\n".join(pose_lines)
            + "\n]\n"
            "TrajectorySummary:\n"
            f"StepsSoFar = {len(self.actions)}/{self.max_steps}\n"
            f"RecentNetDisplacementLast{self.history_size} = {recent_net_displacement:.2f}m\n"
            f"LastActionEffect = {last_effect_text}\n"
            f"RecentMotionState = {recent_motion_state}\n"
            f"RecentTurnState = {recent_turn_state}\n"
            f"RecentActions (oldest to newest) = {recent_actions_text}\n\n"
            "Memory rules:\n"
            "- Memory contains executed history only. The current RGB image and CurrentViewDepth describe the present state.\n"
            "- First inspect the current RGB image. If the target clearly matches the target description, choose stop immediately; memory must not override clear target evidence.\n"
            "- If the target is not clearly confirmed, never choose stop only because of the step count, stagnation, or repeated actions.\n"
            "- A short same-direction turn sequence is normal visual scanning. Change strategy only when RecentTurnState is oscillating or full_rotation_loop, or RecentMotionState is stagnant or revisiting.\n"
            "- For full_rotation_loop or revisiting, do not repeat the same turn pattern. Choose forward only when CurrentViewDepth indicates that the complete 3m path is safe.\n"
            "- For oscillating, do not reverse the most recent turn again; either repeat the most recent turn once to break the alternation, or choose forward when safe.\n"
            "- If a translation action moved less than 1m, do not immediately repeat that translation; inspect another direction first.\n"
            "- Never use descend only to break a loop. Use ascend only when the upper depth cells are clear.\n"
            f"{search_bounds_rule}"
        ).rstrip()

        summary = {
            "prompt_version": "uavon_pose_history_v2",
            "history_size": self.history_size,
            "max_steps": self.max_steps,
            "search_radius": self.search_radius,
            "include_search_bounds": self.include_search_bounds,
            "search_center": [round(v, 3) for v in self.search_center],
            "pose_yaw_input_unit": self.pose_yaw_unit,
            "pose_yaw_output_unit": "degrees",
            "steps_so_far": len(self.actions),
            "recent_net_displacement": round(recent_net_displacement, 3),
            "last_action_effect": (
                {
                    "translation": round(last_effect["translation"], 3),
                    "heading_change": round(last_effect["heading_change"], 3),
                }
                if last_effect is not None
                else None
            ),
            "recent_motion_state": recent_motion_state,
            "recent_turn_state": recent_turn_state,
            "revisiting": revisiting,
            "recent_actions": list(recent_actions),
            "recent_poses": [self._pose_to_record(pose) for pose in recent_poses],
        }
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=prompt,
            summary=summary,
        )

    def _recent_net_displacement(self) -> float:
        step_count = min(self.history_size, len(self.actions))
        if step_count == 0:
            return 0.0
        poses = self.poses[-(step_count + 1) :]
        return float(np.linalg.norm(np.asarray(poses[-1][:3]) - np.asarray(poses[0][:3])))

    def _recent_motion_state(
        self,
        recent_actions: list[str],
        recent_net_displacement: float,
        revisiting: bool,
    ) -> str:
        if not self.actions:
            return "not_started"
        if revisiting:
            return "revisiting"
        if any(action in self._TRANSLATION_ACTIONS for action in recent_actions):
            if recent_net_displacement < self._STAGNANT_DISTANCE_METERS:
                return "stagnant"
            return "progressing"
        return "stationary_scanning"

    def _recent_turn_state(self, recent_net_displacement: float) -> str:
        if self._has_full_rotation_loop():
            return "full_rotation_loop"

        recent_actions = self.actions[-self.history_size :]
        if (
            len(recent_actions) >= 4
            and all(action in self._TURN_DIRECTIONS for action in recent_actions[-4:])
            and all(
                left != right
                for left, right in zip(recent_actions[-4:], recent_actions[-3:])
            )
            and recent_net_displacement < self._STAGNANT_DISTANCE_METERS
        ):
            return "oscillating"

        if not self.actions or self.actions[-1] not in self._TURN_DIRECTIONS:
            return "none"
        last_action = self.actions[-1]
        count = 0
        for action in reversed(self.actions):
            if action != last_action:
                break
            count += 1
        return f"scan_{self._TURN_DIRECTIONS[last_action]}_x{count}"

    def _is_revisiting(self) -> bool:
        if len(self.actions) <= self.history_size:
            return False
        current = np.asarray(self.poses[-1][:3])
        for pose_index, pose in enumerate(self.poses[: -self.history_size]):
            intervening_actions = self.actions[pose_index:]
            if not any(action in self._TRANSLATION_ACTIONS for action in intervening_actions):
                continue
            if np.linalg.norm(current - np.asarray(pose[:3])) < self._REVISIT_DISTANCE_METERS:
                return True
        return False

    def _has_full_rotation_loop(self) -> bool:
        if len(self.actions) < 10:
            return False
        current_pose = self.poses[-1]
        current_position = np.asarray(current_pose[:3])
        for pose_index, previous_pose in enumerate(self.poses[:-10]):
            loop_actions = self.actions[pose_index:]
            if not loop_actions or not all(
                action in self._TURN_DIRECTIONS for action in loop_actions
            ):
                continue
            if sum(self.heading_changes[pose_index:]) < self._FULL_LOOP_HEADING_CHANGE_DEGREES:
                continue
            if (
                np.linalg.norm(current_position - np.asarray(previous_pose[:3]))
                >= self._FULL_LOOP_POSITION_METERS
            ):
                continue
            if (
                abs(_angle_diff(current_pose[3], previous_pose[3]))
                < self._FULL_LOOP_YAW_DEGREES
            ):
                return True
        return False

    def _search_bounds_context(self) -> tuple[str, str]:
        if not self.include_search_bounds:
            return "", ""
        x0, y0, _ = self.search_center
        text = (
            "SearchBounds are centered on the episode start position. "
            "Keep the next horizontal pose inside them.\n"
            f"SearchBounds: x=[{x0 - self.search_radius:.1f}, {x0 + self.search_radius:.1f}], "
            f"y=[{y0 - self.search_radius:.1f}, {y0 + self.search_radius:.1f}]\n"
        )
        rule = "- Keep the next horizontal pose inside SearchBounds.\n"
        return text, rule


class UAVONPoseHistoryPolicyMemory(UAVONPoseHistoryV2Memory):
    """V2 trajectory state with a versioned production policy suffix."""

    policy_name = ""
    policy_text = ""

    def build_context(self) -> MemoryPromptContext:
        base_context = super().build_context()
        marker = "\nMemory rules:\n"
        prefix, separator, _ = base_context.prompt_text.partition(marker)
        if not separator:
            raise ValueError("V2 memory prompt is missing the Memory rules marker")

        summary = dict(base_context.summary or {})
        summary.update(
            {
                "prompt_version": self.name,
                "base_prompt_version": UAVONPoseHistoryV2Memory.name,
                "memory_policy": self.policy_name,
            }
        )
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=f"{prefix}\n{self.policy_text}".rstrip(),
            summary=summary,
        )


class UAVONPoseHistoryTargetDirectedV1Memory(UAVONPoseHistoryPolicyMemory):
    """Exact production form of the fixed-frame target_directed_v1 policy."""

    name = "uavon_pose_history_target_directed_v1"
    policy_name = "target_directed_v1"
    policy_text = TARGET_DIRECTED_V1_POLICY


class UAVONPoseHistoryTargetDirectedV11Memory(UAVONPoseHistoryPolicyMemory):
    """Concrete target-directed policy with V2 trajectory safeguards restored."""

    name = "uavon_pose_history_target_directed_v1_1"
    policy_name = "target_directed_v1_1"
    policy_text = TARGET_DIRECTED_V1_1_POLICY


class UAVONPoseHistoryV3Memory(UAVONPoseHistoryV2Memory):
    """Action-oriented memory that turns trajectory diagnostics into one active directive."""

    name = "uavon_pose_history_v3"

    _TRANSLATION_NAMES = {
        "forward 3m": "forward",
        "ascend 3m": "ascend",
        "descend 3m": "descend",
    }

    def build_context(self) -> MemoryPromptContext:
        recent_poses = self._recent_poses()
        recent_actions = self.actions[-self.history_size :]
        pose_lines = [
            f"  [({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}), {pose[3]:.2f}]"
            for pose in recent_poses
        ]

        recent_net_displacement = self._recent_net_displacement()
        revisiting = self._is_revisiting()
        recent_motion_state = self._recent_motion_state(
            recent_actions,
            recent_net_displacement,
            revisiting,
        )
        recent_turn_state = self._recent_turn_state(recent_net_displacement)
        turns_since_translation = self._turns_since_last_translation()
        turn_direction = self._latest_turn_direction()
        recent_action_pattern = self._compact_action_pattern(recent_actions)
        last_translation_action, last_translation_distance = self._last_translation_effect()
        decision_mode = self._decision_mode(
            recent_turn_state,
            recent_motion_state,
            last_translation_action,
            last_translation_distance,
        )
        avoid_turn_direction = self._avoid_turn_direction(decision_mode, turn_direction)
        active_directive = self._active_directive(
            decision_mode,
            turn_direction,
            avoid_turn_direction,
            last_translation_action,
        )

        search_bounds_text, search_bounds_rule = self._search_bounds_context()
        last_translation_text = (
            "none"
            if last_translation_distance is None
            else f"{last_translation_distance:.2f}m"
        )
        prompt = (
            "EpisodeMemory:\n"
            f"{search_bounds_text}"
            f"{self._pose_history_title()}\n"
            "Format: [(x, y, z), yaw_degrees]\n"
            "[\n"
            + ",\n".join(pose_lines)
            + "\n]\n"
            "NavigationState:\n"
            f"RecentNetDisplacementLast{self.history_size} = {recent_net_displacement:.2f}m\n"
            f"LastTranslationDistance = {last_translation_text}\n"
            f"RecentActionPattern = {recent_action_pattern}\n"
            f"TurnsSinceLastTranslation = {turns_since_translation}\n"
            f"FullViewScanCompleted = {str(recent_turn_state == 'full_rotation_loop').lower()}\n"
            f"DecisionMode = {decision_mode}\n"
            f"AvoidTurnDirection = {avoid_turn_direction}\n"
            f"ActiveMemoryDirective = {active_directive}\n\n"
            "Memory decision policy:\n"
            "- The current RGB image and CurrentViewDepth describe the present state; memory is executed history only.\n"
            "- If one object clearly matches the target description in the current RGB image, choose stop immediately.\n"
            "- Otherwise follow ActiveMemoryDirective, but never violate the Depth rules.\n"
            "- Never choose stop merely to escape a loop, oscillation, failed movement, or high step count.\n"
            f"{search_bounds_rule}"
        ).rstrip()

        summary = {
            "prompt_version": self.name,
            "history_size": self.history_size,
            "max_steps": self.max_steps,
            "search_radius": self.search_radius,
            "include_search_bounds": self.include_search_bounds,
            "search_center": [round(v, 3) for v in self.search_center],
            "pose_yaw_input_unit": self.pose_yaw_unit,
            "pose_yaw_output_unit": "degrees",
            "recent_net_displacement": round(recent_net_displacement, 3),
            "last_translation_action": last_translation_action,
            "last_translation_distance": (
                round(last_translation_distance, 3)
                if last_translation_distance is not None
                else None
            ),
            "recent_motion_state": recent_motion_state,
            "recent_turn_state": recent_turn_state,
            "recent_action_pattern": recent_action_pattern,
            "turn_direction": turn_direction,
            "turns_since_last_translation": turns_since_translation,
            "full_view_scan_completed": recent_turn_state == "full_rotation_loop",
            "decision_mode": decision_mode,
            "avoid_turn_direction": avoid_turn_direction,
            "active_directive": active_directive,
            "revisiting": revisiting,
            "recent_actions": list(recent_actions),
            "recent_poses": [self._pose_to_record(pose) for pose in recent_poses],
        }
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=prompt,
            summary=summary,
        )

    def _turns_since_last_translation(self) -> int:
        count = 0
        for action in reversed(self.actions):
            if action not in self._TURN_DIRECTIONS:
                break
            count += 1
        return count

    def _latest_turn_direction(self) -> str:
        if not self.actions:
            return "none"
        return self._TURN_DIRECTIONS.get(self.actions[-1], "none")

    def _compact_action_pattern(self, recent_actions: list[str]) -> str:
        if not recent_actions:
            return "none"

        directions = [self._TURN_DIRECTIONS.get(action) for action in recent_actions]
        if all(direction is not None for direction in directions):
            if len(set(directions)) == 1:
                return f"{directions[0]}_turn_x{len(directions)}"
            if all(left != right for left, right in zip(directions, directions[1:])):
                return "alternating_turns"
            return "mixed_turns"

        compact = [
            self._TRANSLATION_NAMES.get(action, self._TURN_DIRECTIONS.get(action, "other"))
            for action in recent_actions
        ]
        if len(set(compact)) == 1:
            return f"{compact[0]}_x{len(compact)}"
        return "mixed_motion"

    def _last_translation_effect(self) -> tuple[str | None, float | None]:
        for action, effect in zip(reversed(self.actions), reversed(self.action_effects)):
            if action in self._TRANSLATION_ACTIONS:
                return self._TRANSLATION_NAMES[action], float(effect["translation"])
        return None, None

    def _decision_mode(
        self,
        recent_turn_state: str,
        recent_motion_state: str,
        last_translation_action: str | None,
        last_translation_distance: float | None,
    ) -> str:
        if recent_turn_state == "full_rotation_loop":
            return "LOOP_RECOVERY"
        if recent_turn_state == "oscillating":
            return "OSCILLATION_RECOVERY"
        if recent_motion_state == "revisiting":
            return "POSITION_RECOVERY"
        if (
            last_translation_action is not None
            and last_translation_distance is not None
            and last_translation_distance < self._STAGNANT_DISTANCE_METERS
            and self.actions[-1] in self._TRANSLATION_ACTIONS
        ):
            return "FAILED_TRANSLATION_RECOVERY"
        return "NORMAL"

    @staticmethod
    def _avoid_turn_direction(decision_mode: str, turn_direction: str) -> str:
        if turn_direction == "none":
            return "none"
        if decision_mode == "LOOP_RECOVERY":
            return turn_direction
        if decision_mode == "OSCILLATION_RECOVERY":
            return "left" if turn_direction == "right" else "right"
        return "none"

    @staticmethod
    def _active_directive(
        decision_mode: str,
        turn_direction: str,
        avoid_turn_direction: str,
        last_translation_action: str | None,
    ) -> str:
        if decision_mode == "LOOP_RECOVERY":
            return (
                "A full visual scan is complete. Change position now: move forward when the complete "
                f"3m path is depth-safe; otherwise turn away from {avoid_turn_direction} once and reassess. "
                f"Do not continue turning {turn_direction}."
            )
        if decision_mode == "OSCILLATION_RECOVERY":
            return (
                f"Break the left-right alternation. Do not turn {avoid_turn_direction} next; "
                f"move forward when depth-safe, otherwise continue {turn_direction} once and reassess."
            )
        if decision_mode == "POSITION_RECOVERY":
            return (
                "Leave the revisited position: move forward when depth-safe; otherwise inspect a new "
                "direction without repeating the recent motion pattern."
            )
        if decision_mode == "FAILED_TRANSLATION_RECOVERY":
            return (
                f"The last {last_translation_action} movement made less than 1m progress. "
                "Do not repeat it immediately; inspect another direction first."
            )
        return "Continue normal visual search; short directional scans are allowed."


class UAVONTransitionHistoryMemory(UAVONPoseHistoryMemory):
    """Prompt-time memory with aligned action-to-resulting-pose transitions."""

    name = "uavon_transition_history"

    def build_context(self) -> MemoryPromptContext:
        start_pose = self.poses[0]
        current_pose = self.poses[-1]
        recent_actions = self.actions[-self.history_size :]
        recent_after_poses = self.poses[-len(recent_actions) :] if recent_actions else []
        first_step = len(self.actions) - len(recent_actions) + 1
        recent_transitions = []
        transition_lines = []
        for offset, (action, pose_after) in enumerate(zip(recent_actions, recent_after_poses)):
            step = first_step + offset
            recent_transitions.append(
                {
                    "step": step,
                    "action": action,
                    "pose_after": self._pose_to_record(pose_after),
                }
            )
            transition_lines.append(
                f"  {step}. {action} -> "
                f"[({pose_after[0]:.2f}, {pose_after[1]:.2f}, {pose_after[2]:.2f}), "
                f"{pose_after[3]:.2f}deg]"
            )

        x0, y0, _ = self.search_center
        x_min = x0 - self.search_radius
        x_max = x0 + self.search_radius
        y_min = y0 - self.search_radius
        y_max = y0 + self.search_radius
        avg_heading = float(np.mean(self.heading_changes)) if self.heading_changes else 0.0
        transitions_text = "\n".join(transition_lines) if transition_lines else "  none"
        search_bounds_text = ""
        forward_rule = "- Before choosing forward 3m, consider CurrentViewDepth.\n"
        if self.include_search_bounds:
            search_bounds_text = (
                "SearchBounds are centered on EpisodeStartPose. "
                "Keep the next horizontal pose inside them.\n"
                f"SearchBounds: x=[{x_min:.1f}, {x_max:.1f}], "
                f"y=[{y_min:.1f}, {y_max:.1f}]\n"
            )
            forward_rule = (
                "- Before choosing forward 3m, consider both CurrentViewDepth "
                "and SearchBounds.\n"
            )
        prompt = (
            "EpisodeMemory:\n"
            "Pose format: [(x, y, z), yaw_degrees]\n"
            f"EpisodeStartPose: [({start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[2]:.2f}), "
            f"{start_pose[3]:.2f}deg]\n"
            f"CurrentPose: [({current_pose[0]:.2f}, {current_pose[1]:.2f}, {current_pose[2]:.2f}), "
            f"{current_pose[3]:.2f}deg]\n"
            f"{search_bounds_text}"
            f"RecentTransitions (last {self.history_size} executed steps, oldest to newest):\n"
            f"{transitions_text}\n"
            "TrajectorySummary:\n"
            f"StepsSoFar = {len(self.actions)}\n"
            f"DistanceTraveled = {self.distance_traveled:.2f}m\n"
            f"AvgHeadingChange = {avg_heading:.2f}deg\n\n"
            "Memory rules:\n"
            "- Use RecentTransitions to avoid revisiting the same place and repeating rotation loops.\n"
            "- If several turns barely change position, choose a position-changing action only when CurrentViewDepth is safe.\n"
            f"{forward_rule}"
            "- Do not stop only because many steps were used; stop only when the target is visually confirmed."
        )
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=prompt,
            summary={
                "history_size": self.history_size,
                "search_radius": self.search_radius,
                "include_search_bounds": self.include_search_bounds,
                "search_center": [round(v, 3) for v in self.search_center],
                "pose_yaw_input_unit": self.pose_yaw_unit,
                "pose_yaw_output_unit": "degrees",
                "start_pose": self._pose_to_record(start_pose),
                "current_pose": self._pose_to_record(current_pose),
                "steps_so_far": len(self.actions),
                "distance_traveled": round(self.distance_traveled, 3),
                "avg_heading_change": round(avg_heading, 3),
                "recent_transitions": recent_transitions,
            },
        )


class UAVONPoseActionHistoryMemory(UAVONPoseHistoryMemory):
    """Unpadded, aligned pose-action-pose transitions for prompt-time memory."""

    name = "uavon_pose_action_history"

    def build_context(self) -> MemoryPromptContext:
        recent_actions = self.actions[-self.history_size :]
        recent_poses = self.poses[-(len(recent_actions) + 1) :]
        first_step = len(self.actions) - len(recent_actions) + 1
        chain_lines = []
        recent_transitions = []
        pose_before = recent_poses[0]
        chain_lines.append(
            f"  Pose{first_step - 1}: "
            f"[({pose_before[0]:.2f}, {pose_before[1]:.2f}, {pose_before[2]:.2f}), "
            f"{pose_before[3]:.2f}deg]"
        )
        for offset, (action, pose_after) in enumerate(zip(recent_actions, recent_poses[1:])):
            step = first_step + offset
            chain_lines.append(f"  Action{step}: {action}")
            chain_lines.append(
                f"  Pose{step}: "
                f"[({pose_after[0]:.2f}, {pose_after[1]:.2f}, {pose_after[2]:.2f}), "
                f"{pose_after[3]:.2f}deg]"
            )
            recent_transitions.append(
                {
                    "step": step,
                    "pose_before": self._pose_to_record(pose_before),
                    "action": action,
                    "pose_after": self._pose_to_record(pose_after),
                }
            )
            pose_before = pose_after

        x0, y0, _ = self.search_center
        x_min = x0 - self.search_radius
        x_max = x0 + self.search_radius
        y_min = y0 - self.search_radius
        y_max = y0 + self.search_radius
        search_bounds_text = ""
        forward_rule = "- Before choosing forward 3m, consider CurrentViewDepth.\n"
        if self.include_search_bounds:
            search_bounds_text = (
                "SearchBounds are centered on the episode start position. "
                "Keep the next horizontal pose inside them.\n"
                f"SearchBounds: x=[{x_min:.1f}, {x_max:.1f}], "
                f"y=[{y_min:.1f}, {y_max:.1f}]\n"
            )
            forward_rule = (
                "- Before choosing forward 3m, consider both CurrentViewDepth "
                "and SearchBounds.\n"
            )

        avg_heading = float(np.mean(self.heading_changes)) if self.heading_changes else 0.0
        prompt = (
            "EpisodeMemory:\n"
            f"{search_bounds_text}"
            "Pose format: [(x, y, z), yaw_degrees]\n"
            f"RecentPoseActionChain (up to last {self.history_size} executed transitions, "
            "oldest to newest):\n"
            + "\n".join(chain_lines)
            + "\n"
            "TrajectorySummary:\n"
            f"StepsSoFar = {len(self.actions)}\n"
            f"DistanceTraveled = {self.distance_traveled:.2f}m\n"
            f"AvgHeadingChange = {avg_heading:.2f}deg\n\n"
            "Memory rules:\n"
            "- Read each Pose -> Action -> Pose segment as one executed transition.\n"
            "- Use the chain to avoid revisiting the same place and repeating rotation loops.\n"
            "- If several turns barely change position, choose a position-changing action only when CurrentViewDepth is safe.\n"
            f"{forward_rule}"
            "- Do not stop only because many steps were used; stop only when the target is visually confirmed."
        )
        return MemoryPromptContext(
            enabled=True,
            module=self.name,
            prompt_text=prompt,
            summary={
                "history_size": self.history_size,
                "pose_history_size": self.history_size + 1,
                "action_history_size": self.history_size,
                "search_radius": self.search_radius,
                "include_search_bounds": self.include_search_bounds,
                "search_center": [round(v, 3) for v in self.search_center],
                "pose_yaw_input_unit": self.pose_yaw_unit,
                "pose_yaw_output_unit": "degrees",
                "steps_so_far": len(self.actions),
                "distance_traveled": round(self.distance_traveled, 3),
                "avg_heading_change": round(avg_heading, 3),
                "recent_actions": list(recent_actions),
                "recent_poses": [self._pose_to_record(pose) for pose in recent_poses],
                "recent_transitions": recent_transitions,
            },
        )


def _angle_diff(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def build_episodic_memory(
    name: str,
    start_pose: list[float],
    search_center: list[float] | None = None,
    history_size: int = 5,
    search_radius: float = 50.0,
    pose_yaw_unit: str = "radians",
    include_search_bounds: bool = False,
    max_steps: int = 100,
):
    if name == "none":
        return NoEpisodicMemory()
    if name == "uavon_pose_history":
        return UAVONPoseHistoryMemory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
        )
    if name == "uavon_pose_history_v1":
        return UAVONPoseHistoryV1Memory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
        )
    if name == "uavon_transition_history":
        return UAVONTransitionHistoryMemory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
        )
    if name == "uavon_pose_history_v2":
        return UAVONPoseHistoryV2Memory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
            max_steps=max_steps,
        )
    if name == "uavon_pose_history_target_directed_v1":
        return UAVONPoseHistoryTargetDirectedV1Memory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
            max_steps=max_steps,
        )
    if name == "uavon_pose_history_target_directed_v1_1":
        return UAVONPoseHistoryTargetDirectedV11Memory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
            max_steps=max_steps,
        )
    if name == "uavon_pose_history_v3":
        return UAVONPoseHistoryV3Memory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
            max_steps=max_steps,
        )
    if name == "uavon_pose_action_history":
        return UAVONPoseActionHistoryMemory(
            start_pose=start_pose,
            search_center=search_center,
            history_size=history_size,
            search_radius=search_radius,
            pose_yaw_unit=pose_yaw_unit,
            include_search_bounds=include_search_bounds,
        )
    raise ValueError(f"Unknown memory context module: {name!r}")

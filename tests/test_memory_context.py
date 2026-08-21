from vlm_baseline.memory_context import (
    TARGET_DIRECTED_V1_1_POLICY,
    TARGET_DIRECTED_V1_POLICY,
    UAVONPoseActionHistoryMemory,
    UAVONPoseHistoryMemory,
    UAVONPoseHistoryTargetDirectedV11Memory,
    UAVONPoseHistoryTargetDirectedV1Memory,
    UAVONPoseHistoryV1Memory,
    UAVONPoseHistoryV2Memory,
    UAVONPoseHistoryV3Memory,
    build_episodic_memory,
)


def _pose(step: int) -> list[float]:
    return [float(step), 0.0, 0.0, 0.0]


def test_pose_history_uses_only_real_entries_and_last_ten_actions() -> None:
    memory = UAVONPoseHistoryMemory(start_pose=_pose(0), history_size=10)

    initial = memory.build_context().summary
    assert initial is not None
    assert len(initial["recent_poses"]) == 1
    assert initial["recent_actions"] == []

    for step in range(1, 4):
        memory.update(_pose(step - 1), _pose(step), f"action-{step}")

    early = memory.build_context().summary
    assert early is not None
    assert [pose["x"] for pose in early["recent_poses"]] == [0.0, 1.0, 2.0, 3.0]
    assert early["recent_actions"] == ["action-1", "action-2", "action-3"]

    for step in range(4, 13):
        memory.update(_pose(step - 1), _pose(step), f"action-{step}")

    late = memory.build_context().summary
    assert late is not None
    assert [pose["x"] for pose in late["recent_poses"]] == [
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]
    assert late["recent_actions"] == [f"action-{step}" for step in range(3, 13)]


def test_default_memory_matches_real5_action5_nobounds_prompt() -> None:
    memory = build_episodic_memory(
        "uavon_pose_history",
        start_pose=_pose(0),
        include_search_bounds=False,
    )
    for step in range(1, 8):
        memory.update(_pose(step - 1), _pose(step), f"action-{step}")

    context = memory.build_context()
    assert context.summary is not None
    assert context.summary["history_size"] == 5
    assert [pose["x"] for pose in context.summary["recent_poses"]] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert context.summary["recent_actions"] == [
        "action-3",
        "action-4",
        "action-5",
        "action-6",
        "action-7",
    ]
    assert (
        "Previous UAV Poses (up to last 5 actual poses, oldest to newest):"
        in context.prompt_text
    )
    assert "RecentPoseActionChain" not in context.prompt_text
    assert "SearchBounds" not in context.prompt_text


def test_legacy_pose_history_prompt_is_stable() -> None:
    memory = UAVONPoseHistoryMemory(start_pose=_pose(0), history_size=5)
    memory.update(_pose(0), _pose(1), "forward 3m")

    assert memory.build_context().prompt_text == (
        "EpisodeMemory:\n"
        "Previous UAV Poses (up to last 5 actual poses, oldest to newest):\n"
        "Format: [(x, y, z), yaw_degrees]\n"
        "[\n"
        "  [(0.00, 0.00, 0.00), 0.00],\n"
        "  [(1.00, 0.00, 0.00), 0.00]\n"
        "]\n"
        "TrajectorySummary:\n"
        "StepsSoFar = 1\n"
        "DistanceTraveled = 1.00m\n"
        "AvgHeadingChange = 0.00deg\n"
        "RecentActions = forward 3m\n\n"
        "Memory rules:\n"
        "- Use EpisodeMemory to avoid revisiting the same place and avoid repeated rotations.\n"
        "- If RecentActions show repeated turns, prefer moving forward only when CurrentViewDepth is safe.\n"
        "- Do not stop only because memory shows many steps; stop only when the target is visually confirmed."
    )


def test_versioned_v1_alias_matches_legacy_prompt_exactly() -> None:
    legacy = UAVONPoseHistoryMemory(start_pose=_pose(0), history_size=5)
    versioned = UAVONPoseHistoryV1Memory(start_pose=_pose(0), history_size=5)
    for memory in (legacy, versioned):
        memory.update(_pose(0), _pose(1), "forward 3m")

    assert versioned.name == "uavon_pose_history_v1"
    assert versioned.build_context().prompt_text == legacy.build_context().prompt_text


def test_pose_history_v2_keeps_real5_history_and_precomputes_state() -> None:
    memory = UAVONPoseHistoryV2Memory(
        start_pose=_pose(0),
        history_size=5,
        pose_yaw_unit="legacy",
        max_steps=100,
    )
    for step in range(1, 8):
        memory.update(_pose(step - 1), _pose(step), "forward 3m")

    context = memory.build_context()
    assert context.module == "uavon_pose_history_v2"
    assert context.summary is not None
    assert [pose["x"] for pose in context.summary["recent_poses"]] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert context.summary["recent_actions"] == ["forward 3m"] * 5
    assert context.summary["recent_net_displacement"] == 5.0
    assert context.summary["recent_motion_state"] == "progressing"
    assert context.summary["recent_turn_state"] == "none"
    assert "DistanceTraveled" not in context.prompt_text
    assert "AvgHeadingChange" not in context.prompt_text
    assert "StepsSoFar = 7/100" in context.prompt_text
    assert "RecentNetDisplacementLast5 = 5.00m" in context.prompt_text
    assert "SearchBounds" not in context.prompt_text


def test_pose_history_v2_distinguishes_scan_oscillation_and_full_loop() -> None:
    scan = UAVONPoseHistoryV2Memory(start_pose=_pose(0), pose_yaw_unit="legacy")
    for yaw in (30.0, 60.0, 90.0):
        before = scan.poses[-1]
        scan.update(before, [0.0, 0.0, 0.0, yaw], "turn right 30 degree")
    assert scan.build_context().summary["recent_turn_state"] == "scan_right_x3"

    oscillation = UAVONPoseHistoryV2Memory(start_pose=_pose(0), pose_yaw_unit="legacy")
    for action, yaw in (
        ("turn left 30 degree", -30.0),
        ("turn right 30 degree", 0.0),
        ("turn left 30 degree", -30.0),
        ("turn right 30 degree", 0.0),
    ):
        before = oscillation.poses[-1]
        oscillation.update(before, [0.0, 0.0, 0.0, yaw], action)
    assert oscillation.build_context().summary["recent_turn_state"] == "oscillating"

    full_loop = UAVONPoseHistoryV2Memory(start_pose=_pose(0), pose_yaw_unit="legacy")
    for step in range(1, 13):
        before = full_loop.poses[-1]
        yaw = float((step * 30) % 360)
        full_loop.update(before, [0.0, 0.0, 0.0, yaw], "turn right 30 degree")
    assert full_loop.build_context().summary["recent_turn_state"] == "full_rotation_loop"


def test_pose_history_v2_detects_stagnation_and_revisit() -> None:
    stagnant = UAVONPoseHistoryV2Memory(start_pose=_pose(0), pose_yaw_unit="legacy")
    stagnant.update(_pose(0), [0.2, 0.0, 0.0, 0.0], "forward 3m")
    summary = stagnant.build_context().summary
    assert summary["recent_motion_state"] == "stagnant"
    assert summary["last_action_effect"]["translation"] == 0.2

    revisit = UAVONPoseHistoryV2Memory(start_pose=_pose(0), pose_yaw_unit="legacy")
    positions = [3.0, 6.0, 9.0, 6.0, 3.0, 0.0]
    for x in positions:
        before = revisit.poses[-1]
        revisit.update(before, [x, 0.0, 0.0, 0.0], "forward 3m")
    summary = revisit.build_context().summary
    assert summary["revisiting"] is True
    assert summary["recent_motion_state"] == "revisiting"


def test_target_directed_v1_preserves_v2_state_and_uses_exact_policy() -> None:
    memory = UAVONPoseHistoryTargetDirectedV1Memory(
        start_pose=_pose(0),
        history_size=5,
        pose_yaw_unit="legacy",
    )
    memory.update(_pose(0), _pose(1), "forward 3m")

    context = memory.build_context()
    assert context.module == "uavon_pose_history_target_directed_v1"
    assert context.summary is not None
    assert context.summary["prompt_version"] == context.module
    assert context.summary["base_prompt_version"] == "uavon_pose_history_v2"
    assert context.summary["memory_policy"] == "target_directed_v1"
    assert "RecentNetDisplacementLast5 = 1.00m" in context.prompt_text
    assert context.prompt_text.endswith(TARGET_DIRECTED_V1_POLICY)
    assert "Memory rules:\n- Memory contains executed history only." not in context.prompt_text


def test_target_directed_v1_1_factory_restores_concrete_safeguards() -> None:
    memory = build_episodic_memory(
        "uavon_pose_history_target_directed_v1_1",
        start_pose=_pose(0),
        pose_yaw_unit="legacy",
    )
    assert isinstance(memory, UAVONPoseHistoryTargetDirectedV11Memory)

    memory.update(_pose(0), [0.2, 0.0, 0.0, 0.0], "forward 3m")
    context = memory.build_context()
    assert context.module == "uavon_pose_history_target_directed_v1_1"
    assert context.summary is not None
    assert context.summary["memory_policy"] == "target_directed_v1_1"
    assert context.summary["recent_motion_state"] == "stagnant"
    assert context.prompt_text.endswith(TARGET_DIRECTED_V1_1_POLICY)
    assert "Avoid repeated left-right alternation." in context.prompt_text
    assert "Never use descend only for loop recovery" in context.prompt_text


def test_target_directed_v1_factory_keeps_original_policy_separate() -> None:
    original = build_episodic_memory(
        "uavon_pose_history_target_directed_v1",
        start_pose=_pose(0),
        pose_yaw_unit="legacy",
    )
    revised = build_episodic_memory(
        "uavon_pose_history_target_directed_v1_1",
        start_pose=_pose(0),
        pose_yaw_unit="legacy",
    )

    assert isinstance(original, UAVONPoseHistoryTargetDirectedV1Memory)
    assert isinstance(revised, UAVONPoseHistoryTargetDirectedV11Memory)
    assert original.build_context().prompt_text != revised.build_context().prompt_text


def test_pose_history_v3_compacts_actions_and_activates_loop_recovery() -> None:
    memory = UAVONPoseHistoryV3Memory(
        start_pose=_pose(0),
        history_size=5,
        pose_yaw_unit="legacy",
    )
    for step in range(1, 13):
        before = memory.poses[-1]
        yaw = float((step * 30) % 360)
        memory.update(before, [0.0, 0.0, 0.0, yaw], "turn right 30 degree")

    context = memory.build_context()
    assert context.module == "uavon_pose_history_v3"
    assert context.summary is not None
    assert context.summary["prompt_version"] == "uavon_pose_history_v3"
    assert context.summary["recent_action_pattern"] == "right_turn_x5"
    assert context.summary["turns_since_last_translation"] == 12
    assert context.summary["full_view_scan_completed"] is True
    assert context.summary["decision_mode"] == "LOOP_RECOVERY"
    assert context.summary["avoid_turn_direction"] == "right"
    assert "RecentActions (oldest to newest)" not in context.prompt_text
    assert "DecisionMode = LOOP_RECOVERY" in context.prompt_text
    assert "Do not continue turning right" in context.prompt_text


def test_pose_history_v3_factory_and_oscillation_directive() -> None:
    memory = build_episodic_memory(
        "uavon_pose_history_v3",
        start_pose=_pose(0),
        pose_yaw_unit="legacy",
    )
    for action, yaw in (
        ("turn left 30 degree", -30.0),
        ("turn right 30 degree", 0.0),
        ("turn left 30 degree", -30.0),
        ("turn right 30 degree", 0.0),
    ):
        before = memory.poses[-1]
        memory.update(before, [0.0, 0.0, 0.0, yaw], action)

    context = memory.build_context()
    assert context.summary is not None
    assert context.summary["recent_action_pattern"] == "alternating_turns"
    assert context.summary["decision_mode"] == "OSCILLATION_RECOVERY"
    assert context.summary["avoid_turn_direction"] == "left"
    assert "Do not turn left next" in context.prompt_text


def test_pose_action_history_uses_six_real_poses_and_five_actions_without_padding() -> None:
    memory = UAVONPoseActionHistoryMemory(start_pose=_pose(0), history_size=5)
    for step in range(1, 8):
        memory.update(_pose(step - 1), _pose(step), f"action-{step}")

    context = memory.build_context()
    assert context.summary is not None
    assert [pose["x"] for pose in context.summary["recent_poses"]] == [
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    assert context.summary["recent_actions"] == [
        "action-3",
        "action-4",
        "action-5",
        "action-6",
        "action-7",
    ]
    assert context.summary["pose_history_size"] == 6
    assert context.summary["action_history_size"] == 5
    assert context.summary["include_search_bounds"] is False
    assert "RecentPoseActionChain (up to last 5 executed transitions" in context.prompt_text
    assert "Pose2:" in context.prompt_text
    assert "Action3: action-3" in context.prompt_text
    assert "Pose3:" in context.prompt_text
    assert "SearchBounds" not in context.prompt_text


def test_pose_action_history_early_steps_use_only_real_transitions() -> None:
    memory = UAVONPoseActionHistoryMemory(start_pose=_pose(0), history_size=5)
    memory.update(_pose(0), _pose(1), "forward 3m")

    context = memory.build_context()
    assert context.summary is not None
    assert [pose["x"] for pose in context.summary["recent_poses"]] == [0.0, 1.0]
    assert context.summary["recent_actions"] == ["forward 3m"]
    assert len(context.summary["recent_transitions"]) == 1
    assert "Pose0:" in context.prompt_text
    assert "Action1: forward 3m" in context.prompt_text
    assert "Pose1:" in context.prompt_text

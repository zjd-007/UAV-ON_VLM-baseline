#!/usr/bin/env python3
"""Diagnose recognition, conversion, and exploration failures in an eval run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SUCCESS_RADIUS_METERS = 20.0
TURN_ACTIONS = {"turn left 30 degree", "turn right 30 degree"}
TRANSLATION_ACTIONS = {"forward 3m", "ascend 3m", "descend 3m"}
ACTION_ORDER = (
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("map_name", "")), str(row.get("episode_id", ""))


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = [float(value) for value in values]
    return {
        "count": len(collected),
        "mean": statistics.fmean(collected) if collected else None,
        "median": statistics.median(collected) if collected else None,
        "p75": percentile(collected, 0.75),
        "p90": percentile(collected, 0.90),
        "p95": percentile(collected, 0.95),
        "min": min(collected) if collected else None,
        "max": max(collected) if collected else None,
    }


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left[:3], right[:3])))


def longest_same_turn(actions: list[str]) -> int:
    longest = current = 0
    previous = ""
    for action in actions:
        if action in TURN_ACTIONS and action == previous:
            current += 1
        elif action in TURN_ACTIONS:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous = action
    return longest


def longest_alternating_turn(actions: list[str]) -> int:
    longest = current = 0
    previous = ""
    for action in actions:
        if action not in TURN_ACTIONS:
            current = 0
        elif previous in TURN_ACTIONS and action != previous:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = action
    return longest


def longest_turn_only_run(actions: list[str]) -> int:
    longest = current = 0
    for action in actions:
        current = current + 1 if action in TURN_ACTIONS else 0
        longest = max(longest, current)
    return longest


def build_source_index(run_dir: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    duplicates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(run_dir.glob("lane*/temp/*.json")):
        scene, separator, episode = path.stem.rpartition("-")
        if not separator:
            continue
        key = scene, episode
        if key in result:
            duplicates[key].extend([result[key], path])
        else:
            result[key] = path
    if duplicates:
        keys = ", ".join(f"{scene}/{episode}" for scene, episode in sorted(duplicates)[:5])
        raise RuntimeError(f"Duplicate result JSONs found for {len(duplicates)} tasks: {keys}")
    return result


def resolve_image(source: Path | None, image_path: Any) -> str:
    if source is None or not image_path:
        return ""
    return str((source.parent.parent / str(image_path)).resolve())


def size_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    for group in ("small", "mid", "big"):
        if text.startswith(group):
            return group
    return "unknown"


def forward_clearance(record: dict[str, Any]) -> float | None:
    grid = (record.get("depth_avoidance") or {}).get("depth_grid")
    if not isinstance(grid, list) or len(grid) < 3:
        return None
    try:
        return min(float(grid[1][1]), float(grid[2][1]))
    except (IndexError, TypeError, ValueError):
        return None


def derive_episode(row: dict[str, Any], source: Path | None) -> dict[str, Any]:
    records = row.get("step_records") or []
    actions = [str(record.get("parsed_command", "")) for record in records]
    distances: list[tuple[int, float]] = []
    if records:
        distances.append((-1, float(records[0].get("distance_before", math.inf))))
        distances.extend(
            (index, float(record.get("distance_after", math.inf)))
            for index, record in enumerate(records)
        )
    elif row.get("ne") is not None:
        distances.append((-1, float(row["ne"])))

    closest_step, minimum_distance = min(distances, key=lambda item: item[1])
    reach_steps = [step for step, value in distances if value <= SUCCESS_RADIUS_METERS]
    first_reach_step = min(reach_steps) if reach_steps else None
    observed_reach = bool(reach_steps)

    poses: list[list[float]] = []
    if records:
        poses.append([float(value) for value in records[0].get("pose_before", [])[:3]])
        poses.extend(
            [float(value) for value in record.get("pose_after", [])[:3]] for record in records
        )
    path_length = sum(distance(before, after) for before, after in zip(poses, poses[1:]))
    net_displacement = distance(poses[0], poses[-1]) if len(poses) >= 2 else 0.0

    action_counts = Counter(actions)
    turn_count = sum(action_counts[action] for action in TURN_ACTIONS)
    translation_count = sum(action_counts[action] for action in TRANSLATION_ACTIONS)
    motion_states = Counter(
        str(((record.get("memory_context") or {}).get("summary") or {}).get("recent_motion_state"))
        for record in records
    )
    turn_states = Counter(
        str(((record.get("memory_context") or {}).get("summary") or {}).get("recent_turn_state"))
        for record in records
    )
    memory_state_actions: dict[str, Counter[str]] = defaultdict(Counter)
    full_rotation_clear_3m = Counter()
    for record, action in zip(records, actions):
        summary = ((record.get("memory_context") or {}).get("summary") or {})
        for state in (
            str(summary.get("recent_motion_state")),
            str(summary.get("recent_turn_state")),
        ):
            memory_state_actions[state][action] += 1
        if str(summary.get("recent_turn_state")) == "full_rotation_loop":
            clearance = forward_clearance(record)
            if clearance is not None and clearance > 3.0:
                full_rotation_clear_3m[action] += 1
    collision = bool(row.get("collision"))
    official_osr = bool(row.get("osr"))
    success = bool(row.get("acc"))
    termination = str(row.get("termination_reason", "unknown"))

    if success:
        outcome_group = "success"
    elif official_osr:
        outcome_group = "osr_only"
    else:
        outcome_group = "no_official_osr"

    closest_record = records[closest_step] if closest_step is not None and closest_step >= 0 else None
    final_record = records[-1] if records else None
    final_collision_info = (final_record or {}).get("collision_info") or {}
    initial_distance = distances[0][1]
    final_distance = float(row.get("ne", distances[-1][1]))
    progress_to_closest = initial_distance - minimum_distance
    steps_after_first_reach = (
        len(records) - first_reach_step - 1
        if first_reach_step is not None and first_reach_step >= 0
        else (len(records) if first_reach_step == -1 else None)
    )

    result: dict[str, Any] = {
        "scene": str(row.get("map_name", "")),
        "episode_id": str(row.get("episode_id", "")),
        "target": str(row.get("true_name") or row.get("object_name") or "").strip(),
        "object_name": str(row.get("object_name") or ""),
        "description": str(row.get("description") or "").strip(),
        "size": str(row.get("size") or ""),
        "size_group": size_group(row.get("size")),
        "used_in_train": int(row.get("used-in-train", 0) or 0),
        "outcome_group": outcome_group,
        "success": int(success),
        "official_osr": int(official_osr),
        "observed_reach_20m": int(observed_reach),
        "reached_then_collided": int(observed_reach and collision),
        "collision": int(collision),
        "termination_reason": termination,
        "final_within_20m": int(final_distance <= SUCCESS_RADIUS_METERS),
        "steps": len(records),
        "initial_distance": initial_distance,
        "minimum_distance": minimum_distance,
        "closest_step": closest_step,
        "first_reach_step": first_reach_step,
        "steps_after_first_reach": steps_after_first_reach,
        "final_distance": final_distance,
        "progress_to_closest": progress_to_closest,
        "path_length_3d": path_length,
        "net_displacement_3d": net_displacement,
        "turn_count": turn_count,
        "translation_count": translation_count,
        "turn_fraction": turn_count / len(actions) if actions else 0.0,
        "forward_count": action_counts["forward 3m"],
        "ascend_count": action_counts["ascend 3m"],
        "descend_count": action_counts["descend 3m"],
        "stop_count": action_counts["stop"],
        "final_action": actions[-1] if actions else "",
        "final_forward_clearance": forward_clearance(final_record) if final_record else None,
        "collision_object_name": str(final_collision_info.get("object_name") or ""),
        "longest_turn_only_run": longest_turn_only_run(actions),
        "longest_same_turn_run": longest_same_turn(actions),
        "longest_alternating_turn_run": longest_alternating_turn(actions),
        "memory_stagnant_steps": motion_states["stagnant"],
        "memory_revisiting_steps": motion_states["revisiting"],
        "memory_full_rotation_steps": turn_states["full_rotation_loop"],
        "memory_oscillating_steps": turn_states["oscillating"],
        "source_result": str(source.resolve()) if source else "",
        "closest_frame": resolve_image(source, closest_record.get("image_path") if closest_record else None),
        "final_frame": resolve_image(source, final_record.get("image_path") if final_record else None),
        "_memory_state_actions": {
            state: dict(counts) for state, counts in memory_state_actions.items()
        },
        "_full_rotation_clear_3m_actions": dict(full_rotation_clear_3m),
    }
    for action in ACTION_ORDER:
        result[f"action_{action}"] = action_counts[action]
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    action_counts = {
        action: sum(int(row[f"action_{action}"]) for row in rows) for action in ACTION_ORDER
    }
    action_total = sum(action_counts.values())
    successes = sum(int(row["success"]) for row in rows)
    official_oracle = sum(int(row["official_osr"]) for row in rows)
    observed_reaches = sum(int(row["observed_reach_20m"]) for row in rows)
    return {
        "episodes": count,
        "successes": successes,
        "sr": successes / count if count else 0.0,
        "official_oracle_successes": official_oracle,
        "official_osr": official_oracle / count if count else 0.0,
        "official_osr_to_sr": successes / official_oracle if official_oracle else 0.0,
        "observed_reaches_20m": observed_reaches,
        "observed_reach_rate": observed_reaches / count if count else 0.0,
        "observed_reach_to_sr": successes / observed_reaches if observed_reaches else 0.0,
        "collisions": sum(int(row["collision"]) for row in rows),
        "collision_rate": sum(int(row["collision"]) for row in rows) / count if count else 0.0,
        "terminations": dict(Counter(str(row["termination_reason"]) for row in rows)),
        "action_counts": action_counts,
        "action_shares": {
            action: value / action_total if action_total else 0.0
            for action, value in action_counts.items()
        },
        "steps": describe(row["steps"] for row in rows),
        "minimum_distance": describe(row["minimum_distance"] for row in rows),
        "progress_to_closest": describe(row["progress_to_closest"] for row in rows),
        "path_length_3d": describe(row["path_length_3d"] for row in rows),
        "net_displacement_3d": describe(row["net_displacement_3d"] for row in rows),
        "turn_fraction": describe(row["turn_fraction"] for row in rows),
        "longest_turn_only_run": describe(row["longest_turn_only_run"] for row in rows),
        "longest_same_turn_run": describe(row["longest_same_turn_run"] for row in rows),
        "longest_alternating_turn_run": describe(
            row["longest_alternating_turn_run"] for row in rows
        ),
    }


def stratify(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = {}
    for name, items in sorted(groups.items()):
        item = aggregate(items)
        result[name] = {
            key: item[key]
            for key in (
                "episodes",
                "successes",
                "sr",
                "official_oracle_successes",
                "official_osr",
                "official_osr_to_sr",
                "observed_reaches_20m",
                "observed_reach_rate",
                "collisions",
                "collision_rate",
            )
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "scene",
        "episode_id",
        "target",
        "object_name",
        "description",
        "size",
        "size_group",
        "used_in_train",
        "outcome_group",
        "success",
        "official_osr",
        "observed_reach_20m",
        "reached_then_collided",
        "collision",
        "termination_reason",
        "final_within_20m",
        "steps",
        "initial_distance",
        "minimum_distance",
        "closest_step",
        "first_reach_step",
        "steps_after_first_reach",
        "final_distance",
        "progress_to_closest",
        "path_length_3d",
        "net_displacement_3d",
        "turn_fraction",
        "forward_count",
        "ascend_count",
        "descend_count",
        "final_action",
        "final_forward_clearance",
        "collision_object_name",
        "longest_turn_only_run",
        "longest_same_turn_run",
        "longest_alternating_turn_run",
        "memory_stagnant_steps",
        "memory_revisiting_steps",
        "memory_full_rotation_steps",
        "memory_oscillating_steps",
        "source_result",
        "closest_frame",
        "final_frame",
    )
    return {field: row[field] for field in fields}


def memory_rule_response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_actions: dict[str, Counter[str]] = defaultdict(Counter)
    clear_actions: Counter[str] = Counter()
    for row in rows:
        for state, counts in row["_memory_state_actions"].items():
            state_actions[state].update(counts)
        clear_actions.update(row["_full_rotation_clear_3m_actions"])

    selected_states = ("full_rotation_loop", "oscillating", "stagnant", "revisiting")
    result = {}
    for state in selected_states:
        counts = state_actions[state]
        total = sum(counts.values())
        result[state] = {
            "steps": total,
            "actions": dict(counts),
            "turn_share": sum(counts[action] for action in TURN_ACTIONS) / total if total else 0.0,
            "forward_share": counts["forward 3m"] / total if total else 0.0,
        }
    clear_total = sum(clear_actions.values())
    result["full_rotation_loop_with_center_and_lower_center_depth_gt_3m"] = {
        "steps": clear_total,
        "actions": dict(clear_actions),
        "turn_share": (
            sum(clear_actions[action] for action in TURN_ACTIONS) / clear_total
            if clear_total
            else 0.0
        ),
        "forward_share": clear_actions["forward 3m"] / clear_total if clear_total else 0.0,
    }
    return result


def markdown_percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "failure_mode_analysis_v1").resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    merged_path = run_dir / "all_episodes.jsonl"
    if not merged_path.is_file():
        raise FileNotFoundError(merged_path)

    sources = build_source_index(run_dir)
    rows: list[dict[str, Any]] = []
    with merged_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            rows.append(derive_episode(episode, sources.get(task_key(episode))))
    if len(rows) != len({(row["scene"], row["episode_id"]) for row in rows}):
        raise RuntimeError("Merged input contains duplicate episode keys")

    success = [row for row in rows if row["success"]]
    osr_only = [row for row in rows if row["official_osr"] and not row["success"]]
    no_official_osr = [row for row in rows if not row["official_osr"] and not row["success"]]
    observed_reach_failures = [
        row for row in rows if row["observed_reach_20m"] and not row["success"]
    ]
    reached_then_collision = [row for row in rows if row["reached_then_collided"]]
    false_stop = [
        row for row in rows if row["termination_reason"] == "stop" and not row["success"]
    ]
    false_stop_no_reach = [row for row in false_stop if not row["observed_reach_20m"]]
    reached_no_stop = [
        row
        for row in observed_reach_failures
        if row["termination_reason"] != "stop"
    ]
    final_inside_no_stop = [
        row
        for row in rows
        if row["final_within_20m"] and not row["success"] and row["termination_reason"] != "stop"
    ]
    no_reach = [row for row in rows if not row["observed_reach_20m"]]
    collision_before_reach = [row for row in no_reach if row["collision"]]
    step_limit_no_reach = [
        row for row in no_reach if row["termination_reason"] == "step_limit"
    ]
    near_miss = [row for row in no_reach if row["minimum_distance"] < 25.0]
    turn_loop = [
        row
        for row in no_reach
        if row["steps"] >= 20
        and (row["longest_turn_only_run"] >= 12 or row["turn_fraction"] >= 0.85)
    ]
    low_translation = [
        row
        for row in no_reach
        if row["steps"] >= 50 and row["net_displacement_3d"] < 6.0
    ]

    total_stops = sum(row["termination_reason"] == "stop" for row in rows)
    summary = {
        "source_run": str(run_dir),
        "metric_note": (
            "The evaluator forces official OSR to zero on collision. observed_reach_20m is "
            "reconstructed from recorded per-step distances and can remain true for a later collision."
        ),
        "overall": aggregate(rows),
        "outcome_groups": {
            "success": aggregate(success),
            "official_osr_only": aggregate(osr_only),
            "no_official_osr": aggregate(no_official_osr),
        },
        "recognition_and_stop": {
            "stop_terminations": total_stops,
            "successful_stops": len(success),
            "stop_precision": len(success) / total_stops if total_stops else 0.0,
            "false_stops": len(false_stop),
            "false_stops_without_observed_reach": len(false_stop_no_reach),
            "false_stops_at_step_0": sum(row["steps"] == 1 for row in false_stop),
            "false_stops_within_first_5_actions": sum(row["steps"] <= 5 for row in false_stop),
            "false_stop_distance": describe(row["final_distance"] for row in false_stop),
        },
        "osr_to_sr_conversion": {
            "official_osr_only": len(osr_only),
            "observed_reach_failures_including_later_collision": len(observed_reach_failures),
            "reached_then_collided_official_osr_zero": len(reached_then_collision),
            "reached_but_never_stopped": len(reached_no_stop),
            "ended_inside_20m_without_stop": len(final_inside_no_stop),
            "termination_counts": dict(
                Counter(row["termination_reason"] for row in observed_reach_failures)
            ),
            "final_inside_20m_without_stop_by_termination": dict(
                Counter(row["termination_reason"] for row in final_inside_no_stop)
            ),
            "steps_after_first_reach": describe(
                row["steps_after_first_reach"]
                for row in observed_reach_failures
                if row["steps_after_first_reach"] is not None
            ),
        },
        "exploration": {
            "never_observed_within_20m": len(no_reach),
            "near_miss_20_to_25m": len(near_miss),
            "collision_before_reach": sum(row["collision"] for row in no_reach),
            "false_stop_before_reach": len(false_stop_no_reach),
            "step_limit_before_reach": sum(
                row["termination_reason"] == "step_limit" for row in no_reach
            ),
            "turn_loop_candidates": len(turn_loop),
            "low_net_displacement_candidates": len(low_translation),
            "minimum_distance_bins": {
                "[20,25)": sum(20.0 <= row["minimum_distance"] < 25.0 for row in no_reach),
                "[25,30)": sum(25.0 <= row["minimum_distance"] < 30.0 for row in no_reach),
                "[30,40)": sum(30.0 <= row["minimum_distance"] < 40.0 for row in no_reach),
                "[40,inf)": sum(row["minimum_distance"] >= 40.0 for row in no_reach),
            },
        },
        "collision_diagnosis": {
            "total": sum(row["collision"] for row in rows),
            "before_observed_reach": len(collision_before_reach),
            "after_observed_reach": len(reached_then_collision),
            "final_action_counts": dict(
                Counter(row["final_action"] for row in rows if row["collision"])
            ),
            "collision_object_counts": dict(
                Counter(row["collision_object_name"] for row in rows if row["collision"])
            ),
            "final_forward_clearance": describe(
                row["final_forward_clearance"]
                for row in rows
                if row["collision"] and row["final_forward_clearance"] is not None
            ),
        },
        "memory_rule_response": memory_rule_response(rows),
        "stratified": {
            "scene": stratify(rows, "scene"),
            "size_group": stratify(rows, "size_group"),
            "seen": stratify(rows, "used_in_train"),
        },
    }

    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "episodes.csv", [compact_row(row) for row in rows])

    lists = {
        "official_osr_only": osr_only,
        "observed_reach_failures": observed_reach_failures,
        "reached_then_collision": reached_then_collision,
        "reached_but_never_stopped": reached_no_stop,
        "ended_inside_20m_without_stop": final_inside_no_stop,
        "false_stop_without_reach": false_stop_no_reach,
        "collision_before_reach": collision_before_reach,
        "step_limit_without_reach": step_limit_no_reach,
        "near_miss_20_to_25m": sorted(near_miss, key=lambda row: row["minimum_distance"]),
        "turn_loop_without_reach": sorted(
            turn_loop, key=lambda row: (-row["longest_turn_only_run"], -row["turn_fraction"])
        ),
        "low_displacement_without_reach": sorted(
            low_translation, key=lambda row: (row["net_displacement_3d"], -row["steps"])
        ),
    }
    for name, items in lists.items():
        write_csv(output_dir / "lists" / f"{name}.csv", [compact_row(row) for row in items])

    overall = summary["overall"]
    stop = summary["recognition_and_stop"]
    conversion = summary["osr_to_sr_conversion"]
    exploration = summary["exploration"]
    lines = [
        "# Evaluation Failure-mode Diagnosis",
        "",
        f"- Episodes: **{overall['episodes']}**",
        f"- SR: **{markdown_percent(overall['sr'])}** ({overall['successes']})",
        f"- Official OSR: **{markdown_percent(overall['official_osr'])}** ({overall['official_oracle_successes']})",
        f"- Official OSR -> SR: **{markdown_percent(overall['official_osr_to_sr'])}**",
        f"- Collision rate: **{markdown_percent(overall['collision_rate'])}** ({overall['collisions']})",
        "",
        "## Recognition And Stop",
        "",
        f"- Stop terminations: **{stop['stop_terminations']}**",
        f"- Successful / false stops: **{stop['successful_stops']} / {stop['false_stops']}**",
        f"- Stop precision: **{markdown_percent(stop['stop_precision'])}**",
        f"- False stops without ever entering 20 m: **{stop['false_stops_without_observed_reach']}**",
        f"- False stops at the first frame / first 5 actions: **{stop['false_stops_at_step_0']} / {stop['false_stops_within_first_5_actions']}**",
        "",
        "## OSR To SR Conversion",
        "",
        f"- Official OSR-only failures: **{conversion['official_osr_only']}**",
        f"- All observed-reach failures, including later collisions: **{conversion['observed_reach_failures_including_later_collision']}**",
        f"- Reached then collided (official OSR is reset to 0): **{conversion['reached_then_collided_official_osr_zero']}**",
        f"- Reached but never stopped: **{conversion['reached_but_never_stopped']}**",
        f"- Ended inside 20 m without stop: **{conversion['ended_inside_20m_without_stop']}**",
        "",
        "## Exploration",
        "",
        f"- Never observed within 20 m: **{exploration['never_observed_within_20m']}**",
        f"- Near misses with minimum distance in [20, 25) m: **{exploration['near_miss_20_to_25m']}**",
        f"- Collision before reach: **{exploration['collision_before_reach']}**",
        f"- False stop before reach: **{exploration['false_stop_before_reach']}**",
        f"- Step limit before reach: **{exploration['step_limit_before_reach']}**",
        f"- Turn-loop candidates: **{exploration['turn_loop_candidates']}**",
        f"- Low-displacement candidates: **{exploration['low_net_displacement_candidates']}**",
        "",
        "## Manual Review",
        "",
        "Start with `lists/ended_inside_20m_without_stop.csv`: these are the cleanest missed-stop cases.",
        "Then inspect `lists/official_osr_only.csv` around `closest_frame` and `final_frame` to separate target-visible misses from cases where the UAV was close but looking away or occluded.",
        "For exploration, inspect `lists/near_miss_20_to_25m.csv`, then the highest-ranked rows in `lists/turn_loop_without_reach.csv` and `lists/low_displacement_without_reach.csv`.",
        "The evaluator forces official OSR to zero after any collision; use `lists/reached_then_collision.csv` when diagnosing reach-to-stop conversion.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"analysis={output_dir}", flush=True)


if __name__ == "__main__":
    main()

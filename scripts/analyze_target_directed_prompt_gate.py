#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TURN_ACTIONS = {"turn left 30 degree", "turn right 30 degree"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--revised-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def task_key(row: dict) -> tuple[str, str]:
    return str(row["map_name"]), str(row["episode_id"])


def load_dataset_keys(path: Path) -> set[tuple[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {task_key(row) for row in rows}


def load_episodes(path: Path, allowed: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    episodes: dict[tuple[str, str], dict] = {}
    merged = path / "all_episodes.jsonl"
    if merged.is_file():
        sources = [merged]
    else:
        sources = sorted(path.glob("lane*/temp/*.json"))

    for source in sources:
        if source.suffix == ".jsonl":
            rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
        else:
            rows = [json.loads(source.read_text(encoding="utf-8"))]
        for row in rows:
            key = task_key(row)
            if key in allowed:
                episodes[key] = row
    return episodes


def longest_same_turn(actions: list[str]) -> int:
    longest = current = 0
    previous = None
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
    previous = None
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


def episode_actions(row: dict) -> list[str]:
    return [
        str(step.get("parsed_command"))
        for step in (row.get("step_records") or [])
        if step.get("parsed_command")
    ]


def aggregate(episodes: dict[tuple[str, str], dict]) -> dict[str, Any]:
    rows = list(episodes.values())
    count = len(rows)
    successes = sum(bool(row.get("acc")) for row in rows)
    oracle = sum(bool(row.get("osr")) for row in rows)
    collisions = sum(bool(row.get("collision")) for row in rows)
    actions = [action for row in rows for action in episode_actions(row)]
    same_turns = [longest_same_turn(episode_actions(row)) for row in rows]
    alternating = [longest_alternating_turn(episode_actions(row)) for row in rows]
    steps = [len(episode_actions(row)) for row in rows]
    ne_values = [float(row["ne"]) for row in rows if row.get("ne") is not None]
    return {
        "count": count,
        "successes": successes,
        "sr": successes / count if count else 0.0,
        "oracle_successes": oracle,
        "osr": oracle / count if count else 0.0,
        "osr_to_sr": successes / oracle if oracle else 0.0,
        "collisions": collisions,
        "collision_rate": collisions / count if count else 0.0,
        "mean_ne": sum(ne_values) / len(ne_values) if ne_values else None,
        "mean_steps": sum(steps) / len(steps) if steps else 0.0,
        "mean_longest_same_turn": sum(same_turns) / len(same_turns) if same_turns else 0.0,
        "mean_longest_alternating_turn": (
            sum(alternating) / len(alternating) if alternating else 0.0
        ),
        "action_distribution": dict(Counter(actions)),
    }


def exact_paired_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired(
    left: dict[tuple[str, str], dict],
    right: dict[tuple[str, str], dict],
    field: str,
) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    gains = sum(not bool(left[key].get(field)) and bool(right[key].get(field)) for key in keys)
    losses = sum(bool(left[key].get(field)) and not bool(right[key].get(field)) for key in keys)
    return {
        "paired_count": len(keys),
        "gains": gains,
        "losses": losses,
        "unchanged": len(keys) - gains - losses,
        "exact_p": exact_paired_p(gains, losses),
    }


def main() -> None:
    args = parse_args()
    keys = load_dataset_keys(args.dataset)
    runs = {
        "baseline_v2": load_episodes(args.baseline_dir, keys),
        "target_directed_v1": load_episodes(args.original_dir, keys),
        "target_directed_v1_1": load_episodes(args.revised_dir, keys),
    }
    missing = {name: len(keys - set(rows)) for name, rows in runs.items()}
    if any(missing.values()):
        raise RuntimeError(f"incomplete paired inputs: {missing}")

    metrics = {name: aggregate(rows) for name, rows in runs.items()}
    comparisons = {}
    for left_name, right_name in (
        ("baseline_v2", "target_directed_v1"),
        ("baseline_v2", "target_directed_v1_1"),
        ("target_directed_v1", "target_directed_v1_1"),
    ):
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "success": paired(runs[left_name], runs[right_name], "acc"),
            "oracle_success": paired(runs[left_name], runs[right_name], "osr"),
            "collision": paired(runs[left_name], runs[right_name], "collision"),
        }

    original = metrics["target_directed_v1"]
    revised = metrics["target_directed_v1_1"]
    gate = {
        "collision_not_increased": revised["collision_rate"] <= original["collision_rate"],
        "sr_not_decreased": revised["sr"] >= original["sr"],
        "osr_not_decreased": revised["osr"] >= original["osr"],
        "sr_or_osr_strictly_improved": (
            revised["sr"] > original["sr"] or revised["osr"] > original["osr"]
        ),
    }
    gate["pass_for_full_eval"] = all(gate.values())

    summary = {
        "dataset": str(args.dataset.resolve()),
        "task_count": len(keys),
        "run_dirs": {
            "baseline_v2": str(args.baseline_dir.resolve()),
            "target_directed_v1": str(args.original_dir.resolve()),
            "target_directed_v1_1": str(args.revised_dir.resolve()),
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Target-directed Prompt Closed-loop Gate",
        "",
        "| Run | SR | OSR | OSR->SR | Collision | Mean steps | Same-turn max | Alternating-turn max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        lines.append(
            f"| {name} | {item['sr']:.1%} | {item['osr']:.1%} | "
            f"{item['osr_to_sr']:.1%} | {item['collision_rate']:.1%} | "
            f"{item['mean_steps']:.2f} | {item['mean_longest_same_turn']:.2f} | "
            f"{item['mean_longest_alternating_turn']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Gate pass for full evaluation: **{gate['pass_for_full_eval']}**",
            "",
            "The gate requires no collision increase, no SR/OSR regression, and a strict improvement in SR or OSR versus target_directed_v1.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

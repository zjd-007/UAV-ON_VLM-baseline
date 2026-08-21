#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from eval_utils import process_results  # noqa: E402


def load_rows(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("lane*/temp/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_result_file"] = str(path.relative_to(run_dir))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top_k_failures", type=int, default=200)
    args = parser.parse_args()

    run_dir = args.run_dir
    rows = load_rows(run_dir)
    by_env = defaultdict(list)
    for row in rows:
        by_env[row["map_name"]].append(row)

    metrics = process_results(by_env)
    (run_dir / "merged_results.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with (run_dir / "all_episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    action_counter = Counter()
    command_counter = Counter()
    parse_counter = Counter()
    for row in rows:
        action_counter.update(map(str, row.get("action_ids", [])))
        command_counter.update(row.get("parsed_commands", []))
        parse_counter.update(map(str, row.get("parse_matched", [])))
    action_stats = {
        "action_ids": dict(action_counter),
        "parsed_commands": dict(command_counter),
        "parse_matched": dict(parse_counter),
    }
    (run_dir / "action_distribution.json").write_text(json.dumps(action_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [row for row in rows if int(row.get("acc", 0)) == 0]
    failures.sort(key=lambda x: float(x.get("ne", 0)), reverse=True)
    with (run_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in failures[: args.top_k_failures]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_scene = {}
    for env, env_rows in sorted(by_env.items()):
        per_scene[env] = process_results({env: env_rows})["total_results"]
    (run_dir / "per_scene_results.json").write_text(json.dumps(per_scene, ensure_ascii=False, indent=2), encoding="utf-8")

    total = metrics["total_results"]
    summary = [
        f"# {run_dir.name}",
        "",
        f"- episodes: {len(rows)}",
        f"- acc: {total['acc']:.4f}",
        f"- osr: {total['osr']:.4f}",
        f"- ne: {total['ne']:.4f}",
        f"- failures: {len(failures)}",
        "",
        "## Action Distribution",
        "```json",
        json.dumps(action_stats, ensure_ascii=False, indent=2),
        "```",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary[:7]))


if __name__ == "__main__":
    main()

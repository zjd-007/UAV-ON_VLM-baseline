#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/data/zhujd/miniconda3/envs/octmem_openvla_nomemory/bin/python")
DEFAULT_OLD_MODEL = ROOT / "outputs/phi35_uavon_lora_r256_depth_grid_collision_filtered_20260719_001301/checkpoint-20997"
DEFAULT_NEW_MODEL = ROOT / "outputs/phi35_uavon_lora_r256_depth_grid_stop_visible_v4_per_frame_safe_stopbank_v1_20260813/checkpoint-19764"
DEFAULT_CAPTURE_DIR = ROOT / "results/stop_frame_visibility_ckpt20997_vs_stopbank19764_full_20260815_134453"
DEFAULT_DATA_DIR = Path(
    "/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/processed/"
    "neighborhood_coordinate_repair_v1_20260812_194807/"
    "final_dataset_per_frame_safe_stopbank_v1"
)
DEFAULT_SELECTIONS = [
    Path(
        "/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/processed/"
        "stop_visible_v4_production_20260812/prepared_base_v2/selections.jsonl"
    ),
    Path(
        "/data/zhujd/Aerial-ObjectNav/UAV-ON_dataset/processed/"
        "neighborhood_coordinate_repair_v1_20260812_194807/"
        "stop_visible_prepared_base/selections.jsonl"
    ),
]


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def predictions_complete(sample_file: Path, output_dir: Path) -> bool:
    return jsonl_count(sample_file) > 0 and jsonl_count(output_dir / "predictions.jsonl") == jsonl_count(sample_file)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe scheduler for the small-target Stop threshold evaluation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--old-model", type=Path, default=DEFAULT_OLD_MODEL)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "scheduler.log"
    status_path = args.output_dir / "scheduler_status.json"

    train_samples = args.output_dir / "fixed_frames.jsonl"
    inference_dir = args.output_dir / "inference_stop_union"
    inference_samples = inference_dir / "fixed_frames.jsonl"
    steps: list[dict[str, Any]] = []

    build_train_cmd = [
        str(args.python), "-u", str(ROOT / "scripts/build_small_target_stop_threshold_eval.py"),
        "--selections", *(str(path) for path in DEFAULT_SELECTIONS),
        "--frames", str(DEFAULT_DATA_DIR / "train_frames.jsonl"),
        "--sft", str(DEFAULT_DATA_DIR / "uavon_phi35_sft_depth_grid_stop_visible_v4_per_frame_safe_stopbank_v1.jsonl"),
        "--output-dir", str(args.output_dir),
    ]
    build_inference_cmd = [
        str(args.python), "-u", str(ROOT / "scripts/build_inference_small_stop_fixed_eval.py"),
        "--capture-dir", str(args.capture_dir),
        "--output-dir", str(inference_dir),
    ]

    def model_cmd(sample_file: Path, model: Path, output_dir: Path) -> list[str]:
        return [
            str(args.python), "-u", str(ROOT / "scripts/offline_action_recall.py"),
            "--sample_file", str(sample_file),
            "--model_path", str(model),
            "--output_dir", str(output_dir),
            "--inference_mode", "generate",
            "--num_workers", "1",
            "--log_every", "25",
        ]

    def summary_cmd(parent: Path) -> list[str]:
        return [
            str(args.python), "-u", str(ROOT / "scripts/summarize_small_target_stop_threshold_eval.py"),
            "--old", str(parent / "old_ckpt20997_generate/predictions.jsonl"),
            "--new", str(parent / "new_ckpt19764_generate/predictions.jsonl"),
            "--output-dir", str(parent / "comparison_generate"),
        ]

    specs = [
        ("build_training_fixed_frames", build_train_cmd, lambda: train_samples.is_file()),
        (
            "old_model_training_frames",
            model_cmd(train_samples, args.old_model, args.output_dir / "old_ckpt20997_generate"),
            lambda: predictions_complete(train_samples, args.output_dir / "old_ckpt20997_generate"),
        ),
        (
            "new_model_training_frames",
            model_cmd(train_samples, args.new_model, args.output_dir / "new_ckpt19764_generate"),
            lambda: predictions_complete(train_samples, args.output_dir / "new_ckpt19764_generate"),
        ),
        (
            "summarize_training_frames",
            summary_cmd(args.output_dir),
            lambda: (args.output_dir / "comparison_generate/summary.json").is_file(),
        ),
        ("build_inference_fixed_frames", build_inference_cmd, lambda: inference_samples.is_file()),
        (
            "old_model_inference_frames",
            model_cmd(inference_samples, args.old_model, inference_dir / "old_ckpt20997_generate"),
            lambda: predictions_complete(inference_samples, inference_dir / "old_ckpt20997_generate"),
        ),
        (
            "new_model_inference_frames",
            model_cmd(inference_samples, args.new_model, inference_dir / "new_ckpt19764_generate"),
            lambda: predictions_complete(inference_samples, inference_dir / "new_ckpt19764_generate"),
        ),
        (
            "summarize_inference_frames",
            summary_cmd(inference_dir),
            lambda: (inference_dir / "comparison_generate/summary.json").is_file(),
        ),
    ]

    state: dict[str, Any] = {
        "status": "running",
        "started_at": timestamp(),
        "updated_at": timestamp(),
        "gpu": args.gpu,
        "current_step": None,
        "steps": steps,
    }
    atomic_json(status_path, state)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{timestamp()}] scheduler start gpu={args.gpu}\n")
        log.flush()
        try:
            for name, command, complete in specs:
                record = {"name": name, "status": "pending"}
                steps.append(record)
                state["current_step"] = name
                state["updated_at"] = timestamp()
                if complete():
                    record.update(status="skipped_complete", finished_at=timestamp())
                    log.write(f"[{timestamp()}] {name}: skipped_complete\n")
                    log.flush()
                    atomic_json(status_path, state)
                    continue
                record.update(status="running", started_at=timestamp())
                atomic_json(status_path, state)
                log.write(f"[{timestamp()}] {name}: start\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    record.update(status="failed", returncode=result.returncode, finished_at=timestamp())
                    raise RuntimeError(f"{name} failed with return code {result.returncode}")
                record.update(status="completed", returncode=0, finished_at=timestamp())
                log.write(f"[{timestamp()}] {name}: completed\n")
                log.flush()
                atomic_json(status_path, state)
            state.update(status="completed", current_step=None, finished_at=timestamp(), updated_at=timestamp())
            log.write(f"[{timestamp()}] scheduler completed\n")
        except Exception as exc:
            state.update(status="failed", error=repr(exc), finished_at=timestamp(), updated_at=timestamp())
            log.write(f"[{timestamp()}] scheduler failed: {exc!r}\n")
            raise
        finally:
            log.flush()
            atomic_json(status_path, state)


if __name__ == "__main__":
    main()

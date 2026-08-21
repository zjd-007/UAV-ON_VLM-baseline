#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from offline_action_recall import (  # noqa: E402
    ACTION_LABELS,
    build_phi35_prompt,
    generate_action_text,
    load_jsonl,
    load_model_and_processor,
    move_inputs_to_device,
    patch_transformers_cache_compat,
    processor_call,
)
from vlm_baseline.actions import ACTION_IDS, parse_action_text  # noqa: E402


COMMANDS = [
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
]


def select_balanced(samples: list[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if row["label"] in ACTION_LABELS:
            buckets[row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    for label in ACTION_LABELS:
        bucket = buckets[label]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_class])
    rng.shuffle(selected)
    return selected


def make_variant_image(kind: str, item: dict[str, Any], shuffled_item: dict[str, Any], rng: random.Random) -> Image.Image:
    image = Image.open(item["image"]).convert("RGB")
    if kind == "original":
        return image
    if kind == "black":
        return Image.new("RGB", image.size, (0, 0, 0))
    if kind == "noise":
        data = torch.randint(0, 256, (image.size[1], image.size[0], 3), dtype=torch.uint8).numpy()
        return Image.fromarray(data, "RGB")
    if kind == "shuffled":
        return Image.open(shuffled_item["image"]).convert("RGB").resize(image.size)
    raise ValueError(f"Unknown image variant: {kind}")


def inspect_processor_inputs(processor, item: dict[str, Any], device: str) -> dict[str, Any]:
    image = Image.open(item["image"]).convert("RGB")
    prompt = build_phi35_prompt(processor, item["target_description"])
    inputs = processor_call(processor, prompt, image)
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_1|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None
    input_ids = inputs.get("input_ids")
    result: dict[str, Any] = {
        "prompt_preview": prompt[:500],
        "processor_keys": sorted(inputs.keys()),
        "image_token_id": image_token_id,
    }
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[f"{key}_shape"] = list(value.shape)
            result[f"{key}_dtype"] = str(value.dtype)
    if torch.is_tensor(input_ids) and isinstance(image_token_id, int) and image_token_id >= 0:
        result["image_token_count_in_input_ids"] = int((input_ids == image_token_id).sum().item())
        result["negative_image_placeholder_count"] = int((input_ids < 0).sum().item())
    try:
        moved = move_inputs_to_device(inputs, device, torch.bfloat16)
        with torch.inference_mode():
            outputs = processor_call(processor, prompt, image)
        result["processor_call_ok"] = bool(outputs)
        result["moved_tensor_devices"] = {
            key: str(value.device) for key, value in moved.items() if torch.is_tensor(value)
        }
    except Exception as exc:
        result["processor_inspect_error"] = repr(exc)
    return result


def first_token_logprobs(model, processor, image: Image.Image, target_description: str, device: str) -> dict[str, float]:
    prompt = build_phi35_prompt(processor, target_description)
    inputs = processor_call(processor, prompt, image)
    inputs = move_inputs_to_device(inputs, device, torch.bfloat16)
    tokenizer = getattr(processor, "tokenizer", processor)

    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)[0]

    scores: dict[str, float] = {}
    for command in COMMANDS:
        token_ids = tokenizer(command, add_special_tokens=False).input_ids
        scores[command] = float(log_probs[token_ids[0]].item()) if token_ids else float("-inf")
    return scores


def run_probe(args: argparse.Namespace) -> None:
    patch_transformers_cache_compat()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_jsonl(args.sample_file)
    samples = select_balanced(all_samples, args.samples_per_class, args.seed)
    model, processor = load_model_and_processor(args.model_path, args.base_model_path, args.device)

    inspect = inspect_processor_inputs(processor, samples[0], args.device) if samples else {}
    (args.output_dir / "processor_inspection.json").write_text(
        json.dumps(inspect, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(samples):
        shuffled_item = rng.choice([row for row in all_samples if row["image"] != item["image"]])
        for variant in args.variants.split(","):
            image = make_variant_image(variant, item, shuffled_item, rng)
            raw_text, raw_special = generate_action_text(
                model,
                processor,
                item["image"] if variant == "original" else save_temp_image(args.output_dir, image, idx, variant),
                item["target_description"],
                args.device,
                args.max_new_tokens,
            )
            parsed = parse_action_text(raw_text)
            scores = first_token_logprobs(model, processor, image, item["target_description"], args.device)
            row = {
                "source_index": item["source_index"],
                "label": item["label"],
                "label_id": item["label_id"],
                "variant": variant,
                "raw_action_text": raw_text,
                "raw_action_text_with_special_tokens": raw_special,
                "pred_command": parsed.command,
                "pred_id": parsed.action_id,
                "parse_matched": parsed.matched,
                "first_token_logprobs": scores,
                "first_token_top_command": max(scores, key=scores.get),
            }
            rows.append(row)
            print(
                f"[{idx + 1}/{len(samples)}] {item['label']} | {variant}: "
                f"{raw_text!r} -> {parsed.command}",
                flush=True,
            )

    write_jsonl(args.output_dir / "image_sensitivity_predictions.jsonl", rows)
    summarize(rows, args.output_dir)


def save_temp_image(output_dir: Path, image: Image.Image, idx: int, variant: str) -> str:
    temp_dir = output_dir / "variant_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{idx:04d}_{variant}.jpg"
    image.save(path, quality=90)
    return str(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_source[row["source_index"]].append(row)

    variant_summary = {}
    for variant, variant_rows in sorted(by_variant.items()):
        variant_summary[variant] = {
            "num_samples": len(variant_rows),
            "prediction_distribution": dict(Counter(row["pred_command"] for row in variant_rows)),
            "first_token_top_distribution": dict(Counter(row["first_token_top_command"] for row in variant_rows)),
            "accuracy": sum(row["pred_id"] == row["label_id"] for row in variant_rows) / len(variant_rows),
        }

    invariance = {
        "num_source_samples": len(by_source),
        "same_generated_command_across_all_variants": 0,
        "same_first_token_top_across_all_variants": 0,
    }
    for source_rows in by_source.values():
        if len({row["pred_command"] for row in source_rows}) == 1:
            invariance["same_generated_command_across_all_variants"] += 1
        if len({row["first_token_top_command"] for row in source_rows}) == 1:
            invariance["same_first_token_top_across_all_variants"] += 1

    summary = {
        "variant_summary": variant_summary,
        "invariance": invariance,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether Phi-3.5 UAV-ON policy is sensitive to image content.")
    parser.add_argument("--sample_file", type=Path, default=ROOT / "results" / "offline_action_recall_20260625_104438" / "sampled.jsonl")
    parser.add_argument("--model_path", type=str, default=str(ROOT / "outputs" / "phi35_uavon_lora_r256"))
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "image_sensitivity_probe")
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--variants", type=str, default="original,black,noise,shuffled")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())

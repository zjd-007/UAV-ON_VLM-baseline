#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_qwen25_vl_uavon import build_qwen_inputs, generate_qwen_action_text  # noqa: E402
from vlm_baseline.actions import ACTION_IDS, parse_action_text  # noqa: E402
from vlm_baseline.depth_avoidance import UAVONSingleViewDepthPrompt  # noqa: E402
from vlm_baseline.prompting import build_prompt  # noqa: E402


ACTIONS = list(ACTION_IDS)
DEFAULT_RUN = ROOT / "results" / "qwen25vl7b_zero_shot_cfmem_full_20260727_191644"
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
CONDITIONS = ("original", "black", "shuffled", "mask_left", "mask_center", "mask_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=18)
    parser.add_argument("--caption_samples", type=int, default=6)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--caption_max_new_tokens", type=int, default=96)
    return parser.parse_args()


def load_samples(run_dir: Path, num_samples: int, seed: int) -> list[dict[str, Any]]:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane_dir in sorted(path for path in run_dir.glob("lane*") if path.is_dir()):
        for result_path in sorted((lane_dir / "temp").glob("*.json")):
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if not row.get("step_records"):
                continue
            step = row["step_records"][0]
            image_rel = step.get("image_path")
            if not image_rel:
                continue
            image_path = lane_dir / image_rel
            if not image_path.is_file():
                continue
            grid = step.get("depth_avoidance", {}).get("depth_grid")
            memory_prompt = step.get("memory_context", {}).get("prompt_text", "")
            if grid is None:
                continue
            by_lane[lane_dir.name].append(
                {
                    "sample_id": f"{lane_dir.name}:{row['map_name']}:{row['episode_id']}:0",
                    "lane": lane_dir.name,
                    "map_name": row["map_name"],
                    "episode_id": str(row["episode_id"]),
                    "step": 0,
                    "target_description": row["description"],
                    "image_path": str(image_path),
                    "depth_grid": grid,
                    "memory_prompt": memory_prompt,
                    "recorded_action": step["parsed_command"],
                }
            )

    rng = random.Random(seed)
    for rows in by_lane.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    lane_names = sorted(by_lane)
    while len(selected) < num_samples:
        added = False
        for lane in lane_names:
            if by_lane[lane] and len(selected) < num_samples:
                selected.append(by_lane[lane].pop())
                added = True
        if not added:
            break
    if len(selected) < num_samples:
        raise ValueError(f"Only found {len(selected)} usable samples, requested {num_samples}")
    return selected


def build_condition_image(
    condition: str,
    original: Image.Image,
    shuffled: Image.Image,
) -> Image.Image:
    original = original.convert("RGB")
    if condition == "original":
        return original.copy()
    if condition == "black":
        return Image.new("RGB", original.size, (0, 0, 0))
    if condition == "shuffled":
        return shuffled.convert("RGB").resize(original.size)
    if condition.startswith("mask_"):
        masked = original.copy()
        width, height = masked.size
        third = width // 3
        ranges = {
            "mask_left": (0, 0, third, height),
            "mask_center": (third, 0, 2 * third, height),
            "mask_right": (2 * third, 0, width, height),
        }
        masked.paste((0, 0, 0), ranges[condition])
        return masked
    raise ValueError(f"Unknown condition: {condition}")


def make_action_prompt(sample: dict[str, Any]) -> str:
    depth_module = UAVONSingleViewDepthPrompt()
    depth_prompt = depth_module.format_prompt(np.asarray(sample["depth_grid"], dtype=np.float32))
    return build_prompt(
        sample["target_description"],
        depth_context=depth_prompt,
        memory_context=sample["memory_prompt"],
    ).replace("<image>\n", "", 1)


def make_chat_text(processor, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def move_inputs(inputs, device: str) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def score_action_candidates(model, processor, image: Image.Image, prompt_text: str, device: str) -> dict[str, Any]:
    chat_text = make_chat_text(processor, prompt_text)
    prompt_inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_inputs["input_ids"][0]
    scores: dict[str, dict[str, float | int]] = {}

    for action in ACTIONS:
        full_inputs = processor(
            text=[chat_text + action],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        full_ids_cpu = full_inputs["input_ids"][0]
        prompt_len = int(prompt_ids.numel())
        if full_ids_cpu.numel() <= prompt_len or not torch.equal(full_ids_cpu[:prompt_len], prompt_ids):
            raise RuntimeError(f"Prompt token prefix changed while appending action {action!r}")

        inputs = move_inputs(full_inputs, device)
        input_ids = inputs["input_ids"]
        with torch.inference_mode():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                use_cache=False,
                return_dict=True,
            )
            answer_hidden = outputs[0][:, prompt_len - 1 : -1]
            logits = model.lm_head(answer_hidden).float()
            target_ids = input_ids[:, prompt_len:]
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)

        score_sum = float(token_log_probs.sum().cpu())
        token_count = int(target_ids.shape[1])
        scores[action] = {
            "sum_logprob": score_sum,
            "mean_logprob": score_sum / max(1, token_count),
            "token_count": token_count,
        }
        del inputs, outputs, answer_hidden, logits, target_ids, token_log_probs

    mean_values = torch.tensor([scores[action]["mean_logprob"] for action in ACTIONS])
    pseudo_probs = torch.softmax(mean_values, dim=0).tolist()
    for action, probability in zip(ACTIONS, pseudo_probs):
        scores[action]["pseudo_probability"] = float(probability)
    top_action = max(ACTIONS, key=lambda action: scores[action]["mean_logprob"])
    return {"top_action": top_action, "scores": scores}


def generate_caption(
    model,
    processor,
    image: Image.Image,
    target_description: str,
    device: str,
    max_new_tokens: int,
) -> str:
    prompt = (
        "Inspect only the supplied image. Do not infer objects that are outside the image. "
        "Return exactly four lines:\n"
        "VISIBLE_OBJECTS: brief list of visible objects\n"
        "TARGET_VISIBLE: yes or no\n"
        "TARGET_LOCATION: left, center, right, or none\n"
        "NEAREST_OBSTACLE: left, center, right, or none\n"
        f"Target description: {target_description}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")
    inputs = move_inputs(inputs, device)
    input_len = inputs["input_ids"].shape[-1]
    tokenizer = processor.tokenizer
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()


def js_divergence(p: list[float], q: list[float]) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p_arr + q_arr)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p_arr, midpoint) + 0.5 * kl(q_arr, midpoint)


def build_summary(records: list[dict[str, Any]], pipeline_check: dict[str, Any]) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_sample[record["sample_id"]][record["condition"]] = record

    distributions = defaultdict(Counter)
    for record in records:
        distributions[record["condition"]][record["greedy_action"]] += 1

    comparisons = {}
    for condition in CONDITIONS:
        if condition == "original":
            continue
        pairs = [
            (rows["original"], rows[condition])
            for rows in by_sample.values()
            if "original" in rows and condition in rows
        ]
        if not pairs:
            continue
        greedy_changes = sum(a["greedy_action"] != b["greedy_action"] for a, b in pairs)
        top_changes = sum(a["candidate_top_action"] != b["candidate_top_action"] for a, b in pairs)
        divergences = []
        l1_distances = []
        for original, changed in pairs:
            p = [original["candidate_scores"][action]["pseudo_probability"] for action in ACTIONS]
            q = [changed["candidate_scores"][action]["pseudo_probability"] for action in ACTIONS]
            divergences.append(js_divergence(p, q))
            l1_distances.append(float(np.abs(np.asarray(p) - np.asarray(q)).sum()))
        comparisons[condition] = {
            "pairs": len(pairs),
            "greedy_action_change_rate": greedy_changes / len(pairs),
            "candidate_top_action_change_rate": top_changes / len(pairs),
            "mean_js_divergence": float(np.mean(divergences)),
            "mean_probability_l1": float(np.mean(l1_distances)),
        }

    return {
        "num_samples": len(by_sample),
        "num_condition_records": len(records),
        "actions": ACTIONS,
        "pipeline_check": pipeline_check,
        "greedy_action_distributions": {
            condition: dict(counts) for condition, counts in distributions.items()
        },
        "condition_vs_original": comparisons,
    }


def main() -> None:
    args = parse_args()
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    invalid = sorted(set(conditions) - set(CONDITIONS))
    if invalid:
        raise ValueError(f"Unsupported conditions: {invalid}")
    if "original" not in conditions:
        raise ValueError("Conditions must include original")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    samples = load_samples(args.run_dir, args.num_samples, args.seed)
    (args.output_dir / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    first_original = Image.open(samples[0]["image_path"]).convert("RGB")
    first_shuffled = Image.open(samples[1 % len(samples)]["image_path"]).convert("RGB").resize(first_original.size)
    first_black = Image.new("RGB", first_original.size, (0, 0, 0))
    pipeline_tensors = {}
    for name, image in [
        ("original", first_original),
        ("black", first_black),
        ("shuffled", first_shuffled),
    ]:
        inputs, _ = build_qwen_inputs(
            processor,
            image,
            samples[0]["target_description"],
            depth_context="DepthGrid diagnostic",
            memory_context="EpisodeMemory diagnostic",
        )
        pixels = inputs["pixel_values"].float()
        pipeline_tensors[name] = pixels
    pipeline_check = {
        "pixel_values_shape": list(pipeline_tensors["original"].shape),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "original_vs_black_mean_abs_diff": float(
            torch.mean(torch.abs(pipeline_tensors["original"] - pipeline_tensors["black"]))
        ),
        "original_vs_shuffled_mean_abs_diff": float(
            torch.mean(torch.abs(pipeline_tensors["original"] - pipeline_tensors["shuffled"]))
        ),
    }
    del pipeline_tensors, inputs

    action_path = args.output_dir / "action_ablation.jsonl"
    records = []
    with action_path.open("w", encoding="utf-8") as output:
        for sample_index, sample in enumerate(samples):
            original = Image.open(sample["image_path"]).convert("RGB")
            shuffled_sample = samples[(sample_index + 1) % len(samples)]
            shuffled = Image.open(shuffled_sample["image_path"]).convert("RGB")
            prompt_text = make_action_prompt(sample)
            for condition in conditions:
                image = build_condition_image(condition, original, shuffled)
                raw_text = generate_qwen_action_text(
                    model,
                    processor,
                    image,
                    sample["target_description"],
                    args.device,
                    args.max_new_tokens,
                    depth_context=UAVONSingleViewDepthPrompt().format_prompt(
                        np.asarray(sample["depth_grid"], dtype=np.float32)
                    ),
                    memory_context=sample["memory_prompt"],
                )
                parsed = parse_action_text(raw_text)
                candidate_result = score_action_candidates(
                    model, processor, image, prompt_text, args.device
                )
                record = {
                    **sample,
                    "condition": condition,
                    "shuffled_image_path": shuffled_sample["image_path"] if condition == "shuffled" else None,
                    "raw_text": raw_text,
                    "greedy_action": parsed.command,
                    "parse_matched": parsed.matched,
                    "candidate_top_action": candidate_result["top_action"],
                    "candidate_scores": candidate_result["scores"],
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[{sample_index + 1}/{len(samples)}] {sample['sample_id']} {condition}: "
                    f"greedy={parsed.command}, score_top={candidate_result['top_action']}",
                    flush=True,
                )

    caption_rows = []
    caption_path = args.output_dir / "visual_qa.jsonl"
    with caption_path.open("w", encoding="utf-8") as output:
        for sample in samples[: args.caption_samples]:
            original = Image.open(sample["image_path"]).convert("RGB")
            for condition in ("original", "black"):
                image = build_condition_image(condition, original, original)
                response = generate_caption(
                    model,
                    processor,
                    image,
                    sample["target_description"],
                    args.device,
                    args.caption_max_new_tokens,
                )
                row = {
                    "sample_id": sample["sample_id"],
                    "image_path": sample["image_path"],
                    "target_description": sample["target_description"],
                    "condition": condition,
                    "response": response,
                }
                caption_rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()

    summary = build_summary(records, pipeline_check)
    summary["visual_qa_records"] = len(caption_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

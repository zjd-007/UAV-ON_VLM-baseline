#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_baseline.actions import ACTION_IDS, parse_action_text  # noqa: E402
from vlm_baseline.prompting import build_prompt  # noqa: E402


ACTION_LABELS = [
    "stop",
    "forward 3m",
    "turn left 30 degree",
    "turn right 30 degree",
    "ascend 3m",
    "descend 3m",
]


def patch_transformers_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "get_max_length") or not hasattr(DynamicCache, "get_max_cache_shape"):
        return
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


def read_label(row: dict[str, Any]) -> str:
    if "conversations" in row:
        return row["conversations"][-1]["value"].strip()
    if "messages" in row:
        return row["messages"][-1]["content"].strip()
    raise KeyError("Expected conversations or messages in dataset row.")


def read_prompt(row: dict[str, Any]) -> str:
    if "conversations" in row:
        return row["conversations"][0]["value"]
    if "messages" in row:
        return row["messages"][0]["content"]
    raise KeyError("Expected conversations or messages in dataset row.")


def extract_description(prompt: str) -> str:
    text = prompt.replace("<image>", "", 1).strip()
    prefix = "What action should the UAV take to find "
    if text.startswith(prefix) and "?" in text:
        return text[len(prefix) : text.index("?")]
    suffix = "? Reply with exactly one command."
    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : -len(suffix)]
    return text


def sample_balanced(dataset: Path, samples_per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    with dataset.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            label = read_label(row)
            counts[label] += 1
            if label not in ACTION_LABELS:
                continue
            item = {
                "source_index": idx,
                "image": row["images"][0],
                "prompt": read_prompt(row),
                "target_description": extract_description(read_prompt(row)),
                "label": label,
                "label_id": ACTION_IDS[label],
            }
            # Reservoir sample per label, avoiding loading every row of majority classes.
            bucket = buckets[label]
            seen = counts[label]
            if len(bucket) < samples_per_class:
                bucket.append(item)
            else:
                j = rng.randrange(seen)
                if j < samples_per_class:
                    bucket[j] = item

    print("dataset action counts:", dict(counts))
    samples: list[dict[str, Any]] = []
    for label in ACTION_LABELS:
        bucket = buckets[label]
        rng.shuffle(bucket)
        samples.extend(bucket)
        print(f"sampled {label}: {len(bucket)} / requested {samples_per_class}")
    rng.shuffle(samples)
    return samples


def build_phi35_prompt(processor, target_description: str) -> str:
    plain_prompt = build_prompt(target_description).replace("<image>\n", "")
    content = f"<|image_1|>\n{plain_prompt}"
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"


def build_phi35_prompt_from_text(processor, prompt_text: str) -> str:
    plain_prompt = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
    content = f"<|image_1|>\n{plain_prompt}"
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"


def processor_call(processor, prompt: str, image: Image.Image):
    images = image if isinstance(image, list) else [image]
    try:
        return processor(prompt, images, return_tensors="pt")
    except Exception:
        try:
            return processor(text=prompt, images=images, return_tensors="pt")
        except Exception:
            return processor(text=prompt, images=image, return_tensors="pt")


def move_inputs_to_device(inputs, device: str, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def get_generation_token_ids(processor) -> tuple[int | list[int] | None, int | None]:
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_ids: list[int] = []
    for token_id in (
        tokenizer.convert_tokens_to_ids("<|end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None,
        getattr(tokenizer, "eos_token_id", None),
    ):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
            eos_ids.append(token_id)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[-1]

    if not eos_ids:
        eos_token_id: int | list[int] | None = None
    elif len(eos_ids) == 1:
        eos_token_id = eos_ids[0]
    else:
        eos_token_id = eos_ids
    return eos_token_id, pad_token_id


def load_base_model(model_path: str, torch_dtype):
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        return AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)


def load_model_and_processor(model_path: str, base_model_path: str | None, device: str):
    model_dir = Path(model_path)
    adapter_config = model_dir / "adapter_config.json"
    processor_path = model_path
    torch_dtype = torch.bfloat16

    if adapter_config.is_file():
        if base_model_path is None:
            adapter_meta = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_path = adapter_meta.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError("LoRA adapter requires --base_model_path or base_model_name_or_path in adapter_config.json")
        processor_path = base_model_path
        from peft import PeftModel

        base_model = load_base_model(base_model_path, torch_dtype)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = load_base_model(model_path, torch_dtype)

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, processor


def generate_action_text(
    model,
    processor,
    image_path: str,
    target_description: str,
    device: str,
    max_new_tokens: int,
    prompt_text: str | None = None,
) -> tuple[str, str]:
    image = Image.open(image_path).convert("RGB")
    prompt = (
        build_phi35_prompt_from_text(processor, prompt_text)
        if prompt_text
        else build_phi35_prompt(processor, target_description)
    )
    inputs = processor_call(processor, prompt, image)
    inputs = move_inputs_to_device(inputs, device, torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    eos_token_id, pad_token_id = get_generation_token_ids(processor)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    new_tokens = generated[:, input_len:] if input_len else generated
    tokenizer = getattr(processor, "tokenizer", processor)
    raw_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    raw_text_with_special_tokens = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0].strip()
    return raw_text, raw_text_with_special_tokens


def score_action_candidates(
    model,
    processor,
    image_path: str,
    target_description: str,
    device: str,
    normalization: str,
    prompt_text: str | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    image = Image.open(image_path).convert("RGB")
    prompt = (
        build_phi35_prompt_from_text(processor, prompt_text)
        if prompt_text
        else build_phi35_prompt(processor, target_description)
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    candidate_texts = [f"{label}<|end|>\n" for label in ACTION_LABELS]
    full_texts = [prompt + candidate for candidate in candidate_texts]
    inputs = processor_call(processor, full_texts, [image] * len(full_texts))
    inputs = move_inputs_to_device(inputs, device, torch.bfloat16)

    input_ids = inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    for row_idx, candidate in enumerate(candidate_texts):
        answer_len = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        labels[row_idx, -answer_len:] = input_ids[row_idx, -answer_len:]
    labels[input_ids < 0] = -100

    with torch.inference_mode():
        outputs = model(**inputs)

    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    score_sum = token_log_probs.sum(dim=1)
    token_count = mask.sum(dim=1).clamp_min(1)
    score_mean = score_sum / token_count

    stats: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(ACTION_LABELS):
        stats[label] = {
            "sum_logprob": float(score_sum[idx].detach().cpu()),
            "mean_logprob": float(score_mean[idx].detach().cpu()),
            "token_count": float(token_count[idx].detach().cpu()),
        }

    key = "sum_logprob" if normalization == "sum" else "mean_logprob"
    best = max(ACTION_LABELS, key=lambda label: stats[label][key])
    return best, stats


def worker_main(rank: int, world_size: int, args_dict: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    patch_transformers_cache_compat()
    args = argparse.Namespace(**args_dict)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    shard = samples[rank::world_size]
    out_path = Path(args.output_dir) / f"predictions_shard_{rank}.jsonl"

    model, processor = load_model_and_processor(args.model_path, args.base_model_path, device)
    done = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in shard:
            error = None
            candidate_scores = None
            try:
                if args.inference_mode == "score":
                    pred_command, candidate_scores = score_action_candidates(
                        model,
                        processor,
                        item["image"],
                        item["target_description"],
                        device,
                        args.score_normalization,
                        item.get("prompt"),
                    )
                    raw_text = pred_command
                    raw_text_with_special_tokens = pred_command
                    pred_id = ACTION_IDS[pred_command]
                    matched = True
                else:
                    raw_text, raw_text_with_special_tokens = generate_action_text(
                        model,
                        processor,
                        item["image"],
                        item["target_description"],
                        device,
                        args.max_new_tokens,
                        item.get("prompt"),
                    )
                    parsed = parse_action_text(raw_text)
                    pred_command = parsed.command
                    pred_id = parsed.action_id
                    matched = parsed.matched
            except Exception as exc:
                raw_text = ""
                raw_text_with_special_tokens = ""
                pred_command = "stop"
                pred_id = ACTION_IDS["stop"]
                matched = False
                error = repr(exc)

            row = {
                **item,
                "raw_action_text": raw_text,
                "raw_action_text_with_special_tokens": raw_text_with_special_tokens,
                "pred_command": pred_command,
                "pred_id": pred_id,
                "parse_matched": matched,
                "correct": pred_id == item["label_id"],
                "candidate_scores": candidate_scores,
                "error": error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            if done % args.log_every == 0:
                print(f"[rank {rank}] {done}/{len(shard)} done", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_metrics(predictions: list[dict[str, Any]], output_dir: Path) -> None:
    labels = ACTION_LABELS
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in predictions:
        true_label = row["label"]
        pred_label = row["pred_command"]
        if pred_label not in label_to_idx:
            pred_label = "stop"
        matrix[label_to_idx[true_label]][label_to_idx[pred_label]] += 1

    per_class = {}
    total = sum(sum(r) for r in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(len(labels)))
        tp = matrix[i][i]
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    metrics = {
        "num_samples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(labels),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(labels),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "prediction_distribution": dict(Counter(row["pred_command"] for row in predictions)),
        "parse_matched": dict(Counter(str(row["parse_matched"]) for row in predictions)),
        "errors": sum(1 for row in predictions if row.get("error")),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

    failures = [row for row in predictions if not row["correct"]]
    write_jsonl(output_dir / "wrong_predictions.jsonl", failures)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Offline balanced action recall for Phi-3.5 UAV-ON policy.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "uavon_phi35_sft.jsonl")
    parser.add_argument("--sample_file", type=Path, default=None)
    parser.add_argument("--model_path", type=str, default=str(ROOT / "outputs" / "phi35_uavon_lora_r256"))
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "offline_action_recall")
    parser.add_argument("--samples_per_class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--inference_mode", choices=["generate", "score"], default="generate")
    parser.add_argument("--score_normalization", choices=["mean", "sum"], default="mean")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--skip_inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    patch_transformers_cache_compat()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_file and args.sample_file.is_file():
        samples = load_jsonl(args.sample_file)
    else:
        samples = sample_balanced(args.dataset, args.samples_per_class, args.seed)
        sample_file = args.sample_file or (args.output_dir / "sampled.jsonl")
        write_jsonl(sample_file, samples)
        args.sample_file = sample_file

    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if not args.skip_inference:
        if args.num_workers is None:
            args.num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
        world_size = max(1, min(args.num_workers, len(samples)))
        for old in args.output_dir.glob("predictions_shard_*.jsonl"):
            old.unlink()
        ctx = get_context("spawn")
        procs = []
        args_dict = vars(args)
        for rank in range(world_size):
            proc = ctx.Process(target=worker_main, args=(rank, world_size, args_dict, samples))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode:
                raise RuntimeError(f"Worker exited with code {proc.exitcode}")

    predictions = []
    for path in sorted(args.output_dir.glob("predictions_shard_*.jsonl")):
        predictions.extend(load_jsonl(path))
    predictions.sort(key=lambda row: row["source_index"])
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    compute_metrics(predictions, args.output_dir)


if __name__ == "__main__":
    main()

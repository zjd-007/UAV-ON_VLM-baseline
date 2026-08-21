#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import clip
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = REPO_ROOT / "UAV-ON_dataset"
DEFAULT_METADATA = DATASET_ROOT / "splits" / "uavon_raw_json" / "train.json"
DEFAULT_DOWNLOAD_ROOT = Path.home() / ".cache" / "clip"


def iter_jsonl_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(path)


def load_cache(path: Path) -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source in iter_jsonl_files(path):
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" not in row or "trajectory_key" not in row:
                continue
            rows[str(row["trajectory_key"])] = row
    return rows


def humanize_label(label: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label).strip())
    return value.replace("_", " ").lower()


def load_class_names(metadata_path: Path) -> list[str]:
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    return sorted({str(row.get("true_name") or "").strip() for row in rows if row.get("true_name")})


def padded_bbox_crop(
    image: Image.Image,
    metrics: dict[str, Any],
    padding_ratio: float,
    min_padding: int,
) -> Image.Image:
    bbox = metrics.get("bbox")
    if not bbox:
        raise ValueError("cannot crop a frame without a target bbox")
    mask_width = max(1, int(metrics.get("width", image.width)))
    mask_height = max(1, int(metrics.get("height", image.height)))
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 *= image.width / mask_width
    x1 = (x1 + 1) * image.width / mask_width
    y0 *= image.height / mask_height
    y1 = (y1 + 1) * image.height / mask_height
    padding = max(min_padding, int(max(x1 - x0, y1 - y0) * padding_ratio))
    crop_box = (
        max(0, int(x0) - padding),
        max(0, int(y0) - padding),
        min(image.width, int(x1) + padding),
        min(image.height, int(y1) + padding),
    )
    return image.crop(crop_box)


def build_samples(cache: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for key, trajectory in cache.items():
        if trajectory.get("status") != "ok":
            continue
        target = str(trajectory.get("true_name") or "").strip()
        if not target:
            continue
        for frame in trajectory.get("frames") or []:
            metrics = frame.get("mask") or {}
            if int(metrics.get("pixel_count", 0)) <= 0 or not metrics.get("bbox"):
                continue
            samples.append(
                {
                    "trajectory_key": key,
                    "frame_idx": int(frame["frame_idx"]),
                    "image_path": str(
                        frame.get("replay_image_path") or frame["image_path"]
                    ),
                    "image_source": (
                        "synchronized_replay"
                        if frame.get("replay_image_path")
                        else "original_recording"
                    ),
                    "target": target,
                    "description": str(trajectory.get("target_description") or "").strip(),
                    "pixel_count": int(metrics.get("pixel_count", 0)),
                    "bbox_short_side": min(
                        int(metrics.get("bbox_width", 0)),
                        int(metrics.get("bbox_height", 0)),
                    ),
                    "metrics": metrics,
                }
            )
    return samples


def normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def rank_result(
    similarities: torch.Tensor,
    target_index: int,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    target_score = float(similarities[target_index].item())
    other = similarities.clone()
    other[target_index] = -torch.inf
    best_other_score, best_other_index = other.max(dim=0)
    rank = int((similarities > similarities[target_index]).sum().item()) + 1
    top_scores, top_indices = similarities.topk(min(top_k, len(class_names)))
    return {
        "target_score": target_score,
        "target_rank": rank,
        "target_margin": target_score - float(best_other_score.item()),
        "best_other_class": class_names[int(best_other_index.item())],
        "best_other_score": float(best_other_score.item()),
        "top_classes": [
            {
                "class_name": class_names[int(index.item())],
                "score": float(score.item()),
            }
            for score, index in zip(top_scores, top_indices)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score target recognizability with OpenAI CLIP.")
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.5)
    parser.add_argument("--min-crop-padding", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.torch_num_threads < 1:
        raise ValueError("--torch-num-threads must be at least 1")

    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(
        args.model,
        device=device,
        jit=False,
        download_root=str(args.download_root),
    )
    model.eval()
    class_names = load_class_names(args.metadata)
    class_index = {name: index for index, name in enumerate(class_names)}
    class_prompts = [f"a photo of a {humanize_label(name)}" for name in class_names]
    with torch.no_grad():
        class_features = normalize(
            model.encode_text(clip.tokenize(class_prompts, truncate=True).to(device))
        )

    cache = load_cache(args.visibility_cache)
    all_samples = build_samples(cache)
    sharded_samples = [
        sample
        for index, sample in enumerate(all_samples)
        if index % args.num_shards == args.shard_index
    ]
    existing_keys: set[tuple[str, int]] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_keys.add((str(row["trajectory_key"]), int(row["frame_idx"])))
    samples = [
        sample
        for sample in sharded_samples
        if (sample["trajectory_key"], sample["frame_idx"]) not in existing_keys
    ]
    written = 0
    output_mode = "a" if args.resume else "w"
    with args.output.open(output_mode, encoding="utf-8") as output:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            full_tensors = []
            crop_tensors = []
            descriptions = []
            for sample in batch:
                image = Image.open(sample["image_path"]).convert("RGB")
                crop = padded_bbox_crop(
                    image,
                    sample["metrics"],
                    args.crop_padding_ratio,
                    args.min_crop_padding,
                )
                full_tensors.append(preprocess(image))
                crop_tensors.append(preprocess(crop))
                description = sample["description"] or humanize_label(sample["target"])
                descriptions.append(
                    f"a photo of a {humanize_label(sample['target'])}: {description}"
                )
            images = torch.stack(full_tensors + crop_tensors).to(device)
            with torch.no_grad():
                image_features = normalize(model.encode_image(images))
                description_features = normalize(
                    model.encode_text(clip.tokenize(descriptions, truncate=True).to(device))
                )
            full_features = image_features[: len(batch)]
            crop_features = image_features[len(batch) :]
            full_similarities = full_features @ class_features.T
            crop_similarities = crop_features @ class_features.T
            description_similarities = (crop_features * description_features).sum(dim=-1)
            for index, sample in enumerate(batch):
                target_index = class_index[sample["target"]]
                full_result = rank_result(
                    full_similarities[index],
                    target_index,
                    class_names,
                    args.top_k,
                )
                crop_result = rank_result(
                    crop_similarities[index],
                    target_index,
                    class_names,
                    args.top_k,
                )
                row = {
                    "trajectory_key": sample["trajectory_key"],
                    "frame_idx": sample["frame_idx"],
                    "target": sample["target"],
                    "pixel_count": sample["pixel_count"],
                    "bbox_short_side": sample["bbox_short_side"],
                    "clip_model": args.model,
                    "image_source": sample["image_source"],
                    "clip_crop_description_score": float(description_similarities[index].item()),
                    "clip_crop_target_score": crop_result["target_score"],
                    "clip_crop_target_rank": crop_result["target_rank"],
                    "clip_crop_target_margin": crop_result["target_margin"],
                    "clip_crop_best_other_class": crop_result["best_other_class"],
                    "clip_crop_top_classes": crop_result["top_classes"],
                    "clip_full_target_score": full_result["target_score"],
                    "clip_full_target_rank": full_result["target_rank"],
                    "clip_full_target_margin": full_result["target_margin"],
                    "clip_full_best_other_class": full_result["best_other_class"],
                    "clip_full_top_classes": full_result["top_classes"],
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            output.flush()
            print(
                json.dumps(
                    {"processed": min(start + len(batch), len(samples)), "total": len(samples)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "samples_written_this_run": written,
                "samples_already_present": len(existing_keys),
                "samples": written + len(existing_keys),
                "unsharded_samples": len(all_samples),
                "sharded_samples": len(sharded_samples),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "classes": len(class_names),
                "model": args.model,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

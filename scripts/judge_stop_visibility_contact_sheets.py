#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"


def load_cache(cache_dir: Path) -> list[dict[str, Any]]:
    trajectories = []
    for path in sorted(cache_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def load_requested_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keys = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return keys


def frame_image_path(frame: dict[str, Any]) -> Path:
    return Path(frame.get("replay_image_path") or frame["image_path"])


def draw_candidate_tile(frame: dict[str, Any], tile_size: int) -> Image.Image:
    label_height = 28
    source = Image.open(frame_image_path(frame)).convert("RGB")
    source_width, source_height = source.size
    image = source.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    bbox = (frame.get("mask") or {}).get("bbox")
    if bbox:
        x0, y0, x1, y1 = bbox
        scaled = (
            round(x0 * tile_size / source_width),
            round(y0 * tile_size / source_height),
            round(x1 * tile_size / source_width),
            round(y1 * tile_size / source_height),
        )
        draw.rectangle(scaled, outline=(255, 220, 0), width=4)

    tile = Image.new("RGB", (tile_size, tile_size + label_height), "white")
    tile.paste(image, (0, label_height))
    label = (
        f"f{int(frame['frame_idx']):02d}  "
        f"d={float(frame['distance_to_target']):.1f}m  "
        f"px={int((frame.get('mask') or {}).get('pixel_count', 0))}"
    )
    ImageDraw.Draw(tile).text((6, 6), label, fill="black", font=ImageFont.load_default())
    return tile


def build_contact_sheet(
    trajectory: dict[str, Any],
    output_path: Path,
    tile_size: int,
    columns: int,
) -> None:
    frames = sorted(trajectory.get("frames") or [], key=lambda row: int(row["frame_idx"]))
    if not frames:
        raise ValueError(f"trajectory has no candidate frames: {trajectory['trajectory_key']}")
    tiles = [draw_candidate_tile(frame, tile_size) for frame in frames]
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tiles[0].height), (235, 235, 235))
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_size
        y = (index // columns) * tile.height
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def build_prompt(trajectory: dict[str, Any]) -> str:
    frame_ids = ", ".join(
        f"f{int(frame['frame_idx']):02d}" for frame in trajectory.get("frames") or []
    )
    return (
        "You are auditing candidate STOP images for visual navigation training. "
        "Each tile is one camera frame. A yellow rectangle, when present, marks the exact "
        "target instance; it is an annotation and not part of the scene. Inspect the actual "
        "pixels inside and around it.\n\n"
        f"Target category: {trajectory.get('true_name') or 'unknown'}\n"
        f"Target description: {trajectory.get('target_description') or 'unknown'}\n"
        f"Candidate frame IDs: {frame_ids}\n\n"
        "Choose the single best STOP training frame. The target must be plainly visible and "
        "identifiable from its semantic features, not merely indicated by a tiny box. Prefer a "
        "closer and larger target only while enough of the object remains visible to identify it. "
        "Penalize severe truncation, occlusion, blur, and views showing only an uninformative "
        "fragment. Return NONE if no frame provides reliable visual evidence of the target.\n\n"
        "Return exactly one JSON object with this schema and no markdown:\n"
        '{"choice":"fNN or NONE","confidence":"high, medium, or low",'
        '"reason":"one short visual reason"}'
    )


def move_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
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


def judge_sheet(
    model,
    processor,
    sheet_path: Path,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(sheet_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = move_inputs(inputs, device)
    input_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=(
                processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            ),
        )
    return processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True
    )[0].strip()


def parse_choice(text: str, valid_frames: set[int]) -> int | None:
    if re.search(r"\bNONE\b", text, flags=re.IGNORECASE):
        return None
    match = re.search(r'"choice"\s*:\s*"?f(\d+)', text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\bf(\d+)\b", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"could not parse frame choice: {text!r}")
    frame_idx = int(match.group(1))
    if frame_idx not in valid_frames:
        raise ValueError(f"model selected unavailable frame f{frame_idx}: {text!r}")
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a frozen VLM to audit the most recognizable Stop view per trajectory."
    )
    parser.add_argument("--visibility-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-keys", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    requested = load_requested_keys(args.trajectory_keys)
    trajectories = [
        row
        for row in load_cache(args.visibility_cache)
        if requested is None or row["trajectory_key"] in requested
    ]
    if requested is not None:
        found = {row["trajectory_key"] for row in trajectories}
        missing = sorted(requested - found)
        if missing:
            raise KeyError(f"trajectory keys missing from cache: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sheets_dir = args.output_dir / "sheets"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    output_path = args.output_dir / "judgments.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for index, trajectory in enumerate(trajectories, start=1):
            safe_key = str(trajectory["trajectory_key"]).replace("::", "__")
            sheet_path = sheets_dir / f"{safe_key}.jpg"
            build_contact_sheet(
                trajectory,
                sheet_path,
                tile_size=args.tile_size,
                columns=args.columns,
            )
            prompt = build_prompt(trajectory)
            raw = judge_sheet(
                model,
                processor,
                sheet_path,
                prompt,
                args.device,
                args.max_new_tokens,
            )
            valid_frames = {int(frame["frame_idx"]) for frame in trajectory["frames"]}
            try:
                choice = parse_choice(raw, valid_frames)
                error = None
            except Exception as exc:
                choice = None
                error = repr(exc)
            record = {
                "trajectory_key": trajectory["trajectory_key"],
                "target": trajectory.get("true_name"),
                "chosen_frame_idx": choice,
                "raw_response": raw,
                "parse_error": error,
                "sheet": str(sheet_path.resolve()),
                "model": str(args.model_path.resolve()),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(trajectories)}",
                        "trajectory_key": trajectory["trajectory_key"],
                        "choice": choice,
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

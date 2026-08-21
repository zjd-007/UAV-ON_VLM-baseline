#!/usr/bin/env python3
"""One-sample debug script: verify LLaMA-Factory phi3v preprocessing produces
negative image placeholders, pixel_values, and correct labels.

Usage:
  python scripts/debug_phi3v_preprocess.py [--sample_idx N]

Pass condition (all must be true):
  negative_placeholder_count > 0
  pixel_values exists
  image_sizes exists
  supervised_token_count > 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Register phi3v plugin
import scripts.llamafactory_phi3v as lf_phi3v  # noqa: E402

lf_phi3v.register_phi3v_template()

from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402


def load_one_sample(sft_path: Path, sample_idx: int):
    with sft_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise IndexError(f"Sample index {sample_idx} not found in {sft_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "uavon_phi35_sft.jsonl")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-3.5-vision-instruct", trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    sample = load_one_sample(args.dataset, args.sample_idx)
    human_msg = sample["conversations"][0]["value"]
    gpt_msg = sample["conversations"][1]["value"]
    image_path = sample["images"][0]
    image = Image.open(image_path).convert("RGB")

    # === Before fix: text-only tokenization ===
    text_with_token = human_msg.replace("<image>\n", "<|image_1|>\n", 1)
    formatted_text = (
        f"<|user|>\n{text_with_token}<|end|>\n<|assistant|>\n"
        f"{gpt_msg}<|end|>\n"
    )
    text_only_ids = tokenizer.encode(formatted_text, add_special_tokens=True)
    text_only_neg = sum(1 for t in text_only_ids if t < 0)
    print(f"=== Before fix (text-only tokenization) ===")
    print(f"  input_ids_len: {len(text_only_ids)}")
    print(f"  negative_placeholder_count: {text_only_neg}")

    # === After fix: plugin.process_token_ids ===
    from llamafactory.data.mm_plugin import get_mm_plugin
    plugin = get_mm_plugin(name="phi3v", image_token="<|image_1|>")

    # Simulate LLaMA-Factory's message processing
    msgs = [
        {"role": "user", "content": human_msg},
        {"role": "assistant", "content": gpt_msg},
    ]
    msgs = plugin.process_messages(msgs, [image], [], [], processor)

    # Build text-level input_ids as LLaMA-Factory's encode_multiturn would
    # (using the same format strings as the phi3v template)
    user_text = f"<|user|>\n{msgs[0]['content']}<|end|>\n<|assistant|>\n"
    assistant_text = f"{msgs[1]['content']}<|end|>\n"
    full_text = user_text + assistant_text

    llama_ids = tokenizer.encode(full_text, add_special_tokens=True)

    # Build labels like LLaMA-Factory does: mask user part
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    llama_labels = [IGNORE_INDEX] * len(user_ids) + assistant_ids
    if len(llama_labels) < len(llama_ids):
        llama_labels = llama_labels + [IGNORE_INDEX] * (len(llama_ids) - len(llama_labels))
    else:
        llama_labels = llama_labels[: len(llama_ids)]

    # Call the FIXED process_token_ids
    new_ids, new_labels = plugin.process_token_ids(
        input_ids=llama_ids,
        labels=llama_labels,
        images=[image],
        videos=[],
        audios=[],
        tokenizer=tokenizer,
        processor=processor,
    )

    neg_count = sum(1 for t in new_ids if t < 0)
    sup_count = sum(1 for x in new_labels if x != IGNORE_INDEX)
    # Check for leaked image placeholders in labels (negative but NOT IGNORE_INDEX)
    leaked_placeholders = sum(1 for t in new_labels if t < 0 and t != IGNORE_INDEX)

    print(f"\n=== After fix (Phi processor re-encode) ===")
    print(f"  input_ids_len: {len(new_ids)}")
    print(f"  negative_placeholder_count: {neg_count}")
    print(f"  labels_len: {len(new_labels)}")
    print(f"  supervised_token_count: {sup_count}")
    print(f"  leaked_image_placeholders_in_labels (should be 0): {leaked_placeholders}")

    # Decode supervised labels
    valid_label_ids = [x for x in new_labels if x != IGNORE_INDEX and x >= 0]
    decoded = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
    print(f"  supervised_labels_decoded: {decoded!r}")

    # Check pixel_values from get_mm_inputs
    mm_inputs = plugin.get_mm_inputs(
        images=[image], videos=[], audios=[],
        imglens=[1], vidlens=[0], audlens=[0],
        batch_ids=[new_ids],
        processor=processor,
    )
    has_pv = "pixel_values" in mm_inputs
    has_is = "image_sizes" in mm_inputs
    print(f"  pixel_values present: {has_pv}")
    if has_pv:
        print(f"  pixel_values_shape: {list(mm_inputs['pixel_values'].shape)}")
    print(f"  image_sizes present: {has_is}")
    if has_is:
        print(f"  image_sizes: {mm_inputs.get('image_sizes', 'N/A')}")

    # === Verdict ===
    print(f"\n{'='*60}")
    passed = (
        neg_count > 0
        and has_pv
        and has_is
        and sup_count > 0
        and leaked_placeholders == 0
    )
    if passed:
        print("VERDICT: PASS — Training will use real multimodal encoding.")
    else:
        print("VERDICT: FAIL")
        if neg_count == 0:
            print("  -> negative_placeholder_count == 0 (no images in input)")
        if not has_pv:
            print("  -> pixel_values missing")
        if not has_is:
            print("  -> image_sizes missing")
        if sup_count == 0:
            print("  -> no supervised tokens in labels")
        if leaked_placeholders > 0:
            print(f"  -> {leaked_placeholders} image placeholders NOT masked in labels")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Qwen2.5-VL backend for the shared UAV-ON evaluation loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_phi35_uavon as shared  # noqa: E402
from vlm_baseline.prompting import build_prompt  # noqa: E402


def load_qwen_model_and_processor(model_path: str, base_model_path: str | None, device: str):
    model_dir = Path(model_path)
    is_adapter = (model_dir / "adapter_config.json").is_file()
    resolved_model_path = model_path
    if is_adapter:
        adapter_config = PeftConfig.from_pretrained(model_path)
        resolved_model_path = base_model_path or adapter_config.base_model_name_or_path
        if not resolved_model_path:
            raise ValueError(f"Unable to resolve the base model for Qwen adapter: {model_path}")
    elif base_model_path:
        raise ValueError("--base_model_path is only valid when --model_path is a Qwen LoRA adapter")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if is_adapter:
        model = PeftModel.from_pretrained(model, model_path)
        if os.environ.get("QWEN_MERGE_ADAPTER_FOR_INFERENCE", "0") == "1":
            model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_path if is_adapter else resolved_model_path)
    model.to(device)
    model.eval()
    return model, processor


def build_qwen_inputs(
    processor,
    image: Image.Image,
    target_description: str,
    depth_context: str | None,
    memory_context: str | None,
):
    prompt_text = build_prompt(
        target_description,
        depth_context=depth_context,
        memory_context=memory_context,
    ).replace("<image>\n", "", 1)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ), chat_text


def generate_qwen_action_text(
    model,
    processor,
    image,
    target_description: str,
    device: str,
    max_new_tokens: int,
    depth_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    inputs, _ = build_qwen_inputs(
        processor,
        image,
        target_description,
        depth_context,
        memory_context,
    )
    inputs = shared.move_inputs_to_device(inputs, device, torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    tokenizer = getattr(processor, "tokenizer", processor)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = generated[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = shared.parse_args()
    if args.inference_mode != "generate":
        raise ValueError("The Qwen2.5-VL backend currently supports --inference_mode generate only")
    if not args.skip_kill_env_process:
        shared.kill_all_env_process()

    shared.load_model_and_processor = load_qwen_model_and_processor
    shared.generate_action_text = generate_qwen_action_text
    shared.evaluate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments
from transformers.processing_utils import ProcessorMixin

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


IGNORE_INDEX = -100
ROOT = Path(__file__).resolve().parents[1]
_ORIG_PROCESSOR_SAVE_PRETRAINED = ProcessorMixin.save_pretrained


def _patched_processor_save_pretrained(self, *args, **kwargs):
    if not hasattr(self, "chat_template"):
        self.chat_template = None
    return _ORIG_PROCESSOR_SAVE_PRETRAINED(self, *args, **kwargs)


ProcessorMixin.save_pretrained = _patched_processor_save_pretrained


def patch_transformers_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "get_max_length") or not hasattr(DynamicCache, "get_max_cache_shape"):
        return
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs.")
    data = yaml.safe_load(text)
    return data or {}


def get_arg(config: dict[str, Any], name: str, default: Any = None) -> Any:
    return config.get(name, default)


def read_label(row: dict[str, Any]) -> str:
    if "conversations" in row:
        return row["conversations"][-1]["value"].strip()
    if "messages" in row:
        return row["messages"][-1]["content"].strip()
    raise KeyError("Expected conversations or messages in dataset row.")


def read_user_prompt(row: dict[str, Any]) -> str:
    if "conversations" in row:
        return row["conversations"][0]["value"]
    if "messages" in row:
        return row["messages"][0]["content"]
    raise KeyError("Expected conversations or messages in dataset row.")


def build_phi35_prompt(tokenizer, user_prompt: str) -> str:
    plain_prompt = user_prompt.replace("<image>\n", "", 1)
    content = f"<|image_1|>\n{plain_prompt}"
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{content}<|end|>\n<|assistant|>\n"


class UAVONSFTDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = None, seed: int = 42) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        if max_samples is not None and max_samples > 0 and max_samples < len(self.rows):
            rng = random.Random(seed)
            indices = list(range(len(self.rows)))
            rng.shuffle(indices)
            indices = sorted(indices[:max_samples])
            self.rows = [self.rows[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image": row["images"][0],
            "user_prompt": read_user_prompt(row),
            "label": read_label(row),
            "source_index": index,
        }


@dataclass
class Phi3VDataCollator:
    processor: Any
    cutoff_len: int = 2048
    torch_dtype: torch.dtype = torch.bfloat16
    debug: bool = False

    def __post_init__(self) -> None:
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def _encode_one(self, feature: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = Image.open(feature["image"]).convert("RGB")
        prompt = build_phi35_prompt(self.tokenizer, feature["user_prompt"])
        answer_text = f"{feature['label']}<|end|>\n"
        full_text = prompt + answer_text

        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        answer_len = len(self.tokenizer(answer_text, add_special_tokens=False).input_ids)

        if input_ids.numel() > self.cutoff_len:
            keep_prompt_len = self.cutoff_len - answer_len
            if keep_prompt_len <= 0:
                raise ValueError(f"cutoff_len={self.cutoff_len} is too small for answer {answer_text!r}.")
            input_ids = torch.cat([input_ids[:keep_prompt_len], input_ids[-answer_len:]], dim=0)
            attention_mask = torch.cat([attention_mask[:keep_prompt_len], attention_mask[-answer_len:]], dim=0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[-answer_len:] = input_ids[-answer_len:]
        labels[input_ids < 0] = IGNORE_INDEX

        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"][0],
            "image_sizes": inputs["image_sizes"][0],
        }
        if self.debug:
            encoded["negative_placeholder_count"] = torch.tensor((input_ids < 0).sum().item(), dtype=torch.long)
            encoded["supervised_token_count"] = torch.tensor((labels != IGNORE_INDEX).sum().item(), dtype=torch.long)
        return encoded

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(feature) for feature in features]
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(item["input_ids"].numel() for item in encoded)
        max_crops = max(item["pixel_values"].shape[0] for item in encoded)

        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
            "image_sizes": [],
        }

        for item in encoded:
            seq_pad = max_len - item["input_ids"].numel()
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, seq_pad), value=pad_token_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, seq_pad), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, seq_pad), value=IGNORE_INDEX))

            crop_pad = max_crops - item["pixel_values"].shape[0]
            pixel_values = item["pixel_values"]
            if crop_pad:
                pixel_values = torch.nn.functional.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, crop_pad), value=0)
            batch["pixel_values"].append(pixel_values)
            batch["image_sizes"].append(item["image_sizes"])

        output = {
            "input_ids": torch.stack(batch["input_ids"], dim=0),
            "attention_mask": torch.stack(batch["attention_mask"], dim=0),
            "labels": torch.stack(batch["labels"], dim=0),
            "pixel_values": torch.stack(batch["pixel_values"], dim=0).to(self.torch_dtype),
            "image_sizes": torch.stack(batch["image_sizes"], dim=0),
        }
        if self.debug:
            output["negative_placeholder_count"] = torch.stack(
                [item["negative_placeholder_count"] for item in encoded], dim=0
            )
            output["supervised_token_count"] = torch.stack([item["supervised_token_count"] for item in encoded], dim=0)
        return output


def load_base_model(model_name_or_path: str, torch_dtype: torch.dtype, flash_attn: str | None):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if flash_attn in {"fa2", "flash_attention_2"}:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except Exception:
        return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)


def build_model(config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    torch_dtype = torch.bfloat16 if get_arg(config, "bf16", True) else torch.float16
    model = load_base_model(
        get_arg(config, "model_name_or_path", "microsoft/Phi-3.5-vision-instruct"),
        torch_dtype,
        get_arg(config, "flash_attn", "fa2"),
    )
    model.config.use_cache = False

    target_modules = get_arg(
        config,
        "lora_target",
        "qkv_proj,o_proj,gate_up_proj,down_proj,q_proj,k_proj,v_proj,out_proj,fc1,fc2",
    )
    if isinstance(target_modules, str):
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]

    lora_config = LoraConfig(
        r=int(get_arg(config, "lora_rank", 256)),
        lora_alpha=int(get_arg(config, "lora_alpha", 512)),
        lora_dropout=float(get_arg(config, "lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if bool(get_arg(config, "gradient_checkpointing", False)):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


class Phi3VTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("negative_placeholder_count", None)
        inputs.pop("supervised_token_count", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def save_sanity_batch(output_dir: Path, batch: dict[str, torch.Tensor]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "labels_shape": list(batch["labels"].shape),
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "image_sizes": batch["image_sizes"].tolist(),
        "negative_placeholder_count": int((batch["input_ids"] < 0).sum().item()),
        "supervised_token_count": int((batch["labels"] != IGNORE_INDEX).sum().item()),
        "per_sample_negative_placeholder_count": (batch["input_ids"] < 0).sum(dim=1).tolist(),
        "per_sample_supervised_token_count": (batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist(),
    }
    (output_dir / "native_batch_sanity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["negative_placeholder_count"] <= 0:
        raise RuntimeError("Native Phi-3.5 batch has no negative image placeholders; image features would be unused.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Hugging Face/PEFT Phi-3.5-Vision LoRA SFT for UAV-ON.")
    parser.add_argument("config", nargs="?", type=Path, default=ROOT / "configs" / "train_phi35_lora_native.yaml")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--sanity_check_only", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def merge_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in (
        "dataset",
        "model_name_or_path",
        "output_dir",
        "max_samples",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = str(value) if isinstance(value, Path) else value
    return merged


def main() -> None:
    patch_transformers_cache_compat()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    args = parse_args()
    config = merge_overrides(load_config(args.config), args)

    seed = int(get_arg(config, "seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)

    model_name_or_path = get_arg(config, "model_name_or_path", "microsoft/Phi-3.5-vision-instruct")
    output_dir = Path(get_arg(config, "output_dir", ROOT / "outputs" / "phi35_uavon_lora_r256_native"))
    dataset_path = Path(get_arg(config, "dataset", ROOT / "data" / "uavon_phi35_sft.jsonl"))

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    dataset = UAVONSFTDataset(dataset_path, get_arg(config, "max_samples", None), seed)
    collator = Phi3VDataCollator(
        processor=processor,
        cutoff_len=int(get_arg(config, "cutoff_len", 2048)),
        torch_dtype=torch.bfloat16 if get_arg(config, "bf16", True) else torch.float16,
        debug=True,
    )
    sanity_batch = collator([dataset[0]])
    save_sanity_batch(output_dir, sanity_batch)

    model = build_model(config)
    with torch.no_grad():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        forward_batch = {
            key: value.to(device) for key, value in sanity_batch.items() if key not in {"negative_placeholder_count", "supervised_token_count"}
        }
        loss = model(**forward_batch).loss
        print(f"sanity_forward_loss={float(loss.detach().cpu()):.6f}")
        del forward_batch, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.sanity_check_only:
        processor.save_pretrained(output_dir)
        return

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=bool(get_arg(config, "overwrite_output_dir", False)),
        per_device_train_batch_size=int(get_arg(config, "per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(get_arg(config, "gradient_accumulation_steps", 8)),
        learning_rate=float(get_arg(config, "learning_rate", 1.0e-4)),
        num_train_epochs=float(get_arg(config, "num_train_epochs", 3.0)),
        lr_scheduler_type=str(get_arg(config, "lr_scheduler_type", "cosine")),
        warmup_ratio=float(get_arg(config, "warmup_ratio", 0.03)),
        optim=str(get_arg(config, "optim", "adamw_torch")),
        bf16=bool(get_arg(config, "bf16", True)),
        fp16=bool(get_arg(config, "fp16", False)),
        logging_steps=int(get_arg(config, "logging_steps", 10)),
        save_steps=int(get_arg(config, "save_steps", 1000)),
        save_total_limit=get_arg(config, "save_total_limit", None),
        dataloader_num_workers=int(get_arg(config, "dataloader_num_workers", 4)),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=str(get_arg(config, "report_to", "none")),
        ddp_find_unused_parameters=bool(get_arg(config, "ddp_find_unused_parameters", False)),
        gradient_checkpointing=bool(get_arg(config, "gradient_checkpointing", False)),
        gradient_checkpointing_kwargs=get_arg(config, "gradient_checkpointing_kwargs", None),
        max_grad_norm=float(get_arg(config, "max_grad_norm", 1.0)),
        seed=seed,
    )

    train_collator = Phi3VDataCollator(
        processor=processor,
        cutoff_len=int(get_arg(config, "cutoff_len", 2048)),
        torch_dtype=torch.bfloat16 if get_arg(config, "bf16", True) else torch.float16,
        debug=False,
    )
    trainer = Phi3VTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=train_collator,
        tokenizer=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)
    metrics = trainer.state.log_history
    (output_dir / "trainer_log_history.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

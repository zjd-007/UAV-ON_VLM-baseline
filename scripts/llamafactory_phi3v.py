#!/usr/bin/env python3
from __future__ import annotations

import sys
from copy import deepcopy

from llamafactory.data.formatter import StringFormatter
from llamafactory.data.mm_plugin import IMAGE_PLACEHOLDER, BasePlugin, get_mm_plugin, register_mm_plugin
from llamafactory.data.template import TEMPLATES, register_template
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.tuner import export_model, run_exp
from transformers.processing_utils import ProcessorMixin


_ORIG_PROCESSOR_SAVE_PRETRAINED = ProcessorMixin.save_pretrained


def _patched_processor_save_pretrained(self, *args, **kwargs):
    if not hasattr(self, "chat_template"):
        self.chat_template = None
    return _ORIG_PROCESSOR_SAVE_PRETRAINED(self, *args, **kwargs)


ProcessorMixin.save_pretrained = _patched_processor_save_pretrained


class Phi3VPlugin(BasePlugin):
    """Phi-3.5-Vision plugin for LLaMA-Factory 0.9.3.

    Key fix over the previous minimal plugin: process_token_ids() re-encodes
    with the real Phi-3.5-Vision processor so that input_ids contain negative
    image placeholders, matching the inference path.
    """

    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        image_idx = 1
        for message in messages:
            while IMAGE_PLACEHOLDER in message["content"]:
                message["content"] = message["content"].replace(IMAGE_PLACEHOLDER, f"<|image_{image_idx}|>", 1)
                image_idx += 1
        return messages

    def process_token_ids(
        self,
        input_ids,
        labels,
        images,
        videos,
        audios,
        tokenizer,
        processor,
    ):
        if not images or len(input_ids) == 0:
            return input_ids, labels

        # Only single-image samples are supported (UAV-ON uses one current frame).
        if len(images) != 1:
            raise ValueError(f"Phi3VPlugin supports exactly one image, got {len(images)}.")

        # Load image (Phi-3.5-Vision processor requires PIL.Image, not str/Path).
        from pathlib import Path
        from PIL import Image

        img = images[0]
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")

        # Decode LLaMA-Factory's text-only token ids.
        # 1) Filter out negative tokens (image placeholders) so decode does not overflow.
        # 2) Strip leading BOS to avoid double-<s> when processor re-encodes
        #    (LLaMA-Factory's encode adds <s>; processor's tokenizer add_bos_token=True
        #     would add another <s>, shifting position ids by 1).
        clean_ids = [tid for tid in input_ids if tid >= 0]
        if clean_ids and clean_ids[0] == getattr(tokenizer, "bos_token_id", None):
            clean_ids = clean_ids[1:]

        text = tokenizer.decode(
            clean_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # Re-encode with the real Phi-3.5-Vision processor.
        encoded = processor(text=text, images=img, return_tensors="pt")
        new_input_ids = encoded["input_ids"][0].tolist()

        # Rebuild labels by decoding old supervised tokens and re-tokenizing,
        # then verifying alignment with the tail of new_input_ids.
        if labels is None:
            new_labels = None
        else:
            old_answer_ids = [x for x in labels if x != IGNORE_INDEX]
            new_labels = [IGNORE_INDEX] * len(new_input_ids)

            if old_answer_ids:
                old_answer_text = tokenizer.decode(
                    old_answer_ids, skip_special_tokens=False
                )
                new_answer_ids = tokenizer(
                    old_answer_text, add_special_tokens=False
                ).input_ids

                if new_input_ids[-len(new_answer_ids) :] != new_answer_ids:
                    raise RuntimeError(
                        "Assistant labels do not match processor input tail. "
                        f"answer={old_answer_text!r}"
                    )

                new_labels[-len(new_answer_ids) :] = new_answer_ids

            # Mask all negative image placeholders in labels.
            for idx, token_id in enumerate(new_input_ids):
                if token_id < 0:
                    new_labels[idx] = IGNORE_INDEX

        # Sanity checks: re-encoded ids must contain negative placeholders,
        # and image dimensions must be consistent.
        neg_count = sum(1 for tid in new_input_ids if tid < 0)
        if neg_count <= 0:
            raise RuntimeError(
                "Phi3VPlugin.process_token_ids: no negative image placeholders "
                "in re-encoded input_ids."
            )

        return new_input_ids, new_labels

    def get_mm_inputs(self, images, videos, audios, imglens, vidlens, audlens, batch_ids, processor):
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_img_tokens", None)
        return mm_inputs


def register_phi3v_template() -> None:
    try:
        register_mm_plugin("phi3v", Phi3VPlugin)
    except ValueError:
        pass

    if "phi3v" not in TEMPLATES:
        register_template(
            name="phi3v",
            format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
            format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
            stop_words=["<|end|>"],
            replace_eos=True,
            mm_plugin=get_mm_plugin(name="phi3v", image_token="<|image_1|>"),
        )


def main() -> None:
    register_phi3v_template()
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "train"
    if command == "train":
        run_exp()
    elif command == "export":
        export_model()
    else:
        raise SystemExit(f"Unsupported command for phi3v launcher: {command}")


if __name__ == "__main__":
    main()

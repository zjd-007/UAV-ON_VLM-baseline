# Phi-3.5-Vision + LLaMA-Factory 0.9.3 Plugin Fix Plan

## Goal

Fix the current custom `phi3v` LLaMA-Factory plugin so that Phi-3.5-Vision training is no longer "pseudo multimodal training".

The corrected training path must satisfy:

```text
input_ids contains negative image placeholders
pixel_values is present
image_sizes is present
labels only supervise assistant answer tokens
```

Expected sanity result for one image sample:

```text
input_ids length: about 800+
negative_placeholder_count: > 0, e.g. 700+
pixel_values shape: e.g. [1, 5, 3, 336, 336]
```

The key objective is:

```text
Training processor path must match inference processor path.
```

## Current Problem

Current `scripts/llamafactory_phi3v.py` only does:

```text
<image> -> <|image_1|>
```

Then LLaMA-Factory uses its default tokenizer path to encode text.

This produces:

```text
train_ids_len: 73
negative_count: 0
```

But correct Phi-3.5-Vision processor encoding produces:

```text
infer_ids_len: 818
negative_count: 757
pixel_values: [1, 5, 3, 336, 336]
```

Phi-3.5-Vision only inserts visual features when it sees:

```python
input_ids < 0
```

Therefore, the current plugin likely reads image files and may pass `pixel_values`, but the model does not know where to insert visual features. The model effectively trains on text only:

```text
<|user|>
<|image_1|>
What action should ...
<|assistant|>
forward 3m
```

This can cause the model to collapse to the most frequent action, usually:

```text
forward 3m
```

## High-Level Fix

Add a proper token-id processing stage to `Phi3VPlugin`.

The plugin must not stop at:

```text
<image> -> <|image_1|>
```

It must also ensure that training input ids are produced by Phi-3.5-Vision's processor:

```python
processor(text=..., images=..., return_tensors="pt")
```

instead of plain tokenizer encoding.

## Files To Modify

Primary file:

```text
scripts/llamafactory_phi3v.py
```

Potential supporting debug file:

```text
scripts/debug_phi3v_preprocess.py
```

Config files may also need minor changes:

```text
configs/train_phi35_lora.yaml
configs/train_phi35_lora_smoke.yaml
```

## Step 1: Inspect LLaMA-Factory 0.9.3 BasePlugin Signature

Before editing, inspect the installed `BasePlugin` implementation.

Run:

```bash
python - <<'PY'
import inspect
from llamafactory.data.mm_plugin import BasePlugin

print(inspect.getsource(BasePlugin))
PY
```

Find the exact signature for:

```python
process_token_ids(...)
```

Different LLaMA-Factory versions may use slightly different arguments. The implementation must match the installed 0.9.3 signature exactly.

Expected plugin hooks to inspect:

```text
process_messages
process_token_ids
get_mm_inputs
```

## Step 2: Keep `process_messages()` But Treat It As Text Preparation Only

Keep the current logic:

```python
def process_messages(self, messages, images, videos, audios, processor):
    self._validate_input(processor, images, videos, audios)
    self._validate_messages(messages, images, videos, audios)
    messages = deepcopy(messages)
    image_idx = 1
    for message in messages:
        while IMAGE_PLACEHOLDER in message["content"]:
            message["content"] = message["content"].replace(
                IMAGE_PLACEHOLDER,
                f"<|image_{image_idx}|>",
                1,
            )
            image_idx += 1
    return messages
```

This part is still useful because Phi-3.5-Vision expects:

```text
<|image_1|>
```

not LLaMA-Factory's generic:

```text
<image>
```

But this function alone is not enough.

## Step 3: Add `process_token_ids()` To Re-encode With Phi Processor

Add a new method inside `Phi3VPlugin`.

The exact signature must match the installed `BasePlugin`, but the logic should be:

1. If no images exist, return the original `input_ids` and `labels`.
2. Decode LLaMA-Factory's assembled `input_ids` back to text.
3. Call Phi processor with both text and images.
4. Replace `input_ids` with processor-produced input ids.
5. Rebuild labels so only the assistant answer is supervised.
6. Mask all negative image placeholders with `IGNORE_INDEX`.
7. Assert `negative_placeholder_count > 0`.

Conceptual implementation:

```python
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
    if not images:
        return input_ids, labels

    text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    encoded = processor(
        text=text,
        images=images[0] if len(images) == 1 else images,
        return_tensors="pt",
    )

    new_input_ids = encoded["input_ids"][0].tolist()

    if labels is None:
        new_labels = labels
    else:
        old_supervised = [x for x in labels if x != IGNORE_INDEX]
        supervised_len = len(old_supervised)

        new_labels = [IGNORE_INDEX] * len(new_input_ids)
        if supervised_len > 0:
            new_labels[-supervised_len:] = new_input_ids[-supervised_len:]

        for idx, token_id in enumerate(new_input_ids):
            if token_id < 0:
                new_labels[idx] = IGNORE_INDEX

    neg_count = sum(1 for token_id in new_input_ids if token_id < 0)
    if neg_count <= 0:
        raise RuntimeError(
            "Phi3VPlugin failed: no negative image placeholders in input_ids."
        )

    return new_input_ids, new_labels
```

Important:

```text
Do not manually insert 757 negative ids.
```

The number of image placeholders depends on processor logic, image size, crop count, and model config. Let the processor decide.

## Step 4: Confirm `IGNORE_INDEX` Import

The plugin needs the same ignore label value used by LLaMA-Factory.

Find where LLaMA-Factory defines it. It may be in one of these:

```python
from llamafactory.extras.constants import IGNORE_INDEX
```

or:

```python
IGNORE_INDEX = -100
```

If import path is unavailable in 0.9.3, define locally:

```python
IGNORE_INDEX = -100
```

## Step 5: Revisit `get_mm_inputs()`

Current code:

```python
def get_mm_inputs(self, images, videos, audios, imglens, vidlens, audlens, batch_ids, processor):
    self._validate_input(processor, images, videos, audios)
    mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
    mm_inputs.pop("num_img_tokens", None)
    return mm_inputs
```

Keep this initially, but verify final batch contains:

```text
pixel_values
image_sizes
```

If `pixel_values` or `image_sizes` are missing, modify `get_mm_inputs()` to directly call:

```python
processor.image_processor(...)
```

or:

```python
processor(text=..., images=...)
```

However, avoid double-processing inconsistencies. The safest design is:

```text
process_token_ids and get_mm_inputs must use the same processor behavior.
```

## Step 6: Add A One-Sample Debug Script

Create:

```text
scripts/debug_phi3v_preprocess.py
```

Purpose:

```text
Load one dataset sample, run LLaMA-Factory preprocessing with phi3v template, and print token/image stats.
```

It should print:

```python
print("input_ids_len:", len(input_ids))
print("negative_placeholder_count:", sum(x < 0 for x in input_ids))
print("labels_len:", len(labels))
print("supervised_token_count:", sum(x != -100 for x in labels))
print("pixel_values_shape:", pixel_values.shape if pixel_values is not None else None)
print("image_sizes:", image_sizes)
```

Pass condition:

```text
negative_placeholder_count > 0
pixel_values exists
image_sizes exists
supervised_token_count > 0
```

Fail condition:

```text
negative_placeholder_count == 0
```

This means the plugin is still not correct.

## Step 7: Validate Label Decoding

After preprocessing, decode supervised labels only:

```python
valid_label_ids = [x for x in labels if x != -100 and x >= 0]
print(tokenizer.decode(valid_label_ids, skip_special_tokens=False))
```

Expected:

```text
forward 3m<|end|>
```

or:

```text
turn left<|end|>
```

Suspicious:

```text
full prompt is supervised
empty labels
image placeholders are supervised
missing <|end|>
```

## Step 8: Fix Inference EOS Together

Because the template uses:

```python
stop_words=["<|end|>"]
replace_eos=True
```

the model learns to stop with:

```text
<|end|>
```

Inference must use:

```python
tokenizer = processor.tokenizer
end_id = tokenizer.convert_tokens_to_ids("<|end|>")
eos_id = tokenizer.eos_token_id

outputs = model.generate(
    **inputs,
    max_new_tokens=8,
    do_sample=False,
    eos_token_id=[end_id, eos_id],
    pad_token_id=tokenizer.pad_token_id,
)
```

Debug decode should use:

```python
tokenizer.decode(outputs[0], skip_special_tokens=False)
```

## Step 9: Run Smoke Preprocessing Before Smoke Training

Do not start training until preprocessing passes.

Required checks:

```text
train_ids_len should be close to infer processor ids length
negative_placeholder_count should match or be close to inference path
pixel_values shape should match inference path
labels should supervise only assistant answer
```

Compare:

```text
LLaMA-Factory training preprocessing
native processor inference preprocessing
```

They do not need to be byte-identical, but they must agree on the key property:

```text
input_ids contains negative image placeholders
```

## Step 10: Run 16-Sample Smoke Train

After preprocessing passes:

```bash
CONFIG=configs/train_phi35_lora_smoke.yaml bash scripts/train_phi35_lora.sh
```

Expected outcomes:

```text
It may use much more VRAM than before.
It may OOM if LoRA rank is too high.
This is expected because real visual tokens are now used.
```

If OOM occurs, reduce memory usage:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
optim: adafactor
lora_rank: 32 or 64
lora_alpha: 64 or 128
```

Avoid starting with:

```yaml
lora_rank: 256
```

because real visual sequence training can exceed 24GB quickly.

## Step 11: Run Offline Recall Again

After smoke training succeeds, run the same 3000 offline samples.

Log:

```text
raw generated text with special tokens
cleaned generated text
parsed action
gold action
```

Expected improvement:

```text
Action distribution should no longer collapse entirely to forward 3m.
```

If still collapsed:

1. Check action label distribution.
2. Check parser.
3. Check whether labels are correct.
4. Check whether prompt lists valid actions.
5. Check whether LoRA rank/training steps are too small.

## Risk Notes

This deep LLaMA-Factory fix is fragile because:

```text
process_token_ids signature must match 0.9.3 exactly
processor re-encoding can break label alignment
pixel_values must match the negative placeholder count
padding must preserve negative ids
image placeholders must remain masked from loss
```

If this becomes too complex, prefer the native Hugging Face/PEFT collator route:

```python
inputs = processor(text=full_text, images=image, return_tensors="pt")
```

That route is easier to verify because the collator directly controls:

```text
input_ids
labels
pixel_values
image_sizes
negative_placeholder_count
```

## Final Acceptance Criteria

The LLaMA-Factory plugin fix is acceptable only if all checks pass:

```text
1. Training batch input_ids has negative image placeholders.
2. Training batch has pixel_values and image_sizes.
3. Labels supervise only assistant answer tokens.
4. Negative image placeholders are masked with -100 in labels.
5. Inference uses <|end|> as an eos token.
6. Offline raw decode does not show uncontrolled over-generation.
7. Offline recall action distribution is not trivially collapsed to forward 3m.
```

If any of 1-4 fails, the model is not doing valid Phi-3.5-Vision multimodal training.

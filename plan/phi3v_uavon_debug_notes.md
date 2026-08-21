# Phi-3.5-Vision UAV-ON Fine-tuning Debug Notes

## Background

The current UAV-ON baseline fine-tunes Phi-3.5-Vision with LLaMA-Factory 0.9.3. Since this LLaMA-Factory version does not provide a built-in Phi-vision multimodal plugin, the project adds a custom launcher/plugin:

- `llamafactory_phi3v.py`
- custom `Phi3VPlugin`
- custom `phi3v` chat template
- `prompting.py` for task prompt construction

Current symptom:

```text
The fine-tuned model tends to output "forward 3m" for almost every evaluation step.
```

This document summarizes the most likely issues to check in the repository.

## Relevant Code

### `prompting.py`

```python
PROMPT_TEMPLATE = (
    "<image>\n"
    "What action should the UAV take to find {target_description}? "
    "Reply with exactly one command."
)


def build_prompt(target_description: str) -> str:
    description = target_description.strip().lower().rstrip(" .")
    return PROMPT_TEMPLATE.format(target_description=description)
```

For target `Person.`, the base prompt becomes:

```text
<image>
What action should the UAV take to find person? Reply with exactly one command.
```

### `llamafactory_phi3v.py`

Key plugin logic:

```python
while IMAGE_PLACEHOLDER in message["content"]:
    message["content"] = message["content"].replace(
        IMAGE_PLACEHOLDER, f"<|image_{image_idx}|>", 1
    )
    image_idx += 1
```

So during training:

```text
<image>
```

is converted into:

```text
<|image_1|>
```

This is necessary because Phi-3.5-Vision's processor expects image placeholders such as `<|image_1|>`.

The custom template is:

```python
register_template(
    name="phi3v",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="phi3v", image_token="<|image_1|>"),
)
```

The expected training text structure is therefore:

```text
<|user|>
<|image_1|>
What action should the UAV take to find person? Reply with exactly one command.<|end|>
<|assistant|>
forward 3m<|end|>
```

## Key Issue 1: Inference EOS Token May Not Match Training EOS Token

This is the most important issue to check.

The training template uses:

```python
stop_words=["<|end|>"]
replace_eos=True
```

This means the model is trained to end assistant responses with:

```text
<|end|>
```

However, Phi tokenizer usually has:

| Token | Role |
|---|---|
| `<|endoftext|>` | tokenizer `eos_token` / `pad_token` |
| `<|end|>` | chat turn delimiter |

If evaluation uses:

```python
eos_token_id=processor.tokenizer.eos_token_id
```

then generation waits for `<|endoftext|>`, not `<|end|>`.

Expected bad behavior:

```text
forward 3m<|end|> turn left forward 3m ...
```

The model may already output the correct `<|end|>` token, but `generate()` does not stop because it is waiting for `<|endoftext|>`.

Recommended inference fix:

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

Also decode with special tokens preserved during debugging:

```python
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Key Issue 2: Action Parser May Collapse Messy Generations into `forward 3m`

If `parse_action_text()` uses loose substring matching, it may incorrectly return forward even when the model generated multiple commands.

Risky parser pattern:

```python
if "forward" in text:
    return 1
if "turn left" in text:
    return 2
```

Bad example:

```text
turn left<|end|> forward 3m
```

If parser checks `forward` first, this becomes:

```text
forward 3m
```

Recommended parser strategy:

1. Decode with `skip_special_tokens=False`.
2. Keep only newly generated text, not the full prompt.
3. Truncate at `<|end|>` or `<|endoftext|>`.
4. Use exact matching first.
5. Avoid defaulting unknown text to `forward 3m`.

Example:

```python
def clean_action_text(text: str) -> str:
    text = text.lower().strip()
    text = text.split("<|end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def parse_action_text(text: str) -> int:
    text = clean_action_text(text)

    valid = {
        "stop": 0,
        "forward 3m": 1,
        "turn left": 2,
        "turn left 30 degrees": 2,
        "turn right": 3,
        "turn right 30 degrees": 3,
        "go up 3m": 4,
        "ascend 3m": 4,
        "go down 3m": 5,
        "descend 3m": 5,
    }

    return valid.get(text, 0)
```

During debugging, log both:

```text
raw generated text
parsed action id
parsed action text
```

## Key Issue 3: Training Data May Be Heavily Biased Toward `forward 3m`

Navigation trajectories usually contain many forward actions and fewer turning, vertical, or stop actions.

If `forward 3m` dominates the labels, the model may learn a strong action prior:

```text
When uncertain, output forward 3m.
```

Please add a data distribution check before training:

```python
from collections import Counter
import json

counter = Counter()

with open("TRAIN_DATA.json", "r") as f:
    data = json.load(f)

for item in data:
    answer = item["messages"][-1]["content"].strip().lower()
    counter[answer] += 1

print(counter)
print(counter.most_common())
```

If `forward 3m` is much larger than other classes, consider:

- creating a balanced smoke subset;
- oversampling `turn left`, `turn right`, `stop`, `go up 3m`, `go down 3m`;
- reducing consecutive forward-only samples;
- reporting both original and balanced results.

## Key Issue 4: Prompt Does Not Explicitly List the Valid Action Space

Current prompt:

```text
What action should the UAV take to find person? Reply with exactly one command.
```

This is underspecified. The model is not told what the valid command set is.

Recommended prompt:

```python
PROMPT_TEMPLATE = (
    "<image>\n"
    "What action should the UAV take to find {target_description}? "
    "Choose exactly one command from: "
    "forward 3m, turn left, turn right, go up 3m, go down 3m, stop."
)
```

If the code action space includes extended actions, include them too:

```text
forward 6m, forward 9m, move left 3m, move right 3m
```

Make sure the listed actions exactly match training labels and parser labels.

## Key Issue 5: Verify Images Really Enter the Training Batch

Since Phi-3.5-Vision relies on `<|image_1|>` and processor-generated image fields, confirm that the training batch contains image tensors.

Expected batch keys include:

```text
input_ids
attention_mask
labels
pixel_values
image_sizes
```

Add temporary debug logging near collation/model input:

```python
print(batch.keys())
print(batch["input_ids"].shape)
print(batch["pixel_values"].shape if "pixel_values" in batch else None)
print(batch["image_sizes"] if "image_sizes" in batch else None)
```

If `pixel_values` or `image_sizes` are missing, the model is likely training as text-only. In that case, it will fall back to the most common text-label prior, often `forward 3m`.

## Key Issue 6: Check Label Masking

LLaMA-Factory should compute loss only on assistant response tokens.

Decode active labels:

```python
labels = batch["labels"][0]
valid_ids = labels[labels != -100]
print(tokenizer.decode(valid_ids, skip_special_tokens=False))
```

Expected output:

```text
forward 3m<|end|>
```

or:

```text
turn left<|end|>
```

Suspicious outputs:

```text
empty string
full user prompt
mostly forward 3m only
missing <|end|>
wrong special token
```

## Key Issue 7: Possible `actions.py` File Mix-up

The pasted `actions.py` content is identical to `llamafactory_phi3v.py`.

This is suspicious.

An `actions.py` file should normally contain action mappings such as:

```python
ACTION_ID_TO_TEXT = {
    0: "stop",
    1: "forward 3m",
    2: "turn left",
    3: "turn right",
    4: "go up 3m",
    5: "go down 3m",
}

TEXT_TO_ACTION_ID = {v: k for k, v in ACTION_ID_TO_TEXT.items()}
```

If the repository imports `actions.py` for dataset conversion or evaluation parsing, but the file actually contains the Phi3V launcher, the action mapping may be missing or replaced by unrelated logic.

Please check:

```bash
rg -n "from actions|import actions|ACTION_ID|TEXT_TO_ACTION|parse_action" .
```

## Key Issue 8: Train/Eval Prompt Equivalence Looks Mostly Correct

The core prompt text appears aligned:

Training:

```text
<image>
What action should the UAV take to find person? Reply with exactly one command.
```

then plugin converts:

```text
<image> -> <|image_1|>
```

Inference should produce the same final text:

```text
<|image_1|>
What action should the UAV take to find person? Reply with exactly one command.
```

The chat template also appears consistent with Phi-3.5-Vision:

```text
<|user|>
{content}<|end|>
<|assistant|>
```

Therefore the main bug is probably not the visible question text, but one of:

1. stop token mismatch;
2. parser behavior;
3. action distribution bias;
4. missing image tensors;
5. bad label masking;
6. action mapping file mix-up.

## Recommended Debug Order

1. Print raw generated text with `skip_special_tokens=False`.
2. Change inference `eos_token_id` to include `<|end|>`.
3. Print generated-only text, not the whole prompt.
4. Log parsed action id and raw action text.
5. Audit `parse_action_text()` for substring/default-forward behavior.
6. Count action label distribution in the training data.
7. Decode `labels != -100` from a training batch.
8. Verify training batch has `pixel_values` and `image_sizes`.
9. Confirm `actions.py` has not been accidentally overwritten.
10. Consider adding the valid action list to `PROMPT_TEMPLATE`.

## Most Likely Explanation for "Always forward 3m"

The most likely combined explanation is:

```text
The model learns a strong forward-action prior from imbalanced trajectory data.
At inference, <|end|> is not used as eos_token_id, so generation may continue past the first answer.
The parser then sees "forward" somewhere in the generated text or defaults to forward, causing most steps to become forward 3m.
```

Fixing only one part may not solve the symptom. The generation stopping logic, parser, and action distribution should be checked together.

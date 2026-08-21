from __future__ import annotations


ACTION_SPACE_TEXT = (
    "stop, forward 3m, turn left 30 degree, turn right 30 degree, "
    "ascend 3m, descend 3m"
)

PROMPT_TEMPLATE = (
    "<image>\n"
    "What action should the UAV take to find {target_description}? "
    "Choose exactly one command from: {action_space}. "
    "Reply with only the command."
)

DEPTH_AUGMENTED_PROMPT_TEMPLATE = (
    "<image>\n"
    "Target: {target_description}\n\n"
    "{context}\n\n"
    "What action should the UAV take to find the target? "
    "Choose exactly one command from: {action_space}. "
    "Reply with only the command."
)


def build_prompt(
    target_description: str,
    depth_context: str | None = None,
    memory_context: str | None = None,
) -> str:
    description = target_description.strip().lower().rstrip(" .")
    context_parts = []
    if depth_context:
        context_parts.append(depth_context.strip())
    if memory_context:
        context_parts.append(memory_context.strip())
    if context_parts:
        return DEPTH_AUGMENTED_PROMPT_TEMPLATE.format(
            target_description=description,
            context="\n\n".join(context_parts),
            action_space=ACTION_SPACE_TEXT,
        )
    return PROMPT_TEMPLATE.format(target_description=description, action_space=ACTION_SPACE_TEXT)

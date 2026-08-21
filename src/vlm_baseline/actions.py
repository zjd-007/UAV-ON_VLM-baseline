from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ACTION_COMMANDS = {
    "Stop": "stop",
    "Move Forward": "forward 3m",
    "Turn Left": "turn left 30 degree",
    "Turn Right": "turn right 30 degree",
    "Ascend": "ascend 3m",
    "Descend": "descend 3m",
}

ACTION_IDS = {
    "stop": 0,
    "forward 3m": 1,
    "turn left 30 degree": 2,
    "turn right 30 degree": 3,
    "ascend 3m": 4,
    "descend 3m": 5,
}

ACTION_VECTORS = {
    "Stop": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "Move Forward": np.asarray([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "Turn Left": np.asarray([0, 0, 30, 0, 0, 0, 0, 0], dtype=np.float32),
    "Turn Right": np.asarray([0, 0, 0, 30, 0, 0, 0, 0], dtype=np.float32),
    "Ascend": np.asarray([0, 0, 0, 0, 3, 0, 0, 0], dtype=np.float32),
    "Descend": np.asarray([0, 0, 0, 0, 0, 3, 0, 0], dtype=np.float32),
}


@dataclass(frozen=True)
class ParsedAction:
    command: str
    action_id: int
    matched: bool


def action_name_to_command(action_name: str) -> str:
    try:
        return ACTION_COMMANDS[action_name]
    except KeyError as exc:
        raise ValueError(f"Unknown UAV-ON action name: {action_name!r}") from exc


def action_vector_to_name(action_vector: Iterable[float]) -> str:
    vector = np.asarray(list(action_vector), dtype=np.float32)
    if vector.shape != (8,):
        raise ValueError(f"Expected 8-D action vector, got shape {vector.shape}")
    for name, expected in ACTION_VECTORS.items():
        if np.allclose(vector, expected, atol=1e-4):
            return name
    raise ValueError(f"Unknown UAV-ON action vector: {vector.tolist()}")


def action_vector_to_command(action_vector: Iterable[float]) -> str:
    return action_name_to_command(action_vector_to_name(action_vector))


def normalize_command(text: str) -> str:
    text = text.strip().lower()
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"[`\"'.,;:!?]", " ", text)
    text = re.sub(r"\bmeters?\b", "m", text)
    text = re.sub(r"\bmetres?\b", "m", text)
    text = re.sub(r"\bdegrees?\b", "degree", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("3 m", "3m")

    if re.search(r"\bstop\b", text):
        return "stop"
    if re.search(r"\bforward\b", text) or re.search(r"\bmove forward\b", text):
        return "forward 3m"
    if re.search(r"\bturn left\b", text) or re.search(r"\bleft turn\b", text):
        return "turn left 30 degree"
    if re.search(r"\bturn right\b", text) or re.search(r"\bright turn\b", text):
        return "turn right 30 degree"
    if re.search(r"\bascend\b", text) or re.search(r"\bgo up\b", text) or re.search(r"\bup\b", text):
        return "ascend 3m"
    if re.search(r"\bdescend\b", text) or re.search(r"\bgo down\b", text) or re.search(r"\bdown\b", text):
        return "descend 3m"
    return text


def parse_action_text(text: str) -> ParsedAction:
    command = normalize_command(text)
    if command in ACTION_IDS:
        return ParsedAction(command=command, action_id=ACTION_IDS[command], matched=True)
    return ParsedAction(command="stop", action_id=ACTION_IDS["stop"], matched=False)

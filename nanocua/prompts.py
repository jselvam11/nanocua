"""Prompt templates: how a sample becomes the text a VLM sees.

Keep this file boring and short. Changing the prompt is the cheapest
experiment you can run — swap the system text or the history format here
without touching the trainer.

Default target the model must emit::

    Thought: <why this action>
    Action: <pyautogui-style call>
"""

from __future__ import annotations

import re
from typing import Any

from nanocua.schema import Action, SFTSample, Step, Trajectory

SYSTEM_PROMPT = """You are a computer-use agent. You receive a goal, the actions you already took, and a screenshot of the current desktop.

Reply with exactly two blocks:
Thought: <brief reasoning about what you see and what to do next>
Action: <a single call such as click(x=0.12, y=0.40), type(text="hello"), hotkey(keys=["enter"]), scroll(dx=0, dy=-2), wait(seconds=1), or terminate(status="success")>

Coordinates are normalized to [0, 1] with origin at the top-left of the screenshot.
Output one action only."""

ASSISTANT_TEMPLATE = "Thought: {thought}\nAction: {action}"

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\n\s*Action:|\Z)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(.+)$", re.DOTALL | re.IGNORECASE)


def format_history(history: list[str]) -> str:
    if not history:
        return "(none — this is the first step)"
    return "\n".join(f"{i}. {line}" for i, line in enumerate(history))


def history_from_steps(steps: list[Step], up_to: int) -> list[str]:
    """Build a text history from steps ``0 .. up_to-1``.

    Default: one line per past action (optionally with the thought). This is
    the usual cheap prefix. Multi-image history (past screenshots in the
    prompt) is a common upgrade — see the TODO in ``data/expand.py``.
    """
    lines: list[str] = []
    for step in steps[:up_to]:
        action = step.action.to_string()
        thought = (step.thought or "").strip()
        if thought:
            lines.append(f"{action}  # {thought}")
        else:
            lines.append(action)
    return lines


def format_user_message(sample: SFTSample, *, include_image_token: bool = True) -> str:
    image = "<image>\n" if include_image_token else ""
    return (
        f"{image}"
        f"Goal: {sample.goal}\n"
        f"History:\n{format_history(sample.history)}"
    )


def format_assistant_message(sample: SFTSample) -> str:
    return ASSISTANT_TEMPLATE.format(
        thought=sample.thought.strip(),
        action=sample.action.to_string(),
    )


def format_chat(sample: SFTSample, *, include_image_token: bool = True) -> list[dict[str, str]]:
    """ShareGPT-ish / TRL-friendly chat turns (text only; image is sidecar)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(sample, include_image_token=include_image_token)},
        {"role": "assistant", "content": format_assistant_message(sample)},
    ]


def messages_for_vlm(sample: SFTSample) -> list[dict[str, Any]]:
    """OpenAI-style multimodal messages (image + text) for processors / TRL."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.screenshot},
                {"type": "text", "text": format_user_message(sample, include_image_token=False)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": format_assistant_message(sample)}],
        },
    ]


def parse_assistant_message(text: str) -> tuple[str, Action]:
    """Inverse of ``format_assistant_message``. Used by offline eval."""
    text = text.strip()
    thought_match = _THOUGHT_RE.search(text)
    action_match = _ACTION_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    if action_match:
        raw_action = action_match.group(1).strip().splitlines()[0].strip()
    else:
        # Bare action line, no "Action:" prefix.
        raw_action = text.splitlines()[-1].strip()
    return thought, Action.from_string(raw_action)


def gold_assistant_for_step(traj: Trajectory, index: int) -> str:
    step = traj.steps[index]
    return ASSISTANT_TEMPLATE.format(thought=step.thought.strip(), action=step.action.to_string())

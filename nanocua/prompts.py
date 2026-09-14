"""Prompt templates: how a sample becomes the text a VLM sees.

Keep this file boring and short. Changing the prompt is the cheapest
experiment you can run — swap the system text or the history format here
without touching the trainer.

Default SFT target the model must emit::

    Thought: <why this action>
    Action: <pyautogui-style call>

Default grounding target (Stage-1 localize)::

    point(x=0.52, y=0.08)
"""

from __future__ import annotations

import re
from typing import Any

from nanocua.schema import Action, GroundingExample, SFTSample, Step, Trajectory

SYSTEM_PROMPT = """You are a computer-use agent. You receive a goal, the actions you already took, and a screenshot of the current desktop.

Reply with exactly two blocks:
Thought: <brief reasoning about what you see and what to do next>
Action: <a single call such as click(x=0.12, y=0.40), type(text="hello"), hotkey(keys=["enter"]), scroll(dx=0, dy=-2), wait(seconds=1), or terminate(status="success")>

Coordinates are normalized to [0, 1] with origin at the top-left of the screenshot.
Output one action only."""

ASSISTANT_TEMPLATE = "Thought: {thought}\nAction: {action}"

# Stage-1 sibling: localize, do not plan a multi-step episode.
GROUNDING_SYSTEM_PROMPT = """You locate UI elements on a screenshot. You receive an image and a referring expression (what to find).

Reply with exactly one localization:
point(x=<0-1>, y=<0-1>)
or
bbox(x1=<0-1>, y1=<0-1>, x2=<0-1>, y2=<0-1>)

Coordinates are normalized to [0, 1] with origin at the top-left of the screenshot.
Prefer a point when you can. Output one localization only."""

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\n\s*Action:|\Z)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(.+)$", re.DOTALL | re.IGNORECASE)
_POINT_PAIR_RE = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*(?:,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*)?\]"
)


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


def format_grounding_user(example: GroundingExample, *, include_image_token: bool = True) -> str:
    image = "<image>\n" if include_image_token else ""
    return f"{image}Instruction: {example.instruction}"


def format_grounding_assistant(example: GroundingExample) -> str:
    return example.target_string()


def format_grounding_chat(
    example: GroundingExample, *, include_image_token: bool = True
) -> list[dict[str, str]]:
    """ShareGPT-ish chat turns for a grounding triple (text only; image is sidecar)."""
    return [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_grounding_user(example, include_image_token=include_image_token),
        },
        {"role": "assistant", "content": format_grounding_assistant(example)},
    ]


def parse_grounding_prediction(text: str) -> Action:
    """Inverse of ``format_grounding_assistant``.

    Accepts ``point(...)``, ``bbox(...)``, a CUA ``click(...)``, an
    ``Action:`` line, or a SeeClick-ish ``[x, y]`` / ``[x1, y1, x2, y2]``.
    """
    text = text.strip()
    action_match = _ACTION_RE.search(text)
    raw = action_match.group(1).strip().splitlines()[0].strip() if action_match else ""
    candidates = [raw, text, text.splitlines()[-1].strip()] if raw else [text, text.splitlines()[-1].strip()]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return Action.from_string(candidate)
        except ValueError:
            continue
    pair = _POINT_PAIR_RE.search(text)
    if pair and pair.group(3) is None:
        return Action(type="point", args={"x": float(pair.group(1)), "y": float(pair.group(2))})
    if pair and pair.group(3) is not None:
        return Action(
            type="bbox",
            args={
                "x1": float(pair.group(1)),
                "y1": float(pair.group(2)),
                "x2": float(pair.group(3)),
                "y2": float(pair.group(4)),
            },
        )
    raise ValueError(f"not a grounding localization: {text!r}")

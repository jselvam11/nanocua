"""Export SFT samples to a simple ShareGPT-ish JSONL.

Many trainers (Axolotl, LLaMA-Factory, TRL recipes) accept a conversations
list plus an images list. We emit that and nothing else.

Example row::

    {
      "id": "fixture-weather-search-step0",
      "images": ["/abs/path/to/step_00.png"],
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "<image>\\nGoal: ..."},
        {"from": "gpt",    "value": "Thought: ...\\nAction: click(...)"}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from nanocua.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_assistant_message,
    format_grounding_assistant,
    format_grounding_user,
    format_user_message,
)
from nanocua.schema import GroundingExample, SFTSample


def sample_to_sharegpt(sample: SFTSample) -> dict:
    return {
        "id": f"{sample.trajectory_id}-step{sample.step_index}",
        "images": [sample.screenshot],
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": format_user_message(sample, include_image_token=True)},
            {"from": "gpt", "value": format_assistant_message(sample)},
        ],
        # Extra fields some TRL / datasets recipes look for:
        "goal": sample.goal,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(sample, include_image_token=True)},
            {"role": "assistant", "content": format_assistant_message(sample)},
        ],
    }


def grounding_to_sharegpt(example: GroundingExample) -> dict:
    return {
        "id": example.id,
        "images": [example.screenshot],
        "conversations": [
            {"from": "system", "value": GROUNDING_SYSTEM_PROMPT},
            {"from": "human", "value": format_grounding_user(example, include_image_token=True)},
            {"from": "gpt", "value": format_grounding_assistant(example)},
        ],
        "instruction": example.instruction,
        "messages": [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {"role": "user", "content": format_grounding_user(example, include_image_token=True)},
            {"role": "assistant", "content": format_grounding_assistant(example)},
        ],
    }


def export_jsonl(samples: list[SFTSample] | list[GroundingExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = (
                grounding_to_sharegpt(sample)
                if isinstance(sample, GroundingExample)
                else sample_to_sharegpt(sample)
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

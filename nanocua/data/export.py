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

from nanocua.prompts import format_assistant_message, format_user_message, SYSTEM_PROMPT
from nanocua.schema import SFTSample


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


def export_jsonl(samples: list[SFTSample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample_to_sharegpt(sample), ensure_ascii=False) + "\n")
    return path

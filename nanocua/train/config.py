"""Single training config: a dataclass plus optional YAML/JSON file.

YAML support is a tiny subset (``key: value`` plus comments) so the
base package does not depend on PyYAML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Knobs for the smoke SFT loop. Keep this list short on purpose."""

    # HuggingFaceTB/SmolVLM-256M-Instruct is a *learning* default: small
    # enough to download and run a few steps, not a competitive CUA policy.
    # Swap this string to try Qwen2-VL / InternVL / your fork.
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # "sft" = trajectory behavior cloning. "grounding" = Stage-1 GUI localize.
    task: str = "sft"
    data_path: str = ""  # empty → bundled fixture for the chosen task
    output_dir: str = "outputs/smoke"
    learning_rate: float = 1.0e-5
    batch_size: int = 1
    max_steps: int = 4
    image_size: int = 384
    max_length: int = 1024
    logging_steps: int = 1
    seed: int = 42
    gradient_accumulation_steps: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FIELD_TYPES = {item.name: item.type for item in fields(TrainConfig)}


def _coerce(name: str, value: Any) -> Any:
    hint = _FIELD_TYPES.get(name)
    if hint is int or hint == "int":
        return int(value)
    if hint is float or hint == "float":
        return float(value)
    if hint is str or hint == "str":
        return str(value)
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse ``key: value`` YAML. No nested maps, no lists — smoke configs only."""
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            data[key] = lowered == "true"
        elif lowered in {"null", "none", "~", ""}:
            data[key] = ""
        else:
            try:
                data[key] = int(value)
            except ValueError:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data


def load_train_config(path: str | Path | None = None, **overrides: Any) -> TrainConfig:
    payload: dict[str, Any] = {}
    if path is not None:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            payload = _parse_simple_yaml(text)
        else:
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise TypeError(f"config file must be a mapping: {file_path}")
            payload = loaded
    known = {item.name for item in fields(TrainConfig)}
    filtered = {key: _coerce(key, value) for key, value in payload.items() if key in known}
    for key, value in overrides.items():
        if key in known and value is not None:
            filtered[key] = _coerce(key, value)
    return TrainConfig(**filtered)

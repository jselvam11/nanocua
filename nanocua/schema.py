"""Canonical CUA records: Action, Step, Trajectory, SFTSample, GroundingExample.

A *trajectory* is one recorded attempt at a computer-use task: a goal plus an
ordered list of steps. Each *step* is what the agent saw (screenshot), thought,
and did (action) at that moment.

A *grounding example* is the Stage-1 sibling: one screenshot, one referring
expression, and a point and/or bbox. Papers call this GUI grounding pretrain
(usually continual pretrain of an already-trained VLM, not from scratch).
It is **not** a trajectory — there is no goal-over-time, no history.

This is the whole data model. Fork here if you want a richer action space
(drag, middle-click, …) or extra CoT fields (observation, reflection).

Coordinates
-----------
Mouse positions are **normalized to [0, 1]** relative to the screenshot
(top-left is ``(0, 0)``, bottom-right is ``(1, 1)``). That matches common
open CUA datasets (e.g. AgentNet-style ``pyautogui.click(x=0.16, y=0.27)``).
At execution time you multiply by the real screen width/height.

Action string
-------------
``Action.to_string()`` emits a pyautogui-style call::

    click(x=0.52, y=0.08)
    type(text="weather today")
    hotkey(keys=["ctrl", "l"])
    terminate(status="success")

The same string is what the VLM is trained to emit after ``Action:``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# A small, portable subset — enough to learn the loop, not a full OS driver.
# TODO: add drag, mouse_move, key_down/up when you need them.
ACTION_TYPES = (
    "click",
    "double_click",
    "right_click",
    "type",
    "hotkey",
    "scroll",
    "wait",
    "terminate",
)

_CALL_RE = re.compile(
    r"^(?:pyautogui\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$",
    re.DOTALL,
)


def _fmt_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        # Stable, short floats so exact-match eval is not ruined by 0.5200001.
        return f"{value:.4g}"
    if isinstance(value, list):
        inner = ", ".join(_fmt_value(v) for v in value)
        return f"[{inner}]"
    return repr(value)


@dataclass
class Action:
    """One computer-use action: a type plus JSON-serializable args."""

    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = str(self.type).strip()
        if not self.type:
            raise ValueError("Action.type must be non-empty")
        if not isinstance(self.args, dict):
            raise TypeError("Action.args must be a dict")

    def to_string(self) -> str:
        """Canonical pyautogui-style string used as the SFT target."""
        if not self.args:
            return f"{self.type}()"
        parts = []
        for key in sorted(self.args):
            parts.append(f"{key}={_fmt_value(self.args[key])}")
        return f"{self.type}({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> Action:
        if isinstance(data, str):
            return cls.from_string(data)
        if not isinstance(data, dict):
            raise TypeError(f"action must be dict or str, got {type(data)!r}")
        if "type" in data:
            args = data.get("args") or data.get("params") or {}
            return cls(type=data["type"], args=dict(args))
        # Single-key form: {"click": {"x": 0.2, "y": 0.3}}
        if len(data) == 1:
            kind, args = next(iter(data.items()))
            return cls(type=kind, args=dict(args or {}))
        raise ValueError(f"cannot parse action dict: {data!r}")

    @classmethod
    def from_string(cls, text: str) -> Action:
        """Parse ``click(x=0.5, y=0.2)`` or ``pyautogui.click(...)``."""
        text = text.strip()
        match = _CALL_RE.match(text)
        if not match:
            raise ValueError(f"not an action call: {text!r}")
        name, raw_args = match.group(1), match.group(2).strip()
        if not raw_args:
            return cls(type=name, args={})
        # Parse kwargs via the AST so we never eval() user strings.
        try:
            tree = ast.parse(f"f({raw_args})", mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"cannot parse action args: {text!r}") from exc
        call = tree.body
        if not isinstance(call, ast.Call) or call.args:
            raise ValueError(f"only keyword args are allowed: {text!r}")
        args: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError(f"no **kwargs in action strings: {text!r}")
            args[keyword.arg] = ast.literal_eval(keyword.value)
        return cls(type=name, args=args)


@dataclass
class Step:
    """One timestep in a trajectory.

    ``history`` is usually *derived* at expand-time from earlier steps. You
    may store it explicitly if you imported a dataset that already has it.
    """

    screenshot: str
    thought: str
    action: Action
    observation: str = ""
    index: int = 0
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "screenshot": self.screenshot,
            "observation": self.observation,
            "thought": self.thought,
            "action": self.action.to_dict(),
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_index: int = 0) -> Step:
        action = Action.from_dict(data.get("action", data.get("code", {})))
        return cls(
            screenshot=str(data.get("screenshot") or data.get("image") or ""),
            thought=str(data.get("thought") or data.get("reasoning") or ""),
            action=action,
            observation=str(data.get("observation") or ""),
            index=int(data.get("index", default_index)),
            history=list(data.get("history") or []),
        )


@dataclass
class Trajectory:
    """One recorded episode: a goal and the T steps taken to pursue it."""

    id: str
    goal: str
    steps: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        raw_steps = data.get("steps") or data.get("traj") or []
        steps = [Step.from_dict(item, default_index=i) for i, item in enumerate(raw_steps)]
        return cls(
            id=str(data.get("id") or data.get("task_id") or ""),
            goal=str(data.get("goal") or data.get("instruction") or data.get("task") or ""),
            steps=steps,
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> Trajectory:
        return cls.from_dict(json.loads(text))


@dataclass
class SFTSample:
    """One supervised example produced by step-prefix expansion.

    The model sees: goal + history (steps ``0 .. t-1``) + current screenshot.
    The model should produce: thought + action at step ``t``.
    """

    trajectory_id: str
    step_index: int
    goal: str
    screenshot: str
    history: list[str]
    thought: str
    action: Action

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.to_dict()
        return payload


def _unit(name: str, value: Any) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return number


def _as_point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValueError(f"point dict needs x and y: {value!r}")
        return (_unit("point.x", value["x"]), _unit("point.y", value["y"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_unit("point.x", value[0]), _unit("point.y", value[1]))
    raise TypeError(f"point must be [x, y] or {{x, y}}, got {value!r}")


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if {"x1", "y1", "x2", "y2"} <= set(value):
            x1, y1, x2, y2 = value["x1"], value["y1"], value["x2"], value["y2"]
        elif {"x", "y", "w", "h"} <= set(value):
            x1 = float(value["x"])
            y1 = float(value["y"])
            x2 = x1 + float(value["w"])
            y2 = y1 + float(value["h"])
        else:
            raise ValueError(f"bbox dict needs x1/y1/x2/y2 or x/y/w/h: {value!r}")
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = value
    else:
        raise TypeError(f"bbox must be 4 numbers or a dict, got {value!r}")
    box = (
        _unit("bbox.x1", x1),
        _unit("bbox.y1", y1),
        _unit("bbox.x2", x2),
        _unit("bbox.y2", y2),
    )
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError(f"bbox must have x2>=x1 and y2>=y1, got {box}")
    return box


@dataclass
class GroundingExample:
    """One GUI grounding triple: screenshot + referring expression → point/bbox.

    This is the Stage-1 CUA pretrain record. The VLM sees an image and a
    phrase like ``"the browser address bar"`` and must emit a localization
    in normalized coordinates — typically ``point(x=0.52, y=0.08)``.

    Provide a ``point``, a ``bbox`` ``(x1, y1, x2, y2)``, or both. Training
    defaults to the point (click-style). Eval uses the bbox for
    point-in-bbox when it is present.
    """

    id: str
    screenshot: str
    instruction: str
    point: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.instruction = str(self.instruction).strip()
        if not self.instruction:
            raise ValueError("GroundingExample.instruction must be non-empty")
        self.point = _as_point(self.point)
        self.bbox = _as_bbox(self.bbox)
        if self.point is None and self.bbox is None:
            raise ValueError("GroundingExample needs a point and/or a bbox")

    def gold_point(self) -> tuple[float, float]:
        """Explicit point, or the bbox center if only a box was given."""
        if self.point is not None:
            return self.point
        x1, y1, x2, y2 = self.bbox  # type: ignore[misc]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def as_click(self) -> Action:
        """The CUA action this localization becomes after Stage-2 SFT."""
        x, y = self.gold_point()
        return Action(type="click", args={"x": x, "y": y})

    def target_string(self) -> str:
        """Canonical assistant target: ``point(...)`` or ``bbox(...)``."""
        if self.point is not None:
            return Action(type="point", args={"x": self.point[0], "y": self.point[1]}).to_string()
        x1, y1, x2, y2 = self.bbox  # type: ignore[misc]
        return Action(
            type="bbox",
            args={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        ).to_string()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "screenshot": self.screenshot,
            "instruction": self.instruction,
            "metadata": dict(self.metadata),
        }
        if self.point is not None:
            payload["point"] = [self.point[0], self.point[1]]
        if self.bbox is not None:
            payload["bbox"] = [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundingExample:
        if not isinstance(data, dict):
            raise TypeError(f"grounding example must be a dict, got {type(data)!r}")
        return cls(
            id=str(data.get("id") or data.get("example_id") or ""),
            screenshot=str(data.get("screenshot") or data.get("image") or ""),
            instruction=str(
                data.get("instruction")
                or data.get("expression")
                or data.get("referring_expression")
                or data.get("query")
                or ""
            ),
            point=data.get("point", data.get("center")),
            bbox=data.get("bbox", data.get("box", data.get("bounds"))),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Observation:
    """What an environment returns after ``reset`` / ``step``.

    ``screenshot`` is a filesystem path (or any string ref your env understands).
    ``text`` is reserved for optional a11y / OCR later — unused in the MVP.
    """

    screenshot: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def dump_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([traj.to_dict() for traj in trajectories], indent=2) + "\n",
        encoding="utf-8",
    )


def load_trajectory_dicts(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("trajectories") or raw.get("data") or [raw]
    if not isinstance(raw, list):
        raise TypeError(f"expected a list of trajectories in {path}")
    return raw


def dump_grounding(examples: list[GroundingExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([example.to_dict() for example in examples], indent=2) + "\n",
        encoding="utf-8",
    )


def load_grounding_dicts(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("examples") or raw.get("grounding") or raw.get("data") or [raw]
    if not isinstance(raw, list):
        raise TypeError(f"expected a list of grounding examples in {path}")
    return raw

"""Sketch: plug a real computer in later.

TODO: OSWorld / Docker Ubuntu / raw pyautogui adapters.

This module is intentionally **not** imported by ``nanocua.env`` so a
base install never requires those stacks. Copy it into your fork when
you have a machine you are willing to let an agent drive.
"""

from __future__ import annotations

from typing import Any

from nanocua.schema import Action, Observation


class PyAutoGUIAdapter:
    """Illustrative adapter — methods raise until you fill them in.

    Expected mapping (normalized coords → pixels)::

        click(x, y)  -> pyautogui.click(x * width, y * height)
        type(text)   -> pyautogui.write(text)
        hotkey(keys) -> pyautogui.hotkey(*keys)
        scroll(...)  -> pyautogui.scroll(...)
        terminate    -> end the episode (do not call pyautogui)
    """

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height

    def reset(self, task: str) -> Observation:  # pragma: no cover - sketch
        raise NotImplementedError(
            "TODO: take a real screenshot (mss / pyautogui.screenshot), "
            f"save it, return Observation. task={task!r}"
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError(
            "TODO: dispatch action.to_string() through pyautogui, then screenshot. "
            f"Got {action.to_string()}"
        )


class OSWorldAdapter:
    """TODO: wrap an OSWorld / desktop-in-Docker env behind ComputerEnv.

    Do not import OSWorld at module top-level. Keep that dependency inside
    the method that constructs the real env, so ``import nanocua`` stays light.
    """

    def reset(self, task: str) -> Observation:  # pragma: no cover - sketch
        raise NotImplementedError("TODO: OSWorld reset")

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError("TODO: OSWorld step")

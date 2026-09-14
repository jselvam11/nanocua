"""The only env contract nanocua needs: reset(task) and step(action).

This is *not* an agent framework. A real desktop (OSWorld, a Docker Ubuntu,
pyautogui on your laptop) just has to speak this protocol. See
``nanocua/env/adapter.py`` for a sketch that is not imported by default.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nanocua.schema import Action, Observation


@runtime_checkable
class ComputerEnv(Protocol):
    """Minimal computer-use environment.

    ``reset(task)`` → first observation (usually a screenshot).
    ``step(action)`` → next observation, reward, done, info.

    Reward / done semantics are env-specific. The mock env uses a step
    budget; a real benchmark would use task success.
    """

    def reset(self, task: str) -> Observation:
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        ...

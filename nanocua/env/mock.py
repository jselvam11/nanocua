"""In-process fake desktop: dummy screenshots, no OS, no Docker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanocua.data.images import color_for_index, write_dummy_png
from nanocua.schema import Action, Observation


class MockComputerEnv:
    """Cycles through solid-color frames so unit tests need no display.

    Reward is 0 until ``terminate`` or ``max_steps``, then 1.0 if the agent
    terminated with ``status=success``. This is a stub, not a benchmark.
    """

    def __init__(
        self,
        *,
        max_steps: int = 4,
        frame_dir: str | Path | None = None,
        size: int = 64,
    ) -> None:
        self.max_steps = max_steps
        self.size = size
        self.frame_dir = Path(frame_dir) if frame_dir else Path("outputs/mock_frames")
        self.task = ""
        self.t = 0
        self.done = False
        self.last_action: Action | None = None

    def _frame(self) -> Path:
        path = self.frame_dir / f"t{self.t:02d}.png"
        return write_dummy_png(path, rgb=color_for_index(self.t), size=self.size)

    def reset(self, task: str) -> Observation:
        self.task = task
        self.t = 0
        self.done = False
        self.last_action = None
        screenshot = self._frame()
        return Observation(screenshot=str(screenshot), metadata={"task": task, "t": 0})

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("env is done; call reset()")
        self.last_action = action
        self.t += 1
        terminated = action.type == "terminate"
        timeout = self.t >= self.max_steps
        self.done = terminated or timeout
        reward = 0.0
        if terminated and str(action.args.get("status", "")).lower() == "success":
            reward = 1.0
        screenshot = self._frame()
        info = {
            "t": self.t,
            "terminated": terminated,
            "timeout": timeout,
            "action": action.to_string(),
        }
        obs = Observation(screenshot=str(screenshot), metadata={"task": self.task, "t": self.t})
        return obs, reward, self.done, info

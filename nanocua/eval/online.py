"""Online eval stub: run a policy against any ``ComputerEnv`` for N steps.

This is here so the mental model has a closed loop. The mock env does not
score real computer-use success — it just checks that reset/step/policy
wire together. Swap in a real env (see ``nanocua.env.adapter``) later.

TODO: GRPO / RL rollouts would collect these transitions as trajectories
and feed them back into training. Not implemented.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from nanocua.env.protocol import ComputerEnv
from nanocua.schema import Action, Observation


class Policy(Protocol):
    def __call__(self, obs: Observation, goal: str, history: list[Action]) -> Action:
        ...


def always_wait(_obs: Observation, _goal: str, _history: list[Action]) -> Action:
    return Action(type="wait", args={"seconds": 1})


def online_eval(
    env: ComputerEnv,
    policy: Callable[[Observation, str, list[Action]], Action] | Policy,
    task: str,
    *,
    max_steps: int = 4,
) -> dict[str, Any]:
    obs = env.reset(task)
    history: list[Action] = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    steps_run = 0
    for _ in range(max_steps):
        action = policy(obs, task, history)
        obs, reward, done, info = env.step(action)
        history.append(action)
        total_reward += float(reward)
        steps_run += 1
        if done:
            break
    return {
        "task": task,
        "steps": steps_run,
        "reward": total_reward,
        "done": done,
        "info": info,
        "actions": [action.to_string() for action in history],
    }

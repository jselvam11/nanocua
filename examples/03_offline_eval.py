"""Offline eval on the bundled fixture (no GPU, no network).

    python examples/03_offline_eval.py

Prints gold-copy scores (should be 1.0) and a dummy always-wait policy
(should be near 0.0 except type-match on any wait steps — the fixture has none).
"""

from __future__ import annotations

from nanocua.data import load_trajectories
from nanocua.eval import offline_eval, online_eval
from nanocua.eval.online import always_wait
from nanocua.env import MockComputerEnv
from nanocua.schema import Action, SFTSample


def wait_predictor(_sample: SFTSample) -> Action:
    return Action(type="wait", args={"seconds": 1})


def slightly_noisy(sample: SFTSample) -> Action:
    """Jitter click coords to show normalized match vs exact match."""
    action = Action(type=sample.action.type, args=dict(sample.action.args))
    if "x" in action.args:
        action.args["x"] = float(action.args["x"]) + 0.001
        action.args["y"] = float(action.args["y"]) + 0.001
    return action


def main() -> None:
    trajectories = load_trajectories()
    gold = offline_eval(trajectories)
    noisy = offline_eval(trajectories, predictor=slightly_noisy)
    wait = offline_eval(trajectories, predictor=wait_predictor)
    print("gold copy     ", f"exact={gold['exact']:.2f}  normalized={gold['normalized']:.2f}")
    print("noisy clicks  ", f"exact={noisy['exact']:.2f}  normalized={noisy['normalized']:.2f}")
    print("always wait   ", f"exact={wait['exact']:.2f}  normalized={wait['normalized']:.2f}  type={wait['type']:.2f}")

    env = MockComputerEnv(max_steps=3)
    online = online_eval(env, always_wait, task="demo task", max_steps=3)
    print("online mock   ", online)


if __name__ == "__main__":
    main()

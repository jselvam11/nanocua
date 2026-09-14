"""Step-prefix expansion: the core CUA-SFT transform.

A trajectory of T steps becomes up to T supervised samples::

    sample t = (goal + history[0:t] + screenshot_t)  ->  (thought_t, action_t)

That is behavior cloning: at every prefix of the episode, predict the
action the demonstrator actually took. This is what almost every CUA SFT
recipe does before anyone talks about RL.

TODO: include previous screenshots in the prompt (multi-image history).
TODO: drop redundant / incorrect steps when a dataset marks them.
"""

from __future__ import annotations

from nanocua.prompts import history_from_steps
from nanocua.schema import SFTSample, Trajectory


def expand_trajectory(traj: Trajectory, *, include_terminal: bool = True) -> list[SFTSample]:
    samples: list[SFTSample] = []
    for index, step in enumerate(traj.steps):
        if not include_terminal and step.action.type == "terminate":
            continue
        history = step.history or history_from_steps(traj.steps, index)
        samples.append(
            SFTSample(
                trajectory_id=traj.id,
                step_index=step.index if step.index else index,
                goal=traj.goal,
                screenshot=step.screenshot,
                history=list(history),
                thought=step.thought,
                action=step.action,
            )
        )
    return samples


def expand_trajectories(
    trajectories: list[Trajectory],
    *,
    include_terminal: bool = True,
) -> list[SFTSample]:
    samples: list[SFTSample] = []
    for traj in trajectories:
        samples.extend(expand_trajectory(traj, include_terminal=include_terminal))
    return samples

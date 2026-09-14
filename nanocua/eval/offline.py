"""Offline eval: predicted action vs gold action on recorded trajectories.

Two scores, both computed per step then averaged:

* **exact** — canonical ``Action.to_string()`` equality
  (``click(x=0.52, y=0.08)`` vs the gold string).
* **normalized** — same action type, numeric args rounded, text stripped.
  ``click(x=0.5201, y=0.0802)`` matches gold ``click(x=0.52, y=0.08)``.

This is *not* task success. It is a cheap sanity check that a model still
emits the demonstrator actions on the training distribution. Real CUA
papers report online success on OSWorld / AndroidWorld / WebArena, which
this skeleton does not run.

TODO: element-level matching (click inside a gold bbox) when you have SOMs.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from nanocua.data.expand import expand_trajectories
from nanocua.prompts import parse_assistant_message
from nanocua.schema import Action, SFTSample, Trajectory


def _norm_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_norm_value(v) for v in value]
    return value


def normalize_action(action: Action) -> tuple[str, tuple[tuple[str, Any], ...]]:
    items = tuple(sorted((k, _norm_value(v)) for k, v in action.args.items()))
    return action.type.strip().lower(), items


def match_actions(predicted: Action, gold: Action) -> dict[str, bool]:
    return {
        "exact": predicted.to_string() == gold.to_string(),
        "normalized": normalize_action(predicted) == normalize_action(gold),
        "type": predicted.type.strip().lower() == gold.type.strip().lower(),
    }


def as_action(predicted: Action | str) -> Action:
    if isinstance(predicted, Action):
        return predicted
    text = predicted.strip()
    try:
        return Action.from_string(text)
    except ValueError:
        _, action = parse_assistant_message(text)
        return action


def offline_eval(
    trajectories: Iterable[Trajectory],
    predictions: list[Action | str] | None = None,
    predictor: Callable[[SFTSample], Action | str] | None = None,
) -> dict[str, Any]:
    """Score predictions against gold actions.

    Pass either a list of predictions aligned with the expanded samples,
    or a ``predictor(sample) -> Action | str`` callable. If both are
    omitted, a copy-the-gold predictor is used (sanity check → 1.0).
    """
    if predictions is not None and predictor is not None:
        raise ValueError("pass predictions or predictor, not both")

    samples = expand_trajectories(list(trajectories))
    if predictions is not None and len(predictions) != len(samples):
        raise ValueError(f"{len(predictions)} predictions for {len(samples)} samples")

    rows: list[dict[str, Any]] = []
    exact_n = 0
    norm_n = 0
    type_n = 0
    errors = 0
    for index, sample in enumerate(samples):
        gold = sample.action
        try:
            if predictions is not None:
                raw = predictions[index]
            elif predictor is not None:
                raw = predictor(sample)
            else:
                raw = sample.action
            predicted = as_action(raw)
            scores = match_actions(predicted, gold)
        except (ValueError, TypeError) as exc:
            errors += 1
            rows.append(
                {
                    "id": f"{sample.trajectory_id}-step{sample.step_index}",
                    "gold": gold.to_string(),
                    "predicted": None,
                    "error": str(exc),
                    "exact": False,
                    "normalized": False,
                    "type": False,
                }
            )
            continue
        exact_n += int(scores["exact"])
        norm_n += int(scores["normalized"])
        type_n += int(scores["type"])
        rows.append(
            {
                "id": f"{sample.trajectory_id}-step{sample.step_index}",
                "gold": gold.to_string(),
                "predicted": predicted.to_string(),
                "exact": scores["exact"],
                "normalized": scores["normalized"],
                "type": scores["type"],
            }
        )

    n = len(samples)
    denom = n if n else 1
    return {
        "n": n,
        "parse_errors": errors,
        "exact": exact_n / denom,
        "normalized": norm_n / denom,
        "type": type_n / denom,
        "rows": rows,
    }

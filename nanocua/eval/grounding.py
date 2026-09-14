"""Offline GUI grounding eval: did the model point at the right widget?

Two cheap scores, both on normalized [0, 1] coordinates:

* **point_in_bbox** — predicted point falls inside the gold bbox.
  This is the usual "center accuracy" / loc-acc number in grounding papers
  when you have boxes. Skipped for examples that have no gold bbox.
* **point_acc** — L2 distance from predicted point to gold point
  ``<= point_threshold`` (default 0.05 ≈ 5% of the image width).

This is *not* OS-Atlas / ScreenSpot / SeeClick benchmark score. It is a
sanity check on whatever JSON you pass in (the bundled fixture is four
synthetic boxes). Gold-copy should print 1.0; a policy that always points
at the bottom-right should print 0.0.

TODO: ScreenSpot-style split metrics and IoU@0.5 when you load a real set.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from nanocua.prompts import parse_grounding_prediction
from nanocua.schema import Action, GroundingExample

DEFAULT_POINT_THRESHOLD = 0.05

_POINT_TYPES = {"point", "click", "double_click", "right_click"}


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def action_to_point(action: Action) -> tuple[float, float]:
    """Click-like calls keep (x, y); a bbox becomes its center."""
    kind = action.type.strip().lower()
    if kind in _POINT_TYPES:
        if "x" not in action.args or "y" not in action.args:
            raise ValueError(f"point-like action missing x/y: {action.to_string()}")
        return float(action.args["x"]), float(action.args["y"])
    if kind in {"bbox", "box"}:
        if {"x1", "y1", "x2", "y2"} <= set(action.args):
            x1 = float(action.args["x1"])
            y1 = float(action.args["y1"])
            x2 = float(action.args["x2"])
            y2 = float(action.args["y2"])
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        raise ValueError(f"bbox action needs x1/y1/x2/y2: {action.to_string()}")
    raise ValueError(f"cannot read a point from {action.to_string()}")


def as_grounding_action(predicted: Action | str) -> Action:
    if isinstance(predicted, Action):
        return predicted
    return parse_grounding_prediction(predicted)


def grounding_eval(
    examples: Iterable[GroundingExample],
    predictions: list[Action | str] | None = None,
    predictor: Callable[[GroundingExample], Action | str] | None = None,
    *,
    point_threshold: float = DEFAULT_POINT_THRESHOLD,
) -> dict[str, Any]:
    """Score predicted localizations against gold points/boxes.

    Pass either a list of predictions aligned with ``examples``, or a
    ``predictor(example) -> Action | str`` callable. If both are omitted,
    a copy-the-gold predictor is used (sanity check → 1.0).
    """
    if predictions is not None and predictor is not None:
        raise ValueError("pass predictions or predictor, not both")

    items = list(examples)
    if predictions is not None and len(predictions) != len(items):
        raise ValueError(f"{len(predictions)} predictions for {len(items)} examples")

    rows: list[dict[str, Any]] = []
    pib_hits = 0
    pib_n = 0
    dist_hits = 0
    dist_n = 0
    dist_sum = 0.0
    errors = 0

    for index, example in enumerate(items):
        gold_point = example.gold_point()
        try:
            if predictions is not None:
                raw = predictions[index]
            elif predictor is not None:
                raw = predictor(example)
            else:
                raw = example.target_string()
            predicted = as_grounding_action(raw)
            pred_point = action_to_point(predicted)
        except (ValueError, TypeError) as exc:
            errors += 1
            rows.append(
                {
                    "id": example.id,
                    "gold": example.target_string(),
                    "predicted": None,
                    "error": str(exc),
                    "point_in_bbox": False,
                    "point_acc": False,
                    "distance": None,
                }
            )
            continue

        distance = point_distance(pred_point, gold_point)
        acc = distance <= point_threshold
        dist_n += 1
        dist_sum += distance
        dist_hits += int(acc)

        inside: bool | None = None
        if example.bbox is not None:
            inside = point_in_bbox(pred_point, example.bbox)
            pib_n += 1
            pib_hits += int(inside)

        rows.append(
            {
                "id": example.id,
                "gold": example.target_string(),
                "predicted": predicted.to_string(),
                "pred_point": [pred_point[0], pred_point[1]],
                "point_in_bbox": inside,
                "point_acc": acc,
                "distance": distance,
            }
        )

    def _rate(hits: int, n: int) -> float | None:
        if n == 0:
            return None
        return hits / n

    return {
        "n": len(items),
        "parse_errors": errors,
        "point_in_bbox": _rate(pib_hits, pib_n),
        "point_acc": _rate(dist_hits, dist_n),
        "mean_distance": (dist_sum / dist_n) if dist_n else None,
        "point_threshold": point_threshold,
        "n_bbox": pib_n,
        "n_point": dist_n,
        "rows": rows,
    }

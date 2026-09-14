"""CLI: ``python -m nanocua.eval`` or ``nanocua-eval`` (offline by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanocua.data.load import load_grounding_examples, load_trajectories
from nanocua.eval.grounding import grounding_eval
from nanocua.eval.offline import as_action, offline_eval
from nanocua.schema import Action


def _row_to_prediction(row: object) -> Action | str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if "action" in row:
            action = row["action"]
            return Action.from_dict(action) if isinstance(action, dict) else as_action(str(action))
        if "prediction" in row:
            return str(row["prediction"])
        if "point" in row:
            point = row["point"]
            if isinstance(point, dict):
                return Action(type="point", args={"x": point["x"], "y": point["y"]})
            return Action(type="point", args={"x": point[0], "y": point[1]})
        if "bbox" in row:
            bbox = row["bbox"]
            if isinstance(bbox, dict):
                return Action(
                    type="bbox",
                    args={"x1": bbox["x1"], "y1": bbox["y1"], "x2": bbox["x2"], "y2": bbox["y2"]},
                )
            return Action(
                type="bbox",
                args={"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
            )
        return Action.from_dict(row)
    raise TypeError(f"bad prediction row: {row!r}")


def _load_predictions(path: Path) -> list[Action | str]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else loaded.get("predictions", [])
    return [_row_to_prediction(row) for row in rows]


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanocua.eval",
        description="Offline eval: action-match (SFT) or point-in-bbox (grounding).",
    )
    parser.add_argument(
        "--task",
        choices=("sft", "grounding"),
        default="sft",
        help="sft = gold action match (default); grounding = point-in-bbox / point_acc.",
    )
    parser.add_argument("--data", help="JSON fixture (default: bundled for the task)")
    parser.add_argument(
        "--predictions",
        help="JSON/JSONL of predictions aligned with samples. "
        "Omitted = copy gold (sanity check, scores=1.0).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full result dict")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions = _load_predictions(Path(args.predictions)) if args.predictions else None
    if args.task == "grounding":
        result = grounding_eval(load_grounding_examples(args.data), predictions=predictions)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"n={result['n']}  point_in_bbox={_fmt_rate(result['point_in_bbox'])}  "
                f"point_acc={_fmt_rate(result['point_acc'])}  "
                f"parse_errors={result['parse_errors']}"
            )
        return 0 if result["parse_errors"] == 0 else 1

    result = offline_eval(load_trajectories(args.data), predictions=predictions)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"n={result['n']}  exact={result['exact']:.3f}  "
            f"normalized={result['normalized']:.3f}  type={result['type']:.3f}  "
            f"parse_errors={result['parse_errors']}"
        )
    return 0 if result["parse_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

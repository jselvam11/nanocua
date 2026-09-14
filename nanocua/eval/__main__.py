"""CLI: ``python -m nanocua.eval`` or ``nanocua-eval`` (offline by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanocua.data.load import load_trajectories
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanocua.eval",
        description="Offline action-match eval on trajectory JSON.",
    )
    parser.add_argument("--data", help="Trajectory JSON (default: bundled fixture)")
    parser.add_argument(
        "--predictions",
        help="JSON/JSONL of predicted actions aligned with expanded samples. "
        "Omitted = copy gold (sanity check, exact=1.0).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full result dict")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trajectories = load_trajectories(args.data)
    predictions = _load_predictions(Path(args.predictions)) if args.predictions else None
    result = offline_eval(trajectories, predictions=predictions)
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

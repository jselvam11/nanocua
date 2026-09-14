"""CLI: ``python -m nanocua.train`` or ``nanocua-train``."""

from __future__ import annotations

import argparse
import json
import sys

from nanocua.train.config import load_train_config
from nanocua.train.sft import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanocua.train",
        description="Smoke SFT on CUA trajectories (fixture by default).",
    )
    parser.add_argument("--config", help="YAML or JSON TrainConfig file")
    parser.add_argument("--data", dest="data_path", help="Trajectory JSON (default: bundled fixture)")
    parser.add_argument("--model", dest="model_name", help="Hugging Face VLM id")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--image-size", dest="image_size", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand + format samples and exit (no GPU, no model download).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_train_config(
        args.config,
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
    )
    try:
        result = train(cfg, dry_run=args.dry_run)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

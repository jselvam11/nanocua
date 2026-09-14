"""Smoke SFT on the synthetic fixture.

Needs the train extra (and a model download on first real run)::

    pip install -e ".[train]"
    python examples/02_smoke_train.py --dry-run          # no GPU, no download
    python examples/02_smoke_train.py --config configs/smoke.yaml

CPU can finish ``max_steps=4`` but it is slow. A GPU is the comfortable path.
If transformers/torch are missing, this script prints how to install them
and exits 2 — it does not fake a training run.
"""

from __future__ import annotations

import argparse
import json
import sys

from nanocua.train import load_train_config, train


def main() -> int:
    parser = argparse.ArgumentParser(description="nanocua smoke train")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_train_config(args.config, max_steps=args.max_steps)
    try:
        result = train(cfg, dry_run=args.dry_run)
    except (ImportError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

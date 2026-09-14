"""GUI grounding pretrain on the bundled fixture (no GPU, no network).

    python examples/04_grounding.py

Prints the Stage-1 chat format, gold-copy scores (should be 1.0), a slightly
jittered pointer (still inside the box), and a far-off pointer (should miss).
"""

from __future__ import annotations

from pathlib import Path

from nanocua.data import (
    bundled_grounding_path,
    export_jsonl,
    load_grounding_examples,
)
from nanocua.eval import grounding_eval
from nanocua.prompts import format_grounding_assistant, format_grounding_user
from nanocua.schema import GroundingExample


def jitter_inside(example: GroundingExample) -> str:
    x, y = example.gold_point()
    return f"point(x={x + 0.01}, y={y + 0.01})"


def far_corner(_example: GroundingExample) -> str:
    return "point(x=0.99, y=0.99)"


def main() -> None:
    path = bundled_grounding_path()
    examples = load_grounding_examples(path)
    print(f"loaded {len(examples)} grounding triples from {path}\n")
    for example in examples:
        print(f"--- {example.id} ---")
        print(format_grounding_user(example, include_image_token=False))
        print(format_grounding_assistant(example))
        print()

    gold = grounding_eval(examples)
    jitter = grounding_eval(examples, predictor=jitter_inside)
    far = grounding_eval(examples, predictor=far_corner)
    print(
        "gold copy     ",
        f"point_in_bbox={gold['point_in_bbox']:.2f}  point_acc={gold['point_acc']:.2f}",
    )
    print(
        "jitter +0.01  ",
        f"point_in_bbox={jitter['point_in_bbox']:.2f}  point_acc={jitter['point_acc']:.2f}",
    )
    print(
        "far corner    ",
        f"point_in_bbox={far['point_in_bbox']:.2f}  point_acc={far['point_acc']:.2f}",
    )

    out = Path("outputs/grounding_sharegpt_tiny.jsonl")
    export_jsonl(examples, out)
    print(f"\nwrote ShareGPT-ish JSONL → {out}")
    print("train dry-run:  python -m nanocua.train --task grounding --dry-run")


if __name__ == "__main__":
    main()

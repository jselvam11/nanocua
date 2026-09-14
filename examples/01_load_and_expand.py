"""Load the bundled fixture and expand it into SFT samples.

No GPU, no network::

    python examples/01_load_and_expand.py
"""

from __future__ import annotations

from pathlib import Path

from nanocua.data import bundled_fixture_path, expand_trajectories, export_jsonl, load_trajectories
from nanocua.prompts import format_assistant_message, format_user_message


def main() -> None:
    path = bundled_fixture_path()
    trajectories = load_trajectories(path)
    samples = expand_trajectories(trajectories)
    print(f"loaded {len(trajectories)} trajectories from {path}")
    print(f"expanded to {len(samples)} SFT samples (one per step)\n")
    for sample in samples:
        print(f"--- {sample.trajectory_id} step {sample.step_index} ---")
        print(format_user_message(sample, include_image_token=False))
        print(format_assistant_message(sample))
        print()
    out = Path("outputs/sharegpt_tiny.jsonl")
    export_jsonl(samples, out)
    print(f"wrote ShareGPT-ish JSONL → {out}")


if __name__ == "__main__":
    main()

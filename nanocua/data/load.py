"""Load trajectories and grounding triples from local JSON.

Hugging Face loaders are documented stubs so the base install stays offline.
"""

from __future__ import annotations

from pathlib import Path

from nanocua.data.images import color_for_index, write_dummy_png
from nanocua.schema import (
    GroundingExample,
    Trajectory,
    load_grounding_dicts,
    load_trajectory_dicts,
)

# Shipped synthetic datasets — small enough to read, works offline.
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_trajectories.json"
_GROUNDING_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_grounding.json"


def bundled_fixture_path() -> Path:
    """Path to the tiny offline trajectory fixture shipped with the package."""
    if not _FIXTURE.is_file():
        raise FileNotFoundError(
            f"bundled fixture missing: {_FIXTURE}. Reinstall the package or "
            "pass an explicit JSON path to load_trajectories()."
        )
    return _FIXTURE


def bundled_grounding_path() -> Path:
    """Path to the tiny offline grounding fixture shipped with the package."""
    if not _GROUNDING_FIXTURE.is_file():
        raise FileNotFoundError(
            f"bundled grounding fixture missing: {_GROUNDING_FIXTURE}. "
            "Reinstall the package or pass an explicit JSON path to "
            "load_grounding_examples()."
        )
    return _GROUNDING_FIXTURE


def _resolve_screenshot(raw: str, json_dir: Path, index: int) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = json_dir / path
    if not path.exists():
        write_dummy_png(path, rgb=color_for_index(index))
    return str(path)


def load_trajectories(path: str | Path | None = None) -> list[Trajectory]:
    """Load a JSON list of trajectories and resolve screenshot paths.

    Screenshot paths are relative to the JSON file. Missing images are
    replaced with a solid-color dummy PNG so demos run with no assets.
    """
    json_path = Path(path) if path is not None else bundled_fixture_path()
    json_dir = json_path.parent
    trajectories: list[Trajectory] = []
    for item in load_trajectory_dicts(json_path):
        traj = Trajectory.from_dict(item)
        for step in traj.steps:
            step.screenshot = _resolve_screenshot(step.screenshot, json_dir, step.index)
        trajectories.append(traj)
    return trajectories


def load_grounding_examples(path: str | Path | None = None) -> list[GroundingExample]:
    """Load a JSON list of grounding triples and resolve screenshot paths.

    Same rules as :func:`load_trajectories`: paths are relative to the JSON
    file, and missing images become a solid-color dummy PNG.
    """
    json_path = Path(path) if path is not None else bundled_grounding_path()
    json_dir = json_path.parent
    examples: list[GroundingExample] = []
    for index, item in enumerate(load_grounding_dicts(json_path)):
        example = GroundingExample.from_dict(item)
        if not example.id:
            example.id = f"grounding-{index:02d}"
        example.screenshot = _resolve_screenshot(example.screenshot, json_dir, index)
        examples.append(example)
    return examples


def load_hf_dataset(name: str, split: str = "train") -> list[Trajectory]:
    """Hook for later: map a Hugging Face dataset onto :class:`Trajectory`.

    TODO: implement column mapping for AgentNet / OS-Atlas-style datasets.
    Until then, export those rows to nanocua JSON (see the bundled fixture)
    and use :func:`load_trajectories`.
    """
    raise NotImplementedError(
        "Hugging Face loading is a stub so the base install stays offline. "
        f"Asked for {name!r} split={split!r}. Convert the split to nanocua "
        "JSON or implement this mapper (goal/instruction + steps/traj)."
    )


def load_hf_grounding(name: str, split: str = "train") -> list[GroundingExample]:
    """Hook for later: OS-Atlas / SeeClick / UGround-style HF datasets.

    TODO: map columns (image, instruction/expression, bbox or point) onto
    :class:`GroundingExample`. Until then, export a tiny JSON (see the
    bundled grounding fixture) and use :func:`load_grounding_examples`.
    """
    raise NotImplementedError(
        "Hugging Face grounding load is a stub so the base install stays "
        f"offline. Asked for {name!r} split={split!r}. Convert OS-Atlas / "
        "SeeClick rows to nanocua grounding JSON or implement this mapper."
    )

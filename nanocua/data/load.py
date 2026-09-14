"""Load trajectories from local JSON. Hugging Face is a documented stub."""

from __future__ import annotations

from pathlib import Path

from nanocua.data.images import color_for_index, write_dummy_png
from nanocua.schema import Trajectory, load_trajectory_dicts

# Shipped synthetic dataset — small enough to read, works offline.
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_trajectories.json"


def bundled_fixture_path() -> Path:
    """Path to the tiny offline fixture shipped with the package."""
    if not _FIXTURE.is_file():
        raise FileNotFoundError(
            f"bundled fixture missing: {_FIXTURE}. Reinstall the package or "
            "pass an explicit JSON path to load_trajectories()."
        )
    return _FIXTURE


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

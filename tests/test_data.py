"""Load + expand + export on the bundled fixture (offline)."""

import json
from pathlib import Path

import pytest

from nanocua.data import (
    bundled_fixture_path,
    bundled_grounding_path,
    expand_trajectories,
    expand_trajectory,
    export_jsonl,
    grounding_to_sharegpt,
    load_grounding_examples,
    load_hf_dataset,
    load_hf_grounding,
    load_trajectories,
    sample_to_sharegpt,
)


def test_bundled_fixture_expands_one_sample_per_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trajectories = load_trajectories()
    assert len(trajectories) == 2
    samples = expand_trajectories(trajectories)
    total_steps = sum(len(t.steps) for t in trajectories)
    assert len(samples) == total_steps == 6
    first = samples[0]
    assert first.history == []
    assert first.action.type == "click"
    second = samples[1]
    assert len(second.history) == 1
    assert "click" in second.history[0]


def test_prefix_grows_with_step_index():
    traj = load_trajectories()[0]
    samples = expand_trajectory(traj)
    for i, sample in enumerate(samples):
        assert len(sample.history) == i
        assert sample.goal == traj.goal
        assert Path(sample.screenshot).is_file()


def test_sharegpt_export(tmp_path):
    samples = expand_trajectories(load_trajectories())
    out = tmp_path / "out.jsonl"
    export_jsonl(samples, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(samples)
    row = json.loads(lines[0])
    roles = [turn["from"] for turn in row["conversations"]]
    assert roles == ["system", "human", "gpt"]
    assert "<image>" in row["conversations"][1]["value"]
    assert row["conversations"][2]["value"].startswith("Thought:")
    assert "Action:" in row["conversations"][2]["value"]
    packed = sample_to_sharegpt(samples[0])
    assert packed["images"][0] == samples[0].screenshot


def test_load_from_explicit_json_path():
    path = bundled_fixture_path()
    trajectories = load_trajectories(path)
    assert trajectories[0].id == "fixture-weather-search"


def test_hf_hook_is_explicit_stub():
    with pytest.raises(NotImplementedError, match="stub"):
        load_hf_dataset("xlangai/AgentNet")


def test_load_tests_fixtures_copy():
    path = Path(__file__).parent / "fixtures" / "tiny_trajectories.json"
    trajectories = load_trajectories(path)
    assert len(trajectories) == 2
    assert Path(trajectories[0].steps[0].screenshot).is_file()
    assert Path(trajectories[0].steps[0].screenshot).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_bundled_grounding_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    examples = load_grounding_examples()
    assert len(examples) == 4
    assert {ex.id for ex in examples} == {"g-address-bar", "g-new-tab", "g-file-menu", "g-save-button"}
    for example in examples:
        assert Path(example.screenshot).is_file()
        assert example.point is not None
        assert example.bbox is not None
        assert example.bbox[0] <= example.point[0] <= example.bbox[2]
        assert example.bbox[1] <= example.point[1] <= example.bbox[3]


def test_grounding_address_bar_matches_sft_click():
    """Stage-1 localize and Stage-2 SFT meet at the same address-bar click."""
    grounding = {ex.id: ex for ex in load_grounding_examples()}["g-address-bar"]
    first_sft = expand_trajectories(load_trajectories())[0]
    assert grounding.as_click().to_string() == first_sft.action.to_string()


def test_grounding_export(tmp_path):
    examples = load_grounding_examples()
    out = tmp_path / "g.jsonl"
    export_jsonl(examples, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    row = json.loads(lines[0])
    roles = [turn["from"] for turn in row["conversations"]]
    assert roles == ["system", "human", "gpt"]
    assert "Instruction:" in row["conversations"][1]["value"]
    assert row["conversations"][2]["value"].startswith("point(")
    packed = grounding_to_sharegpt(examples[0])
    assert packed["images"][0] == examples[0].screenshot


def test_load_grounding_from_explicit_json_path():
    path = bundled_grounding_path()
    examples = load_grounding_examples(path)
    assert examples[0].id == "g-address-bar"


def test_hf_grounding_hook_is_explicit_stub():
    with pytest.raises(NotImplementedError, match="stub"):
        load_hf_grounding("OS-Atlas/OS-Atlas-data")


def test_load_tests_grounding_copy():
    path = Path(__file__).parent / "fixtures" / "tiny_grounding.json"
    examples = load_grounding_examples(path)
    assert len(examples) == 4
    assert Path(examples[0].screenshot).is_file()

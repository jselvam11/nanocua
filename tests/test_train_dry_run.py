"""Train formatting + CLI dry-run (no GPU, no model download)."""

import json

import pytest

from nanocua.train import load_train_config, train
from nanocua.train.__main__ import main as train_main
from nanocua.eval.__main__ import main as eval_main


def test_simple_yaml_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("max_steps: 7\nlearning_rate: 2e-5\nmodel_name: tiny\n", encoding="utf-8")
    cfg = load_train_config(path)
    assert cfg.max_steps == 7
    assert cfg.learning_rate == 2e-5
    assert cfg.model_name == "tiny"


def test_dry_run_returns_fixture_sample_ids():
    cfg = load_train_config(max_steps=2)
    result = train(cfg, dry_run=True)
    assert result["dry_run"] is True
    assert result["task"] == "sft"
    assert result["n_samples"] == 6
    assert any("weather" in i for i in result["ids"])


def test_train_cli_dry_run(capsys):
    code = train_main(["--dry-run", "--max-steps", "1"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_samples"] == 6


def test_eval_cli_gold_copy(capsys):
    code = eval_main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "exact=1.000" in out


def test_grounding_dry_run_formats_instruction():
    from nanocua.train.sft import format_plain_example, samples_from_config

    cfg = load_train_config(task="grounding")
    sample = samples_from_config(cfg)[0]
    row = format_plain_example(sample)
    assert "Instruction:" in row["text"]
    assert "point(" in row["completion"]
    assert row["id"] == "g-address-bar"


def test_grounding_dry_run_returns_fixture_ids():
    cfg = load_train_config(task="grounding", max_steps=2)
    result = train(cfg, dry_run=True)
    assert result["dry_run"] is True
    assert result["task"] == "grounding"
    assert result["n_samples"] == 4
    assert "g-address-bar" in result["ids"]


def test_grounding_train_cli_dry_run(capsys):
    code = train_main(["--task", "grounding", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "grounding"
    assert payload["n_samples"] == 4


def test_grounding_eval_cli_gold_copy(capsys):
    code = eval_main(["--task", "grounding"])
    assert code == 0
    out = capsys.readouterr().out
    assert "point_in_bbox=1.000" in out
    assert "point_acc=1.000" in out


def test_unknown_task_is_rejected():
    cfg = load_train_config(task="rl")
    with pytest.raises(ValueError, match="unknown task"):
        train(cfg, dry_run=True)


def test_grounding_yaml_config(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("task: grounding\nmax_steps: 3\n", encoding="utf-8")
    cfg = load_train_config(path)
    assert cfg.task == "grounding"
    assert cfg.max_steps == 3


def test_real_train_raises_without_extras():
    from nanocua.train.sft import smoke_available, train

    if smoke_available():
        pytest.skip("train extras are installed in this environment")
    cfg = load_train_config(max_steps=1)
    with pytest.raises(ImportError, match="Training extras"):
        train(cfg, dry_run=False)

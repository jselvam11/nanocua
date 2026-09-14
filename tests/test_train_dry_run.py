"""Train formatting + CLI dry-run (no GPU, no model download)."""

import json

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


def test_real_train_raises_without_extras():
    import pytest
    from nanocua.train.sft import smoke_available, train

    if smoke_available():
        pytest.skip("train extras are installed in this environment")
    cfg = load_train_config(max_steps=1)
    with pytest.raises(ImportError, match="Training extras"):
        train(cfg, dry_run=False)

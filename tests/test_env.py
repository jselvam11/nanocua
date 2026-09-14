"""Mock env + protocol (no real computer)."""

import pytest

from nanocua.env import ComputerEnv, MockComputerEnv
from nanocua.env.adapter import OSWorldAdapter, PyAutoGUIAdapter
from nanocua.schema import Action


def test_mock_is_a_computer_env():
    env = MockComputerEnv(max_steps=2)
    assert isinstance(env, ComputerEnv)


def test_terminate_success_gives_reward(tmp_path):
    env = MockComputerEnv(max_steps=5, frame_dir=tmp_path)
    env.reset("save the file")
    _, reward, done, _ = env.step(Action(type="hotkey", args={"keys": ["ctrl", "s"]}))
    assert done is False
    assert reward == 0.0
    _, reward, done, info = env.step(Action(type="terminate", args={"status": "success"}))
    assert done is True
    assert reward == 1.0
    assert info["terminated"] is True


def test_adapter_sketches_exist_but_are_stubs():
    with pytest.raises(NotImplementedError, match="TODO"):
        PyAutoGUIAdapter().reset("task")
    with pytest.raises(NotImplementedError, match="TODO"):
        OSWorldAdapter().step(Action(type="wait", args={"seconds": 1}))

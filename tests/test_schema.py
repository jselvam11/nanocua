"""Schema round-trip and action parsing."""

from nanocua.schema import Action, Trajectory


def test_action_string_roundtrip():
    original = Action(type="click", args={"x": 0.52, "y": 0.08})
    parsed = Action.from_string(original.to_string())
    assert parsed.type == "click"
    assert parsed.args["x"] == 0.52
    assert parsed.args["y"] == 0.08


def test_pyautogui_prefix_and_type_text():
    action = Action.from_string('pyautogui.type(text="weather today")')
    assert action.type == "type"
    assert action.args["text"] == "weather today"
    assert "weather today" in action.to_string()


def test_hotkey_list_args():
    action = Action.from_string('hotkey(keys=["ctrl", "s"])')
    assert action.args["keys"] == ["ctrl", "s"]


def test_trajectory_json_roundtrip():
    raw = {
        "id": "t1",
        "goal": "do a thing",
        "steps": [
            {
                "screenshot": "a.png",
                "thought": "click it",
                "action": "click(x=0.1, y=0.2)",
            }
        ],
    }
    traj = Trajectory.from_dict(raw)
    assert traj.steps[0].action.type == "click"
    again = Trajectory.from_dict(traj.to_dict())
    assert again.goal == "do a thing"
    assert again.steps[0].action.to_string() == traj.steps[0].action.to_string()

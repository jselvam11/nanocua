"""Schema round-trip and action parsing."""

import pytest

from nanocua.schema import Action, GroundingExample, Trajectory


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


def test_grounding_roundtrip_and_as_click():
    example = GroundingExample.from_dict(
        {
            "id": "g1",
            "screenshot": "a.png",
            "instruction": "the browser address bar",
            "point": [0.52, 0.08],
            "bbox": [0.20, 0.03, 0.85, 0.13],
        }
    )
    assert example.target_string() == "point(x=0.52, y=0.08)"
    assert example.as_click().to_string() == "click(x=0.52, y=0.08)"
    again = GroundingExample.from_dict(example.to_dict())
    assert again.point == (0.52, 0.08)
    assert again.bbox == (0.20, 0.03, 0.85, 0.13)


def test_grounding_bbox_only_uses_center():
    example = GroundingExample(
        id="box",
        screenshot="a.png",
        instruction="Save",
        bbox=(0.0, 0.0, 0.2, 0.4),
    )
    assert example.gold_point() == (0.1, 0.2)
    assert example.target_string().startswith("bbox(")


def test_grounding_xywh_and_rejects_oob():
    example = GroundingExample.from_dict(
        {
            "id": "xywh",
            "image": "a.png",
            "expression": "OK",
            "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        }
    )
    assert example.bbox[0] == pytest.approx(0.1)
    assert example.bbox[1] == pytest.approx(0.2)
    assert example.bbox[2] == pytest.approx(0.4)
    assert example.bbox[3] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="non-empty"):
        GroundingExample(id="x", screenshot="a.png", instruction="  ")
    with pytest.raises(ValueError, match="point and/or"):
        GroundingExample(id="x", screenshot="a.png", instruction="OK")
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        GroundingExample(id="x", screenshot="a.png", instruction="OK", point=(1.5, 0.2))

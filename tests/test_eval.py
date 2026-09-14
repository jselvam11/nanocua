"""Offline metrics + mock-env online stub (no GPU)."""

from nanocua.data import load_grounding_examples, load_trajectories
from nanocua.env import MockComputerEnv
from nanocua.eval import grounding_eval, match_actions, offline_eval, online_eval, point_in_bbox
from nanocua.eval.offline import as_action
from nanocua.eval.online import always_wait
from nanocua.prompts import parse_assistant_message, parse_grounding_prediction
from nanocua.schema import Action, GroundingExample, SFTSample


def test_exact_and_normalized_match():
    gold = Action(type="click", args={"x": 0.52, "y": 0.08})
    close = Action(type="click", args={"x": 0.521, "y": 0.079})
    scores = match_actions(close, gold)
    assert scores["exact"] is False
    assert scores["normalized"] is True
    assert scores["type"] is True


def test_gold_copy_is_perfect():
    result = offline_eval(load_trajectories())
    assert result["n"] == 6
    assert result["exact"] == 1.0
    assert result["normalized"] == 1.0
    assert result["parse_errors"] == 0


def test_always_wait_is_wrong():
    def wait(_sample: SFTSample) -> Action:
        return Action(type="wait", args={"seconds": 1})

    result = offline_eval(load_trajectories(), predictor=wait)
    assert result["exact"] == 0.0
    assert result["type"] == 0.0


def test_parse_assistant_and_eval_from_strings():
    gold = load_trajectories()[0].steps[0].action
    text = "Thought: focus the bar\nAction: click(x=0.52, y=0.08)"
    thought, parsed = parse_assistant_message(text)
    assert "focus" in thought
    assert match_actions(parsed, gold)["exact"]
    result = offline_eval(load_trajectories()[:1], predictions=[text] * 4)
    assert result["n"] == 4
    assert result["exact"] == 0.25  # only step 0 is that click


def test_as_action_accepts_bare_call():
    action = as_action("hotkey(keys=[\"enter\"])")
    assert action.type == "hotkey"


def test_mock_env_and_online_stub(tmp_path):
    env = MockComputerEnv(max_steps=3, frame_dir=tmp_path / "frames")
    result = online_eval(env, always_wait, task="do nothing", max_steps=3)
    assert result["steps"] == 3
    assert result["done"] is True
    assert result["actions"][0].startswith("wait")
    obs = env.reset("again")
    assert obs.screenshot.endswith(".png")


def test_grounding_gold_copy_is_perfect():
    result = grounding_eval(load_grounding_examples())
    assert result["n"] == 4
    assert result["parse_errors"] == 0
    assert result["point_in_bbox"] == 1.0
    assert result["point_acc"] == 1.0
    assert result["mean_distance"] == 0.0


def test_grounding_jitter_inside_bbox_and_far_miss():
    examples = load_grounding_examples()

    def jitter(example: GroundingExample) -> str:
        x, y = example.gold_point()
        return f"point(x={x + 0.01}, y={y + 0.01})"

    jittered = grounding_eval(examples, predictor=jitter)
    assert jittered["point_in_bbox"] == 1.0
    assert jittered["point_acc"] == 1.0

    def far(_example: GroundingExample) -> str:
        return "point(x=0.99, y=0.99)"

    missed = grounding_eval(examples, predictor=far)
    assert missed["point_in_bbox"] == 0.0
    assert missed["point_acc"] == 0.0


def test_grounding_accepts_click_and_bracket_forms():
    gold = load_grounding_examples()[0]
    click = parse_grounding_prediction("Action: click(x=0.52, y=0.08)")
    bracket = parse_grounding_prediction("the element is at [0.52, 0.08]")
    assert click.type == "click"
    assert bracket.type == "point"
    result = grounding_eval([gold], predictions=["click(x=0.52, y=0.08)"])
    assert result["point_in_bbox"] == 1.0
    assert point_in_bbox((0.52, 0.08), gold.bbox) is True
    assert point_in_bbox((0.99, 0.99), gold.bbox) is False

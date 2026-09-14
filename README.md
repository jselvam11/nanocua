# nanocua

A nanoGPT-style skeleton for **training computer-use agents** (CUAs).

The code is intentionally small. The point is not to ship a sandbox fleet or
another agent framework. The point is that you can **read this repo in an
afternoon**, understand the CUA training loop, and fork it to try an idea.

```
Stage 1  (screenshot, referring expression)  →  point / bbox     grounding pretrain
Stage 2  trajectory  →  step samples  →  SFT train  →  eval      behavior cloning
```

That is the whole mental model. Grounding is a sibling of trajectory SFT, not
a second framework. Everything in the tree maps onto one of those two arrows.

---

## What is CUA training?

A computer-use agent looks at a **screenshot** of a desktop (or browser),
optionally writes a **thought**, and emits an **action**: click, type, hotkey,
scroll, wait, or give up / succeed.

Training data is not a pile of unrelated (image, caption) pairs. It is a
**trajectory**: one goal, pursued over T steps.

```
goal: "Search for today's weather."

t=0  screenshot_0  →  thought_0  +  click(address bar)
t=1  screenshot_1  →  thought_1  +  type("weather today")
t=2  screenshot_2  →  thought_2  +  hotkey(["enter"])
t=3  screenshot_3  →  thought_3  +  terminate(success)
```

Supervised fine-tuning (SFT) turns that episode into **T samples**, one per
step. Sample `t` sees the goal, the actions already taken, and the current
screenshot, and must produce the demonstrator's next thought + action. That
transform is called **step-prefix expansion**. It is behavior cloning.

Later papers add more: richer chain-of-thought, multiple past screenshots in
the prompt, then RL (GRPO/PPO) on live rollouts. Those are marked TODO here
on purpose. Read the SFT loop first.

Before that SFT loop, many CUA papers (SeeClick, OS-Atlas, UGround, …) run a
**GUI grounding** stage: continual pretrain of an already-trained VLM on
`(screenshot, referring expression) → point or bbox`. That is Stage 1 here.
It is not training a VLM from scratch, and it is not a multi-step episode.

This is **not** trycua, OpenCUA, or ScaleCUA. Those are full stacks
(annotation, sandboxes, large datasets, trained models). nanocua is the
minimum you need to *see the loop*.

---

## Repo map

| Path | Role in the loop |
|------|------------------|
| `nanocua/schema.py` | `Action`, `Step`, `Trajectory`, `SFTSample`, `GroundingExample` |
| `nanocua/prompts.py` | How a sample becomes the text a VLM sees (SFT *and* grounding) |
| `nanocua/data/load.py` | JSON in (HF datasets / OS-Atlas = stub) |
| `nanocua/data/expand.py` | Trajectory of T steps → T SFT samples |
| `nanocua/data/export.py` | ShareGPT-ish JSONL for other trainers |
| `nanocua/data/fixtures/` | Tiny synthetic datasets (offline) |
| `nanocua/train/sft.py` | Thin `transformers.Trainer` loop (`--task sft` or `grounding`) |
| `nanocua/train/config.py` | One dataclass / YAML of knobs |
| `nanocua/env/` | `ComputerEnv` protocol + mock desktop |
| `nanocua/eval/` | Offline action-match, grounding point-in-bbox, online stub |
| `examples/` | Four scripts: expand, smoke train, eval, grounding |

Start with `schema.py`, then `data/expand.py`, then `train/sft.py`. Those
three files are the course. For Stage 1, read `GroundingExample` in
`schema.py` and `python -m nanocua.train --task grounding --dry-run`.

---

## Schema (the format)

A trajectory JSON looks like this (see
`nanocua/data/fixtures/tiny_trajectories.json`):

```json
{
  "id": "fixture-weather-search",
  "goal": "Open the browser address bar and search for today's weather.",
  "steps": [
    {
      "index": 0,
      "screenshot": "screenshots/step_00.png",
      "thought": "The browser is already open. I should focus the address bar.",
      "action": {"type": "click", "args": {"x": 0.52, "y": 0.08}}
    }
  ]
}
```

Actions can also be pyautogui-style strings: `"click(x=0.52, y=0.08)"`.

**Coordinates are normalized to [0, 1]** (top-left origin), matching common
open CUA datasets. At execution time you multiply by screen width/height.

Default action types: `click`, `double_click`, `right_click`, `type`,
`hotkey`, `scroll`, `wait`, `terminate`. That is a teaching subset, not a
complete OS driver. Fork `schema.py` to extend it.

The assistant target the VLM is trained to emit is always:

```
Thought: <brief reasoning>
Action: click(x=0.52, y=0.08)
```

---

## Stage-1: GUI grounding pretrain

Trajectory SFT teaches *what to do next* given a goal and a history.
Grounding teaches *where a widget is* given a phrase. Same VLM, same
Trainer, different record:

```
(screenshot, "the browser address bar")  →  point(x=0.52, y=0.08)
```

In the literature this is usually **continual pretrain** on a VLM that
already reads images (SeeClick-style element grounding, OS-Atlas-style
boxes). nanocua does not download those datasets. The bundled fixture is
four synthetic triples that reuse the dummy screenshots from the
trajectory fixture — including the same address-bar point that step 0 of
`fixture-weather-search` later clicks. That is the whole punchline:
grounding localizes, SFT clicks.

A grounding JSON row looks like this (see
`nanocua/data/fixtures/tiny_grounding.json`):

```json
{
  "id": "g-address-bar",
  "screenshot": "screenshots/step_00.png",
  "instruction": "the browser address bar",
  "point": [0.52, 0.08],
  "bbox": [0.20, 0.03, 0.85, 0.13]
}
```

Coordinates are still **normalized to [0, 1]**. Give a point, a bbox, or
both. Training emits `point(...)` when a point is present (click-style);
offline eval reports **point-in-bbox** and a distance-threshold
**point_acc** (default 0.05). Gold-copy on the fixture is 1.0. This repo
does not report ScreenSpot / OS-Atlas numbers.

The HF loaders (`load_hf_dataset`, `load_hf_grounding`) are explicit
stubs. Dump OS-Atlas / SeeClick rows into this JSON shape when you want
a real run.

---

## Install

Python 3.10+. The **base install has no third-party dependencies** so you can
inspect data and run offline eval on a laptop.

```bash
pip install -e ".[dev]"     # pytest; enough for data + eval
pip install -e ".[train]"   # torch, transformers, trl, pillow, …
```

`[train]` is optional. You do not need it to understand the repo.

---

## Smoke path (run this first)

From the repo root, after `pip install -e ".[dev]"`:

```bash
# 1. Trajectories → SFT samples → ShareGPT JSONL  (no GPU, no network)
python examples/01_load_and_expand.py

# 2. Offline eval against gold actions             (no GPU, no network)
python examples/03_offline_eval.py
python -m nanocua.eval

# 3. GUI grounding fixture (Stage-1 localize)      (no GPU, no network)
python examples/04_grounding.py
python -m nanocua.eval --task grounding
python -m nanocua.train --task grounding --dry-run

# 4. Prove the SFT train CLI wires up without downloading a model
python -m nanocua.train --dry-run
```

Expected: 2 synthetic trajectories expand to **6 SFT samples**; gold-copy
offline eval prints `exact=1.000`. Grounding loads **4** triples; gold-copy
prints `point_in_bbox=1.000`.

### Smoke train (optional, needs `[train]`)

```bash
pip install -e ".[train]"
python examples/02_smoke_train.py --config configs/smoke.yaml
# or
python -m nanocua.train --config configs/smoke.yaml
# Stage-1 grounding, same Trainer:
python -m nanocua.train --task grounding --config configs/smoke_grounding.yaml
```

Default model: `HuggingFaceTB/SmolVLM-256M-Instruct` — small enough to *learn
the loop*, not a competitive CUA. First run downloads weights from Hugging
Face.

| Hardware | Honest expectation |
|----------|--------------------|
| GPU | A 4-step smoke run should finish in seconds to a couple of minutes |
| CPU | It can run; it will be slow (minutes). This is for wiring, not accuracy |
| No `torch` / no download | The CLI exits with a clear message. Data + offline eval still work |

This repo does **not** report task-success numbers. A smoke run only checks
that data → collate → loss → optimizer step is connected.

---

## Where to hack next (an afternoon)

Pick one. The files are short.

1. **New action space** — `nanocua/schema.py` (`ACTION_TYPES`, `Action.to_string`).
2. **New prompt / CoT** — `nanocua/prompts.py` (add observation, reflection, …).
3. **Multi-image history** — `nanocua/data/expand.py` (past screenshots in the user turn).
4. **Different loss** — `nanocua/train/sft.py` (mask everything except the `Action:` line; swap in TRL `SFTTrainer`).
5. **Real desktop** — fill in `nanocua/env/adapter.py` (not imported by default).
6. **Your own JSON** — dump trajectories in the fixture format and pass `--data path.json`.
7. **Grounding target** — emit `bbox(...)` instead of `point(...)`, or 0–1000 integer coords like SeeClick (`prompts.py` / `GroundingExample.target_string`).

### TODO (deliberately out of scope)

- Hugging Face dataset mapper (`nanocua.data.load_hf_dataset`, `load_hf_grounding`)
- OS-Atlas / SeeClick / ScreenSpot downloaders
- Real OSWorld / Docker / pyautogui execution
- RL / GRPO on rollouts
- Competitive VLMs, packing, multi-GPU recipes

When you add one of these, keep it in one file if you can. Resist the
framework.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests stay offline: no GPU, no Hugging Face download. Train is covered by
`--dry-run` (format samples, do not load weights). Grounding is covered the
same way (`--task grounding`).

---

## License

MIT. See `LICENSE`.

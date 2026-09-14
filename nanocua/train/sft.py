"""Supervised fine-tuning for computer-use agents.

Read this file top to bottom — it is the whole training story:

1. Load data for the chosen task:
   * ``sft`` — trajectories expanded into step-prefix samples
     (goal + history + screenshot → thought + action).
   * ``grounding`` — screenshot + referring expression → point/bbox
     (Stage-1 GUI localize; same Trainer, different prompt).
2. Format each sample as a short chat (see ``nanocua.prompts``).
3. Run a few optimizer steps with ``transformers.Trainer``.

TRL's ``SFTTrainer`` is the usual next step when you scale (packing,
completion-only loss, easier VLM collators). We keep a plain Trainer
here so the loop stays visible. ``trl`` is still an extra so you can
swap it in without inventing a new stack.

GPU vs CPU
----------
A SmolVLM-class model *will* run a 4-step smoke train on CPU. It is
slow (minutes, not seconds) and needs a one-time Hugging Face download.
A GPU makes the same smoke run finish in seconds. This library does
not invent benchmark scores; a smoke run only checks that the loop
connects.

If transformers/torch are missing, or the model cannot be downloaded,
``train()`` raises a clear error instead of pretending to succeed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanocua.data.expand import expand_trajectories
from nanocua.data.load import load_grounding_examples, load_trajectories
from nanocua.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_assistant_message,
    format_grounding_assistant,
    format_grounding_user,
    format_user_message,
)
from nanocua.schema import GroundingExample, SFTSample
from nanocua.train.config import TrainConfig

_TRAIN_INSTALL = 'pip install -e ".[train]"'
_TASKS = ("sft", "grounding")


def _require_train_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Training extras are not installed. The data/eval path does not "
            f"need them. For smoke train: {_TRAIN_INSTALL}\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, AutoProcessor, AutoModelForImageTextToText


def samples_from_config(cfg: TrainConfig) -> list[SFTSample] | list[GroundingExample]:
    task = cfg.task.strip().lower()
    if task not in _TASKS:
        raise ValueError(
            f"unknown task {cfg.task!r}. Use 'sft' (trajectory behavior cloning) "
            "or 'grounding' (Stage-1 GUI localize)."
        )
    data_path = cfg.data_path or None
    if task == "grounding":
        return load_grounding_examples(data_path)
    return expand_trajectories(load_trajectories(data_path))


def format_plain_example(sample: SFTSample | GroundingExample) -> dict[str, Any]:
    """Plain-text view of one sample (used by --dry-run and as a fallback)."""
    if isinstance(sample, GroundingExample):
        user = format_grounding_user(sample, include_image_token=False)
        assistant = format_grounding_assistant(sample)
        system = GROUNDING_SYSTEM_PROMPT
        sample_id = sample.id
    else:
        user = format_user_message(sample, include_image_token=False)
        assistant = format_assistant_message(sample)
        system = SYSTEM_PROMPT
        sample_id = f"{sample.trajectory_id}-step{sample.step_index}"
    text = f"{system}\n\nUser: {user}\n\nAssistant: {assistant}"
    return {
        "text": text,
        "prompt": f"{system}\n\nUser: {user}\n\nAssistant:",
        "completion": f" {assistant}",
        "screenshot": sample.screenshot,
        "id": sample_id,
    }


def _build_collate(processor: Any, tokenizer: Any, cfg: TrainConfig) -> Any:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image

        prompts: list[str] = []
        full_texts: list[str] = []
        images = []
        for example in examples:
            image = Image.open(example["screenshot"]).convert("RGB")
            if cfg.image_size:
                image = image.resize((cfg.image_size, cfg.image_size))
            images.append(image)
            prompts.append(example["prompt"])
            full_texts.append(example["text"])

        encoded = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        labels = encoded["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Best-effort prompt mask. If the processor's prompt tokenization
        # disagrees with the full sequence (common with some VLMs), we keep
        # the pad-only mask rather than crash the smoke run.
        try:
            prompt_batch = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            )
            for i in range(labels.shape[0]):
                prompt_len = int((prompt_batch["input_ids"][i] != pad_id).sum().item())
                labels[i, : min(prompt_len, labels.shape[1])] = -100
        except Exception:
            pass

        encoded["labels"] = labels
        return encoded

    return collate


def train(cfg: TrainConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Run SFT or grounding pretrain (or stop after formatting when ``dry_run=True``)."""
    samples = samples_from_config(cfg)
    formatted = [format_plain_example(sample) for sample in samples]
    summary = {
        "n_samples": len(formatted),
        "task": cfg.task,
        "model_name": cfg.model_name,
        "max_steps": cfg.max_steps,
        "dry_run": dry_run,
        "ids": [row["id"] for row in formatted],
    }
    if dry_run:
        return summary
    if not formatted:
        raise ValueError("no training samples — check data_path, task, and the fixture JSON")

    torch, AutoProcessor, AutoModelForImageTextToText = _require_train_stack()

    # Some VLMs still live under AutoModelForVision2Seq in older transformers.
    try:
        processor = AutoProcessor.from_pretrained(cfg.model_name)
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_name)
    except Exception as exc:
        try:
            from transformers import AutoModelForVision2Seq

            processor = AutoProcessor.from_pretrained(cfg.model_name)
            model = AutoModelForVision2Seq.from_pretrained(cfg.model_name)
        except Exception as inner:
            raise RuntimeError(
                f"Could not load {cfg.model_name!r}. Smoke train needs the "
                "[train] extras and a one-time Hugging Face download. "
                "Data expand + offline eval do not.\n"
                f"Last error: {inner or exc}"
            ) from inner

    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    class _ListDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.rows[index]

    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        print(
            "No CUDA GPU detected — running on CPU. "
            f"{cfg.max_steps} smoke steps may take several minutes."
        )

    args_kwargs: dict[str, Any] = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=max(cfg.max_steps, 1),
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
        dataloader_num_workers=0,
    )
    # transformers 4.46+ renamed this; older versions want evaluation_strategy.
    try:
        train_args = TrainingArguments(**args_kwargs, save_strategy="no")
    except TypeError:
        train_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=_ListDataset(formatted),
        data_collator=_build_collate(processor, tokenizer, cfg),
    )
    result = trainer.train()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(cfg.output_dir) / "train_metrics.json"
    metrics = dict(result.metrics) if getattr(result, "metrics", None) else {}
    metrics.update(summary)
    metrics["device"] = "cpu" if use_cpu else "cuda"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")
    return metrics


def smoke_available() -> bool:
    """True when the optional train stack can be imported (no download)."""
    try:
        _require_train_stack()
    except ImportError:
        return False
    return True

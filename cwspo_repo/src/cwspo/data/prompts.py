from __future__ import annotations

import random

from cwspo.config import AppConfig
from cwspo.schemas import PromptRecord
from cwspo.utils.io import read_jsonl


def _sample_rows(rows: list[PromptRecord], limit: int | None, seed: int) -> list[PromptRecord]:
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def load_train_prompts(cfg: AppConfig) -> list[PromptRecord]:
    rows = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    return _sample_rows(rows, cfg.data.max_train_prompts, cfg.data.prompt_sampling_seed)


def load_eval_prompts(cfg: AppConfig) -> list[PromptRecord]:
    path = cfg.paths.eval_prompt_file or cfg.paths.prompt_file
    rows = read_jsonl(path, PromptRecord)
    return _sample_rows(rows, cfg.data.max_eval_prompts, cfg.data.prompt_sampling_seed + 1)

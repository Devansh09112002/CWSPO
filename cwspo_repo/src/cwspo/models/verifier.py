from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from cwspo.models.hf import load_causal_lm


class BaseVerifier(Protocol):
    def score_prefix(self, prompt: str, prefix: str) -> float:
        ...


@dataclass
class JudgeTokenVerifier:
    model_name: str
    prompt_template: str
    positive_token: str
    negative_token: str
    dtype: str = "bf16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    attn_implementation: str | None = None

    def __post_init__(self):
        self.model, self.tokenizer = load_causal_lm(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            load_in_4bit=self.load_in_4bit,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        self.pos_id = self.tokenizer.encode(self.positive_token, add_special_tokens=False)
        self.neg_id = self.tokenizer.encode(self.negative_token, add_special_tokens=False)
        if len(self.pos_id) != 1 or len(self.neg_id) != 1:
            raise ValueError(
                f"positive_token={self.positive_token!r} and negative_token={self.negative_token!r} must each map to exactly one token"
            )
        self.pos_id = self.pos_id[0]
        self.neg_id = self.neg_id[0]

    @torch.no_grad()
    def score_prefix(self, prompt: str, prefix: str) -> float:
        text = self.prompt_template.format(prompt=prompt, prefix=prefix)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        pos = probs[0, self.pos_id].item()
        neg = probs[0, self.neg_id].item()
        denom = pos + neg + 1e-8
        return float(pos / denom)


@dataclass
class MeanLogProbVerifier:
    model_name: str
    dtype: str = "bf16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    attn_implementation: str | None = None

    def __post_init__(self):
        self.model, self.tokenizer = load_causal_lm(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            load_in_4bit=self.load_in_4bit,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

    @torch.no_grad()
    def score_prefix(self, prompt: str, prefix: str) -> float:
        text = f"Question:\n{prompt}\n\nPartial solution:\n{prefix}"
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if enc.input_ids.shape[1] < 2:
            return 0.0
        out = self.model(**enc)
        logits = out.logits[:, :-1, :]
        target = enc.input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.mean().item())


def build_verifier(cfg, dtype: str):
    if cfg.mode == "judge_token":
        return JudgeTokenVerifier(
            model_name=cfg.model_name,
            prompt_template=cfg.prompt_template,
            positive_token=cfg.positive_token,
            negative_token=cfg.negative_token,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            load_in_4bit=cfg.load_in_4bit,
            attn_implementation=cfg.attn_implementation,
        )
    if cfg.mode == "mean_logprob":
        return MeanLogProbVerifier(
            model_name=cfg.model_name,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            load_in_4bit=cfg.load_in_4bit,
            attn_implementation=cfg.attn_implementation,
        )
    raise ValueError(f"Unknown verifier mode: {cfg.mode}")

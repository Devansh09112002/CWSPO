from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from cwspo.models.hf import load_auto_model, load_causal_lm, resolve_device_map


class BaseVerifier(Protocol):
    def score_prefix(self, prompt: str, prefix: str) -> float:
        ...


def split_prefix_lines(prefix: str) -> list[str]:
    steps = [line.strip() for line in prefix.splitlines() if line.strip()]
    if steps:
        return steps
    text = prefix.strip()
    return [text] if text else []


def extract_process_reward_probability(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    positive_label_index: int,
) -> float:
    probabilities = F.softmax(logits, dim=-1)
    selected = probabilities[token_mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.view(-1, probabilities.shape[-1])[-1, positive_label_index].item())


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
    device: str = "cuda"

    def __post_init__(self):
        self.model, self.tokenizer = load_causal_lm(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=resolve_device_map(self.device),
            load_in_4bit=self.load_in_4bit,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        self.pos_ids = self.tokenizer.encode(self.positive_token, add_special_tokens=False)
        self.neg_ids = self.tokenizer.encode(self.negative_token, add_special_tokens=False)
        if not self.pos_ids or not self.neg_ids:
            raise ValueError("Verifier labels must tokenize to at least one token each.")

    @torch.no_grad()
    def _label_logprob(self, prompt_text: str, label_ids: list[int]) -> float:
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        label = torch.tensor([label_ids], dtype=torch.long, device=self.model.device)
        full = torch.cat([prompt_ids, label], dim=1)
        out = self.model(full)
        logits = out.logits[:, :-1, :]
        target = full[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        start = prompt_ids.shape[1] - 1
        return float(token_lp[:, start:].sum().item())

    @torch.no_grad()
    def score_prefix(self, prompt: str, prefix: str) -> float:
        text = self.prompt_template.format(prompt=prompt, prefix=prefix)
        pos_lp = self._label_logprob(text, self.pos_ids)
        neg_lp = self._label_logprob(text, self.neg_ids)
        pair = torch.tensor([pos_lp, neg_lp], dtype=torch.float32)
        probs = torch.softmax(pair, dim=0)
        return float(probs[0].item())


@dataclass
class ProcessRewardModelVerifier:
    model_name: str
    dtype: str = "bf16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    device: str = "cuda"
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    step_token: str = "<extra_0>"
    positive_label_index: int = 1

    def __post_init__(self):
        self.model, self.tokenizer = load_auto_model(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=resolve_device_map(self.device),
            load_in_4bit=self.load_in_4bit,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        step_token_ids = self.tokenizer.encode(self.step_token, add_special_tokens=False)
        if len(step_token_ids) != 1:
            raise ValueError(f"Process reward step token must map to exactly one token, got {step_token_ids!r}.")
        self.step_token_id = step_token_ids[0]

    def _conversation_text(self, prompt: str, prefix: str) -> str:
        steps = split_prefix_lines(prefix)
        response = self.step_token.join(steps) + self.step_token
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return f"{self.system_prompt}\n\nQuestion:\n{prompt}\n\nPartial solution:\n{response}"

    @torch.no_grad()
    def score_prefix(self, prompt: str, prefix: str) -> float:
        text = self._conversation_text(prompt, prefix)
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model(**enc, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_mask = enc.input_ids == self.step_token_id
        return extract_process_reward_probability(
            logits,
            token_mask,
            positive_label_index=self.positive_label_index,
        )


@dataclass
class MeanLogProbVerifier:
    model_name: str
    dtype: str = "bf16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    device: str = "cuda"

    def __post_init__(self):
        self.model, self.tokenizer = load_causal_lm(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=resolve_device_map(self.device),
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
            device=getattr(cfg, "device", "cuda"),
        )
    if cfg.mode == "process_reward_model":
        return ProcessRewardModelVerifier(
            model_name=cfg.model_name,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            load_in_4bit=cfg.load_in_4bit,
            attn_implementation=cfg.attn_implementation,
            device=getattr(cfg, "device", "cuda"),
            system_prompt=getattr(cfg, "prm_system_prompt", "Please reason step by step, and put your final answer within \\boxed{}."),
            step_token=getattr(cfg, "prm_step_token", "<extra_0>"),
            positive_label_index=getattr(cfg, "prm_positive_label_index", 1),
        )
    if cfg.mode == "mean_logprob":
        return MeanLogProbVerifier(
            model_name=cfg.model_name,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            load_in_4bit=cfg.load_in_4bit,
            attn_implementation=cfg.attn_implementation,
            device=getattr(cfg, "device", "cuda"),
        )
    raise ValueError(f"Unknown verifier mode: {cfg.mode}")

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DeviceConfig(BaseModel):
    policy: str = "cuda"
    verifier: str = "cuda"


class PathsConfig(BaseModel):
    prompt_file: str
    output_dir: str
    traces_file: str
    scored_file: str
    pairs_file: str
    train_metrics_file: str
    final_eval_file: str
    process_eval_file: str
    checkpoint_dir: str


class PolicyConfig(BaseModel):
    model_name: str
    trust_remote_code: bool = True
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    num_return_sequences: int = 4
    load_in_4bit: bool = False
    attn_implementation: str | None = None


class VerifierConfig(BaseModel):
    mode: Literal["judge_token", "mean_logprob"] = "judge_token"
    model_name: str
    trust_remote_code: bool = True
    positive_token: str = "good"
    negative_token: str = "bad"
    prompt_template: str
    load_in_4bit: bool = False
    attn_implementation: str | None = None


class PairConfig(BaseModel):
    window_H: int = 2
    alpha_local: float = 0.8
    tau_pair: float = 0.1
    min_weight: float = 0.2
    max_pairs_per_prompt: int = 12


class ConfidenceConfig(BaseModel):
    tau_margin: float = 0.05
    scale_margin: float = 0.10
    scale_drop: float = 0.25
    gamma_margin: float = 0.40
    gamma_sharp: float = 0.20
    gamma_agree: float = 0.20
    gamma_out: float = 0.20


class TrainingConfig(BaseModel):
    model_name: str
    reference_model_name: str
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=list)
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    beta: float = 0.1
    lambda_ref: float = 0.0
    log_every: int = 10
    save_every: int = 200
    max_length: int = 2048
    load_in_4bit: bool = False
    attn_implementation: str | None = None


class EvaluationConfig(BaseModel):
    batch_size: int = 4
    max_new_tokens: int = 256


class AppConfig(BaseModel):
    seed: int = 42
    dtype: Literal["float16", "bf16", "float32"] = "bf16"
    device: DeviceConfig
    paths: PathsConfig
    policy: PolicyConfig
    verifier: VerifierConfig
    pair: PairConfig
    confidence: ConfidenceConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def ensure_dirs(cfg: AppConfig) -> None:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

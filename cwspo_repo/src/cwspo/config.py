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
    eval_prompt_file: str | None = None
    process_ground_truth_file: str | None = None
    output_dir: str
    traces_file: str
    scored_file: str
    pairs_file: str
    train_metrics_file: str
    final_eval_file: str
    process_eval_file: str
    checkpoint_dir: str
    confidence_analysis_file: str | None = None
    confidence_report_file: str | None = None
    pair_audit_low_file: str | None = None
    pair_audit_mid_file: str | None = None
    pair_audit_high_file: str | None = None
    pair_purity_report_file: str | None = None
    pair_orientation_audit_file: str | None = None
    training_report_file: str | None = None
    training_report_md_file: str | None = None
    process_failure_report_file: str | None = None
    run_summary_file: str | None = None


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
    mode: Literal["judge_token", "mean_logprob", "process_reward_model"] = "judge_token"
    model_name: str
    trust_remote_code: bool = True
    positive_token: str = "good"
    negative_token: str = "bad"
    prompt_template: str = ""
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    prm_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    prm_step_token: str = "<extra_0>"
    prm_positive_label_index: int = 1


class PairConfig(BaseModel):
    pair_mode: Literal[
        "current_utility",
        "correctness_priority",
        "strict_purified",
        "semi_purified",
    ] = "current_utility"
    window_H: int = 2
    alpha_local: float = 0.8
    tau_pair: float = 0.1
    min_weight: float = 0.2
    max_pairs_per_prompt: int = 12
    min_divergent_chars: int = 24
    max_near_identical_similarity: float = 0.94
    semi_purified_min_confidence: float = 0.82
    semi_purified_min_utility_margin: float = 0.35
    semi_purified_min_local_gap: float = 0.35


class ConfidenceConfig(BaseModel):
    tau_margin: float = 0.05
    scale_margin: float = 0.10
    scale_drop: float = 0.25
    gamma_margin: float = 0.40
    gamma_sharp: float = 0.20
    gamma_agree: float = 0.20
    gamma_out: float = 0.20
    use_margin: bool = True
    use_sharp: bool = True
    use_agree: bool = True
    use_out: bool = True
    low_threshold: float = 0.33
    high_threshold: float = 0.66


class DataConfig(BaseModel):
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    max_train_prompts: int | None = None
    max_eval_prompts: int | None = None
    max_process_examples: int | None = None
    prompt_sampling_seed: int = 42
    append_step_by_step_suffix: bool = True


class MethodConfig(BaseModel):
    name: Literal[
        "answer_dpo",
        "step_dpo",
        "confidence_filter_only",
        "confidence_weighted_step_dpo",
    ] = "confidence_weighted_step_dpo"
    confidence_threshold: float = 0.0


class ResumeConfig(BaseModel):
    use_existing_traces: bool = False
    use_existing_scored: bool = False
    use_existing_pairs: bool = False
    use_existing_checkpoint: bool = False
    use_existing_final_eval: bool = False
    use_existing_process_eval: bool = False


class DiagnosticsConfig(BaseModel):
    num_audit_samples_per_bucket: int = 8
    pair_audit_seed: int = 42
    histogram_bins: int = 10
    num_orientation_audit_samples: int = 4


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
    data: DataConfig = Field(default_factory=DataConfig)
    method: MethodConfig = Field(default_factory=MethodConfig)
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
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

    for key, value in cfg.paths.model_dump().items():
        if not value:
            continue
        path = Path(value)
        if key in {"output_dir", "checkpoint_dir"}:
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)

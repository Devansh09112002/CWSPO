from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def build_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_causal_lm(
    model_name: str,
    dtype: str = "bf16",
    trust_remote_code: bool = True,
    device_map: str | dict | None = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str | None = None,
):
    quant_config = build_quant_config(load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype_from_name(dtype),
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import torch
from peft import AutoPeftModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def torch_dtype_from_name(name: str):
    if not torch.cuda.is_available() and name in {"bf16", "float16"}:
        return torch.float32
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


def resolve_device_map(device: str | None):
    if device is None or device == "auto":
        return "auto"
    if device.startswith("cuda"):
        return "auto"
    return {"": device}


def _supports_attn_implementation(attn_implementation: str | None) -> bool:
    if attn_implementation is None:
        return True
    if attn_implementation != "flash_attention_2":
        return True
    try:
        import flash_attn  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_dynamic_cache_compat() -> None:
    if hasattr(DynamicCache, "from_legacy_cache"):
        needs_from_legacy = False
    else:
        needs_from_legacy = True

    if needs_from_legacy:
        @classmethod
        def _from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()
            if isinstance(past_key_values, cls):
                return past_key_values
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            del new_seq_length
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]


def _load_auto_model(
    loader,
    model_name: str,
    *,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | dict | None,
    load_in_4bit: bool,
    attn_implementation: str | None,
):
    _ensure_dynamic_cache_compat()
    effective_attn = attn_implementation if _supports_attn_implementation(attn_implementation) else None
    if attn_implementation and effective_attn is None:
        warn(
            f"attn_implementation={attn_implementation!r} is unavailable in this environment; falling back to the model default."
        )

    dtype_value = torch_dtype_from_name(dtype)
    config = None
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception:
        config = None
    if config is not None and getattr(config, "pad_token_id", None) is None:
        eos_token_id = getattr(config, "eos_token_id", None)
        bos_token_id = getattr(config, "bos_token_id", None)
        if eos_token_id is not None:
            config.pad_token_id = eos_token_id
        elif bos_token_id is not None:
            config.pad_token_id = bos_token_id
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "quantization_config": build_quant_config(load_in_4bit),
    }
    if config is not None:
        kwargs["config"] = config
    if effective_attn is not None:
        kwargs["attn_implementation"] = effective_attn

    last_exc: Exception | None = None
    for dtype_key in ("dtype", "torch_dtype"):
        try_kwargs = dict(kwargs)
        try_kwargs[dtype_key] = dtype_value
        try:
            return loader.from_pretrained(model_name, **try_kwargs)
        except TypeError as exc:
            last_exc = exc
            if "attn_implementation" in try_kwargs:
                retry_kwargs = dict(try_kwargs)
                retry_kwargs.pop("attn_implementation", None)
                try:
                    return loader.from_pretrained(model_name, **retry_kwargs)
                except TypeError as retry_exc:
                    last_exc = retry_exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to load model {model_name!r}.")


def _load_tokenizer(model_name: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    dtype: str = "bf16",
    trust_remote_code: bool = True,
    device_map: str | dict | None = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str | None = None,
):
    model = _load_auto_model(
        AutoModelForCausalLM,
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
    )
    tokenizer = _load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    return model, tokenizer


def load_auto_model(
    model_name: str,
    dtype: str = "bf16",
    trust_remote_code: bool = True,
    device_map: str | dict | None = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str | None = None,
):
    model = _load_auto_model(
        AutoModel,
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
    )
    tokenizer = _load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    return model, tokenizer


def load_causal_lm_or_adapter(
    model_name: str,
    dtype: str = "bf16",
    trust_remote_code: bool = True,
    device_map: str | dict | None = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str | None = None,
):
    model_path = Path(model_name)
    if model_path.exists() and (model_path / "adapter_config.json").exists():
        model = _load_auto_model(
            AutoPeftModelForCausalLM,
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
        )
        tokenizer = _load_tokenizer(model_name, trust_remote_code=trust_remote_code)
        return model, tokenizer
    return load_causal_lm(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
    )

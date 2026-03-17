from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from cwspo.models.hf import load_causal_lm_or_adapter, resolve_device_map
from cwspo.schemas import PromptRecord
from cwspo.utils.math import extract_final_answer, is_correct_answer


def resolve_eval_model_name(cfg, model_name: str | None = None) -> str:
    if model_name:
        return model_name
    final_ckpt = Path(cfg.paths.checkpoint_dir) / "final"
    if final_ckpt.exists():
        return str(final_ckpt)
    return cfg.training.model_name


def evaluate_final(cfg, prompts: list[PromptRecord], model_name: str | None = None) -> dict:
    eval_model_name = resolve_eval_model_name(cfg, model_name)
    adapter_path = None
    adapter_loaded = False
    if Path(eval_model_name).exists() and (Path(eval_model_name) / "adapter_config.json").exists():
        adapter_path = eval_model_name
        adapter_loaded = True
    model, tokenizer = load_causal_lm_or_adapter(
        eval_model_name,
        dtype=cfg.dtype,
        trust_remote_code=cfg.training.trust_remote_code,
        device_map=resolve_device_map(cfg.device.policy),
        load_in_4bit=cfg.training.load_in_4bit,
        attn_implementation=cfg.training.attn_implementation,
    )
    tokenizer.padding_side = "left"
    model.eval()

    correct = 0
    rows = []
    batch_size = max(1, int(cfg.evaluation.batch_size))
    for i in tqdm(range(0, len(prompts), batch_size), desc="Final evaluation"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer([row.prompt for row in batch], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs.input_ids.shape[1]
        for row, seq in zip(batch, out):
            decoded = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            final_answer = extract_final_answer(decoded)
            ok = is_correct_answer(final_answer, row.answer)
            correct += int(ok)
            rows.append({"id": row.id, "pred": final_answer, "gold": row.answer, "correct": ok})
    return {
        "model_name": eval_model_name,
        "base_model_name": cfg.training.model_name,
        "adapter_path": adapter_path,
        "adapter_loaded": adapter_loaded,
        "accuracy": correct / max(1, len(prompts)),
        "num_examples": len(prompts),
        "rows": rows,
    }

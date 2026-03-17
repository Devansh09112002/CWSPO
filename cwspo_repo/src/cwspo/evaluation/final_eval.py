from __future__ import annotations

from tqdm import tqdm

from cwspo.models.hf import load_causal_lm
from cwspo.schemas import PromptRecord
from cwspo.utils.math import extract_final_answer, is_correct_answer


def evaluate_final(cfg, prompts: list[PromptRecord], model_name: str | None = None) -> dict:
    eval_model_name = model_name or cfg.training.model_name
    model, tokenizer = load_causal_lm(
        eval_model_name,
        dtype=cfg.dtype,
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=False,
    )
    model.eval()

    correct = 0
    rows = []
    for row in tqdm(prompts, desc="Final evaluation"):
        inputs = tokenizer(row.prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.evaluation.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        final_answer = extract_final_answer(decoded)
        ok = is_correct_answer(final_answer, row.answer)
        correct += int(ok)
        rows.append({"id": row.id, "pred": final_answer, "gold": row.answer, "correct": ok})
    return {
        "accuracy": correct / max(1, len(prompts)),
        "num_examples": len(prompts),
        "rows": rows,
    }

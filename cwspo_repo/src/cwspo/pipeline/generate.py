from __future__ import annotations

from tqdm import tqdm

from cwspo.models.hf import load_causal_lm, resolve_device_map
from cwspo.schemas import PromptRecord, TraceRecord
from cwspo.utils.math import extract_final_answer
from cwspo.utils.steps import split_steps


def generate_traces(cfg, prompts: list[PromptRecord]) -> list[TraceRecord]:
    model, tokenizer = load_causal_lm(
        cfg.policy.model_name,
        dtype=cfg.dtype,
        trust_remote_code=cfg.policy.trust_remote_code,
        device_map=resolve_device_map(cfg.device.policy),
        load_in_4bit=cfg.policy.load_in_4bit,
        attn_implementation=cfg.policy.attn_implementation,
    )
    model.eval()

    results: list[TraceRecord] = []
    for row in tqdm(prompts, desc="Generating traces"):
        inputs = tokenizer(row.prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.policy.max_new_tokens,
            do_sample=cfg.policy.do_sample,
            temperature=cfg.policy.temperature,
            top_p=cfg.policy.top_p,
            num_return_sequences=cfg.policy.num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i in range(outputs.shape[0]):
            decoded = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            steps = split_steps(decoded)
            final_answer = extract_final_answer(decoded)
            results.append(
                TraceRecord(
                    id=row.id,
                    prompt=row.prompt,
                    answer=row.answer,
                    trace_id=i,
                    reasoning=decoded,
                    steps=steps,
                    final_answer=final_answer,
                )
            )
    return results

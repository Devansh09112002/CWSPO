from __future__ import annotations

from tqdm import tqdm

from cwspo.models.verifier import build_verifier
from cwspo.schemas import ScoredTraceRecord, TraceRecord
from cwspo.utils.math import is_correct_answer


def score_traces(cfg, traces: list[TraceRecord]) -> list[ScoredTraceRecord]:
    verifier_cfg = cfg.verifier.model_copy(update={"device": cfg.device.verifier})
    verifier = build_verifier(verifier_cfg, dtype=cfg.dtype)
    scored: list[ScoredTraceRecord] = []
    for tr in tqdm(traces, desc="Scoring traces"):
        prefix = ""
        scores: list[float] = []
        for step in tr.steps:
            prefix = prefix + "\n" + step if prefix else step
            scores.append(verifier.score_prefix(tr.prompt, prefix))
        scored.append(
            ScoredTraceRecord(
                id=tr.id,
                prompt=tr.prompt,
                answer=tr.answer,
                trace_id=tr.trace_id,
                steps=tr.steps,
                step_scores=scores,
                reasoning=tr.reasoning,
                final_answer=tr.final_answer,
                final_correct=is_correct_answer(tr.final_answer, tr.answer),
            )
        )
    return scored

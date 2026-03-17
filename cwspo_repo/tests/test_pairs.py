from types import SimpleNamespace

from cwspo.pipeline.build_pairs import build_pairs
from cwspo.schemas import ScoredTraceRecord


def test_build_pairs_smoke():
    cfg = SimpleNamespace(
        pair=SimpleNamespace(window_H=2, alpha_local=0.8, tau_pair=0.05, min_weight=0.0, max_pairs_per_prompt=10),
        confidence=SimpleNamespace(
            tau_margin=0.05,
            scale_margin=0.1,
            scale_drop=0.25,
            gamma_margin=0.4,
            gamma_sharp=0.2,
            gamma_agree=0.2,
            gamma_out=0.2,
        ),
    )
    traces = [
        ScoredTraceRecord(
            id="ex1",
            prompt="Q",
            answer="8",
            trace_id=0,
            steps=["There are 3 red.", "There are 5 blue.", "3+5=8"],
            step_scores=[0.9, 0.85, 0.92],
            reasoning="",
            final_answer="8",
            final_correct=True,
        ),
        ScoredTraceRecord(
            id="ex1",
            prompt="Q",
            answer="8",
            trace_id=1,
            steps=["There are 3 red.", "There are 5 blue.", "3+5=7"],
            step_scores=[0.9, 0.4, 0.1],
            reasoning="",
            final_answer="7",
            final_correct=False,
        ),
    ]

    pairs = build_pairs(cfg, traces)
    assert len(pairs) >= 1
    assert pairs[0].weight >= 0.0
    assert pairs[0].meta["k"] >= 0

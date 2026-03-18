from math import isclose
from types import SimpleNamespace

from cwspo.pipeline.build_pairs import build_pair_artifacts, build_pairs, confidence_features
from cwspo.pipeline.diagnostics import analyze_pairs
from cwspo.schemas import PairRecord, ScoredTraceRecord


def _cfg(
    method_name: str = "confidence_weighted_step_dpo",
    min_weight: float = 0.0,
    confidence_threshold: float = 0.0,
    pair_mode: str = "current_utility",
):
    return SimpleNamespace(
        pair=SimpleNamespace(
            pair_mode=pair_mode,
            window_H=2,
            alpha_local=0.8,
            tau_pair=0.05,
            min_weight=min_weight,
            max_pairs_per_prompt=10,
            min_divergent_chars=8,
            max_near_identical_similarity=0.9,
            semi_purified_min_confidence=0.82,
            semi_purified_min_utility_margin=0.35,
            semi_purified_min_local_gap=0.35,
            semi_purified_both_correct_min_confidence=0.76,
            semi_purified_both_correct_min_utility_margin=0.25,
            semi_purified_both_correct_min_local_gap=0.20,
            semi_purified_both_correct_min_support_gap=0.05,
            semi_purified_both_wrong_min_confidence=0.88,
            semi_purified_both_wrong_min_utility_margin=0.45,
            semi_purified_both_wrong_min_local_gap=0.35,
            semi_purified_both_wrong_min_support_gap=0.10,
            semi_purified_both_wrong_min_drop_advantage=0.15,
        ),
        confidence=SimpleNamespace(
            tau_margin=0.05,
            scale_margin=0.1,
            scale_drop=0.25,
            gamma_margin=0.4,
            gamma_sharp=0.2,
            gamma_agree=0.2,
            gamma_out=0.2,
            use_margin=True,
            use_sharp=True,
            use_agree=True,
            use_out=True,
        ),
        method=SimpleNamespace(name=method_name, confidence_threshold=confidence_threshold),
    )


def _traces():
    return [
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


def test_confidence_features_stay_bounded():
    feats = confidence_features(
        pref_info={"utility": 0.9, "drop_at_k": 0.05, "local_score": 0.8},
        disp_info={"utility": 0.1, "drop_at_k": 0.5, "local_score": 0.1},
        support_stats={"n_pref": 4, "n_other": 1, "p_corr_pref": 1.0, "p_corr_other": 0.0},
        cfg=_cfg().confidence,
    )

    assert 0.0 <= feats["f_margin"] <= 1.0
    assert 0.0 <= feats["f_sharp"] <= 1.0
    assert 0.0 <= feats["f_agree"] <= 1.0
    assert 0.0 <= feats["f_out"] <= 1.0
    assert 0.0 <= feats["confidence"] <= 1.0
    assert feats["confidence"] > 0.5


def test_build_pairs_prefers_higher_utility_branch():
    pairs = build_pairs(_cfg(), _traces())
    assert len(pairs) == 1
    assert pairs[0].prefix_steps == ["There are 3 red.", "There are 5 blue."]
    assert pairs[0].preferred_steps == ["3+5=8"]
    assert pairs[0].dispreferred_steps == ["3+5=7"]
    assert pairs[0].meta["k"] == 2
    assert pairs[0].weight >= 0.0
    assert pairs[0].confidence is not None
    assert isclose(pairs[0].confidence, pairs[0].features["confidence"])


def test_step_dpo_uses_uniform_training_weights():
    artifacts = build_pair_artifacts(_cfg(method_name="step_dpo"), _traces())
    assert len(artifacts.pairs) == 1
    assert artifacts.pairs[0].weight == 1.0
    assert artifacts.pairs[0].confidence is not None


def test_answer_dpo_builds_full_trace_pairs():
    artifacts = build_pair_artifacts(_cfg(method_name="answer_dpo"), _traces())
    assert len(artifacts.pairs) == 1
    pair = artifacts.pairs[0]
    assert pair.meta["pair_type"] == "answer_level"
    assert pair.prefix_steps == []
    assert pair.preferred_steps == _traces()[0].steps
    assert pair.dispreferred_steps == _traces()[1].steps
    assert pair.weight == 1.0
    assert pair.confidence is None


def test_build_pairs_skips_empty_post_divergence_segments():
    traces = [
        ScoredTraceRecord(
            id="ex2",
            prompt="Q",
            answer="8",
            trace_id=0,
            steps=["There are 3 red.", "There are 5 blue."],
            step_scores=[0.9, 0.85],
            reasoning="",
            final_answer="8",
            final_correct=True,
        ),
        ScoredTraceRecord(
            id="ex2",
            prompt="Q",
            answer="8",
            trace_id=1,
            steps=["There are 3 red."],
            step_scores=[0.9],
            reasoning="",
            final_answer="3",
            final_correct=False,
        ),
    ]

    pairs = build_pairs(_cfg(), traces)
    assert pairs == []


def test_analyze_pairs_reports_ambiguous_pair_rates():
    pairs = [
        PairRecord(
            id="ex1",
            prompt="Q",
            prefix_steps=[],
            preferred_steps=["good"],
            dispreferred_steps=["bad"],
            weight=0.9,
            confidence=0.9,
            features={"confidence": 0.9},
            meta={"pref_final_correct": True, "disp_final_correct": False},
        ),
        PairRecord(
            id="ex2",
            prompt="Q",
            prefix_steps=[],
            preferred_steps=["still wrong"],
            dispreferred_steps=["also wrong"],
            weight=0.85,
            confidence=0.85,
            features={"confidence": 0.85},
            meta={"pref_final_correct": False, "disp_final_correct": False},
        ),
    ]

    analysis = analyze_pairs(pairs, low_threshold=0.33, high_threshold=0.66)
    assert isclose(analysis["preferred_branch_final_correct_rate"], 0.5)
    assert isclose(analysis["both_branches_incorrect_fraction"], 0.5)
    assert isclose(analysis["decisive_pair_fraction"], 0.5)
    assert isclose(analysis["bucket_summary"]["high"]["both_branches_incorrect_fraction"], 0.5)


def test_correctness_priority_prefers_correct_branch_over_higher_utility_incorrect_branch():
    traces = [
        ScoredTraceRecord(
            id="ex3",
            prompt="Q",
            answer="8",
            trace_id=0,
            steps=["Add the counts.", "3+5=8"],
            step_scores=[0.1, 0.1],
            reasoning="",
            final_answer="8",
            final_correct=True,
        ),
        ScoredTraceRecord(
            id="ex3",
            prompt="Q",
            answer="8",
            trace_id=1,
            steps=["Add the counts.", "3+5=7"],
            step_scores=[0.9, 0.9],
            reasoning="",
            final_answer="7",
            final_correct=False,
        ),
    ]

    current = build_pair_artifacts(_cfg(method_name="step_dpo", pair_mode="current_utility"), traces)
    assert len(current.pairs) == 1
    assert current.pairs[0].meta["pref_final_correct"] is False
    assert current.pairs[0].meta["correctness_pattern"] == "incorrect_vs_correct"

    refined = build_pair_artifacts(_cfg(method_name="step_dpo", pair_mode="correctness_priority"), traces)
    assert len(refined.pairs) == 1
    assert refined.pairs[0].meta["pref_final_correct"] is True
    assert refined.pairs[0].meta["correctness_pattern"] == "correct_vs_incorrect"
    assert refined.pairs[0].meta["orientation_reason"] == "correctness_priority_final_correctness"


def test_strict_purified_drops_same_correctness_pairs():
    traces = [
        ScoredTraceRecord(
            id="ex4",
            prompt="Q",
            answer="10",
            trace_id=0,
            steps=["Start.", "Add the values and wrongly claim the total is five after combining them."],
            step_scores=[0.8, 0.2],
            reasoning="",
            final_answer="5",
            final_correct=False,
        ),
        ScoredTraceRecord(
            id="ex4",
            prompt="Q",
            answer="10",
            trace_id=1,
            steps=["Start.", "Multiply unrelated quantities and wrongly conclude the total should be six instead."],
            step_scores=[0.7, 0.1],
            reasoning="",
            final_answer="6",
            final_correct=False,
        ),
    ]

    artifacts = build_pair_artifacts(_cfg(method_name="step_dpo", pair_mode="strict_purified"), traces)
    assert artifacts.pairs == []
    assert artifacts.pair_purity_report["status_counts"]["dropped_both_wrong_uninformative"] == 1


def test_strict_purified_drops_weak_near_identical_mixed_correctness_pairs():
    traces = [
        ScoredTraceRecord(
            id="ex5",
            prompt="Q",
            answer="54",
            trace_id=0,
            steps=["Add salary raise.", "Teacher makes 54000."],
            step_scores=[0.2, 0.2],
            reasoning="",
            final_answer="54",
            final_correct=True,
        ),
        ScoredTraceRecord(
            id="ex5",
            prompt="Q",
            answer="54",
            trace_id=1,
            steps=["Add salary raise.", "Teacher makes $54,000."],
            step_scores=[0.9, 0.9],
            reasoning="",
            final_answer="000",
            final_correct=False,
        ),
    ]

    artifacts = build_pair_artifacts(_cfg(method_name="step_dpo", pair_mode="strict_purified"), traces)
    assert artifacts.pairs == []
    assert artifacts.pair_purity_report["status_counts"]["dropped_no_divergence"] == 1


def test_pair_purity_report_tracks_orientation_and_taxonomy():
    artifacts = build_pair_artifacts(_cfg(method_name="step_dpo", pair_mode="current_utility"), _traces())
    report = artifacts.pair_purity_report
    assert report["pair_mode"] == "current_utility"
    assert "pair_taxonomy" in report
    assert "pair_purity_metrics" in report
    assert report["pair_taxonomy"]["correct_vs_incorrect"] + report["pair_taxonomy"]["incorrect_vs_correct"] == 1


def test_semi_purified_rejects_both_wrong_pairs_without_delayed_error_evidence():
    traces = [
        ScoredTraceRecord(
            id="ex6",
            prompt="Q",
            answer="10",
            trace_id=0,
            steps=["Start.", "Add the values and wrongly claim the total is five after combining them."],
            step_scores=[0.9, 0.9],
            reasoning="",
            final_answer="5",
            final_correct=False,
        ),
        ScoredTraceRecord(
            id="ex6",
            prompt="Q",
            answer="10",
            trace_id=1,
            steps=["Start.", "Multiply unrelated quantities and wrongly conclude the total should be seven instead."],
            step_scores=[0.1, 0.1],
            reasoning="",
            final_answer="7",
            final_correct=False,
        ),
    ]

    artifacts = build_pair_artifacts(_cfg(method_name="confidence_weighted_step_dpo", pair_mode="semi_purified"), traces)
    assert artifacts.pairs == []
    assert artifacts.pair_purity_report["status_counts"]["dropped_same_correctness_low_confidence"] == 1

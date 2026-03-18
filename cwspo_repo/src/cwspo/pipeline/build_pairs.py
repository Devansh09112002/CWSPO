from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

import numpy as np

from cwspo.schemas import PairRecord, ScoredTraceRecord
from cwspo.utils.steps import canon


@dataclass
class PairBuildArtifacts:
    pairs: list[PairRecord]
    num_raw_pairs: int
    num_kept_pairs: int
    num_dropped_by_confidence: int
    method_name: str
    pair_type: str
    pair_mode: str
    pair_purity_report: dict[str, Any]
    orientation_audit_rows: list[dict[str, Any]]


def first_divergence(steps_a: list[str], steps_b: list[str]) -> int | None:
    t = min(len(steps_a), len(steps_b))
    for i in range(t):
        if canon(steps_a[i]) != canon(steps_b[i]):
            return i
    if len(steps_a) != len(steps_b):
        return t
    return None


def normalize_scores_across_prompt(traces: list[ScoredTraceRecord]) -> dict[int, list[float]]:
    flat = np.array([s for tr in traces for s in tr.step_scores], dtype=float)
    mu = float(flat.mean()) if flat.size else 0.0
    std = float(flat.std()) + 1e-6
    return {tr.trace_id: [float((x - mu) / std) for x in tr.step_scores] for tr in traces}


def local_segment_score(scores: list[float], k: int, h: int) -> float:
    seg = scores[k : min(len(scores), k + h)]
    if not seg:
        return -1e9
    return float(sum(seg) / len(seg))


def drop_at_k(scores: list[float], k: int) -> float:
    if k <= 0 or k >= len(scores):
        return 0.0
    return max(0.0, scores[k - 1] - scores[k])


def utility(local_score: float, final_correct: bool, alpha: float) -> float:
    return alpha * local_score + (1.0 - alpha) * float(final_correct)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


_GENERIC_PREAMBLE_PHRASES = (
    "let's break down",
    "lets break down",
    "let's solve",
    "lets solve",
    "step by step",
    "follow these steps",
    "we need to",
    "to determine",
    "let us",
    "first,",
    "first ",
)

_CONTENT_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "there",
    "then",
    "into",
    "have",
    "has",
    "had",
    "will",
    "would",
    "should",
    "could",
    "need",
    "needs",
    "show",
    "your",
    "reasoning",
    "step",
    "steps",
    "problem",
    "follow",
    "these",
    "solve",
    "determine",
    "calculate",
    "using",
    "after",
    "before",
    "each",
    "total",
}


def _alnum_signature(text: str) -> str:
    return "".join(ch for ch in canon(text) if ch.isalnum())


def _content_tokens(text: str) -> set[str]:
    tokens = {
        token
        for token in canon(text).split()
        if len(token) >= 3 and token not in _CONTENT_STOPWORDS
    }
    return tokens


def _contains_math_signal(text: str) -> bool:
    canonical = canon(text)
    return any(ch.isdigit() for ch in canonical) or any(op in canonical for op in ("+", "-", "*", "/", "="))


def _looks_like_generic_preamble(text: str) -> bool:
    canonical = canon(text)
    return any(phrase in canonical for phrase in _GENERIC_PREAMBLE_PHRASES)


def estimate_support_stats(
    traces: list[ScoredTraceRecord],
    k: int,
    pref_trace: ScoredTraceRecord,
    disp_trace: ScoredTraceRecord,
) -> dict[str, float]:
    pref_sig = canon(pref_trace.steps[k]) if k < len(pref_trace.steps) else "<eos>"
    disp_sig = canon(disp_trace.steps[k]) if k < len(disp_trace.steps) else "<eos>"
    n_pref = 0
    n_other = 0
    pref_correct: list[int] = []
    other_correct: list[int] = []
    for tr in traces:
        sig = canon(tr.steps[k]) if k < len(tr.steps) else "<eos>"
        if sig == pref_sig:
            n_pref += 1
            pref_correct.append(int(tr.final_correct))
        elif sig == disp_sig:
            n_other += 1
            other_correct.append(int(tr.final_correct))
    p_corr_pref = sum(pref_correct) / max(1, len(pref_correct))
    p_corr_other = sum(other_correct) / max(1, len(other_correct))
    return {
        "n_pref": n_pref,
        "n_other": n_other,
        "p_corr_pref": p_corr_pref,
        "p_corr_other": p_corr_other,
        "support_gap": p_corr_pref - p_corr_other,
    }


def confidence_features(
    pref_info: dict[str, float],
    disp_info: dict[str, float],
    support_stats: dict[str, float],
    cfg,
) -> dict[str, float]:
    margin = abs(pref_info["utility"] - disp_info["utility"])
    f_margin = sigmoid((margin - cfg.tau_margin) / cfg.scale_margin)

    sharp = disp_info["drop_at_k"] - pref_info["drop_at_k"]
    f_sharp = sigmoid(sharp / cfg.scale_drop)

    n_pref = support_stats["n_pref"]
    n_other = support_stats["n_other"]
    f_agree = n_pref / (n_pref + n_other + 1e-6)

    p_corr_pref = support_stats["p_corr_pref"]
    p_corr_other = support_stats["p_corr_other"]
    f_out = (1.0 + (p_corr_pref - p_corr_other)) / 2.0
    f_out = max(0.0, min(1.0, f_out))

    weighted_terms: list[tuple[float, float]] = []
    if getattr(cfg, "use_margin", True):
        weighted_terms.append((cfg.gamma_margin, f_margin))
    if getattr(cfg, "use_sharp", True):
        weighted_terms.append((cfg.gamma_sharp, f_sharp))
    if getattr(cfg, "use_agree", True):
        weighted_terms.append((cfg.gamma_agree, f_agree))
    if getattr(cfg, "use_out", True):
        weighted_terms.append((cfg.gamma_out, f_out))

    denom = sum(weight for weight, _ in weighted_terms) or 1.0
    w = sum(weight * value for weight, value in weighted_terms) / denom
    w = max(0.0, min(1.0, w))
    return {
        "f_margin": float(f_margin),
        "f_sharp": float(f_sharp),
        "f_agree": float(f_agree),
        "f_out": float(f_out),
        "confidence": float(w),
    }


def _method_name(cfg) -> str:
    return getattr(getattr(cfg, "method", None), "name", "confidence_weighted_step_dpo")


def _pair_mode_name(cfg) -> str:
    return getattr(getattr(cfg, "pair", None), "pair_mode", "current_utility")


def _confidence_threshold(cfg) -> float:
    method_name = _method_name(cfg)
    configured = float(getattr(getattr(cfg, "method", None), "confidence_threshold", 0.0))
    legacy_min = float(getattr(cfg.pair, "min_weight", 0.0))
    if method_name in {"confidence_filter_only", "confidence_weighted_step_dpo"}:
        return max(configured, legacy_min)
    return 0.0


def _pair_type(cfg) -> str:
    return "answer_level" if _method_name(cfg) == "answer_dpo" else "local_step"


def _segment_text(steps: list[str]) -> str:
    return "\n".join(steps).strip()


def _segment_signature(steps: list[str]) -> str:
    text = " ".join(canon(step) for step in steps if step.strip())
    return " ".join(text.split())


def _segment_similarity(steps_a: list[str], steps_b: list[str]) -> float:
    sig_a = _segment_signature(steps_a)
    sig_b = _segment_signature(steps_b)
    if not sig_a and not sig_b:
        return 1.0
    return float(SequenceMatcher(None, sig_a, sig_b).ratio())


def _reconverges_after_boundary(a: ScoredTraceRecord, b: ScoredTraceRecord, k: int, h: int) -> bool:
    tail_a = [canon(step) for step in a.steps[k + 1 : min(len(a.steps), k + h)]]
    tail_b = [canon(step) for step in b.steps[k + 1 : min(len(b.steps), k + h)]]
    return bool(tail_a and tail_b and tail_a == tail_b)


def _divergence_diagnostics(
    a: ScoredTraceRecord,
    b: ScoredTraceRecord,
    *,
    k: int,
    h: int,
    cfg,
) -> dict[str, Any]:
    a_seg = a.steps[k : min(len(a.steps), k + h)]
    b_seg = b.steps[k : min(len(b.steps), k + h)]
    a_text = _segment_text(a_seg)
    b_text = _segment_text(b_seg)
    sig_a = _segment_signature(a_seg)
    sig_b = _segment_signature(b_seg)
    similarity = _segment_similarity(a_seg, b_seg)
    min_chars = min(len(sig_a), len(sig_b))
    too_short = min_chars < int(getattr(cfg.pair, "min_divergent_chars", 24))
    exact_match = sig_a == sig_b
    near_identical = similarity >= float(getattr(cfg.pair, "max_near_identical_similarity", 0.94))
    reconverges = _reconverges_after_boundary(a, b, k, h)
    formatting_only = bool(_alnum_signature(a_text) == _alnum_signature(b_text))
    a_tokens = _content_tokens(a_text)
    b_tokens = _content_tokens(b_text)
    trivial_difference = bool(
        too_short
        or (
            a_tokens == b_tokens
            and not _contains_math_signal(a_text)
            and not _contains_math_signal(b_text)
        )
    )
    boilerplate_only = bool(
        _looks_like_generic_preamble(a_text)
        and _looks_like_generic_preamble(b_text)
        and not _contains_math_signal(a_text)
        and not _contains_math_signal(b_text)
    )
    weak_divergence = bool(
        too_short
        or exact_match
        or formatting_only
        or near_identical
        or reconverges
        or trivial_difference
        or boilerplate_only
    )
    return {
        "similarity": float(similarity),
        "min_chars": int(min_chars),
        "too_short": bool(too_short),
        "exact_match": bool(exact_match),
        "formatting_only": bool(formatting_only),
        "near_identical": bool(near_identical),
        "trivial_difference": bool(trivial_difference),
        "boilerplate_only": bool(boilerplate_only),
        "reconverges_after_boundary": bool(reconverges),
        "weak_divergence": bool(weak_divergence),
    }


def _correctness_pattern(pref_correct: bool, disp_correct: bool) -> str:
    if pref_correct and not disp_correct:
        return "correct_vs_incorrect"
    if (not pref_correct) and disp_correct:
        return "incorrect_vs_correct"
    if pref_correct and disp_correct:
        return "both_correct"
    return "both_wrong"


def _correctness_bucket(a_correct: bool | None, b_correct: bool | None) -> str:
    if a_correct is None or b_correct is None:
        return "unresolved"
    if a_correct != b_correct:
        return "mixed_correctness"
    if a_correct:
        return "both_correct"
    return "both_wrong"


def _is_kept_status(status: str | None) -> bool:
    return bool(status and status.startswith("kept"))


def _confidence_bucket(confidence: float | None, cfg) -> str:
    if confidence is None:
        return "unknown"
    if confidence < float(getattr(cfg.confidence, "low_threshold", 0.33)):
        return "low"
    if confidence >= float(getattr(cfg.confidence, "high_threshold", 0.66)):
        return "high"
    return "medium"


def _value_counts(decisions: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in decisions:
        value = row.get(key) or "unknown"
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _divergence_quality_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "weak_divergence": sum(int(bool(row.get("weak_divergence"))) for row in decisions),
        "formatting_only": sum(int(bool(row.get("formatting_only_divergence"))) for row in decisions),
        "near_identical": sum(int(bool(row.get("near_identical_divergence"))) for row in decisions),
        "trivial_difference": sum(int(bool(row.get("trivial_difference_divergence"))) for row in decisions),
        "boilerplate_only": sum(int(bool(row.get("boilerplate_only_divergence"))) for row in decisions),
        "unstable_boundary": sum(int(bool(row.get("boundary_unstable"))) for row in decisions),
        "degenerate_divergence": sum(int(bool(row.get("degenerate_divergence"))) for row in decisions),
    }


def _orientation_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in decisions:
        if not _is_kept_status(row.get("final_status")):
            continue
        reason = row.get("orientation_reason") or "unknown"
        counts[reason] += 1
    return dict(sorted(counts.items()))


def _pair_taxonomy_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "correct_vs_incorrect": 0,
        "incorrect_vs_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "unresolved_or_skipped": 0,
        "same_prefix_but_weak_divergence": 0,
        "same_correctness_but_utility_oriented": 0,
    }
    for row in decisions:
        if row.get("weak_divergence"):
            counts["same_prefix_but_weak_divergence"] += 1
        if not _is_kept_status(row.get("final_status")):
            counts["unresolved_or_skipped"] += 1
            continue
        pattern = row.get("correctness_pattern")
        if pattern in counts:
            counts[pattern] += 1
        if row.get("orientation_reason") in {
            "current_utility",
            "correctness_priority_same_correctness_utility",
            "semi_purified_same_correctness_utility_screening",
        } and pattern in {"both_correct", "both_wrong"}:
            counts["same_correctness_but_utility_oriented"] += 1
    return counts


def _build_pair_purity_report(
    *,
    cfg,
    method_name: str,
    pair_type: str,
    pair_mode: str,
    traces: list[ScoredTraceRecord],
    decisions: list[dict[str, Any]],
    num_raw_pairs: int,
    num_dropped_by_confidence: int,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter(row.get("final_status", "unknown") for row in decisions)
    total = len(decisions)
    kept = [row for row in decisions if _is_kept_status(row.get("final_status"))]
    weak = [row for row in decisions if row.get("weak_divergence")]
    near_identical = [row for row in decisions if row.get("near_identical_divergence")]
    degenerate = [row for row in decisions if row.get("degenerate_divergence")]
    unstable = [row for row in decisions if row.get("boundary_unstable")]
    short_regions = [
        row
        for row in decisions
        if (row.get("divergent_min_chars") if row.get("divergent_min_chars") is not None else 0)
        < getattr(cfg.pair, "min_divergent_chars", 24)
    ]

    correctness_driven = sum(
        1
        for row in kept
        if row.get("orientation_reason")
        in {
            "correctness_priority_final_correctness",
            "strict_purified_final_correctness",
            "semi_purified_final_correctness",
        }
    )
    utility_driven = sum(
        1
        for row in kept
        if row.get("orientation_reason")
        in {
            "current_utility",
            "correctness_priority_same_correctness_utility",
            "semi_purified_same_correctness_utility_screening",
        }
    )
    taxonomy = _pair_taxonomy_counts(decisions)
    avg_steps = float(np.mean([len(trace.steps) for trace in traces])) if traces else None

    return {
        "method_name": method_name,
        "pair_type": pair_type,
        "pair_mode": pair_mode,
        "num_trace_comparisons": total,
        "num_builder_candidates_before_method": num_raw_pairs,
        "num_kept_pairs": len(kept),
        "num_dropped_by_confidence": num_dropped_by_confidence,
        "status_counts": dict(sorted(status_counts.items())),
        "admissibility_diagnostics": {
            "reason_code_counts": dict(sorted(status_counts.items())),
            "correctness_bucket_counts": _value_counts(decisions, "correctness_bucket"),
            "kept_correctness_pattern_counts": _value_counts(kept, "correctness_pattern"),
            "pair_mode_counts": _value_counts(decisions, "pair_mode"),
            "confidence_bucket_counts": _value_counts(decisions, "confidence_bucket"),
            "divergence_quality_counts": _divergence_quality_counts(decisions),
        },
        "pair_taxonomy": taxonomy,
        "pair_purity_metrics": {
            "fraction_strictly_instructional_pairs": float(taxonomy["correct_vs_incorrect"] / max(1, len(kept))) if kept else None,
            "fraction_ambiguous_pairs": float((taxonomy["both_correct"] + taxonomy["both_wrong"]) / max(1, len(kept))) if kept else None,
            "fraction_misoriented_mixed_correctness_pairs": float(taxonomy["incorrect_vs_correct"] / max(1, len(kept))) if kept else None,
            "fraction_dropped_by_purification": float(
                (
                    status_counts.get("dropped_both_correct_ambiguous", 0)
                    + status_counts.get("dropped_both_wrong_uninformative", 0)
                    + status_counts.get("dropped_weak_divergence", 0)
                    + status_counts.get("dropped_trivial_segment_difference", 0)
                    + status_counts.get("dropped_near_identical", 0)
                    + status_counts.get("dropped_formatting_only", 0)
                    + status_counts.get("dropped_unstable_boundary", 0)
                )
                / max(1, total)
            ),
            "fraction_orientation_driven_by_correctness": float(correctness_driven / max(1, len(kept))) if kept else None,
            "fraction_orientation_driven_only_by_utility": float(utility_driven / max(1, len(kept))) if kept else None,
        },
        "boundary_diagnostics": {
            "average_steps_per_trace": avg_steps,
            "boundary_instability_rate": float(len(unstable) / max(1, total)),
            "weak_divergence_fraction": float(len(weak) / max(1, total)),
            "degenerate_divergence_fraction": float(len(degenerate) / max(1, total)),
            "near_identical_divergence_fraction": float(len(near_identical) / max(1, total)),
            "very_short_divergent_region_fraction": float(len(short_regions) / max(1, total)),
        },
        "orientation_counts": _orientation_counts(decisions),
    }


def _divergence_reason_code(pair_mode: str, divergence: dict[str, Any]) -> str | None:
    if pair_mode not in {"strict_purified", "semi_purified"}:
        return None
    if divergence.get("formatting_only"):
        return "dropped_formatting_only"
    if divergence.get("near_identical"):
        return "dropped_near_identical"
    if divergence.get("reconverges_after_boundary"):
        return "dropped_unstable_boundary"
    if divergence.get("trivial_difference") or divergence.get("boilerplate_only"):
        return "dropped_trivial_segment_difference"
    if divergence.get("weak_divergence"):
        return "dropped_weak_divergence"
    return None


def _semi_purified_same_correctness_reason_code(
    *,
    cfg,
    pref: ScoredTraceRecord,
    disp: ScoredTraceRecord,
    pref_info: dict[str, Any],
    disp_info: dict[str, Any],
    support: dict[str, float],
    feats: dict[str, float],
) -> str:
    confidence = float(feats["confidence"])
    utility_margin = abs(float(pref_info["utility"]) - float(disp_info["utility"]))
    local_gap = abs(float(pref_info["local_score"]) - float(disp_info["local_score"]))
    support_gap = float(support.get("support_gap", 0.0))
    drop_advantage = float(disp_info["drop_at_k"]) - float(pref_info["drop_at_k"])

    if confidence < float(getattr(cfg.pair, "semi_purified_min_confidence", 0.82)):
        return "dropped_same_correctness_low_confidence"

    if pref.final_correct and disp.final_correct:
        if (
            utility_margin >= float(getattr(cfg.pair, "semi_purified_both_correct_min_utility_margin", 0.25))
            and local_gap >= float(getattr(cfg.pair, "semi_purified_both_correct_min_local_gap", 0.20))
            and confidence >= float(getattr(cfg.pair, "semi_purified_both_correct_min_confidence", 0.76))
            and support_gap >= float(getattr(cfg.pair, "semi_purified_both_correct_min_support_gap", 0.05))
        ):
            return "kept_both_correct_strong_local_preference"
        return "dropped_both_correct_ambiguous"

    if (
        utility_margin >= float(getattr(cfg.pair, "semi_purified_both_wrong_min_utility_margin", 0.45))
        and local_gap >= float(getattr(cfg.pair, "semi_purified_both_wrong_min_local_gap", 0.35))
        and confidence >= float(getattr(cfg.pair, "semi_purified_both_wrong_min_confidence", 0.88))
        and support_gap >= float(getattr(cfg.pair, "semi_purified_both_wrong_min_support_gap", 0.10))
        and drop_advantage >= float(getattr(cfg.pair, "semi_purified_both_wrong_min_drop_advantage", 0.15))
    ):
        return "kept_both_wrong_delayed_error"
    return "dropped_both_wrong_uninformative"


def _pair_mode_decision(
    *,
    cfg,
    pair_mode: str,
    a: ScoredTraceRecord,
    b: ScoredTraceRecord,
    k: int,
    a_info: dict[str, Any],
    b_info: dict[str, Any],
    divergence: dict[str, Any],
) -> tuple[ScoredTraceRecord, ScoredTraceRecord, dict[str, Any], dict[str, Any], str] | tuple[None, None, None, None, str]:
    a_correct = bool(a.final_correct)
    b_correct = bool(b.final_correct)
    utility_margin = abs(float(a_info["utility"]) - float(b_info["utility"]))

    if pair_mode == "current_utility":
        if utility_margin < cfg.pair.tau_pair:
            return None, None, None, None, "dropped_utility_margin_below_tau_pair"
        if float(a_info["utility"]) >= float(b_info["utility"]):
            return a, b, a_info, b_info, "current_utility"
        return b, a, b_info, a_info, "current_utility"

    if pair_mode == "correctness_priority":
        if a_correct != b_correct:
            if a_correct:
                return a, b, a_info, b_info, "correctness_priority_final_correctness"
            return b, a, b_info, a_info, "correctness_priority_final_correctness"
        if utility_margin < cfg.pair.tau_pair:
            return None, None, None, None, "dropped_utility_margin_below_tau_pair"
        if float(a_info["utility"]) >= float(b_info["utility"]):
            return a, b, a_info, b_info, "correctness_priority_same_correctness_utility"
        return b, a, b_info, a_info, "correctness_priority_same_correctness_utility"

    if pair_mode == "strict_purified":
        if a_correct == b_correct:
            return None, None, None, None, "dropped_both_correct_ambiguous" if a_correct else "dropped_both_wrong_uninformative"
        if a_correct:
            return a, b, a_info, b_info, "strict_purified_final_correctness"
        return b, a, b_info, a_info, "strict_purified_final_correctness"

    if pair_mode == "semi_purified":
        if a_correct != b_correct:
            if a_correct:
                return a, b, a_info, b_info, "semi_purified_final_correctness"
            return b, a, b_info, a_info, "semi_purified_final_correctness"
        if utility_margin < cfg.pair.tau_pair:
            return None, None, None, None, "dropped_utility_margin_below_tau_pair"
        if float(a_info["utility"]) >= float(b_info["utility"]):
            return a, b, a_info, b_info, "semi_purified_same_correctness_utility_screening"
        return b, a, b_info, a_info, "semi_purified_same_correctness_utility_screening"

    raise ValueError(f"Unknown pair mode: {pair_mode}")


def _make_decision_row(
    *,
    low_threshold: float,
    high_threshold: float,
    pair_mode: str,
    prompt_id: str,
    prompt: str,
    prefix_steps: list[str],
    preferred_steps: list[str],
    dispreferred_steps: list[str],
    trace_a: ScoredTraceRecord | None,
    trace_b: ScoredTraceRecord | None,
    pref: ScoredTraceRecord | None,
    disp: ScoredTraceRecord | None,
    k: int | None,
    orientation_reason: str | None,
    confidence: float | None,
    features: dict[str, float] | None,
    a_info: dict[str, Any] | None,
    b_info: dict[str, Any] | None,
    support: dict[str, Any] | None,
    divergence: dict[str, Any] | None,
    final_status: str,
) -> dict[str, Any]:
    a_correct = bool(trace_a.final_correct) if trace_a is not None else None
    b_correct = bool(trace_b.final_correct) if trace_b is not None else None
    pref_correct = bool(pref.final_correct) if pref is not None else None
    disp_correct = bool(disp.final_correct) if disp is not None else None
    pattern = _correctness_pattern(pref_correct, disp_correct) if pref is not None and disp is not None else "unresolved"
    weak_divergence = bool(divergence["weak_divergence"]) if divergence else False
    conf_value = float(confidence) if confidence is not None else None
    if conf_value is None:
        confidence_bucket = "unknown"
    elif conf_value < low_threshold:
        confidence_bucket = "low"
    elif conf_value >= high_threshold:
        confidence_bucket = "high"
    else:
        confidence_bucket = "medium"

    pref_utility = (
        float((a_info or {}).get("utility"))
        if a_info and pref is not None and a_info.get("trace_id") == pref.trace_id
        else float((b_info or {}).get("utility"))
        if b_info and pref is not None
        else None
    )
    disp_utility = (
        float((a_info or {}).get("utility"))
        if a_info and disp is not None and a_info.get("trace_id") == disp.trace_id
        else float((b_info or {}).get("utility"))
        if b_info and disp is not None
        else None
    )
    pref_local_score = (
        float((a_info or {}).get("local_score"))
        if a_info and pref is not None and a_info.get("trace_id") == pref.trace_id
        else float((b_info or {}).get("local_score"))
        if b_info and pref is not None
        else None
    )
    disp_local_score = (
        float((a_info or {}).get("local_score"))
        if a_info and disp is not None and a_info.get("trace_id") == disp.trace_id
        else float((b_info or {}).get("local_score"))
        if b_info and disp is not None
        else None
    )
    pref_drop_at_k = (
        float((a_info or {}).get("drop_at_k"))
        if a_info and pref is not None and a_info.get("trace_id") == pref.trace_id
        else float((b_info or {}).get("drop_at_k"))
        if b_info and pref is not None
        else None
    )
    disp_drop_at_k = (
        float((a_info or {}).get("drop_at_k"))
        if a_info and disp is not None and a_info.get("trace_id") == disp.trace_id
        else float((b_info or {}).get("drop_at_k"))
        if b_info and disp is not None
        else None
    )
    return {
        "id": prompt_id,
        "prompt": prompt,
        "pair_mode": pair_mode,
        "k": k,
        "prefix_steps": prefix_steps,
        "preferred_steps": preferred_steps,
        "dispreferred_steps": dispreferred_steps,
        "pref_trace_id": pref.trace_id if pref is not None else None,
        "disp_trace_id": disp.trace_id if disp is not None else None,
        "pref_final_correct": pref_correct,
        "disp_final_correct": disp_correct,
        "correctness_bucket": _correctness_bucket(a_correct, b_correct),
        "correctness_pattern": pattern,
        "orientation_reason": orientation_reason,
        "confidence": conf_value,
        "confidence_bucket": confidence_bucket,
        "confidence_features": dict(features or {}),
        "utility_margin": float(abs((a_info or {}).get("utility", 0.0) - (b_info or {}).get("utility", 0.0))) if a_info and b_info else None,
        "local_score_gap": float(abs((a_info or {}).get("local_score", 0.0) - (b_info or {}).get("local_score", 0.0))) if a_info and b_info else None,
        "pref_utility": pref_utility,
        "disp_utility": disp_utility,
        "pref_local_score": pref_local_score,
        "disp_local_score": disp_local_score,
        "support_gap": float((support or {}).get("support_gap", 0.0)) if support else None,
        "drop_advantage": float(disp_drop_at_k - pref_drop_at_k) if pref_drop_at_k is not None and disp_drop_at_k is not None else None,
        "weak_divergence": weak_divergence,
        "degenerate_divergence": bool(divergence["too_short"] or divergence["exact_match"]) if divergence else False,
        "formatting_only_divergence": bool(divergence["formatting_only"]) if divergence else False,
        "near_identical_divergence": bool(divergence["near_identical"]) if divergence else False,
        "trivial_difference_divergence": bool(divergence["trivial_difference"]) if divergence else False,
        "boilerplate_only_divergence": bool(divergence["boilerplate_only"]) if divergence else False,
        "boundary_unstable": bool((divergence or {}).get("reconverges_after_boundary")) if divergence else False,
        "divergent_similarity": float((divergence or {}).get("similarity", 0.0)) if divergence else None,
        "divergent_min_chars": int((divergence or {}).get("min_chars", 0)) if divergence else None,
        "reconverges_after_boundary": bool((divergence or {}).get("reconverges_after_boundary", False)) if divergence else False,
        "final_status": final_status,
        "admissibility_reason_code": final_status,
    }


def _apply_method(
    cfg,
    raw_pairs: list[PairRecord],
    *,
    pair_type: str,
    pair_mode: str,
    pair_purity_report: dict[str, Any],
    decision_rows: list[dict[str, Any]],
) -> PairBuildArtifacts:
    method_name = _method_name(cfg)
    threshold = _confidence_threshold(cfg)
    kept: list[PairRecord] = []
    dropped = 0

    decision_index = {row["decision_id"]: row for row in decision_rows if "decision_id" in row}
    for row in raw_pairs:
        confidence = row.confidence if row.confidence is not None else row.features.get("confidence")
        decision_id = row.meta.get("decision_id")
        if pair_type == "local_step":
            if method_name == "confidence_filter_only" and confidence is not None and confidence < threshold:
                dropped += 1
                if decision_id in decision_index:
                    decision_index[decision_id]["final_status"] = "dropped_low_confidence"
                    decision_index[decision_id]["admissibility_reason_code"] = "dropped_low_confidence"
                continue
            if method_name == "confidence_weighted_step_dpo" and confidence is not None and confidence < threshold:
                dropped += 1
                if decision_id in decision_index:
                    decision_index[decision_id]["final_status"] = "dropped_low_confidence"
                    decision_index[decision_id]["admissibility_reason_code"] = "dropped_low_confidence"
                continue
            if method_name == "step_dpo":
                row.weight = 1.0
            elif method_name == "confidence_filter_only":
                row.weight = 1.0
            else:
                row.weight = float(confidence if confidence is not None else row.weight)
        else:
            row.weight = 1.0
        row.meta["method_name"] = method_name
        row.meta["pair_mode"] = pair_mode
        kept.append(row)
        if decision_id in decision_index:
            keep_reason = str(row.meta.get("admissibility_reason_code", "kept_selected"))
            decision_index[decision_id]["final_status"] = keep_reason
            decision_index[decision_id]["admissibility_reason_code"] = keep_reason
            decision_index[decision_id]["training_weight"] = float(row.weight)

    kept.sort(
        key=lambda p: (
            p.confidence if p.confidence is not None else p.features.get("confidence", p.weight),
            p.weight,
        ),
        reverse=True,
    )

    pair_purity_report = dict(pair_purity_report)
    pair_purity_report["num_kept_pairs"] = len(kept)
    pair_purity_report["num_dropped_by_confidence"] = dropped
    pair_purity_report["status_counts"] = dict(sorted(Counter(row.get("final_status", "unknown") for row in decision_rows).items()))
    pair_purity_report["pair_taxonomy"] = _pair_taxonomy_counts(decision_rows)
    pair_purity_report["orientation_counts"] = _orientation_counts(decision_rows)
    kept_count = max(1, len(kept))
    taxonomy = pair_purity_report["pair_taxonomy"]
    correctness_driven = sum(pair_purity_report["orientation_counts"].get(key, 0) for key in (
        "correctness_priority_final_correctness",
        "strict_purified_final_correctness",
        "semi_purified_final_correctness",
        "answer_level_final_correctness",
    ))
    utility_driven = sum(pair_purity_report["orientation_counts"].get(key, 0) for key in (
        "current_utility",
        "correctness_priority_same_correctness_utility",
        "semi_purified_same_correctness_utility_screening",
    ))
    pair_purity_report["pair_purity_metrics"] = {
        "fraction_strictly_instructional_pairs": float(taxonomy["correct_vs_incorrect"] / kept_count) if kept else None,
        "fraction_ambiguous_pairs": float((taxonomy["both_correct"] + taxonomy["both_wrong"]) / kept_count) if kept else None,
        "fraction_misoriented_mixed_correctness_pairs": float(taxonomy["incorrect_vs_correct"] / kept_count) if kept else None,
        "fraction_dropped_by_purification": float(
            (
                pair_purity_report["status_counts"].get("dropped_both_correct_ambiguous", 0)
                + pair_purity_report["status_counts"].get("dropped_both_wrong_uninformative", 0)
                + pair_purity_report["status_counts"].get("dropped_weak_divergence", 0)
                + pair_purity_report["status_counts"].get("dropped_trivial_segment_difference", 0)
                + pair_purity_report["status_counts"].get("dropped_near_identical", 0)
                + pair_purity_report["status_counts"].get("dropped_formatting_only", 0)
                + pair_purity_report["status_counts"].get("dropped_unstable_boundary", 0)
            )
            / max(1, len(decision_rows))
        ),
        "fraction_orientation_driven_by_correctness": float(correctness_driven / kept_count) if kept else None,
        "fraction_orientation_driven_only_by_utility": float(utility_driven / kept_count) if kept else None,
    }
    pair_purity_report["admissibility_diagnostics"] = {
        "reason_code_counts": dict(sorted(pair_purity_report["status_counts"].items())),
        "correctness_bucket_counts": _value_counts(decision_rows, "correctness_bucket"),
        "kept_correctness_pattern_counts": _value_counts(
            [row for row in decision_rows if _is_kept_status(row.get("final_status"))],
            "correctness_pattern",
        ),
        "pair_mode_counts": _value_counts(decision_rows, "pair_mode"),
        "confidence_bucket_counts": _value_counts(decision_rows, "confidence_bucket"),
        "divergence_quality_counts": _divergence_quality_counts(decision_rows),
    }

    return PairBuildArtifacts(
        pairs=kept,
        num_raw_pairs=len(raw_pairs),
        num_kept_pairs=len(kept),
        num_dropped_by_confidence=dropped,
        method_name=method_name,
        pair_type=pair_type,
        pair_mode=pair_mode,
        pair_purity_report=pair_purity_report,
        orientation_audit_rows=decision_rows,
    )


def _build_local_pairs(cfg, traces: list[ScoredTraceRecord]) -> tuple[list[PairRecord], dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[ScoredTraceRecord]] = defaultdict(list)
    for tr in traces:
        grouped[tr.id].append(tr)

    pair_mode = _pair_mode_name(cfg)
    all_pairs: list[PairRecord] = []
    decisions: list[dict[str, Any]] = []

    for _, group in grouped.items():
        norm_scores = normalize_scores_across_prompt(group)
        low_threshold = float(getattr(cfg.confidence, "low_threshold", 0.33))
        high_threshold = float(getattr(cfg.confidence, "high_threshold", 0.66))
        prompt_pairs: list[PairRecord] = []
        prompt_decisions: list[dict[str, Any]] = []
        for a, b in combinations(group, 2):
            k = first_divergence(a.steps, b.steps)
            if k is None:
                prompt_decisions.append(
                    _make_decision_row(
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        pair_mode=pair_mode,
                        prompt_id=a.id,
                        prompt=a.prompt,
                        prefix_steps=[],
                        preferred_steps=[],
                        dispreferred_steps=[],
                        trace_a=a,
                        trace_b=b,
                        pref=None,
                        disp=None,
                        k=None,
                        orientation_reason=None,
                        confidence=None,
                        features=None,
                        a_info=None,
                        b_info=None,
                        support=None,
                        divergence=None,
                        final_status="dropped_no_divergence",
                    )
                )
                continue

            a_scores = norm_scores[a.trace_id]
            b_scores = norm_scores[b.trace_id]
            a_seg = a.steps[k : min(len(a.steps), k + cfg.pair.window_H)]
            b_seg = b.steps[k : min(len(b.steps), k + cfg.pair.window_H)]
            if not a_seg or not b_seg:
                prompt_decisions.append(
                    _make_decision_row(
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        pair_mode=pair_mode,
                        prompt_id=a.id,
                        prompt=a.prompt,
                        prefix_steps=a.steps[:k],
                        preferred_steps=a_seg,
                        dispreferred_steps=b_seg,
                        trace_a=a,
                        trace_b=b,
                        pref=None,
                        disp=None,
                        k=k,
                        orientation_reason=None,
                        confidence=None,
                        features=None,
                        a_info=None,
                        b_info=None,
                        support=None,
                        divergence=None,
                        final_status="dropped_empty_divergent_segment",
                    )
                )
                continue

            Ra = local_segment_score(a_scores, k, cfg.pair.window_H)
            Rb = local_segment_score(b_scores, k, cfg.pair.window_H)
            Ua = utility(Ra, a.final_correct, cfg.pair.alpha_local)
            Ub = utility(Rb, b.final_correct, cfg.pair.alpha_local)
            divergence = _divergence_diagnostics(a, b, k=k, h=cfg.pair.window_H, cfg=cfg)

            a_info = {
                "trace_id": a.trace_id,
                "utility": Ua,
                "drop_at_k": drop_at_k(a_scores, k),
                "local_score": Ra,
            }
            b_info = {
                "trace_id": b.trace_id,
                "utility": Ub,
                "drop_at_k": drop_at_k(b_scores, k),
                "local_score": Rb,
            }
            divergence_reason = _divergence_reason_code(pair_mode, divergence)
            if divergence_reason is not None:
                prompt_decisions.append(
                    _make_decision_row(
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        pair_mode=pair_mode,
                        prompt_id=a.id,
                        prompt=a.prompt,
                        prefix_steps=a.steps[:k],
                        preferred_steps=a_seg,
                        dispreferred_steps=b_seg,
                        trace_a=a,
                        trace_b=b,
                        pref=None,
                        disp=None,
                        k=k,
                        orientation_reason=None,
                        confidence=None,
                        features=None,
                        a_info=a_info,
                        b_info=b_info,
                        support=None,
                        divergence=divergence,
                        final_status=divergence_reason,
                    )
                )
                continue

            pref, disp, pref_info, disp_info, decision_reason = _pair_mode_decision(
                cfg=cfg,
                pair_mode=pair_mode,
                a=a,
                b=b,
                k=k,
                a_info=a_info,
                b_info=b_info,
                divergence=divergence,
            )

            if pref is None or disp is None or pref_info is None or disp_info is None:
                prompt_decisions.append(
                    _make_decision_row(
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        pair_mode=pair_mode,
                        prompt_id=a.id,
                        prompt=a.prompt,
                        prefix_steps=a.steps[:k],
                        preferred_steps=a_seg,
                        dispreferred_steps=b_seg,
                        trace_a=a,
                        trace_b=b,
                        pref=None,
                        disp=None,
                        k=k,
                        orientation_reason=None,
                        confidence=None,
                        features=None,
                        a_info=a_info,
                        b_info=b_info,
                        support=None,
                        divergence=divergence,
                        final_status=decision_reason,
                    )
                )
                continue

            support = estimate_support_stats(group, k, pref, disp)
            feats = confidence_features(pref_info, disp_info, support, cfg.confidence)
            preferred_steps = pref.steps[k : min(len(pref.steps), k + cfg.pair.window_H)]
            dispreferred_steps = disp.steps[k : min(len(disp.steps), k + cfg.pair.window_H)]
            if not preferred_steps or not dispreferred_steps:
                prompt_decisions.append(
                    _make_decision_row(
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        pair_mode=pair_mode,
                        prompt_id=pref.id,
                        prompt=pref.prompt,
                        prefix_steps=pref.steps[:k],
                        preferred_steps=preferred_steps,
                        dispreferred_steps=dispreferred_steps,
                        trace_a=a,
                        trace_b=b,
                        pref=pref,
                        disp=disp,
                        k=k,
                        orientation_reason=decision_reason,
                        confidence=float(feats["confidence"]),
                        features=feats,
                        a_info=a_info,
                        b_info=b_info,
                        support=support,
                        divergence=divergence,
                        final_status="dropped_empty_divergent_segment",
                    )
                )
                continue

            keep_reason = "kept_mixed_correctness" if pref.final_correct != disp.final_correct else "kept_same_correctness_utility_oriented"
            if pair_mode == "semi_purified" and pref.final_correct == disp.final_correct:
                keep_reason = _semi_purified_same_correctness_reason_code(
                    cfg=cfg,
                    pref=pref,
                    disp=disp,
                    pref_info=pref_info,
                    disp_info=disp_info,
                    support=support,
                    feats=feats,
                )
                if not keep_reason.startswith("kept_"):
                    prompt_decisions.append(
                        _make_decision_row(
                            low_threshold=low_threshold,
                            high_threshold=high_threshold,
                            pair_mode=pair_mode,
                            prompt_id=pref.id,
                            prompt=pref.prompt,
                            prefix_steps=pref.steps[:k],
                            preferred_steps=preferred_steps,
                            dispreferred_steps=dispreferred_steps,
                            trace_a=a,
                            trace_b=b,
                            pref=pref,
                            disp=disp,
                            k=k,
                            orientation_reason=decision_reason,
                            confidence=float(feats["confidence"]),
                            features=feats,
                            a_info=a_info,
                            b_info=b_info,
                            support=support,
                            divergence=divergence,
                            final_status=keep_reason,
                        )
                    )
                    continue

            decision_id = f"{pref.id}:{pair_mode}:{pref.trace_id}:{disp.trace_id}:{k}"
            pair = PairRecord(
                id=pref.id,
                prompt=pref.prompt,
                prefix_steps=pref.steps[:k],
                preferred_steps=preferred_steps,
                dispreferred_steps=dispreferred_steps,
                weight=float(feats["confidence"]),
                confidence=float(feats["confidence"]),
                features=feats,
                meta={
                    "decision_id": decision_id,
                    "pair_type": "local_step",
                    "pair_mode": pair_mode,
                    "admissibility_reason_code": keep_reason,
                    "orientation_reason": decision_reason,
                    "pref_trace_id": pref.trace_id,
                    "disp_trace_id": disp.trace_id,
                    "pref_final_answer": pref.final_answer,
                    "disp_final_answer": disp.final_answer,
                    "pref_final_correct": pref.final_correct,
                    "disp_final_correct": disp.final_correct,
                    "correctness_bucket": _correctness_bucket(bool(a.final_correct), bool(b.final_correct)),
                    "correctness_pattern": _correctness_pattern(bool(pref.final_correct), bool(disp.final_correct)),
                    "k": k,
                    "utility_margin": float(abs(float(pref_info["utility"]) - float(disp_info["utility"]))),
                    "pref_utility": float(pref_info["utility"]),
                    "disp_utility": float(disp_info["utility"]),
                    "pref_local_score": float(pref_info["local_score"]),
                    "disp_local_score": float(disp_info["local_score"]),
                    "local_score_gap": float(abs(float(pref_info["local_score"]) - float(disp_info["local_score"]))),
                    "support_gap": float(support.get("support_gap", 0.0)),
                    "drop_advantage": float(disp_info["drop_at_k"] - pref_info["drop_at_k"]),
                    "weak_divergence": bool(divergence["weak_divergence"]),
                    "formatting_only_divergence": bool(divergence["formatting_only"]),
                    "near_identical_divergence": bool(divergence["near_identical"]),
                    "trivial_difference_divergence": bool(divergence["trivial_difference"]),
                    "boilerplate_only_divergence": bool(divergence["boilerplate_only"]),
                    "reconverges_after_boundary": bool(divergence["reconverges_after_boundary"]),
                    "divergent_similarity": float(divergence["similarity"]),
                    "divergent_min_chars": int(divergence["min_chars"]),
                },
            )
            prompt_pairs.append(pair)
            decision_row = _make_decision_row(
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                pair_mode=pair_mode,
                prompt_id=pref.id,
                prompt=pref.prompt,
                prefix_steps=pref.steps[:k],
                preferred_steps=preferred_steps,
                dispreferred_steps=dispreferred_steps,
                trace_a=a,
                trace_b=b,
                pref=pref,
                disp=disp,
                k=k,
                orientation_reason=decision_reason,
                confidence=float(feats["confidence"]),
                features=feats,
                a_info=a_info,
                b_info=b_info,
                support=support,
                divergence=divergence,
                final_status="builder_candidate",
            )
            decision_row["decision_id"] = decision_id
            decision_row["admissibility_reason_code"] = keep_reason
            prompt_decisions.append(decision_row)

        prompt_pairs.sort(key=lambda p: p.confidence or 0.0, reverse=True)
        kept_ids = set()
        for pair in prompt_pairs[: cfg.pair.max_pairs_per_prompt]:
            kept_ids.add(pair.meta["decision_id"])
            all_pairs.append(pair)
        for row in prompt_decisions:
            if row.get("final_status") != "builder_candidate":
                continue
            if row.get("decision_id") in kept_ids:
                row["final_status"] = "builder_candidate"
            else:
                row["final_status"] = "dropped_prompt_pair_cap"
                row["admissibility_reason_code"] = "dropped_prompt_pair_cap"
        decisions.extend(prompt_decisions)

    pair_purity_report = _build_pair_purity_report(
        cfg=cfg,
        method_name=_method_name(cfg),
        pair_type="local_step",
        pair_mode=pair_mode,
        traces=traces,
        decisions=decisions,
        num_raw_pairs=len(all_pairs),
        num_dropped_by_confidence=0,
    )
    return all_pairs, pair_purity_report, decisions


def _answer_trace_score(tr: ScoredTraceRecord) -> tuple[int, float]:
    return int(tr.final_correct), float(sum(tr.step_scores) / max(1, len(tr.step_scores)))


def _build_answer_pairs(cfg, traces: list[ScoredTraceRecord]) -> tuple[list[PairRecord], dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[ScoredTraceRecord]] = defaultdict(list)
    for tr in traces:
        grouped[tr.id].append(tr)

    all_pairs: list[PairRecord] = []
    decisions: list[dict[str, Any]] = []
    for _, group in grouped.items():
        correct = [tr for tr in group if tr.final_correct]
        incorrect = [tr for tr in group if not tr.final_correct]
        if not correct or not incorrect:
            continue

        prompt_pairs: list[PairRecord] = []
        ranked_correct = sorted(correct, key=_answer_trace_score, reverse=True)
        ranked_incorrect = sorted(incorrect, key=_answer_trace_score)
        for pref in ranked_correct:
            for disp in ranked_incorrect:
                margin = _answer_trace_score(pref)[1] - _answer_trace_score(disp)[1]
                decision_id = f"{pref.id}:answer:{pref.trace_id}:{disp.trace_id}"
                pair = PairRecord(
                    id=pref.id,
                    prompt=pref.prompt,
                    prefix_steps=[],
                    preferred_steps=pref.steps,
                    dispreferred_steps=disp.steps,
                    weight=1.0,
                    confidence=None,
                    features={"answer_score_margin": float(margin)},
                    meta={
                        "decision_id": decision_id,
                        "pair_type": "answer_level",
                        "pair_mode": "answer_level",
                        "admissibility_reason_code": "kept_answer_level_mixed_correctness",
                        "orientation_reason": "answer_level_final_correctness",
                        "pref_trace_id": pref.trace_id,
                        "disp_trace_id": disp.trace_id,
                        "pref_final_answer": pref.final_answer,
                        "disp_final_answer": disp.final_answer,
                        "pref_final_correct": pref.final_correct,
                        "disp_final_correct": disp.final_correct,
                        "correctness_bucket": "mixed_correctness",
                        "correctness_pattern": _correctness_pattern(True, False),
                    },
                )
                prompt_pairs.append(pair)
                decisions.append(
                    {
                        "decision_id": decision_id,
                        "id": pref.id,
                        "prompt": pref.prompt,
                        "pair_mode": "answer_level",
                        "k": None,
                        "prefix_steps": [],
                        "preferred_steps": pref.steps,
                        "dispreferred_steps": disp.steps,
                        "pref_trace_id": pref.trace_id,
                        "disp_trace_id": disp.trace_id,
                        "pref_final_correct": True,
                        "disp_final_correct": False,
                        "correctness_bucket": "mixed_correctness",
                        "correctness_pattern": "correct_vs_incorrect",
                        "orientation_reason": "answer_level_final_correctness",
                        "confidence": None,
                        "confidence_bucket": "unknown",
                        "confidence_features": {"answer_score_margin": float(margin)},
                        "utility_margin": None,
                        "local_score_gap": None,
                        "pref_utility": None,
                        "disp_utility": None,
                        "pref_local_score": None,
                        "disp_local_score": None,
                        "weak_divergence": False,
                        "degenerate_divergence": False,
                        "near_identical_divergence": False,
                        "boundary_unstable": False,
                        "divergent_similarity": None,
                        "divergent_min_chars": None,
                        "reconverges_after_boundary": False,
                        "admissibility_reason_code": "kept_answer_level_mixed_correctness",
                        "final_status": "builder_candidate",
                    }
                )
        kept_ids = {pair.meta["decision_id"] for pair in prompt_pairs[: cfg.pair.max_pairs_per_prompt]}
        for row in decisions:
            if row.get("id") != group[0].id or row.get("final_status") != "builder_candidate":
                continue
            if row.get("decision_id") not in kept_ids:
                row["final_status"] = "dropped_prompt_pair_cap"
                row["admissibility_reason_code"] = "dropped_prompt_pair_cap"
        all_pairs.extend(prompt_pairs[: cfg.pair.max_pairs_per_prompt])

    report = {
        "method_name": _method_name(cfg),
        "pair_type": "answer_level",
        "pair_mode": "answer_level",
        "num_trace_comparisons": len(decisions),
        "num_builder_candidates_before_method": len(all_pairs),
        "num_kept_pairs": len(all_pairs),
        "num_dropped_by_confidence": 0,
        "status_counts": {"builder_candidate": len(decisions)},
        "pair_taxonomy": _pair_taxonomy_counts([{**row, "final_status": "kept"} for row in decisions]),
        "pair_purity_metrics": {
            "fraction_strictly_instructional_pairs": 1.0 if all_pairs else None,
            "fraction_ambiguous_pairs": 0.0 if all_pairs else None,
            "fraction_misoriented_mixed_correctness_pairs": 0.0 if all_pairs else None,
            "fraction_dropped_by_purification": 0.0,
            "fraction_orientation_driven_by_correctness": 1.0 if all_pairs else None,
            "fraction_orientation_driven_only_by_utility": 0.0 if all_pairs else None,
        },
        "boundary_diagnostics": {
            "average_steps_per_trace": float(np.mean([len(trace.steps) for trace in traces])) if traces else None,
            "boundary_instability_rate": 0.0,
            "weak_divergence_fraction": 0.0,
            "degenerate_divergence_fraction": 0.0,
            "near_identical_divergence_fraction": 0.0,
            "very_short_divergent_region_fraction": 0.0,
        },
        "orientation_counts": {"answer_level_final_correctness": len(all_pairs)},
    }
    return all_pairs, report, decisions


def build_pair_artifacts(cfg, traces: list[ScoredTraceRecord]) -> PairBuildArtifacts:
    if _pair_type(cfg) == "answer_level":
        raw_pairs, report, decisions = _build_answer_pairs(cfg, traces)
        return _apply_method(
            cfg,
            raw_pairs,
            pair_type="answer_level",
            pair_mode="answer_level",
            pair_purity_report=report,
            decision_rows=decisions,
        )

    raw_pairs, report, decisions = _build_local_pairs(cfg, traces)
    return _apply_method(
        cfg,
        raw_pairs,
        pair_type="local_step",
        pair_mode=_pair_mode_name(cfg),
        pair_purity_report=report,
        decision_rows=decisions,
    )


def build_pairs(cfg, traces: list[ScoredTraceRecord]) -> list[PairRecord]:
    return build_pair_artifacts(cfg, traces).pairs

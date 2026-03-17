from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from cwspo.schemas import PairRecord, ScoredTraceRecord
from cwspo.utils.steps import canon


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
    seg = scores[k:min(len(scores), k + h)]
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


def estimate_support_stats(traces: list[ScoredTraceRecord], k: int, pref_trace: ScoredTraceRecord, disp_trace: ScoredTraceRecord) -> dict[str, float]:
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
    }


def confidence_features(pref_info: dict[str, float], disp_info: dict[str, float], support_stats: dict[str, float], cfg) -> dict[str, float]:
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

    w = (
        cfg.gamma_margin * f_margin
        + cfg.gamma_sharp * f_sharp
        + cfg.gamma_agree * f_agree
        + cfg.gamma_out * f_out
    )
    w = max(0.0, min(1.0, w))
    return {
        "f_margin": float(f_margin),
        "f_sharp": float(f_sharp),
        "f_agree": float(f_agree),
        "f_out": float(f_out),
        "weight": float(w),
    }


def build_pairs(cfg, traces: list[ScoredTraceRecord]) -> list[PairRecord]:
    grouped: dict[str, list[ScoredTraceRecord]] = defaultdict(list)
    for tr in traces:
        grouped[tr.id].append(tr)

    all_pairs: list[PairRecord] = []
    for _, group in grouped.items():
        norm_scores = normalize_scores_across_prompt(group)
        prompt_pairs: list[PairRecord] = []
        for a, b in combinations(group, 2):
            k = first_divergence(a.steps, b.steps)
            if k is None:
                continue
            a_scores = norm_scores[a.trace_id]
            b_scores = norm_scores[b.trace_id]
            Ra = local_segment_score(a_scores, k, cfg.pair.window_H)
            Rb = local_segment_score(b_scores, k, cfg.pair.window_H)
            Ua = utility(Ra, a.final_correct, cfg.pair.alpha_local)
            Ub = utility(Rb, b.final_correct, cfg.pair.alpha_local)
            if abs(Ua - Ub) < cfg.pair.tau_pair:
                continue

            if Ua >= Ub:
                pref, disp = a, b
                Up, Ud = Ua, Ub
                pref_scores, disp_scores = a_scores, b_scores
                Rp, Rd = Ra, Rb
            else:
                pref, disp = b, a
                Up, Ud = Ub, Ua
                pref_scores, disp_scores = b_scores, a_scores
                Rp, Rd = Rb, Ra

            pref_info = {"utility": Up, "drop_at_k": drop_at_k(pref_scores, k), "local_score": Rp}
            disp_info = {"utility": Ud, "drop_at_k": drop_at_k(disp_scores, k), "local_score": Rd}
            support = estimate_support_stats(group, k, pref, disp)
            feats = confidence_features(pref_info, disp_info, support, cfg.confidence)
            if feats["weight"] < cfg.pair.min_weight:
                continue

            pair = PairRecord(
                id=pref.id,
                prompt=pref.prompt,
                prefix_steps=pref.steps[:k],
                preferred_steps=pref.steps[k:min(len(pref.steps), k + cfg.pair.window_H)],
                dispreferred_steps=disp.steps[k:min(len(disp.steps), k + cfg.pair.window_H)],
                weight=feats["weight"],
                features=feats,
                meta={
                    "pref_trace_id": pref.trace_id,
                    "disp_trace_id": disp.trace_id,
                    "pref_final_correct": pref.final_correct,
                    "disp_final_correct": disp.final_correct,
                    "k": k,
                },
            )
            prompt_pairs.append(pair)
        prompt_pairs.sort(key=lambda p: p.weight, reverse=True)
        all_pairs.extend(prompt_pairs[: cfg.pair.max_pairs_per_prompt])
    return all_pairs

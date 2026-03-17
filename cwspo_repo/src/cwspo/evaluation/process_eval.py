from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from cwspo.pipeline.build_pairs import build_pair_artifacts
from cwspo.pipeline.diagnostics import analyze_pairs
from cwspo.pipeline.score import score_traces
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord, TraceRecord


def _pair_score(pair: PairRecord) -> float:
    if pair.confidence is not None:
        return float(pair.confidence)
    if "confidence" in pair.features:
        return float(pair.features["confidence"])
    return float(pair.weight)


def predict_earliest_error_steps(pairs: list[PairRecord]) -> dict[str, dict]:
    pred_steps: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for p in pairs:
        k = p.meta.get("k")
        if k is None:
            continue
        pred_steps[p.id].append((int(k), _pair_score(p)))

    predictions: dict[str, dict] = {}
    for ex_id, candidates in pred_steps.items():
        pred_k, score = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        predictions[ex_id] = {
            "pred": pred_k,
            "score": score,
            "num_candidates": len(candidates),
        }
    return predictions


def evaluate_process_predictions(predictions: dict[str, dict], ground_truth: list[ProcessGroundTruthRecord]) -> dict:
    exact = 0
    near = 0
    covered = 0
    correct_rows = 0
    incorrect_rows = 0
    rows = []
    confusion: Counter[tuple[int | None, int | None]] = Counter()

    for gt in ground_truth:
        pred_info = predictions.get(gt.id)
        pred_k = pred_info["pred"] if pred_info else None
        if pred_info is not None:
            covered += 1
        exact_hit = pred_k == gt.gold_earliest_error_step
        near_hit = pred_k is not None and abs(pred_k - gt.gold_earliest_error_step) <= 1
        exact += int(exact_hit)
        near += int(near_hit)
        correct_rows += int(exact_hit)
        incorrect_rows += int(not exact_hit)
        rows.append(
            {
                "id": gt.id,
                "prompt": gt.prompt,
                "gold": gt.gold_earliest_error_step,
                "pred": pred_k,
                "correct": exact_hit,
                "near_miss": near_hit,
                "prediction_score": pred_info["score"] if pred_info else None,
                "num_candidates": pred_info["num_candidates"] if pred_info else 0,
                "step_count_correct": len(gt.correct_steps) if gt.correct_steps else None,
                "step_count_incorrect": len(gt.incorrect_steps) if gt.incorrect_steps else None,
            }
        )
        confusion[(gt.gold_earliest_error_step, pred_k)] += 1

    return {
        "evaluation_mode": "offline_pair_predictions",
        "depends_on_trained_policy": False,
        "earliest_error_exact": exact / max(1, len(ground_truth)),
        "earliest_error_near_miss": near / max(1, len(ground_truth)),
        "coverage": covered / max(1, len(ground_truth)),
        "num_evaluated_examples": len(ground_truth),
        "num_correct": correct_rows,
        "num_incorrect": incorrect_rows,
        "rows": rows,
        "boundary_confusion": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in sorted(confusion.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]))
        ],
    }


def evaluate_process(pairs: list[PairRecord], ground_truth: list[ProcessGroundTruthRecord]) -> dict:
    predictions = predict_earliest_error_steps(pairs)
    return evaluate_process_predictions(predictions, ground_truth)


def evaluate_process_dataset(cfg, ground_truth: list[ProcessGroundTruthRecord]) -> dict:
    if getattr(cfg.method, "name", "") == "answer_dpo":
        return {
            "not_applicable": True,
            "reason": "answer-level DPO does not produce local-boundary training pairs for process evaluation.",
            "evaluation_mode": "offline_fixed_trace_boundary_detection",
            "depends_on_trained_policy": False,
            "earliest_error_exact": None,
            "earliest_error_near_miss": None,
            "coverage": None,
            "num_evaluated_examples": len(ground_truth),
            "rows": [],
            "boundary_confusion": [],
            "num_process_pairs": 0,
            "process_pair_build": {
                "method_name": getattr(cfg.method, "name", "answer_dpo"),
                "pair_type": "answer_level",
                "num_raw_pairs": 0,
                "num_kept_pairs": 0,
                "num_dropped_by_confidence": 0,
            },
            "process_confidence_analysis": None,
        }

    traces: list[TraceRecord] = []
    for gt in ground_truth:
        if not gt.prompt or not gt.correct_steps or not gt.incorrect_steps or gt.answer is None:
            raise ValueError("Rich process evaluation requires prompt, answer, correct_steps, and incorrect_steps.")
        traces.append(
            TraceRecord(
                id=gt.id,
                prompt=gt.prompt,
                answer=gt.answer,
                trace_id=0,
                reasoning="\n".join(gt.correct_steps),
                steps=gt.correct_steps,
                final_answer=gt.answer,
            )
        )
        traces.append(
            TraceRecord(
                id=gt.id,
                prompt=gt.prompt,
                answer=gt.answer,
                trace_id=1,
                reasoning="\n".join(gt.incorrect_steps),
                steps=gt.incorrect_steps,
                final_answer=gt.incorrect_final_answer or "",
            )
        )

    scored = score_traces(cfg, traces)
    artifacts = build_pair_artifacts(cfg, scored)
    summary = evaluate_process(artifacts.pairs, ground_truth)
    summary["evaluation_mode"] = "offline_fixed_trace_boundary_detection"
    summary["depends_on_trained_policy"] = False
    summary["process_confidence_analysis"] = analyze_pairs(
        artifacts.pairs,
        low_threshold=cfg.confidence.low_threshold,
        high_threshold=cfg.confidence.high_threshold,
        histogram_bins=cfg.diagnostics.histogram_bins,
        num_raw_pairs=artifacts.num_raw_pairs,
        num_dropped_by_confidence=artifacts.num_dropped_by_confidence,
        method_name=artifacts.method_name,
        pair_mode=artifacts.pair_mode,
        pair_purity_report=artifacts.pair_purity_report,
        process_ground_truth=ground_truth,
    )
    summary["num_process_pairs"] = len(artifacts.pairs)
    summary["process_pair_build"] = {
        "method_name": artifacts.method_name,
        "pair_type": artifacts.pair_type,
        "pair_mode": artifacts.pair_mode,
        "num_raw_pairs": artifacts.num_raw_pairs,
        "num_kept_pairs": artifacts.num_kept_pairs,
        "num_dropped_by_confidence": artifacts.num_dropped_by_confidence,
    }
    return summary


def render_process_failure_report(summary: dict) -> str:
    lines = [
        "# Process Evaluation Failures",
        "",
        f"- Exact: `{summary.get('earliest_error_exact')}`",
        f"- Near miss: `{summary.get('earliest_error_near_miss')}`",
        f"- Coverage: `{summary.get('coverage')}`",
        "",
    ]
    process_confidence = summary.get("process_confidence_analysis") or {}
    bucket_summary = process_confidence.get("bucket_summary") or {}
    if bucket_summary:
        lines.extend(["## Confidence Buckets", ""])
        for bucket_name, stats in bucket_summary.items():
            lines.extend(
                [
                    f"### {bucket_name.title()}",
                    f"- Count: `{stats.get('count')}`",
                    f"- Mean confidence: `{stats.get('mean_confidence')}`",
                    f"- Process-boundary accuracy: `{stats.get('process_boundary_accuracy')}`",
                    "",
                ]
            )
    failures = [row for row in summary.get("rows", []) if not row.get("correct")]
    if not failures:
        lines.append("No process-eval failures recorded.")
        return "\n".join(lines) + "\n"

    for row in failures[:15]:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- Gold step: `{row['gold']}`",
                f"- Predicted step: `{row['pred']}`",
                f"- Near miss: `{row['near_miss']}`",
                f"- Prediction score: `{row['prediction_score']}`",
                "",
                "### Prompt",
                row.get("prompt") or "(prompt unavailable)",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_process_failure_report(summary: dict, path: str | Path | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_process_failure_report(summary), encoding="utf-8")


def write_process_summary(summary: dict, path: str | Path | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

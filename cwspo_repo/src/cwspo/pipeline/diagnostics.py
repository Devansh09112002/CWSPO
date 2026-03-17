from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from cwspo.schemas import PairRecord, ProcessGroundTruthRecord


def pair_confidence(pair: PairRecord) -> float | None:
    if pair.confidence is not None:
        return float(pair.confidence)
    if "confidence" in pair.features:
        return float(pair.features["confidence"])
    if "weight" in pair.features:
        return float(pair.features["weight"])
    return None


def pair_reliability_label(pair: PairRecord) -> int | None:
    pref = pair.meta.get("pref_final_correct")
    disp = pair.meta.get("disp_final_correct")
    if pref is None or disp is None or pref == disp:
        return None
    return int(bool(pref) and not bool(disp))


def pair_process_hit(pair: PairRecord, gt_map: dict[str, int]) -> int | None:
    gold = gt_map.get(pair.id)
    if gold is None:
        return None
    k = pair.meta.get("k")
    if k is None:
        return None
    return int(int(k) == int(gold))


def confidence_bucket(value: float, low_threshold: float, high_threshold: float) -> str:
    if value < low_threshold:
        return "low"
    if value >= high_threshold:
        return "high"
    return "medium"


def _histogram(values: list[float], bins: int) -> dict:
    if not values:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(np.array(values, dtype=float), bins=bins, range=(0.0, 1.0))
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _pearson(xs: list[float], ys: list[int]) -> float | None:
    if len(xs) < 2 or len(set(ys)) < 2:
        return None
    arr_x = np.array(xs, dtype=float)
    arr_y = np.array(ys, dtype=float)
    corr = np.corrcoef(arr_x, arr_y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def _branch_stats(rows: list[PairRecord]) -> dict[str, float | int | None]:
    pref_vals = []
    disp_vals = []
    decisive = 0
    both_wrong = 0
    both_correct = 0
    eligible = 0

    for row in rows:
        pref = row.meta.get("pref_final_correct")
        disp = row.meta.get("disp_final_correct")
        if pref is None or disp is None:
            continue
        pref_bool = bool(pref)
        disp_bool = bool(disp)
        pref_vals.append(int(pref_bool))
        disp_vals.append(int(disp_bool))
        eligible += 1
        decisive += int(pref_bool != disp_bool)
        both_wrong += int((not pref_bool) and (not disp_bool))
        both_correct += int(pref_bool and disp_bool)

    return {
        "eligible_pair_count": eligible,
        "decisive_pair_fraction": float(decisive / max(1, eligible)) if eligible else None,
        "preferred_branch_final_correct_rate": float(sum(pref_vals) / max(1, len(pref_vals))) if pref_vals else None,
        "dispreferred_branch_final_correct_rate": float(sum(disp_vals) / max(1, len(disp_vals))) if disp_vals else None,
        "both_branches_incorrect_fraction": float(both_wrong / max(1, eligible)) if eligible else None,
        "both_branches_correct_fraction": float(both_correct / max(1, eligible)) if eligible else None,
    }


def _bucket_summary(
    pairs: list[PairRecord],
    low_threshold: float,
    high_threshold: float,
    gt_map: dict[str, int] | None = None,
) -> dict[str, dict]:
    buckets = {"low": [], "medium": [], "high": []}
    for pair in pairs:
        conf = pair_confidence(pair)
        if conf is None:
            continue
        buckets[confidence_bucket(conf, low_threshold, high_threshold)].append(pair)

    result: dict[str, dict] = {}
    for name, rows in buckets.items():
        confs = [pair_confidence(row) for row in rows if pair_confidence(row) is not None]
        rel = [pair_reliability_label(row) for row in rows]
        rel = [x for x in rel if x is not None]
        proc = []
        if gt_map is not None:
            proc = [pair_process_hit(row, gt_map) for row in rows]
            proc = [x for x in proc if x is not None]
        bucket_stats = {
            "count": len(rows),
            "fraction": float(len(rows) / max(1, len(pairs))),
            "mean_confidence": float(sum(confs) / max(1, len(confs))) if confs else None,
            "preferred_branch_accuracy": float(sum(rel) / max(1, len(rel))) if rel else None,
            "process_boundary_accuracy": float(sum(proc) / max(1, len(proc))) if proc else None,
        }
        bucket_stats.update(_branch_stats(rows))
        result[name] = bucket_stats
    return result


def analyze_pairs(
    pairs: list[PairRecord],
    *,
    low_threshold: float,
    high_threshold: float,
    histogram_bins: int = 10,
    num_raw_pairs: int | None = None,
    num_dropped_by_confidence: int | None = None,
    method_name: str | None = None,
    pair_mode: str | None = None,
    pair_purity_report: dict[str, Any] | None = None,
    process_ground_truth: list[ProcessGroundTruthRecord] | None = None,
) -> dict:
    confidences = [pair_confidence(pair) for pair in pairs if pair_confidence(pair) is not None]
    gt_map = {row.id: row.gold_earliest_error_step for row in process_ground_truth or []}

    decisive = []
    decisive_conf = []
    for pair in pairs:
        label = pair_reliability_label(pair)
        conf = pair_confidence(pair)
        if label is None or conf is None:
            continue
        decisive.append(label)
        decisive_conf.append(conf)

    overall_branch_stats = _branch_stats(pairs)
    analysis = {
        "method_name": method_name,
        "pair_mode": pair_mode,
        "num_pairs": len(pairs),
        "num_raw_pairs": num_raw_pairs,
        "num_dropped_by_confidence": num_dropped_by_confidence,
        "fraction_dropped_by_threshold": (
            float(num_dropped_by_confidence / max(1, num_raw_pairs))
            if num_raw_pairs is not None and num_dropped_by_confidence is not None
            else None
        ),
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "median_confidence": float(np.median(confidences)) if confidences else None,
        "histogram": _histogram([float(x) for x in confidences], histogram_bins),
        "bucket_summary": _bucket_summary(
            pairs,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            gt_map=gt_map if gt_map else None,
        ),
        "decisive_pair_count": len(decisive),
        "overall_decisive_pair_accuracy": float(sum(decisive) / max(1, len(decisive))) if decisive else None,
        "confidence_correctness_correlation": _pearson(decisive_conf, decisive),
        "high_confidence_pair_accuracy": None,
        "confidence_calibration_proxy": None,
        "pair_purity_report": pair_purity_report,
    }
    analysis.update(overall_branch_stats)

    high_summary = analysis["bucket_summary"]["high"]
    analysis["high_confidence_pair_accuracy"] = high_summary["preferred_branch_accuracy"]
    if analysis["mean_confidence"] is not None and analysis["overall_decisive_pair_accuracy"] is not None:
        analysis["confidence_calibration_proxy"] = float(
            analysis["mean_confidence"] - analysis["overall_decisive_pair_accuracy"]
        )
    return analysis


def render_confidence_report(analysis: dict) -> str:
    lines = [
        "# Confidence Report",
        "",
        f"- Method: `{analysis.get('method_name')}`",
        f"- Pair mode: `{analysis.get('pair_mode')}`",
        f"- Pairs kept: `{analysis.get('num_pairs')}`",
        f"- Raw candidate pairs: `{analysis.get('num_raw_pairs')}`",
        f"- Dropped by threshold: `{analysis.get('num_dropped_by_confidence')}`",
        f"- Fraction dropped: `{analysis.get('fraction_dropped_by_threshold')}`",
        f"- Mean confidence: `{analysis.get('mean_confidence')}`",
        f"- Median confidence: `{analysis.get('median_confidence')}`",
        f"- Decisive pair count: `{analysis.get('decisive_pair_count')}`",
        f"- Overall decisive pair accuracy: `{analysis.get('overall_decisive_pair_accuracy')}`",
        f"- Preferred branch final-correct rate: `{analysis.get('preferred_branch_final_correct_rate')}`",
        f"- Dispreferred branch final-correct rate: `{analysis.get('dispreferred_branch_final_correct_rate')}`",
        f"- Decisive pair fraction: `{analysis.get('decisive_pair_fraction')}`",
        f"- Both-branches-wrong fraction: `{analysis.get('both_branches_incorrect_fraction')}`",
        f"- Confidence/correctness correlation: `{analysis.get('confidence_correctness_correlation')}`",
        f"- Confidence calibration proxy: `{analysis.get('confidence_calibration_proxy')}`",
        "",
    ]
    purity = analysis.get("pair_purity_report") or {}
    purity_metrics = purity.get("pair_purity_metrics") or {}
    boundary = purity.get("boundary_diagnostics") or {}
    taxonomy = purity.get("pair_taxonomy") or {}
    if purity:
        lines.extend(
            [
                "## Pair Purity",
                "",
                f"- Strictly instructional fraction: `{purity_metrics.get('fraction_strictly_instructional_pairs')}`",
                f"- Ambiguous fraction: `{purity_metrics.get('fraction_ambiguous_pairs')}`",
                f"- Misoriented mixed-correctness fraction: `{purity_metrics.get('fraction_misoriented_mixed_correctness_pairs')}`",
                f"- Orientation by correctness: `{purity_metrics.get('fraction_orientation_driven_by_correctness')}`",
                f"- Orientation by utility only: `{purity_metrics.get('fraction_orientation_driven_only_by_utility')}`",
                f"- Weak divergence fraction: `{boundary.get('weak_divergence_fraction')}`",
                f"- Boundary instability rate: `{boundary.get('boundary_instability_rate')}`",
                f"- Near-identical divergence fraction: `{boundary.get('near_identical_divergence_fraction')}`",
                f"- Correct-vs-incorrect kept: `{taxonomy.get('correct_vs_incorrect')}`",
                f"- Incorrect-vs-correct kept: `{taxonomy.get('incorrect_vs_correct')}`",
                f"- Both-correct kept: `{taxonomy.get('both_correct')}`",
                f"- Both-wrong kept: `{taxonomy.get('both_wrong')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Buckets",
        "",
        ]
    )
    for bucket_name, stats in analysis["bucket_summary"].items():
        lines.extend(
            [
                f"### {bucket_name.title()}",
                f"- Count: `{stats['count']}`",
                f"- Fraction: `{stats['fraction']}`",
                f"- Mean confidence: `{stats['mean_confidence']}`",
                f"- Preferred-branch accuracy: `{stats['preferred_branch_accuracy']}`",
                f"- Preferred branch final-correct rate: `{stats['preferred_branch_final_correct_rate']}`",
                f"- Dispreferred branch final-correct rate: `{stats['dispreferred_branch_final_correct_rate']}`",
                f"- Decisive pair fraction: `{stats['decisive_pair_fraction']}`",
                f"- Both-branches-wrong fraction: `{stats['both_branches_incorrect_fraction']}`",
                f"- Process-boundary accuracy: `{stats['process_boundary_accuracy']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_pair(row: PairRecord) -> str:
    return "\n".join(
        [
            f"## {row.id}",
            "",
            f"- Confidence: `{pair_confidence(row)}`",
            f"- Training weight: `{row.weight}`",
            f"- Meta: `{json.dumps(row.meta, sort_keys=True)}`",
            f"- Features: `{json.dumps(row.features, sort_keys=True)}`",
            "",
            "### Prompt",
            row.prompt,
            "",
            "### Shared Prefix",
            "\n".join(row.prefix_steps) if row.prefix_steps else "(empty)",
            "",
            "### Preferred Segment",
            "\n".join(row.preferred_steps),
            "",
            "### Rejected Segment",
            "\n".join(row.dispreferred_steps),
            "",
        ]
    )


def _render_orientation_row(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {row.get('id')}",
            "",
            f"- Final status: `{row.get('final_status')}`",
            f"- Pair mode: `{row.get('pair_mode')}`",
            f"- Orientation reason: `{row.get('orientation_reason')}`",
            f"- Correctness pattern: `{row.get('correctness_pattern')}`",
            f"- Confidence: `{row.get('confidence')}`",
            f"- Utility margin: `{row.get('utility_margin')}`",
            f"- Local score gap: `{row.get('local_score_gap')}`",
            f"- Weak divergence: `{row.get('weak_divergence')}`",
            f"- Divergent similarity: `{row.get('divergent_similarity')}`",
            f"- Final correctness pref/disp: `{row.get('pref_final_correct')}` / `{row.get('disp_final_correct')}`",
            f"- Features: `{json.dumps(row.get('confidence_features', {}), sort_keys=True)}`",
            "",
            "### Prompt",
            row.get("prompt") or "(prompt unavailable)",
            "",
            "### Shared Prefix",
            "\n".join(row.get("prefix_steps") or []) if row.get("prefix_steps") else "(empty)",
            "",
            "### Preferred Segment",
            "\n".join(row.get("preferred_steps") or []) if row.get("preferred_steps") else "(none)",
            "",
            "### Rejected Segment",
            "\n".join(row.get("dispreferred_steps") or []) if row.get("dispreferred_steps") else "(none)",
            "",
        ]
    )


def write_pair_audits(
    pairs: list[PairRecord],
    *,
    low_threshold: float,
    high_threshold: float,
    sample_count: int,
    seed: int,
    low_path: str | Path | None,
    mid_path: str | Path | None,
    high_path: str | Path | None,
) -> None:
    buckets = {"low": [], "medium": [], "high": []}
    for pair in pairs:
        conf = pair_confidence(pair)
        if conf is None:
            continue
        buckets[confidence_bucket(conf, low_threshold, high_threshold)].append(pair)

    rng = random.Random(seed)
    path_map = {"low": low_path, "medium": mid_path, "high": high_path}
    for bucket_name, rows in buckets.items():
        path = path_map[bucket_name]
        if not path:
            continue
        sample = list(rows)
        rng.shuffle(sample)
        sample = sample[:sample_count]
        lines = [f"# Pair Audit: {bucket_name.title()}", ""]
        if not sample:
            lines.append("No pairs available in this bucket.")
        else:
            for row in sample:
                lines.append(_render_pair(row))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confidence_artifacts(analysis: dict, *, json_path: str | Path | None, report_path: str | Path | None) -> None:
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(render_confidence_report(analysis), encoding="utf-8")


def write_pair_purity_report(report: dict, *, json_path: str | Path | None) -> None:
    if not json_path:
        return
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_pair_orientation_audit(
    decision_rows: list[dict[str, Any]],
    *,
    path: str | Path | None,
    sample_count: int,
    seed: int,
) -> None:
    if not path:
        return

    sections: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "Kept: Correctness-Driven",
            [
                row
                for row in decision_rows
                if row.get("final_status") == "kept"
                and row.get("orientation_reason")
                in {
                    "correctness_priority_final_correctness",
                    "strict_purified_final_correctness",
                    "semi_purified_final_correctness",
                    "answer_level_final_correctness",
                }
            ],
        ),
        (
            "Kept: Utility-Driven",
            [
                row
                for row in decision_rows
                if row.get("final_status") == "kept"
                and row.get("orientation_reason")
                in {
                    "current_utility",
                    "correctness_priority_same_correctness_utility",
                    "semi_purified_same_correctness_strong_local_preference",
                }
            ],
        ),
        (
            "Dropped: Weak Divergence",
            [row for row in decision_rows if "weak_divergence" in (row.get("final_status") or "")],
        ),
        (
            "Dropped: Same-Correctness Purification",
            [
                row
                for row in decision_rows
                if row.get("final_status")
                in {
                    "dropped_strict_purified_same_correctness",
                    "dropped_semi_purified_same_correctness_not_stable",
                }
            ],
        ),
        (
            "Dropped: Confidence Threshold",
            [row for row in decision_rows if row.get("final_status") == "dropped_confidence_threshold"],
        ),
    ]

    rng = random.Random(seed)
    lines = ["# Pair Orientation Audit", ""]
    for title, rows in sections:
        lines.extend([f"## {title}", ""])
        sample = list(rows)
        rng.shuffle(sample)
        sample = sample[:sample_count]
        if not sample:
            lines.append("No examples in this section.")
            lines.append("")
            continue
        for row in sample:
            lines.append(_render_orientation_row(row))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

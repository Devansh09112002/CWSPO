from __future__ import annotations

import argparse
from pathlib import Path

from cwspo.config import ensure_dirs, load_config
from cwspo.data.prompts import load_eval_prompts, load_train_prompts
from cwspo.evaluation.final_eval import evaluate_final
from cwspo.evaluation.process_eval import evaluate_process, evaluate_process_dataset, write_process_failure_report
from cwspo.pipeline.build_pairs import build_pair_artifacts
from cwspo.pipeline.diagnostics import (
    analyze_pairs,
    write_confidence_artifacts,
    write_pair_audits,
    write_pair_orientation_audit,
    write_pair_purity_report,
)
from cwspo.pipeline.generate import generate_traces
from cwspo.pipeline.score import score_traces
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord, PromptRecord, ScoredTraceRecord, TraceRecord
from cwspo.training.train_step_dpo import train
from cwspo.utils.io import read_json, read_jsonl, write_json, write_jsonl
from cwspo.utils.seed import set_seed


def _override_method(cfg, method_name: str | None):
    if not method_name:
        return cfg
    return cfg.model_copy(update={"method": cfg.method.model_copy(update={"name": method_name})})


def _resume_stage_index(name: str) -> int:
    order = {
        "none": 0,
        "traces": 1,
        "scored": 2,
        "pairs": 3,
        "checkpoint": 4,
        "final_eval": 5,
        "process_eval": 6,
    }
    return order[name]


def _should_use_existing(path: str, *, stage_name: str, resume_from: str, config_flag: bool) -> bool:
    stage_index = _resume_stage_index(stage_name)
    return (Path(path).exists() and config_flag) or (_resume_stage_index(resume_from) >= stage_index and Path(path).exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--with-train', action='store_true')
    parser.add_argument('--with-final-eval', action='store_true')
    parser.add_argument('--with-process-eval', action='store_true')
    parser.add_argument('--ground-truth', default=None)
    parser.add_argument('--eval-model-name', default=None)
    parser.add_argument('--method-name', default=None)
    parser.add_argument(
        '--resume-from',
        choices=['none', 'traces', 'scored', 'pairs', 'checkpoint', 'final_eval', 'process_eval'],
        default='none',
    )
    args = parser.parse_args()

    cfg = _override_method(load_config(args.config), args.method_name)
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    train_prompts = load_train_prompts(cfg)
    eval_prompts = load_eval_prompts(cfg)

    if _should_use_existing(
        cfg.paths.traces_file,
        stage_name='traces',
        resume_from=args.resume_from,
        config_flag=cfg.resume.use_existing_traces,
    ):
        traces = read_jsonl(cfg.paths.traces_file, TraceRecord)
        print(f"[pipeline] reusing {len(traces)} traces -> {cfg.paths.traces_file}")
    else:
        traces = generate_traces(cfg, train_prompts)
        write_jsonl(cfg.paths.traces_file, traces)
        print(f"[pipeline] wrote {len(traces)} traces -> {cfg.paths.traces_file}")

    if _should_use_existing(
        cfg.paths.scored_file,
        stage_name='scored',
        resume_from=args.resume_from,
        config_flag=cfg.resume.use_existing_scored,
    ):
        scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
        print(f"[pipeline] reusing {len(scored)} scored traces -> {cfg.paths.scored_file}")
    else:
        traces = read_jsonl(cfg.paths.traces_file, TraceRecord)
        scored = score_traces(cfg, traces)
        write_jsonl(cfg.paths.scored_file, scored)
        print(f"[pipeline] wrote {len(scored)} scored traces -> {cfg.paths.scored_file}")

    if _should_use_existing(
        cfg.paths.pairs_file,
        stage_name='pairs',
        resume_from=args.resume_from,
        config_flag=cfg.resume.use_existing_pairs,
    ):
        pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord)
        print(f"[pipeline] reusing {len(pair_rows)} pairs -> {cfg.paths.pairs_file}")
    else:
        scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
        pair_artifacts = build_pair_artifacts(cfg, scored)
        pair_rows = pair_artifacts.pairs
        write_jsonl(cfg.paths.pairs_file, pair_rows)
        print(f"[pipeline] wrote {len(pair_rows)} pairs -> {cfg.paths.pairs_file}")
        process_gt = None
        if cfg.paths.process_ground_truth_file and Path(cfg.paths.process_ground_truth_file).exists():
            process_gt = read_jsonl(cfg.paths.process_ground_truth_file, ProcessGroundTruthRecord)
        analysis = analyze_pairs(
            pair_rows,
            low_threshold=cfg.confidence.low_threshold,
            high_threshold=cfg.confidence.high_threshold,
            histogram_bins=cfg.diagnostics.histogram_bins,
            num_raw_pairs=pair_artifacts.num_raw_pairs,
            num_dropped_by_confidence=pair_artifacts.num_dropped_by_confidence,
            method_name=pair_artifacts.method_name,
            pair_mode=pair_artifacts.pair_mode,
            pair_purity_report=pair_artifacts.pair_purity_report,
            process_ground_truth=process_gt,
        )
        write_confidence_artifacts(
            analysis,
            json_path=cfg.paths.confidence_analysis_file,
            report_path=cfg.paths.confidence_report_file,
        )
        write_pair_audits(
            pair_rows,
            low_threshold=cfg.confidence.low_threshold,
            high_threshold=cfg.confidence.high_threshold,
            sample_count=cfg.diagnostics.num_audit_samples_per_bucket,
            seed=cfg.diagnostics.pair_audit_seed,
            low_path=cfg.paths.pair_audit_low_file,
            mid_path=cfg.paths.pair_audit_mid_file,
            high_path=cfg.paths.pair_audit_high_file,
        )
        write_pair_purity_report(
            pair_artifacts.pair_purity_report,
            json_path=cfg.paths.pair_purity_report_file,
        )
        write_pair_orientation_audit(
            pair_artifacts.orientation_audit_rows,
            path=cfg.paths.pair_orientation_audit_file,
            sample_count=cfg.diagnostics.num_orientation_audit_samples,
            seed=cfg.diagnostics.pair_audit_seed,
        )

    train_summary = None
    if args.with_train:
        if _should_use_existing(
            str(Path(cfg.paths.checkpoint_dir) / "final"),
            stage_name='checkpoint',
            resume_from=args.resume_from,
            config_flag=cfg.resume.use_existing_checkpoint,
        ) and Path(cfg.paths.train_metrics_file).exists():
            train_summary = read_json(cfg.paths.train_metrics_file)
            print(f"[pipeline] reusing training summary: {train_summary}")
        else:
            pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord)
            train_summary = train(cfg, pair_rows)
            print(f"[pipeline] training summary: {train_summary}")

    final_summary = None
    if args.with_final_eval:
        if _should_use_existing(
            cfg.paths.final_eval_file,
            stage_name='final_eval',
            resume_from=args.resume_from,
            config_flag=cfg.resume.use_existing_final_eval,
        ):
            final_summary = read_json(cfg.paths.final_eval_file)
            print(f"[pipeline] reusing final eval: {final_summary}")
        else:
            final_summary = evaluate_final(cfg, eval_prompts, model_name=args.eval_model_name)
            write_json(cfg.paths.final_eval_file, final_summary)
            print(f"[pipeline] final eval: {final_summary}")

    process_summary = None
    if args.with_process_eval:
        gt_path = args.ground_truth or cfg.paths.process_ground_truth_file
        if not gt_path:
            raise SystemExit('--with-process-eval requires --ground-truth or paths.process_ground_truth_file')
        if _should_use_existing(
            cfg.paths.process_eval_file,
            stage_name='process_eval',
            resume_from=args.resume_from,
            config_flag=cfg.resume.use_existing_process_eval,
        ):
            process_summary = read_json(cfg.paths.process_eval_file)
            print(f"[pipeline] reusing process eval: {process_summary}")
        else:
            gt = read_jsonl(gt_path, ProcessGroundTruthRecord)
            if gt and gt[0].correct_steps:
                process_summary = evaluate_process_dataset(cfg, gt)
            else:
                pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord)
                process_summary = evaluate_process(pair_rows, gt)
            write_json(cfg.paths.process_eval_file, process_summary)
            write_process_failure_report(process_summary, cfg.paths.process_failure_report_file)
            print(f"[pipeline] process eval: {process_summary}")

    if cfg.paths.run_summary_file:
        pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord) if Path(cfg.paths.pairs_file).exists() else []
        confidence_summary = read_json(cfg.paths.confidence_analysis_file) if cfg.paths.confidence_analysis_file and Path(cfg.paths.confidence_analysis_file).exists() else {}
        process_confidence_summary = (process_summary or {}).get("process_confidence_analysis") or {}
        run_summary = {
            "seed": cfg.seed,
            "method_name": cfg.method.name,
            "pair_mode": getattr(cfg.pair, "pair_mode", "current_utility"),
            "policy_model_name": cfg.policy.model_name,
            "verifier_model_name": cfg.verifier.model_name,
            "dataset_name": cfg.data.dataset_name,
            "dataset_config_name": cfg.data.dataset_config_name,
            "train_prompt_count": len(train_prompts),
            "eval_prompt_count": len(eval_prompts),
            "num_traces": len(read_jsonl(cfg.paths.traces_file, TraceRecord)) if Path(cfg.paths.traces_file).exists() else None,
            "num_scored_traces": len(read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)) if Path(cfg.paths.scored_file).exists() else None,
            "num_pairs": len(pair_rows),
            "num_raw_pairs": confidence_summary.get("num_raw_pairs"),
            "num_dropped_by_confidence": confidence_summary.get("num_dropped_by_confidence"),
            "fraction_dropped_by_threshold": confidence_summary.get("fraction_dropped_by_threshold"),
            "train_steps": train_summary.get("num_steps") if train_summary else None,
            "effective_batch_size": train_summary.get("effective_batch_size") if train_summary else None,
            "train_wall_clock_time_sec": train_summary.get("wall_clock_time_sec") if train_summary else None,
            "final_accuracy": final_summary.get("accuracy") if final_summary else None,
            "final_num_examples": final_summary.get("num_examples") if final_summary else None,
            "adapter_path": final_summary.get("adapter_path") if final_summary else None,
            "adapter_loaded": final_summary.get("adapter_loaded") if final_summary else None,
            "process_earliest_error_exact": process_summary.get("earliest_error_exact") if process_summary else None,
            "process_earliest_error_near_miss": process_summary.get("earliest_error_near_miss") if process_summary else None,
            "process_coverage": process_summary.get("coverage") if process_summary else None,
            "process_num_examples": process_summary.get("num_evaluated_examples") if process_summary else None,
            "num_process_pairs": process_summary.get("num_process_pairs") if process_summary else None,
            "process_evaluation_mode": process_summary.get("evaluation_mode") if process_summary else None,
            "process_depends_on_trained_policy": process_summary.get("depends_on_trained_policy") if process_summary else None,
            "mean_confidence": confidence_summary.get("mean_confidence"),
            "median_confidence": confidence_summary.get("median_confidence"),
            "high_confidence_pair_accuracy": confidence_summary.get("high_confidence_pair_accuracy"),
            "overall_decisive_pair_accuracy": confidence_summary.get("overall_decisive_pair_accuracy"),
            "preferred_branch_final_correct_rate": confidence_summary.get("preferred_branch_final_correct_rate"),
            "dispreferred_branch_final_correct_rate": confidence_summary.get("dispreferred_branch_final_correct_rate"),
            "decisive_pair_fraction": confidence_summary.get("decisive_pair_fraction"),
            "both_branches_incorrect_fraction": confidence_summary.get("both_branches_incorrect_fraction"),
            "both_branches_correct_fraction": confidence_summary.get("both_branches_correct_fraction"),
            "confidence_correctness_correlation": confidence_summary.get("confidence_correctness_correlation"),
            "confidence_bucket_summary": confidence_summary.get("bucket_summary"),
            "pair_purity_report": confidence_summary.get("pair_purity_report"),
            "process_confidence_bucket_summary": process_confidence_summary.get("bucket_summary"),
            "artifacts": {
                "output_dir": cfg.paths.output_dir,
                "traces_file": cfg.paths.traces_file,
                "scored_file": cfg.paths.scored_file,
                "pairs_file": cfg.paths.pairs_file,
                "pair_purity_report_file": cfg.paths.pair_purity_report_file,
                "pair_orientation_audit_file": cfg.paths.pair_orientation_audit_file,
                "checkpoint_dir": cfg.paths.checkpoint_dir,
                "final_eval_file": cfg.paths.final_eval_file,
                "process_eval_file": cfg.paths.process_eval_file,
            },
        }
        write_json(cfg.paths.run_summary_file, run_summary)


if __name__ == '__main__':
    main()

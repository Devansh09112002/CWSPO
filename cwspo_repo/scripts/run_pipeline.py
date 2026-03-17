from __future__ import annotations

import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.evaluation.final_eval import evaluate_final
from cwspo.evaluation.process_eval import evaluate_process
from cwspo.pipeline.build_pairs import build_pairs
from cwspo.pipeline.generate import generate_traces
from cwspo.pipeline.score import score_traces
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord, PromptRecord, ScoredTraceRecord, TraceRecord
from cwspo.training.train_step_dpo import train
from cwspo.utils.io import read_jsonl, write_json, write_jsonl
from cwspo.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--with-train', action='store_true')
    parser.add_argument('--with-final-eval', action='store_true')
    parser.add_argument('--with-process-eval', action='store_true')
    parser.add_argument('--ground-truth', default=None)
    parser.add_argument('--eval-model-name', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    prompts = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    traces = generate_traces(cfg, prompts)
    write_jsonl(cfg.paths.traces_file, traces)
    print(f"[pipeline] wrote {len(traces)} traces -> {cfg.paths.traces_file}")

    traces = read_jsonl(cfg.paths.traces_file, TraceRecord)
    scored = score_traces(cfg, traces)
    write_jsonl(cfg.paths.scored_file, scored)
    print(f"[pipeline] wrote {len(scored)} scored traces -> {cfg.paths.scored_file}")

    scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
    pairs = build_pairs(cfg, scored)
    write_jsonl(cfg.paths.pairs_file, pairs)
    print(f"[pipeline] wrote {len(pairs)} pairs -> {cfg.paths.pairs_file}")

    if args.with_train:
        pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord)
        summary = train(cfg, pair_rows)
        print(f"[pipeline] training summary: {summary}")

    if args.with_final_eval:
        summary = evaluate_final(cfg, prompts, model_name=args.eval_model_name)
        write_json(cfg.paths.final_eval_file, summary)
        print(f"[pipeline] final eval: {summary}")

    if args.with_process_eval:
        if not args.ground_truth:
            raise SystemExit('--with-process-eval requires --ground-truth')
        gt = read_jsonl(args.ground_truth, ProcessGroundTruthRecord)
        pair_rows = read_jsonl(cfg.paths.pairs_file, PairRecord)
        summary = evaluate_process(pair_rows, gt)
        write_json(cfg.paths.process_eval_file, summary)
        print(f"[pipeline] process eval: {summary}")


if __name__ == '__main__':
    main()

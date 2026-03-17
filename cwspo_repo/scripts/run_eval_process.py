import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.evaluation.process_eval import evaluate_process, evaluate_process_dataset, write_process_failure_report
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord
from cwspo.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    gt_path = args.ground_truth or cfg.paths.process_ground_truth_file
    if not gt_path:
        raise SystemExit("--ground-truth is required unless paths.process_ground_truth_file is set in the config.")
    gt = read_jsonl(gt_path, ProcessGroundTruthRecord)
    if gt and gt[0].correct_steps:
        summary = evaluate_process_dataset(cfg, gt)
    else:
        pairs = read_jsonl(cfg.paths.pairs_file, PairRecord)
        summary = evaluate_process(pairs, gt)
    write_json(cfg.paths.process_eval_file, summary)
    write_process_failure_report(summary, cfg.paths.process_failure_report_file)
    print(summary)


if __name__ == "__main__":
    main()

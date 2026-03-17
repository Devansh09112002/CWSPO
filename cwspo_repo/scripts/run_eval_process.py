import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.evaluation.process_eval import evaluate_process
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord
from cwspo.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    pairs = read_jsonl(cfg.paths.pairs_file, PairRecord)
    gt = read_jsonl(args.ground_truth, ProcessGroundTruthRecord)
    summary = evaluate_process(pairs, gt)
    write_json(cfg.paths.process_eval_file, summary)
    print(summary)


if __name__ == "__main__":
    main()

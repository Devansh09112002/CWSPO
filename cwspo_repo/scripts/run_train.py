import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.schemas import PairRecord
from cwspo.training.train_step_dpo import train
from cwspo.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    pairs = read_jsonl(cfg.paths.pairs_file, PairRecord)
    summary = train(cfg, pairs)
    print(summary)


if __name__ == "__main__":
    main()

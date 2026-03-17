import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.pipeline.build_pairs import build_pairs
from cwspo.schemas import ScoredTraceRecord
from cwspo.utils.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
    pairs = build_pairs(cfg, scored)
    write_jsonl(cfg.paths.pairs_file, pairs)
    print(f"Wrote {len(pairs)} pairs to {cfg.paths.pairs_file}")


if __name__ == "__main__":
    main()

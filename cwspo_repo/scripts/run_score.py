import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.pipeline.score import score_traces
from cwspo.schemas import TraceRecord
from cwspo.utils.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    traces = read_jsonl(cfg.paths.traces_file, TraceRecord)
    scored = score_traces(cfg, traces)
    write_jsonl(cfg.paths.scored_file, scored)
    print(f"Wrote {len(scored)} scored traces to {cfg.paths.scored_file}")


if __name__ == "__main__":
    main()

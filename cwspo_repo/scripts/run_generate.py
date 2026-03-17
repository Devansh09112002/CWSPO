import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.pipeline.generate import generate_traces
from cwspo.schemas import PromptRecord
from cwspo.utils.io import read_jsonl, write_jsonl
from cwspo.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    prompts = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    traces = generate_traces(cfg, prompts)
    write_jsonl(cfg.paths.traces_file, traces)
    print(f"Wrote {len(traces)} traces to {cfg.paths.traces_file}")


if __name__ == "__main__":
    main()

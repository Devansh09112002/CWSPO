import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.evaluation.final_eval import evaluate_final
from cwspo.schemas import PromptRecord
from cwspo.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    prompts = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    summary = evaluate_final(cfg, prompts, model_name=args.model_name)
    write_json(cfg.paths.final_eval_file, summary)
    print(summary)


if __name__ == "__main__":
    main()

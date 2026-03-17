import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.data.prompts import load_eval_prompts
from cwspo.evaluation.final_eval import evaluate_final
from cwspo.utils.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    prompts = load_eval_prompts(cfg)
    summary = evaluate_final(cfg, prompts, model_name=args.model_name)
    write_json(cfg.paths.final_eval_file, summary)
    print(summary)


if __name__ == "__main__":
    main()

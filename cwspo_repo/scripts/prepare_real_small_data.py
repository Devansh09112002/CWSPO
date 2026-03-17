from __future__ import annotations

import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.data.real_small import build_gsm8k_prompt_rows, build_process_eval_rows
from cwspo.utils.io import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if cfg.data.max_train_prompts is None or cfg.data.max_eval_prompts is None:
        raise SystemExit("Real-small data prep requires data.max_train_prompts and data.max_eval_prompts in the config.")

    train_rows = build_gsm8k_prompt_rows(
        split="train",
        count=cfg.data.max_train_prompts,
        seed=cfg.data.prompt_sampling_seed,
        append_step_by_step_suffix=cfg.data.append_step_by_step_suffix,
        dataset_name=cfg.data.dataset_name or "openai/gsm8k",
        dataset_config_name=cfg.data.dataset_config_name or "main",
    )
    eval_rows = build_gsm8k_prompt_rows(
        split="test",
        count=cfg.data.max_eval_prompts,
        seed=cfg.data.prompt_sampling_seed + 1,
        append_step_by_step_suffix=cfg.data.append_step_by_step_suffix,
        dataset_name=cfg.data.dataset_name or "openai/gsm8k",
        dataset_config_name=cfg.data.dataset_config_name or "main",
    )

    write_jsonl(cfg.paths.prompt_file, train_rows)
    print(f"Wrote {len(train_rows)} training prompts to {cfg.paths.prompt_file}")

    if cfg.paths.eval_prompt_file:
        write_jsonl(cfg.paths.eval_prompt_file, eval_rows)
        print(f"Wrote {len(eval_rows)} eval prompts to {cfg.paths.eval_prompt_file}")

    if cfg.paths.process_ground_truth_file and cfg.data.max_process_examples:
        process_rows = build_process_eval_rows(
            count=cfg.data.max_process_examples,
            seed=cfg.data.prompt_sampling_seed + 2,
        )
        write_jsonl(cfg.paths.process_ground_truth_file, process_rows)
        print(f"Wrote {len(process_rows)} process eval examples to {cfg.paths.process_ground_truth_file}")


if __name__ == "__main__":
    main()

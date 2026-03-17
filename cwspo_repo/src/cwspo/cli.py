from __future__ import annotations

from pathlib import Path

import typer

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

app = typer.Typer(add_completion=False)


@app.command()
def generate(config: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    prompts = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    traces = generate_traces(cfg, prompts)
    write_jsonl(cfg.paths.traces_file, traces)
    typer.echo(f"Wrote {len(traces)} traces to {cfg.paths.traces_file}")


@app.command()
def score(config: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    traces = read_jsonl(cfg.paths.traces_file, TraceRecord)
    scored = score_traces(cfg, traces)
    write_jsonl(cfg.paths.scored_file, scored)
    typer.echo(f"Wrote {len(scored)} scored traces to {cfg.paths.scored_file}")


@app.command()
def pairs(config: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
    pairs_ = build_pairs(cfg, scored)
    write_jsonl(cfg.paths.pairs_file, pairs_)
    typer.echo(f"Wrote {len(pairs_)} pairs to {cfg.paths.pairs_file}")


@app.command("train")
def train_cmd(config: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    pairs_ = read_jsonl(cfg.paths.pairs_file, PairRecord)
    summary = train(cfg, pairs_)
    typer.echo(summary)


@app.command("eval-final")
def eval_final(config: str, model_name: str | None = None):
    cfg = load_config(config)
    ensure_dirs(cfg)
    prompts = read_jsonl(cfg.paths.prompt_file, PromptRecord)
    summary = evaluate_final(cfg, prompts, model_name=model_name)
    write_json(cfg.paths.final_eval_file, summary)
    typer.echo(summary)


@app.command("eval-process")
def eval_process(config: str, ground_truth: str):
    cfg = load_config(config)
    ensure_dirs(cfg)
    pairs_ = read_jsonl(cfg.paths.pairs_file, PairRecord)
    gt = read_jsonl(ground_truth, ProcessGroundTruthRecord)
    summary = evaluate_process(pairs_, gt)
    write_json(cfg.paths.process_eval_file, summary)
    typer.echo(summary)


def main():
    app()


if __name__ == "__main__":
    main()

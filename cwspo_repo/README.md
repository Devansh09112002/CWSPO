# CW-SPO: Confidence-Weighted Step Preference Optimization

A config-driven research scaffold for **Confidence-Weighted Step Preference Optimization** on mathematical reasoning with noisy process reward signals.

This repository implements an end-to-end offline pipeline:

1. Generate multiple reasoning traces from a policy model.
2. Score each trace step-by-step with a frozen verifier / PRM.
3. Build local step-level preference pairs around divergence points or likely error boundaries.
4. Compute a confidence weight for each pair.
5. Fine-tune the policy with a **confidence-weighted Step-DPO** objective.
6. Evaluate final-answer accuracy and process-level quality.

The code is designed so you can start with a small stack, for example:
- **Policy:** `Qwen2.5-Math-1.5B-Instruct`
- **Verifier:** a small PRM or a prompted verifier LM

and later swap in a stronger offline verifier, for example:
- **Verifier:** `Qwen2.5-Math-7B-PRM800K`

## What is implemented

- Config-driven pipeline with YAML configs
- Offline generation / scoring / pair-building / training / evaluation
- Modular verifier interface with two usable backends:
  - `judge_token`: generic LM-as-verifier using positive/negative token probabilities
  - `mean_logprob`: cheap fallback scorer based on prefix likelihood
- Weighted Step-DPO loss
- Optional low-confidence trust penalty
- JSONL-based datasets and artifacts
- Sample prompts and tests for core pair-building logic
- `Taskfile.yml` and `Makefile` shortcuts

## Repository layout

```text
cwspo_repo/
  configs/
    minimal.yaml
    strong.yaml
    data.yaml
  examples/
    sample_math.jsonl
    sample_processbench_like.jsonl
  scripts/
    run_generate.py
    run_score.py
    run_pairs.py
    run_train.py
    run_eval_final.py
    run_eval_process.py
  src/cwspo/
    cli.py
    config.py
    schemas.py
    data/
    models/
    pipeline/
    training/
    evaluation/
    utils/
  tests/
  README.md
  requirements.txt
  pyproject.toml
  Taskfile.yml
  Makefile
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Quick start

### 1) Edit the minimal config

Open `configs/minimal.yaml` and set:
- `policy.model_name`
- `verifier.model_name`
- `paths.prompt_file`
- `paths.output_dir`

### 2) Generate traces

```bash
task generate CONFIG=configs/minimal.yaml
```

Artifacts:
- `outputs/.../traces.jsonl`

### 3) Score traces with the verifier / PRM

```bash
task score CONFIG=configs/minimal.yaml
```

Artifacts:
- `outputs/.../scored.jsonl`

### 4) Build step-level preference pairs

```bash
task pairs CONFIG=configs/minimal.yaml
```

Artifacts:
- `outputs/.../pairs.jsonl`

### 5) Train weighted Step-DPO

```bash
task train CONFIG=configs/minimal.yaml
```

Artifacts:
- `outputs/.../checkpoints/`

### 6) Evaluate final-answer accuracy

```bash
task eval-final CONFIG=configs/minimal.yaml
```

### 7) Evaluate process-level accuracy

```bash
task eval-process CONFIG=configs/minimal.yaml GT=examples/sample_processbench_like.jsonl
```

## Recommended development path

### Minimal version
- Policy: 1.5B model
- Verifier: 1.5B PRM or prompted verifier
- Training-time method only
- Math reasoning only
- Confidence from 2–4 simple signals

### Stronger version
- Keep policy small, swap in stronger offline verifier
- Add trust penalty
- Add more difficult benchmarks
- Optionally scale policy to 7B

## Key config knobs

### Pair construction
- `pair.window_H`: local segment length after divergence
- `pair.alpha_local`: weight on verifier local score vs final correctness
- `pair.min_weight`: minimum confidence to keep a pair
- `pair.max_pairs_per_prompt`: cap to stabilize training size

### Confidence weights
- `confidence.gamma_margin`
- `confidence.gamma_sharp`
- `confidence.gamma_agree`
- `confidence.gamma_out`

### Training
- `training.beta`: DPO temperature
- `training.lambda_ref`: trust penalty strength for low-confidence pairs
- `training.use_lora`: enable PEFT LoRA
- `training.batch_size`, `training.grad_accum_steps`, `training.lr`

## JSONL schemas

### Prompt file
```json
{"id": "gsm8k_0001", "prompt": "Question text...", "answer": "42"}
```

### Traces
```json
{
  "id": "gsm8k_0001",
  "prompt": "...",
  "answer": "42",
  "trace_id": 0,
  "reasoning": "Step 1: ...\nStep 2: ...",
  "steps": ["...", "..."],
  "final_answer": "41"
}
```

### Scored traces
```json
{
  "id": "gsm8k_0001",
  "trace_id": 0,
  "steps": ["...", "..."],
  "step_scores": [0.72, 0.18],
  "final_answer": "41",
  "final_correct": false
}
```

### Pairs
```json
{
  "id": "gsm8k_0001",
  "prompt": "...",
  "prefix_steps": ["..."],
  "preferred_steps": ["..."],
  "dispreferred_steps": ["..."],
  "weight": 0.84,
  "features": {
    "f_margin": 0.91,
    "f_sharp": 0.75,
    "f_agree": 0.66,
    "f_out": 1.0,
    "weight": 0.84
  },
  "meta": {
    "pref_trace_id": 1,
    "disp_trace_id": 3,
    "k": 2
  }
}
```

## Verifier backends

### `judge_token`
Uses any causal LM as a verifier by prompting it to answer with one positive token and one negative token, for example `good` vs `bad`. The score is the probability of the positive token.

This is the most general backend and works even when you do not have a specialized PRM.

### `mean_logprob`
A lightweight fallback that uses average token log-probability of the prefix under the verifier model. This is not a true PRM, but it is useful for smoke tests.

## Notes on compute

The policy and verifier do **not** need to be resident on GPU at the same time.

Recommended workflow:
1. Load policy -> generate traces -> unload
2. Load verifier -> score traces -> unload
3. Load policy -> train on saved pairs

This keeps peak memory much lower than a naïve two-model live pipeline.

## Testing

```bash
pytest -q
```

## Limitations

- The default verifier backends are modular and practical, but they are not tailored to every public PRM architecture.
- Specialized PRMs can be wrapped by subclassing `BaseVerifier`.
- Math answer parsing is heuristic and should be customized for your benchmark.
- Process-level evaluation expects earliest-error annotations in a simple JSONL schema.

## Citation / related work

This repo is a research scaffold inspired by:
- step-level preference optimization
- process verifiers / PRMs
- confidence-aware pair construction
- earliest-error process benchmarks

You should cite the original papers for any publication.

## RTX 4090 48 GB workflow

For your setup, use the dedicated configs and scripts:

```bash
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl
```

Why this is practical on 48 GB:
- the **policy** and **verifier** are run in separate phases, not held on GPU together all the time,
- the minimal config uses a **1.5B policy** for training speed,
- the stronger config keeps the policy at 1.5B and upgrades only the offline verifier to reduce training cost.

## One-command tasks

```bash
# full pipeline without training
python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml

# full pipeline with training and evaluation
bash scripts/run_minimal_4090.sh

# stronger offline-verifier version
bash scripts/run_strong_4090.sh
```

## Codex-friendly files

- `AGENTS.md` gives repository-level guardrails and workflow.
- `docs/CODEX.md` explains the expected run order.
- `tasks/PIPELINE.md` is a milestone checklist.

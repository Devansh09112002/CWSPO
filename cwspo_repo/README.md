# CW-SPO: Confidence-Weighted Step Preference Optimization

This repository implements a config-driven offline pipeline for:
- math reasoning only,
- offline verifier / PRM scoring,
- earliest-error / local-divergence pair construction,
- confidence-weighted Step-DPO as the main training path.

The intended flow is:
1. generate multiple traces from a policy model,
2. score trace prefixes with a frozen verifier,
3. build local preference pairs around divergence points,
4. attach interpretable confidence signals,
5. train a LoRA adapter with Step-DPO or a baseline variant,
6. evaluate final-answer accuracy and process-level earliest-error prediction.

Detailed technical note:
- `docs/TECHNICAL_PIPELINE.md`

## Current runnable paths

- Smoke test: `configs/rtx4090_48gb_minimal.yaml`
- Real-small baseline matrix on a 48 GB RTX 4090:
  - `configs/rtx4090_48gb_real_small_step_dpo.yaml`
  - `configs/rtx4090_48gb_real_small_conf_filter.yaml`
  - `configs/rtx4090_48gb_real_small.yaml`
  - `configs/rtx4090_48gb_real_small_answer_dpo.yaml`
  - `configs/rtx4090_48gb_real_small_strong_verifier.yaml`

The real-small setup uses:
- `100` GSM8K train prompts
- `24` GSM8K eval prompts
- `48` fixed process-eval examples
- `4` policy traces per train prompt

## Refinement configs

The pair-purification / orientation refinement phase uses:
- `configs/rtx4090_48gb_refine_step_current.yaml`
- `configs/rtx4090_48gb_refine_step_correctness.yaml`
- `configs/rtx4090_48gb_refine_step_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_current.yaml`
- `configs/rtx4090_48gb_refine_cw_correctness.yaml`
- `configs/rtx4090_48gb_refine_cw_strict.yaml`
- `configs/rtx4090_48gb_refine_conf_filter_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_semi.yaml`
- `configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml`

These configs keep:
- the same `100 / 24 / 48` prompt slice,
- the same `1.5B` policy model,
- the same offline phase separation,
- and change only the pair construction mode and, in the strong-verifier run, the verifier backend.

## Bootstrap

```bash
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
export HF_HOME=$PWD/.hf_home
```

Notes:
- `scripts/bootstrap_venv.sh` works on systems with `python3` but no `python`.
- `HF_HOME=$PWD/.hf_home` keeps model downloads local to this repo.
- The editable install exposes `cwspo`, but the `scripts/run_*.py` entrypoints are the clearest path.

## Smoke test

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_minimal.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --ground-truth examples/sample_processbench_like.jsonl
```

Use this once to confirm the repo and local environment are healthy. After that, move to the real-small configs.

## Real-small experiment workflow

### 1. Prepare the reproducible data slice

```bash
python scripts/prepare_real_small_data.py \
  --config configs/rtx4090_48gb_real_small_step_dpo.yaml
```

This writes:
- `data/real_small/gsm8k_train_100_seed42.jsonl`
- `data/real_small/gsm8k_eval_24_seed43.jsonl`
- `data/real_small/process_eval_48_seed44.jsonl`

### 2. Run the small-verifier baseline matrix

Run A: plain Step-DPO

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small_step_dpo.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval
```

Run B: confidence-filter-only

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small_conf_filter.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval
```

Run C: confidence-weighted Step-DPO

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval
```

Optional answer-level DPO baseline

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small_answer_dpo.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval
```

These runs share:
- the same train prompt slice,
- the same eval prompt slice,
- the same base policy model,
- the same shared train traces and small-verifier scored traces when present.

Shared artifacts live under:
- `outputs/real_small/shared/train_traces.jsonl`
- `outputs/real_small/shared/scored_small_verifier.jsonl`

### 3. Strong-verifier comparison

Run D: confidence-weighted Step-DPO with a stronger offline verifier

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small_strong_verifier.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval
```

This keeps the policy at `Qwen/Qwen2.5-Math-1.5B-Instruct` and changes only the verifier path. The stronger verifier rescoring is still offline, so it does not stay loaded during policy training.
The strong-verifier config uses `verifier.mode: process_reward_model` for `Qwen/Qwen2.5-Math-7B-PRM800K`, which scores step boundaries through the PRM `<extra_0>` process-reward interface rather than judge-token generation.

## Refinement workflow: pair purification and orientation

Use this after the baseline matrix if you want to test whether local pair quality is the real bottleneck.

### Compare current utility vs correctness-priority vs strict purification

Step-DPO with the current local utility rule:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_step_current.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Step-DPO with correctness-priority orientation:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_step_correctness.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Step-DPO with strict purified pairs:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_step_strict.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Confidence-weighted Step-DPO with the current local utility rule:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_cw_current.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Confidence-weighted Step-DPO with correctness-priority orientation:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_cw_correctness.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Confidence-weighted Step-DPO with strict purified pairs:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_cw_strict.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Post-purification strong-verifier comparison:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Strict hard-filter tiebreak on the purified target:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_conf_filter_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

Low-priority semi-purified recovery attempt:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_semi.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

Pair-mode definitions:
- `current_utility`: prefer the branch with the larger mixed local utility.
- `correctness_priority`: if final correctness differs, always prefer the finally correct branch; otherwise fall back to local utility.
- `strict_purified`: keep only mixed-correctness pairs and drop both-correct, both-wrong, and weak-divergence local pairs.
- `semi_purified`: keep mixed-correctness pairs always and admit same-correctness pairs only under conservative local criteria.

Observed checked-in refinement results on the seed-42 slice:
- `step_current = 0.4167`
- `step_correctness = 0.4583`
- `step_strict = 0.4583`
- `cw_current = 0.4167`
- `cw_correctness = 0.4167`
- `cw_strict = 0.5000`
- `conf_filter_strict = 0.5000`
- `cw_strict_strongverifier = 0.4583`

Interpretation:
- purification mattered more than rescoring,
- confidence-aware training became competitive only after purification,
- soft weighting tied hard filtering on the purified target,
- and strict process scores must be read together with coverage because many weak-divergence process examples are intentionally dropped.

## Resume commands

The pipeline supports resumable stages through `--resume-from`.

Resume from traces:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from traces
```

Resume from scored traces:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_real_small.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from scored
```

Resume from built pairs:

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_refine_cw_strict.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Train only:

```bash
python scripts/run_train.py --config configs/rtx4090_48gb_real_small.yaml
```

Evaluation only:

```bash
python scripts/run_eval_final.py --config configs/rtx4090_48gb_real_small.yaml
python scripts/run_eval_process.py --config configs/rtx4090_48gb_real_small.yaml
```

Direct stage-by-stage:

```bash
python scripts/run_generate.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
python scripts/run_score.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
python scripts/run_pairs.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
python scripts/run_train.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
python scripts/run_eval_final.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
python scripts/run_eval_process.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml
```

## What each stage does

- `scripts/run_generate.py`: samples multiple reasoning traces per train prompt and writes `traces.jsonl`.
- `scripts/run_score.py`: scores each step prefix offline with the configured verifier and writes `scored.jsonl`.
- `scripts/run_pairs.py`: builds local or answer-level preference pairs and writes confidence diagnostics plus pair audits.
- `scripts/run_train.py`: trains the selected DPO variant and writes a LoRA adapter plus training reports.
- `scripts/run_eval_final.py`: runs deterministic held-out answer evaluation and records whether the adapter was successfully loaded.
- `scripts/run_eval_process.py`: evaluates earliest-error prediction on the process set and writes both machine-readable and human-readable reports.
- `scripts/run_pipeline.py`: orchestrates the same stages end-to-end with resume support.

Process-eval scope:
- `process_eval.json` is currently an offline fixed-trace boundary-detection diagnostic. It does not measure the trained policy adapter directly, so treat it as verifier/pair-quality evidence rather than end-task policy evidence.

## Output layout

Real-small artifacts are separated by run:
- `outputs/real_small/shared/`
- `outputs/real_small/step_dpo_smallverifier/`
- `outputs/real_small/conf_filter_smallverifier/`
- `outputs/real_small/cwspo_smallverifier/`
- `outputs/real_small/answer_dpo_smallverifier/`
- `outputs/real_small/cwspo_strongverifier/`
- `outputs/refinement/step_current/`
- `outputs/refinement/step_correctness/`
- `outputs/refinement/step_strict/`
- `outputs/refinement/cw_current/`
- `outputs/refinement/cw_correctness/`
- `outputs/refinement/cw_strict/`
- `outputs/refinement/conf_filter_strict/`
- `outputs/refinement/cw_semi/`
- `outputs/refinement/cw_strict_strongverifier/`

Important files inside each run directory:
- `pairs.jsonl`: training pairs for that method
- `confidence_analysis.json`: machine-readable confidence diagnostics
- `confidence_report.md`: readable confidence summary
- `pair_purity_report.json`: pair taxonomy, purity metrics, and boundary diagnostics
- `pair_orientation_audit.md`: sampled kept and dropped pairs with orientation reasons
- `pair_audit_low.md`
- `pair_audit_mid.md`
- `pair_audit_high.md`
- `training_report.json`
- `training_report.md`
- `checkpoints/final/`
- `final_eval.json`
- `process_eval.json`
- `process_failures.md`
- `run_summary.json`

## Confidence diagnostics

`confidence_analysis.json` and `confidence_report.md` summarize:
- pair counts and threshold drops,
- mean and median confidence,
- low / medium / high confidence bucket sizes,
- preferred-branch correctness by bucket,
- confidence vs correctness correlation,
- a simple calibration proxy,
- high-confidence pair accuracy.

The pair audit markdown files show sampled pairs from low / medium / high confidence buckets with:
- prompt,
- shared prefix,
- preferred segment,
- rejected segment,
- confidence score,
- confidence features,
- branch correctness metadata.

The refinement phase adds:
- `pair_purity_report.json`, which reports `correct_vs_incorrect`, `incorrect_vs_correct`, `both_correct`, `both_wrong`, weak-divergence counts, and orientation fractions;
- `pair_orientation_audit.md`, which shows kept correctness-driven pairs, kept utility-driven pairs, and dropped pairs from weak-divergence and purification rules.

Use these files to compare:
- `current_utility` vs `correctness_priority` vs `strict_purified`
- small verifier vs strong verifier after purification
- whether pair cleaning improves downstream answer accuracy or only pair-quality proxies

Helpful inspection commands:

```bash
sed -n '1,220p' outputs/refinement/cw_strict/pair_orientation_audit.md
sed -n '1,220p' outputs/refinement/cw_strict/confidence_report.md
sed -n '1,220p' tasks/DIAGNOSIS_AND_REFINEMENT.md
```

Use these audit files before trusting any training run.

## Switching verifiers

The verifier is modular and config-driven.

Small verifier path:
- `verifier.model_name: Qwen/Qwen2.5-Math-1.5B-Instruct`
- config examples: `configs/rtx4090_48gb_real_small_step_dpo.yaml`, `configs/rtx4090_48gb_real_small.yaml`

Stronger verifier path:
- `verifier.model_name: Qwen/Qwen2.5-Math-7B-PRM800K`
- config example: `configs/rtx4090_48gb_real_small_strong_verifier.yaml`

When switching verifiers, keep the same policy config first and rerun only:
1. scoring,
2. pair building,
3. training,
4. evaluation.

Do not scale the policy to 7B until the 1.5B-policy experiments are stable and the confidence diagnostics look plausible.

## Common failure cases

- `python: command not found` during bootstrap:
  - rerun `bash scripts/bootstrap_venv.sh`; it now falls back to `python3`.
- `flash_attention_2` warning:
  - the loader automatically falls back to the model default attention path in this environment.
- HF download throttling or authorization errors:
  - set `HF_TOKEN`, keep `HF_HOME=$PWD/.hf_home`, and retry.
- No pairs produced:
  - inspect `scored.jsonl`,
  - increase `policy.num_return_sequences`,
  - lower `pair.tau_pair`,
  - lower `method.confidence_threshold` for the filter baseline.
- Final eval unexpectedly using the base model:
  - inspect `final_eval.json`,
  - verify `adapter_loaded: true`,
  - check that `checkpoints/final/adapter_config.json` exists.
- Process coverage below `1.0`:
  - inspect `process_failures.md`; missing predictions usually mean no usable local pair was built for that example.
  - in `strict_purified` runs this is expected on the current offline process set because weak-divergence examples are intentionally rejected.

## Testing

```bash
pytest -q
```

## Repository notes

- `AGENTS.md`: repo-level workflow guardrails
- `docs/CODEX.md`: recommended run order and scaling guidance
- `tasks/RUN_LOG.md`: execution log
- `tasks/EXPERIMENT_LOG.md`: experiment log
- `tasks/RESULTS_SUMMARY.md`: scoreboard and critique

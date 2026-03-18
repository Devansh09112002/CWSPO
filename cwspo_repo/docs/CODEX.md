# Codex / Agent workflow

## Recommended run order

1. Bootstrap the repo.

```bash
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
export HF_HOME=$PWD/.hf_home
```

2. Run the smoke test once.

```bash
python scripts/run_pipeline.py \
  --config configs/rtx4090_48gb_minimal.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --ground-truth examples/sample_processbench_like.jsonl
```

3. Prepare the real-small data slice.

```bash
python scripts/prepare_real_small_data.py \
  --config configs/rtx4090_48gb_real_small_step_dpo.yaml
```

4. Run the small-verifier baseline matrix in this order:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml --with-train --with-final-eval --with-process-eval
python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_conf_filter.yaml --with-train --with-final-eval --with-process-eval
python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small.yaml --with-train --with-final-eval --with-process-eval
python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_answer_dpo.yaml --with-train --with-final-eval --with-process-eval
```

5. Run the pair-refinement matrix before scaling anything else.

Start with the Step-DPO pair-mode comparison:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_current.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_correctness.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

Then compare the confidence-aware purified path:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_current.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_correctness.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

Optional tiebreak and low-priority recovery run:

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_conf_filter_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_semi.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict_lambda_ref.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

6. Only after the purified small-verifier runs are stable, run the stronger-offline-verifier purified comparison.

```bash
python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

## Why this order matters

- The smoke test proves wiring, not research value.
- The real-small small-verifier matrix is where you check whether the method is scientifically alive.
- The next bottleneck after the baseline matrix is pair quality, not model scale.
- Pair purification should come before any larger dataset or 7B-policy push because the current risk is contaminated supervision, not lack of capacity.
- The stronger verifier should be retested after purification, because a better scorer can still underperform downstream if the pair target is noisy.
- Do not move to a 7B policy until:
  - pair audits look plausible,
  - confidence diagnostics look meaningful,
  - the purified-pair comparison is complete,
  - the stronger verifier path is stable after purification.
- The checked-in refinement runs currently support this narrower conclusion:
  - `strict_purified` is the right next local-pair path,
  - `current_utility` is too contaminated to trust as the main training target,
  - and the strong verifier is not the main lever yet.

## Resume order

Resume from generated traces:

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
  --config configs/rtx4090_48gb_real_small.yaml \
  --with-train \
  --with-final-eval \
  --with-process-eval \
  --resume-from pairs
```

Train only:

```bash
python scripts/run_train.py --config configs/rtx4090_48gb_real_small.yaml
```

Eval only:

```bash
python scripts/run_eval_final.py --config configs/rtx4090_48gb_real_small.yaml
python scripts/run_eval_process.py --config configs/rtx4090_48gb_real_small.yaml
```

## Expected artifacts

Shared real-small artifacts:
- `outputs/real_small/shared/train_traces.jsonl`
- `outputs/real_small/shared/scored_small_verifier.jsonl`
- `outputs/real_small/shared/scored_strong_verifier.jsonl`

Per-run artifacts:
- `pairs.jsonl`
- `confidence_analysis.json`
- `confidence_report.md`
- `pair_purity_report.json`
- `pair_orientation_audit.md`
- `diagnosis_summary.md`
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

## First debugging checklist

1. Inspect generated traces before trusting scores.
2. Inspect scored traces before trusting pairs.
3. Read the pair audit markdown files before trusting training.
4. Inspect `pair_purity_report.json` and `pair_orientation_audit.md` before trusting local supervision.
5. Inspect `confidence_report.md` before trusting the confidence mechanism.
6. Inspect `diagnosis_summary.md` before writing any run-level conclusion.
7. Compare `current_utility` vs `correctness_priority` vs `strict_purified` before scaling.
8. Confirm `final_eval.json` shows `adapter_loaded: true` before interpreting answer accuracy.
9. Read `process_failures.md` before trusting process metrics.

## Runtime caveats on a 48 GB RTX 4090

Observed on the real-small Step-DPO run:
- small-verifier scoring for `400` train traces: about `2-3` minutes
- training on `536` local pairs for `68` optimizer steps: about `6.9` minutes
- final eval on `24` GSM8K prompts: about `1.3` minutes
- process eval on `48` fixed examples: about `15-20` seconds plus verifier load time

Practical notes:
- the policy and verifier stay in separate phases; do not keep both large models resident together
- the first HF download will be much slower than a cached rerun
- the stronger verifier path should be expected to take noticeably longer than the small-verifier path because rescoring is heavier
- the `Qwen/Qwen2.5-Math-7B-PRM800K` path now uses `verifier.mode: process_reward_model`, not judge-token prompting
- the current `process_eval.json` artifact is an offline fixed-trace boundary diagnostic, not a trained-policy process benchmark
- the refinement configs under `configs/rtx4090_48gb_refine_*.yaml` reuse the same `100 / 24 / 48` slice, so they are directly comparable if you keep `--resume-from pairs`
- from `--resume-from pairs`, the `549`-pair correctness-priority runs take about `8-9` minutes end-to-end, while the `153`-pair strict-purified runs take about `4-5` minutes end-to-end
- the repaired `semi_purified` builder now writes explicit keep/drop reason codes, so use `pair_purity_report.json` and `diagnosis_summary.md` together when checking whether same-correctness recovery was real or just noisy

## Progression path

### Stage 1: 1.5B policy + small verifier

Use:
- `configs/rtx4090_48gb_real_small_step_dpo.yaml`
- `configs/rtx4090_48gb_real_small_conf_filter.yaml`
- `configs/rtx4090_48gb_real_small.yaml`
- `configs/rtx4090_48gb_real_small_answer_dpo.yaml`

Goal:
- establish a baseline matrix,
- audit pair quality,
- test whether confidence is meaningful.

### Stage 2: 1.5B policy + strong verifier

Use:
- `configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml`

Goal:
- keep the same policy and prompt slice,
- change only the verifier after pair purification,
- measure whether rescoring produces cleaner pairs and better downstream behavior.

Current checked-in outcome:
- the strong verifier did not beat the small-verifier `strict_purified` run on held-out answer accuracy,
- so keep it as an optional comparison backend, not the default next step.

## Current research stance

- Read `tasks/SERIOUS_DIAGNOSIS.md` before making any project-level claim.
- Treat `strict_purified` as the default local target.
- Treat `semi_purified` as a controlled admissibility experiment, not as a larger unfiltered target.
- Use `lambda_ref` only on purified targets first.

### Stage 3: optional 7B policy later

Only consider this after:
- Stage 1 and Stage 2 are stable,
- the confidence mechanism looks worth investing in,
- the bottleneck is clearly policy capacity rather than pair quality or verifier quality.

## Do not do this yet

- Do not switch to a 7B policy before the strong-verifier comparison is complete.
- Do not treat a perfect or near-perfect strict-mode process score as meaningful unless you also inspect coverage.
- Do not increase dataset size aggressively before reading the pair audits and pair-purity report.
- Do not interpret a successful pipeline completion as method validation.
- Do not compare baselines on different prompt subsets unless you document the mismatch.
- Do not keep the policy model and strong verifier in memory together during training.
- Do not overclaim strong process metrics: `process_eval.json` is still an offline boundary diagnostic, not direct evidence that the trained policy improved step-by-step reasoning.
- Do not prioritize `semi_purified` or larger-scale runs before repeating `step_strict`, `conf_filter_strict`, and `cw_strict` across multiple seeds.

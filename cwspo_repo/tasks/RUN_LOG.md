# Run Log

## Assumptions

- The working repository root is `/workspace/CWSPO/cwspo_repo`.
- The minimal first experiment should prioritize end-to-end correctness on a tiny subset over benchmark scale.
- If full real-model execution is blocked by local hardware, auth, or interface issues, a clearly marked mock fallback is acceptable only to validate pipeline wiring while keeping the real path intact.

## Intended Pipeline Summary

- Read a config-driven math prompt dataset from JSONL, where each row contains at least `id`, `prompt`, and `answer`.
- Generate multiple candidate reasoning traces per prompt with the policy model and persist them as `traces.jsonl`.
- Split each generated reasoning trace into ordered intermediate steps so later stages can score and compare local reasoning segments.
- Score every step prefix offline with a frozen verifier or PRM and write per-step scores plus final-answer correctness to `scored.jsonl`.
- Compare traces for the same prompt to find earliest local divergence or likely error boundaries rather than only whole-trace winners.
- Build local preference pairs with `prefix_steps`, `preferred_steps`, and `dispreferred_steps`, then cap and filter them according to config.
- Compute an interpretable confidence weight for each pair from simple signals such as score margin, score sharpness/drop, agreement with outcome, and outcome correctness.
- Train the policy with confidence-weighted Step-DPO as the main path, optionally including a low-confidence trust penalty.
- Save reproducible training artifacts including checkpoints and training metrics under the configured output directory.
- Evaluate the trained policy on final-answer accuracy and save structured outputs to `final_eval.json`.
- Evaluate process-level quality against process supervision data and save structured outputs to `process_eval.json`.
- Keep policy generation/training and verifier scoring separated so both larger models do not need to stay loaded together.

## Progress

- 2026-03-17: Read `README.md`, `AGENTS.md`, `docs/CODEX.md`, `tasks/PIPELINE.md`, and `configs/rtx4090_48gb_minimal.yaml`.
- 2026-03-17: Added the initial intended pipeline summary and assumptions.
- 2026-03-17: Fixed `scripts/bootstrap_venv.sh` so setup works on machines with `python3` but no `python`.
- 2026-03-17: Added a packaged `cwspo` CLI entrypoint and aligned the Typer training command to `cwspo train`.
- 2026-03-17: Made model loading more robust by honoring configured devices, auto-falling back when `flash_attention_2` is unavailable, and loading PEFT adapter checkpoints for final evaluation.
- 2026-03-17: Hardened the verifier backend so `judge_token` works even when verifier labels tokenize to multiple tokens.
- 2026-03-17: Fixed tiny-run training behavior by flushing leftover gradient accumulation steps and handling empty-pair runs cleanly.
- 2026-03-17: Tightened pair building to skip empty post-divergence segments and expanded tests for step splitting, divergence detection, confidence weighting, pair construction, and weighted loss forward pass.
- 2026-03-17: Validated the repo environment with `pip check`, import checks, CLI help, and `pytest -q` (`8` tests passing).
- 2026-03-17: Ran the real 4090 minimal pipeline stage-by-stage and via `scripts/run_pipeline.py` on the bundled 3-example sample set.
- 2026-03-17: Updated `README.md` and `docs/CODEX.md` with verified commands, resume order, output locations, and common failure fixes.
- 2026-03-17: Added a real-small GSM8K data-prep path plus seeded train/eval/process split generation under `data/real_small/`.
- 2026-03-17: Extended config handling for method selection, resume flags, eval prompt files, process ground-truth files, diagnostics outputs, and per-run summaries.
- 2026-03-17: Implemented the baseline matrix wiring for `answer_dpo`, `step_dpo`, `confidence_filter_only`, and `confidence_weighted_step_dpo`.
- 2026-03-17: Added confidence analysis reports, low/mid/high pair audits, training reports, and richer process-eval failure reporting.
- 2026-03-17: Added process-evaluation edge-case tests and expanded pair-construction tests; `pytest -q` now passes with `12` tests.
- 2026-03-17: Started Run A for the real-small matrix with `configs/rtx4090_48gb_real_small_step_dpo.yaml`; shared generation and small-verifier scoring completed, and `536` pairs were produced for training.
- 2026-03-17: Completed Run A (`step_dpo` + small verifier) end-to-end on the real-small setup with `400` traces, `536` pairs, `68` train steps, `0.4583` final accuracy, and `0.8958` process earliest-error exact.
- 2026-03-17: Refreshed Run A process evaluation with the richer process-confidence bucket reporting and updated `run_summary.json`.
- 2026-03-17: Started Run B (`confidence_filter_only` + small verifier) from the prebuilt pair set using `--resume-from pairs`.
- 2026-03-17: Expanded confidence diagnostics to report preferred-branch final-correct rate, decisive-pair fraction, and both-branches-wrong contamination.
- 2026-03-17: Rewrote `README.md` and `docs/CODEX.md` around the real-small baseline matrix, resume points, confidence diagnostics, and verifier progression.
- 2026-03-17: Preflighted the strong-verifier path; `bitsandbytes` is installed and `Qwen/Qwen2.5-Math-7B-PRM800K` resolves from the HF Hub.
- 2026-03-17: Completed Run B (`confidence_filter_only` + small verifier) with `458` kept pairs, `78` dropped pairs, `58` train steps, `0.4167` final accuracy, and `0.8958` process earliest-error exact.
- 2026-03-17: Refreshed Run B pair diagnostics and process evaluation with the richer confidence schema.
- 2026-03-17: Clarified the process-eval artifact schema so rich process evaluation is explicitly marked as an offline fixed-trace boundary diagnostic that does not depend on the trained policy adapter.
- 2026-03-17: Added `docs/TECHNICAL_PIPELINE.md`, a dedicated implementation note covering the real pipeline, equations, pair-construction logic, confidence computation, verifier modes, training objective, and evaluation caveats.

## Phase: Moving from smoke test to real experiment

### Current State of the Repo

- The repository is operational end-to-end on the 4090 using the minimal 1.5B policy path.
- Generation, offline verifier scoring, local pair construction, weighted Step-DPO training, final evaluation, and process evaluation all execute without mock mode.
- The current implementation has basic tests and a documented smoke-test run.

### What the Smoke Test Already Proved

- The core pipeline wiring works across all stages.
- The LoRA training path writes usable adapter checkpoints and final evaluation can load them.
- The verifier backend, pair construction logic, and JSONL artifact flow are functional.
- The repo can run honestly on local hardware with offline model phases separated in memory.

### What the Smoke Test Did Not Prove

- It did not establish any real research signal because the sample set has only `3` prompts.
- It did not compare plain Step-DPO, answer-level DPO, hard confidence filtering, and confidence weighting.
- It did not test confidence quality on a non-trivial number of pairs.
- It did not evaluate whether a stronger offline verifier improves pair quality or downstream performance.
- It did not harden process evaluation beyond the tiny bundled proxy dataset.

### Next Experimental Objective

- Build a real-small experiment pipeline on a materially larger math subset that stays runnable on a 48 GB RTX 4090.
- Reuse the same prompt slice where possible to compare:
  - answer-level DPO
  - plain Step-DPO
  - confidence filtering only
  - confidence-weighted Step-DPO
- Add confidence diagnostics, pair audits, richer process-eval reporting, and machine-readable run summaries.
- Run a stronger-offline-verifier comparison while keeping the 1.5B policy fixed.

### Exact Run Plan for This Phase

1. Prepare a reproducible real-small dataset split with substantially more prompts than the smoke test.
2. Add config support for prompt limits, eval split, method selection, diagnostics, and resume behavior.
3. Create `configs/rtx4090_48gb_real_small*.yaml` experiment configs with separated output directories under `outputs/real_small/`.
4. Generate and score shared training traces once with the small verifier.
5. Build and train baseline runs for:
   - `answer_dpo`
   - `step_dpo`
   - `confidence_filter_only`
   - `confidence_weighted_step_dpo`
6. Rescore the same trace set with the stronger offline verifier and rerun the confidence-weighted method.
7. Generate confidence analysis files, pair audits, process-eval reports, training reports, and per-run JSON summaries.
8. Write the honest comparison table and critique in `tasks/RESULTS_SUMMARY.md`.

## Commands Run

- `bash scripts/bootstrap_venv.sh`
- `source .venv/bin/activate && pip check`
- `source .venv/bin/activate && pytest -q`
- `source .venv/bin/activate && cwspo --help`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_generate.py --config configs/rtx4090_48gb_minimal.yaml`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_score.py --config configs/rtx4090_48gb_minimal.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_minimal.yaml`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_train.py --config configs/rtx4090_48gb_minimal.yaml`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_eval_final.py --config configs/rtx4090_48gb_minimal.yaml`
- `source .venv/bin/activate && python scripts/run_eval_process.py --config configs/rtx4090_48gb_minimal.yaml --ground-truth examples/sample_processbench_like.jsonl`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl`
- `source .venv/bin/activate && python scripts/prepare_real_small_data.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml`
- `source .venv/bin/activate && python -m compileall src scripts tests`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml --with-train --with-final-eval --with-process-eval --resume-from final_eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_conf_filter.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_real_small_step_dpo.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_real_small.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_real_small_answer_dpo.yaml`
- `source .venv/bin/activate && python - <<'PY' ... import bitsandbytes ... PY`
- `source .venv/bin/activate && python - <<'PY' ... model_info(\"Qwen/Qwen2.5-Math-7B-PRM800K\") ... PY`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_real_small_conf_filter.yaml`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_conf_filter.yaml --with-train --with-final-eval --with-process-eval --resume-from final_eval`
- `source .venv/bin/activate && sed -n '1,220p' src/cwspo/pipeline/build_pairs.py`
- `source .venv/bin/activate && sed -n '1,220p' src/cwspo/training/losses.py`
- `source .venv/bin/activate && sed -n '1,260p' src/cwspo/models/verifier.py`

## Final Status

### What Was Broken

- The bootstrap script required `python` even when only `python3` was available.
- The repo had no installed console entrypoint for the Typer CLI.
- `judge_token` verifier scoring failed if label strings were not single-token encodings.
- `device.policy`, `device.verifier`, `training.log_every`, and `evaluation.batch_size` were declared in config but not fully honored.
- Tiny training runs could finish with zero optimizer steps when the dataloader length was smaller than the gradient accumulation schedule remainder.
- Final evaluation did not reliably load the trained LoRA adapter checkpoint.
- Batched final evaluation emitted a decoder right-padding warning.
- The docs referenced `.env.example`, which does not exist.

### What I Changed

- Patched bootstrap and packaging so `.venv` setup and `cwspo` CLI installation work directly.
- Added model-loading helpers for device-aware loading, attention fallback, and PEFT adapter checkpoint loading.
- Updated verifier scoring, pair building, training, and final evaluation to make the minimal real path robust.
- Expanded the unit tests from `1` smoke test to `8` focused tests covering the core research logic.
- Updated `README.md` and `docs/CODEX.md` with verified commands, resume points, expected outputs, and failure fixes.

### What Still Remains Optional

- Swapping the verifier to the stronger offline PRM in `configs/rtx4090_48gb_strong.yaml`.
- Running larger prompt subsets than the bundled 3-example smoke dataset.
- Adding richer confidence features or learned calibration beyond the simple interpretable signals already in place.
- Comparing baseline Step-DPO versus confidence-weighted Step-DPO as a follow-up experiment.

### Exact Next Command

```bash
source .venv/bin/activate && export HF_HOME=$PWD/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl
```

## Real-small completion update

### What changed in this phase

- Added a real process-reward-model verifier backend for `Qwen/Qwen2.5-Math-7B-PRM800K` and fixed the malformed strong-verifier YAML configs.
- Hardened the HF loader for remote PRM code by normalizing missing `pad_token_id`, adding `DynamicCache` compatibility shims, and preserving LoRA adapter loading for final evaluation.
- Added `tests/test_verifier.py` and reran the suite so the stronger verifier path is covered by at least one focused unit test.
- Completed the full real-small matrix:
  - `answer_dpo`
  - `step_dpo`
  - `confidence_filter_only`
  - `confidence_weighted_step_dpo`
  - `confidence_weighted_step_dpo` with the strong offline verifier
- Refreshed the earlier small-verifier runs so every `run_summary.json` uses the same richer process-eval schema.
- Updated `README.md`, `docs/CODEX.md`, `tasks/EXPERIMENT_LOG.md`, and `tasks/RESULTS_SUMMARY.md` with the real-small commands, the PRM backend note, and the honest results critique.

### Files edited in this phase

- `src/cwspo/config.py`
- `src/cwspo/models/hf.py`
- `src/cwspo/models/verifier.py`
- `configs/rtx4090_48gb_real_small_strong_verifier.yaml`
- `configs/rtx4090_48gb_strong.yaml`
- `configs/strong.yaml`
- `tests/test_verifier.py`
- `README.md`
- `docs/CODEX.md`
- `tasks/EXPERIMENT_LOG.md`
- `tasks/RESULTS_SUMMARY.md`
- `tasks/RUN_LOG.md`

### Commands run in this phase

- `source .venv/bin/activate && pytest -q`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python - <<'PY' ... build_verifier(load_config('configs/rtx4090_48gb_real_small_strong_verifier.yaml').verifier, ...) ... PY`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_strong_verifier.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_strong_verifier.yaml --with-train --with-final-eval --with-process-eval --resume-from checkpoint`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_answer_dpo.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python - <<'PY' ... subprocess.run(['python', 'scripts/run_pipeline.py', '--config', cfg, '--with-train', '--with-final-eval', '--with-process-eval', '--resume-from', 'final_eval']) ... PY`

### Runs completed

- Run A: `step_dpo` + small verifier
  - `400` traces, `536` pairs, `68` steps, final accuracy `0.4583`, process exact `0.8958`
- Run B: `confidence_filter_only` + small verifier
  - `400` traces, `458` kept pairs, `58` steps, final accuracy `0.4167`, process exact `0.8958`
- Run C: `confidence_weighted_step_dpo` + small verifier
  - `400` traces, `536` pairs, `68` steps, final accuracy `0.4167`, process exact `0.8958`
- Run D: `confidence_weighted_step_dpo` + strong verifier
  - `400` traces, `488` pairs, `62` steps, final accuracy `0.3333`, process exact `1.0`
- Answer-level DPO baseline
  - `400` traces, `160` pairs, `20` steps, final accuracy `0.5000`, process eval `n/a`

### Reproducibility impact

- Positive overall. The strong verifier is now a real config-driven backend rather than an unvalidated placeholder, and the completed runs are saved under stable per-run directories.
- The PRM path still depends on HF downloads the first time it is used.
- Generation remains sampled, so exact trace text is stochastic even though the prompt subsets and process-eval data are seed-controlled.

### Exact next recommended command

```bash
source .venv/bin/activate && export HF_HOME=$PWD/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_real_small_strong_verifier.yaml --with-train --with-final-eval --with-process-eval
```

## 2026-03-18 - Post-repair full rerun execution

### What changed in this phase

- Created fresh post-repair configs so the rerun matrix would not reuse the earlier refinement output directories:
  - `configs/rtx4090_48gb_postrepair_step_strict.yaml`
  - `configs/rtx4090_48gb_postrepair_conf_filter_strict.yaml`
  - `configs/rtx4090_48gb_postrepair_cw_strict.yaml`
  - `configs/rtx4090_48gb_postrepair_cw_strict_lambda_ref.yaml`
  - `configs/rtx4090_48gb_postrepair_step_semi.yaml`
  - `configs/rtx4090_48gb_postrepair_cw_semi.yaml`
- Reused the shared seed-42 traces and small-verifier scored traces, but rebuilt pairs inside fresh `outputs/post_repair/*` directories under the repaired pair logic.
- Completed the full requested train/final-eval/process-eval matrix with no technical block.
- Saved verbose run logs under `outputs/post_repair/logs/*.log`.

### Commands run

- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_step_strict.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_conf_filter_strict.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_cw_strict.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_cw_strict_lambda_ref.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_step_semi.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_postrepair_cw_semi.yaml --with-train --with-final-eval --with-process-eval`
- `source .venv/bin/activate && python - <<'PY' ... compare post-repair pair JSONLs on training-relevant fields ... PY`

### Runs completed

- `step_strict`: `148` pairs, `20` train steps, `0.5417` final accuracy, `0.2917` process exact, `0.2917` process coverage.
- `conf_filter_strict`: `133` pairs, `18` train steps, `0.4583` final accuracy, `0.2500` process exact, `0.2500` process coverage.
- `cw_strict`: `148` pairs, `20` train steps, `0.4583` final accuracy, `0.2917` process exact, `0.2917` process coverage.
- `cw_strict_lambda_ref`: `148` pairs, `20` train steps, `0.4583` final accuracy, `0.2917` process exact, `0.2917` process coverage.
- `step_semi`: `148` pairs, `20` train steps, `0.4583` final accuracy, `0.2917` process exact, `0.2917` process coverage.
- `cw_semi`: `148` pairs, `20` train steps, `0.4167` final accuracy, `0.2917` process exact, `0.2917` process coverage.

### Key notes

- All six requested reruns completed end-to-end. There was no remaining technical blocker in the repaired pipeline.
- The repaired strict target is smaller than the earlier strict target: `153 -> 148` kept pairs for `step_strict` and `cw_strict`, and `138 -> 133` for `conf_filter_strict`.
- `semi_purified` admitted no same-correctness kept pairs on this slice. Its saved reason histograms are dominated by `dropped_same_correctness_low_confidence`, not by successful same-correctness recovery.
- After projecting the pair JSONLs down to training-relevant fields, `step_strict` and `step_semi` match exactly, and `cw_strict` and `cw_semi` match exactly. The metric gap between strict and semi therefore reflects small-slice run variance rather than different supervision content.

## Phase: Pair purification and orientation refinement

### Current state entering this phase

- The repo was already beyond smoke-test status.
- The real-small matrix had shown that:
  - answer-level DPO was the best held-out final-answer baseline on the checked-in slice,
  - plain Step-DPO beat the original CW-SPO setup,
  - confidence was real as a diagnostic,
  - stronger verifier scores improved pair-quality proxies without improving downstream answer accuracy.

### What the earlier phases proved

- Engineering:
  - end-to-end generation, scoring, pair building, training, and evaluation all worked on the 4090 path;
  - the repo was reproducible enough to run a real-small matrix honestly.
- What they did **not** prove:
  - that the current local pair target was valid,
  - that confidence weighting itself was the right lever,
  - or that stronger verifier scoring solved the downstream training problem.

### Experimental objective for this phase

- test whether pair contamination and pair orientation were the real bottleneck,
- implement explicit pair modes rather than rely on one opaque utility rule,
- add pair-purity and orientation diagnostics,
- rerun the local-pair comparison on the same `100 / 24 / 48` slice,
- and make a decisive recommendation on whether purified local pairs should become the main research path.

### Exact run plan executed

1. Read the existing pair audits, confidence reports, and run summaries.
2. Write `tasks/DIAGNOSIS_AND_REFINEMENT.md`.
3. Implement `current_utility`, `correctness_priority`, `strict_purified`, and `semi_purified`.
4. Add pair taxonomy, purity metrics, orientation audits, and weak-divergence diagnostics.
5. Pair-build all refinement configs.
6. Run the focused Step-DPO matrix:
   - `step_current`
   - `step_correctness`
   - `step_strict`
7. Run the focused CW-SPO matrix:
   - `cw_current`
   - `cw_correctness`
   - `cw_strict`
8. Recheck the stronger verifier on the purified target.
9. Run `conf_filter_strict` as a low-cost tiebreak between hard filtering and soft weighting on the purified target.

### What changed in this phase

- Added pair modes, pair-purity reports, and pair-orientation audits.
- Made divergence filtering more conservative for weak or near-identical local branches.
- Added config support for the new pair-mode and divergence-quality controls.
- Updated the refinement docs and wrote the final diagnosis and results critique.

### Files edited in this phase

- `configs/rtx4090_48gb_refine_step_current.yaml`
- `configs/rtx4090_48gb_refine_step_correctness.yaml`
- `configs/rtx4090_48gb_refine_step_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_current.yaml`
- `configs/rtx4090_48gb_refine_cw_correctness.yaml`
- `configs/rtx4090_48gb_refine_cw_strict.yaml`
- `configs/rtx4090_48gb_refine_conf_filter_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_semi.yaml`
- `configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml`
- `src/cwspo/config.py`
- `src/cwspo/pipeline/build_pairs.py`
- `src/cwspo/pipeline/diagnostics.py`
- `src/cwspo/evaluation/process_eval.py`
- `scripts/run_pairs.py`
- `scripts/run_pipeline.py`
- `tests/test_pairs.py`
- `README.md`
- `docs/CODEX.md`
- `tasks/DIAGNOSIS_AND_REFINEMENT.md`
- `tasks/RESULTS_SUMMARY.md`
- `tasks/EXPERIMENT_LOG.md`
- `tasks/RUN_LOG.md`

### Commands run in this phase

- `source .venv/bin/activate && pytest -q tests/test_pairs.py tests/test_steps.py tests/test_losses.py`
- `source .venv/bin/activate && python -m compileall src scripts tests`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_step_current.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_step_correctness.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_step_strict.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_cw_current.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_cw_correctness.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_cw_strict.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_conf_filter_strict.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_cw_semi.yaml`
- `source .venv/bin/activate && python scripts/run_pairs.py --config configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_current.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_correctness.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_step_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_current.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_correctness.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`
- `source .venv/bin/activate && export HF_HOME=/workspace/CWSPO/cwspo_repo/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_conf_filter_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs`

### Runs completed

- `step_current`: final accuracy `0.4167`
- `step_correctness`: final accuracy `0.4583`
- `step_strict`: final accuracy `0.4583`
- `cw_current`: final accuracy `0.4167`
- `cw_correctness`: final accuracy `0.4167`
- `cw_strict`: final accuracy `0.5000`
- `conf_filter_strict`: final accuracy `0.5000`
- `cw_strict_strongverifier`: final accuracy `0.4583`

### Honest status after this phase

- Engineering success: yes.
- Pair-quality success: yes, strongly.
- Method success: partial but now scientifically alive again.
- Research evidence:
  - pair purification was the missing ingredient,
  - stronger verifier was not,
  - and soft weighting still has not beaten hard filtering on the purified target.

### Exact next recommended command

```bash
source .venv/bin/activate && export HF_HOME=$PWD/.hf_home && python scripts/run_pipeline.py --config configs/rtx4090_48gb_refine_cw_strict.yaml --with-train --with-final-eval --with-process-eval --resume-from pairs
```

## Documentation update

### What changed

- Added `docs/REFINEMENT_TRIAL_REPORT.md`, a standalone report for the latest refinement phase.
- Added a pointer to that report in `README.md`.

### Why

- The refinement logic, equations, configs, failure modes, and results were spread across multiple files.
- This new document is meant to give one clear place to understand the latest trial end-to-end.

### Files touched

- `docs/REFINEMENT_TRIAL_REPORT.md`
- `README.md`
- `tasks/RUN_LOG.md`

### Reproducibility impact

- No code-path change.
- Documentation only.

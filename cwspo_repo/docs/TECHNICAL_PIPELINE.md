# Technical Pipeline Note

This file explains the actual pipeline implemented and run in this repository during the real-small experiment phase.

It is intentionally implementation-facing:
- it follows the code that exists in `src/cwspo/`,
- it explains the equations actually used,
- it distinguishes the main method from the baselines,
- it calls out the known caveats of the current setup.

## 1. Goal and scope

The repo is currently scoped to:
- math reasoning only,
- offline verifier scoring,
- training-time preference optimization,
- earliest-error / local-divergence pair construction,
- confidence-weighted Step-DPO as the main path.

The real-small experiments used:
- a `1.5B` policy model,
- a smaller offline verifier for the main baseline matrix,
- an optional stronger offline verifier based on `Qwen/Qwen2.5-Math-7B-PRM800K`,
- separate generation, scoring, pair-building, training, and evaluation phases.

Relevant configs:
- `configs/rtx4090_48gb_real_small_step_dpo.yaml`
- `configs/rtx4090_48gb_real_small_conf_filter.yaml`
- `configs/rtx4090_48gb_real_small.yaml`
- `configs/rtx4090_48gb_real_small_answer_dpo.yaml`
- `configs/rtx4090_48gb_real_small_strong_verifier.yaml`

## 2. End-to-end artifact flow

The pipeline stages are:

1. Prompt data
2. Trace generation
3. Step splitting
4. Offline verifier scoring
5. Pair construction
6. Confidence computation
7. Baseline-specific pair weighting/filtering
8. DPO training
9. Final-answer evaluation
10. Process evaluation

Primary artifacts:

- `traces.jsonl`: generated reasoning traces
- `scored.jsonl`: traces plus per-step verifier scores and final-answer correctness
- `pairs.jsonl`: local or answer-level preference pairs
- `confidence_analysis.json`: diagnostics over the pair set
- `pair_audit_*.md`: sampled human-readable pair inspections
- `training_report.json`: training metadata and optimizer-step logs
- `final_eval.json`: held-out final-answer evaluation
- `process_eval.json`: process-level boundary evaluation
- `run_summary.json`: consolidated machine-readable run summary

## 3. Notation

For a problem `x`, the generator produces multiple traces:

- `tau_i = [s_{i,0}, s_{i,1}, ..., s_{i,T_i-1}]`

where each `s_{i,t}` is a reasoning step after step splitting.

For each prefix ending at step `t`, the verifier returns a scalar score:

- `q_{i,t} in [0, 1]`

Each trace also has a final-answer correctness label:

- `c_i in {0, 1}`

The local pair builder compares two traces `tau_a` and `tau_b` for the same prompt.

## 4. Data and prompt preparation

Code:
- `scripts/prepare_real_small_data.py`
- `src/cwspo/data/real_small.py`
- `src/cwspo/data/prompts.py`

The real-small setup prepares:
- a seeded GSM8K training slice,
- a seeded GSM8K evaluation slice,
- a seeded synthetic process-eval set with explicit gold earliest-error indices.

The synthetic process-eval set is important:
- it is useful for auditing verifier and pair-builder behavior,
- it is not the same thing as evaluating a newly generated policy trajectory.

## 5. Trace generation

Code:
- `src/cwspo/pipeline/generate.py`

For each prompt, the policy model generates `num_return_sequences` sampled traces.

The generator uses:
- `max_new_tokens`
- `do_sample`
- `temperature`
- `top_p`

The raw generated text is then step-split and a final boxed or numeric answer is extracted.

Formally:

- input: prompt `x`
- output: `N` sampled traces `tau_1, ..., tau_N`

This stage is stochastic by design.

## 6. Step splitting

Code:
- `src/cwspo/utils/steps.py`

The splitter is heuristic:
- first try explicit `Step 1`, `1.`, `2)` style markers,
- then try inline step markers,
- then split on newlines,
- finally fall back to sentence splitting.

This means step boundaries are operational rather than oracle-correct.

That matters because every later stage depends on these step boundaries:
- verifier prefix scoring,
- divergence detection,
- pair localization,
- process evaluation.

## 7. Offline verifier scoring

Code:
- `src/cwspo/pipeline/score.py`
- `src/cwspo/models/verifier.py`

For each trace and each prefix:

- prefix `p_{i,t} = [s_{i,0}, ..., s_{i,t}]`
- score `q_{i,t} = verifier(x, p_{i,t})`

The repository currently supports three verifier modes.

### 7.1 Judge-token verifier

Used for the smaller verifier path.

The verifier prompt is formatted and the model is asked to score two label strings, for example:
- positive label: `good`
- negative label: `bad`

For prompt text `z`, the implementation computes:

- `log P(pos | z)`
- `log P(neg | z)`

and converts them to a probability with a 2-way softmax:

`score(z) = softmax([log P(pos | z), log P(neg | z)])[0]`

So the final verifier score is the model's normalized preference for the positive verdict.

### 7.2 Process reward model verifier

Used for the stronger verifier path with:
- `Qwen/Qwen2.5-Math-7B-PRM800K`

This backend uses the process-reward interface from the model card:
- steps are separated by `<extra_0>`
- the model returns token-level logits at those separators
- the implementation takes the positive-label probability at the last separator

If the prefix contains steps `[u_0, ..., u_k]`, the assistant response becomes:

- `u_0 <extra_0> u_1 <extra_0> ... u_k <extra_0>`

Then:

- apply softmax to the model logits at the separator positions
- keep the final separator
- read the positive class probability

So:

`score(x, p_{i,t}) = P(label = positive at final <extra_0> boundary)`

### 7.3 Mean-logprob verifier

This is a generic fallback verifier:

`score(x, p) = mean token log-probability of the text representation of the prefix`

It is not the main experimental path.

## 8. Earliest-divergence pair construction

Code:
- `src/cwspo/pipeline/build_pairs.py`

This is the core local-pair construction logic.

### 8.1 First divergence

For two traces `tau_a` and `tau_b`, let:

`k = first_divergence(tau_a, tau_b)`

where `k` is the first index where the canonicalized step texts differ.

If one trace ends earlier, the shorter-length boundary is treated as the divergence.

If no divergence exists, the pair is skipped.

### 8.2 Prompt-level score normalization

Within a prompt group, all step scores are z-normalized:

`z_{i,t} = (q_{i,t} - mu) / (sigma + 1e-6)`

where `mu` and `sigma` are computed over all step scores from all traces for that prompt.

This is done to compare traces relative to the prompt-local score scale.

### 8.3 Local segment score

For a divergence at step `k`, the local segment score is:

`R_i(k) = mean(z_{i,u} for u in [k, min(T_i, k + H)))`

where `H = pair.window_H`.

If the segment is empty, the trace is effectively invalid for that local pair.

### 8.4 Local utility

The local utility combines verifier-local evidence and final-answer correctness:

`U_i(k) = alpha_local * R_i(k) + (1 - alpha_local) * c_i`

where:
- `R_i(k)` is the local normalized verifier score,
- `c_i` is final-answer correctness,
- `alpha_local` is the config weight.

This is the current implementation's way of making local preference construction "local divergence oriented" while still using outcome information.

### 8.5 Pair eligibility

For two traces `a` and `b`, define:

`Delta_U = |U_a(k) - U_b(k)|`

If:

`Delta_U < tau_pair`

the pair is discarded as too ambiguous.

Otherwise:
- the higher-utility trace becomes preferred,
- the lower-utility trace becomes dispreferred.

The pair stores:
- the shared prefix `s_{0:k}`,
- the preferred segment `s_{k:k+H}`,
- the dispreferred segment `s_{k:k+H}`,
- metadata such as `k`, final answers, and final correctness.

### 8.6 Prompt-level cap

After all raw local pairs are built for a prompt, they are sorted by confidence and capped at:

- `pair.max_pairs_per_prompt`

## 9. Confidence computation

Code:
- `src/cwspo/pipeline/build_pairs.py`

Confidence is deliberately simple and interpretable in the current implementation.

For a preferred trace `p` and dispreferred trace `d`, the code computes four features.

### 9.1 Margin feature

`margin = |U_p - U_d|`

`f_margin = sigmoid((margin - tau_margin) / scale_margin)`

This increases when the utility gap between branches is large.

### 9.2 Sharpness / drop feature

First define verifier drop at divergence:

`drop_i(k) = max(0, z_{i,k-1} - z_{i,k})`

Then:

`sharp = drop_d(k) - drop_p(k)`

`f_sharp = sigmoid(sharp / scale_drop)`

This rewards pairs where the rejected branch drops more sharply than the preferred branch.

### 9.3 Branch-agreement feature

At divergence step `k`, the code checks how many traces in the same prompt group match each branch signature:

- `n_pref`
- `n_other`

Then:

`f_agree = n_pref / (n_pref + n_other + 1e-6)`

This is a crude branch-consensus signal.

### 9.4 Outcome-support feature

Among traces that match the preferred or alternative branch signature at `k`, compute:

- `p_corr_pref`
- `p_corr_other`

Then:

`f_out = clip((1 + (p_corr_pref - p_corr_other)) / 2, 0, 1)`

This is an interpretable outcome-difference proxy.

### 9.5 Final confidence weight

Let the enabled feature weights be:

- `gamma_margin`
- `gamma_sharp`
- `gamma_agree`
- `gamma_out`

The final confidence is the normalized weighted average:

`w = sum(gamma_j * f_j) / sum(gamma_j)`

clipped to `[0, 1]`.

This `w` is stored both as:
- `pair.confidence`
- and, for the main method, the training weight.

## 10. Baselines implemented

Code:
- `src/cwspo/pipeline/build_pairs.py`

The repo currently supports four training modes.

### 10.1 Answer-level DPO

- pair type: full-trace answer-level
- preferred trace: correct final answer
- dispreferred trace: incorrect final answer
- training weight: `1.0`
- no confidence weighting

This baseline ignores local divergence structure.

### 10.2 Step-DPO

- pair type: local step pair
- same local pair builder as the main method
- training weight: `1.0`

This tests whether local pair construction helps without confidence weighting.

### 10.3 Confidence-filter-only

- pair type: local step pair
- drop pairs with `confidence < threshold`
- training weight of kept pairs: `1.0`

This isolates hard filtering from soft weighting.

### 10.4 Confidence-weighted Step-DPO

- pair type: local step pair
- optional thresholding may still apply
- training weight: `w = confidence`

This is the main method implemented in the repo.

## 11. Training objective

Code:
- `src/cwspo/training/dataset.py`
- `src/cwspo/training/losses.py`
- `src/cwspo/training/train_step_dpo.py`

Each pair is converted into:
- `prefix_ids`
- `pref_ids`
- `disp_ids`
- `weight`

The prefix text is:

- `prompt + "\n\n" + shared_prefix_steps`

The preferred and dispreferred texts are the selected continuation segments only.

### 11.1 Sequence log-probability

For a model `pi`, the code computes the log-probability of a continuation segment `y` given prefix `x`:

`log pi(y | x)`

by concatenating prefix and segment and summing token log-probabilities over the segment tokens only.

### 11.2 DPO delta

For preferred continuation `y+` and rejected continuation `y-`:

`Delta = (log pi(y+|x) - log pi(y-|x)) - (log pi_ref(y+|x) - log pi_ref(y-|x))`

### 11.3 DPO term

`L_dpo = -log sigma(beta * Delta)`

where `beta` is the DPO inverse-temperature parameter from config.

### 11.4 Confidence-weighted loss

For pair weight `w`:

`L = mean(w * L_dpo)`

This is the active objective in the real-small runs because `lambda_ref = 0.0`.

### 11.5 Optional low-confidence trust penalty

The code also supports an optional reference-stability penalty:

`L = mean(w * L_dpo + lambda_ref * (1 - w) * L_ref)`

where `L_ref` is a tokenwise squared deviation from the reference model on both branches.

In the real-small baseline matrix, `lambda_ref` was `0.0`, so this term was inactive.

### 11.6 LoRA training

The policy is fine-tuned through LoRA adapters only.

Training reports record:
- batch size,
- gradient accumulation,
- effective batch size,
- number of optimizer steps,
- mean loss,
- per-step logs,
- adapter output path.

## 12. Final-answer evaluation

Code:
- `src/cwspo/evaluation/final_eval.py`

Final eval is deterministic:
- no sampling,
- held-out eval prompt file,
- adapter-loaded model if a checkpoint exists.

The output records:
- `base_model_name`
- `adapter_path`
- `adapter_loaded`
- per-example predictions
- overall accuracy

This check matters because the repo explicitly verifies that evaluation is not silently using the base model when an adapter is expected.

## 13. Process evaluation

Code:
- `src/cwspo/evaluation/process_eval.py`

There are two conceptual layers here.

### 13.1 What the current process metric actually does

For the real-small experiments, process evaluation is currently an offline fixed-trace boundary-detection diagnostic.

It does not directly evaluate trajectories generated by the newly trained policy.

Instead:
1. load synthetic correct and incorrect step sequences with a known earliest-error index,
2. score those fixed traces with the current verifier,
3. build local pairs with the current method,
4. predict the earliest error as the `k` from the highest-confidence candidate pair,
5. compare predicted boundary to the gold boundary.

So the metric is mainly telling us about:
- verifier behavior,
- local pair-builder behavior,
- confidence ranking behavior.

It is not yet a direct learned-policy process benchmark.

### 13.2 Metrics reported

The evaluator reports:
- exact earliest-error accuracy,
- near-miss accuracy,
- coverage,
- boundary confusion summary,
- row-level examples,
- confidence-bucket summaries,
- a human-readable failure report.

For answer-level DPO, process evaluation is marked not applicable.

## 14. Confidence diagnostics and pair audits

Code:
- `src/cwspo/pipeline/diagnostics.py`

The diagnostics compute:
- mean confidence,
- median confidence,
- histogram,
- low / medium / high bucket counts,
- decisive-pair accuracy,
- confidence/correctness correlation,
- calibration proxy,
- preferred-branch final-correct rate,
- dispreferred-branch final-correct rate,
- both-branches-wrong fraction,
- both-branches-correct fraction.

The pair audit markdown files sample pairs from low / medium / high buckets and print:
- prompt,
- shared prefix,
- preferred segment,
- rejected segment,
- confidence,
- feature values,
- branch correctness metadata.

These audits are essential because the pipeline can produce apparently strong scalar diagnostics while still containing many ambiguous or contaminated local pairs.

## 15. What was actually tried in the real-small experiments

The completed matrix was:

1. Answer-level DPO
2. Step-DPO
3. Confidence-filter-only
4. Confidence-weighted Step-DPO
5. Confidence-weighted Step-DPO with strong verifier

The comparison logic was:
- same `1.5B` policy base model,
- same `100`-prompt train slice,
- same `24`-prompt eval slice,
- shared generated traces wherever possible,
- verifier changed only in the strong-verifier run.

## 16. Current known caveats

The current pipeline is scientifically useful, but it still has important limitations.

### 16.1 Local pair purity is still imperfect

Even high-confidence local pairs can contain:
- both branches wrong,
- both branches correct,
- noisy step boundaries,
- utility disagreements caused by heuristics rather than true local reasoning quality.

### 16.2 Process evaluation is not yet a learned-policy process benchmark

The current `process_eval.json` is an offline boundary-detection diagnostic.

It is useful, but it should not be over-interpreted as direct evidence that the trained adapter improved process reasoning.

### 16.3 Confidence is interpretable but simple

The current confidence mechanism is intentionally simple:
- margin,
- drop sharpness,
- branch support,
- branch outcome support.

That is a good first implementation, but it is not a learned calibration model.

### 16.4 Strong verifier improvements do not automatically mean better training data

In the real-small experiments, the stronger verifier improved several pair-quality proxies but still did not improve downstream final-answer accuracy.

So "better verifier scores" and "better training signal" are not equivalent in the current pipeline.

## 17. Code map

Core files for the pipeline:

- generation: `src/cwspo/pipeline/generate.py`
- scoring: `src/cwspo/pipeline/score.py`
- verifier backends: `src/cwspo/models/verifier.py`
- pair building: `src/cwspo/pipeline/build_pairs.py`
- diagnostics: `src/cwspo/pipeline/diagnostics.py`
- training dataset: `src/cwspo/training/dataset.py`
- training loss: `src/cwspo/training/losses.py`
- training loop: `src/cwspo/training/train_step_dpo.py`
- final eval: `src/cwspo/evaluation/final_eval.py`
- process eval: `src/cwspo/evaluation/process_eval.py`

## 18. Practical reading order

If you want to understand the repo quickly, read in this order:

1. `src/cwspo/pipeline/build_pairs.py`
2. `src/cwspo/models/verifier.py`
3. `src/cwspo/training/losses.py`
4. `src/cwspo/training/train_step_dpo.py`
5. `src/cwspo/pipeline/diagnostics.py`
6. `src/cwspo/evaluation/process_eval.py`

That will give you the real algorithmic path used in the experiments.

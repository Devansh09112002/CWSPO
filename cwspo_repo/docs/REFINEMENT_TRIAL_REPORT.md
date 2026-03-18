# Refinement Trial Report

This file is a single, implementation-facing report for the latest refinement phase.

It is meant to answer:
- what was tried,
- what changed relative to the earlier real-small matrix,
- what logic and equations were used,
- what hyperparameters were used,
- what failed,
- what improved,
- and what the latest trial actually says about the project.

It should be read together with:
- `docs/TECHNICAL_PIPELINE.md`
- `tasks/DIAGNOSIS_AND_REFINEMENT.md`
- `tasks/RESULTS_SUMMARY.md`

## 1. Goal of this refinement phase

The previous real-small matrix established:
- the repo was operational,
- the end-to-end pipeline was beyond smoke-test status,
- answer-level DPO was the strongest held-out final-answer baseline,
- plain Step-DPO beat the original confidence-weighted Step-DPO setup,
- confidence looked real as a diagnostic signal,
- the stronger offline verifier improved pair-quality proxies but not downstream answer accuracy.

So this refinement phase did not try to redesign the project.

It focused on one narrower question:

> Is the main failure mode contaminated local supervision targets rather than confidence weighting itself?

Concretely, the phase tried to fix:
- local pair orientation,
- local pair purity,
- weak or cosmetic divergence boundaries,
- and ambiguity from both-correct / both-wrong local pairs.

## 2. Controlled experiment setup

This phase kept the main experimental scaffold fixed:
- domain: math reasoning only
- policy model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- training style: LoRA + DPO-style training
- supervision style: offline traces, offline verifier, training-time method first
- pair concept: earliest-divergence / local-step preference pairs
- hardware target: RTX 4090 with 48 GB VRAM

The data slice was held fixed for comparability:
- train prompts: `100`
- eval prompts: `24`
- process-eval examples: `48`
- train traces per prompt: `4`
- total shared train traces: `400`

The point was to change only pair construction and, in one follow-up run, the verifier backend.

## 3. Starting diagnosis before refinement

The working diagnosis entering this phase was:

1. `current_utility` was too noisy as an orientation rule.
2. Too many local pairs were not actually instructional.
3. Confidence was measuring something real, but it could not rescue a contaminated target.
4. The stronger verifier was improving pair-quality proxies more than training usefulness.
5. The next step was pair purification, not scale.

The pair audits and confidence reports strongly supported that diagnosis.

In the old `current_utility` local pair pool:
- only about `20.0%` of kept pairs were strictly instructional,
- about `72.6%` were ambiguous,
- about `7.5%` of mixed-correctness pairs were oriented against final correctness,
- many divergences were weak, stylistic, or near-identical.

## 4. What changed in this refinement

### 4.1 New pair modes

Four explicit pair modes were implemented:

- `current_utility`
- `correctness_priority`
- `strict_purified`
- `semi_purified`

The purpose was to isolate whether the bottleneck was:
- wrong local orientation,
- ambiguous same-correctness pairs,
- weak divergence boundaries,
- or confidence weighting itself.

### 4.2 New pair diagnostics

Every run now writes:
- `pair_purity_report.json`
- `pair_orientation_audit.md`
- `confidence_analysis.json`
- `confidence_report.md`
- `pair_audit_low.md`
- `pair_audit_mid.md`
- `pair_audit_high.md`

These report:
- pair taxonomy counts,
- pair purity fractions,
- orientation reasons,
- divergence quality diagnostics,
- and sampled kept/dropped pairs.

### 4.3 Divergence-quality filtering

Conservative local divergence filters were added to suppress obviously bad local targets:
- minimum divergent text length,
- near-identical segment filtering,
- weak-divergence tagging,
- degenerate or empty segment rejection.

This was intentionally conservative. The code does not do semantic rewriting or aggressive paraphrase normalization.

## 5. Core logic and equations used

### 5.1 Step-level verifier scoring

For prompt `x`, trace `i`, and step prefix ending at step `t`, the verifier returns:

`q_{i,t} in [0, 1]`

Scores are normalized within each prompt group:

`z_{i,t} = (q_{i,t} - mu_x) / (sigma_x + epsilon)`

where:
- `mu_x` is the prompt-local mean of all step scores,
- `sigma_x` is the prompt-local standard deviation,
- `epsilon = 1e-6`.

### 5.2 Local segment score

For two traces `a` and `b`, let `k` be the earliest step index where the canonicalized step texts diverge.

The local segment score for trace `i` is:

`R_i(k) = mean(z_{i,t} for t in [k, min(T_i, k + H)))`

where:
- `H = pair.window_H`

This is the prompt-local verifier score averaged over the first few divergent steps.

### 5.3 Original local utility

The original local utility used by `current_utility` is:

`U_i(k) = alpha_local * R_i(k) + (1 - alpha_local) * y_i`

where:
- `y_i in {0, 1}` is final-answer correctness,
- `alpha_local = 0.8` in the refinement configs.

The pair is kept only if:

`|U_a(k) - U_b(k)| >= tau_pair`

with:
- `tau_pair = 0.08`

Under `current_utility`, the preferred branch is simply the branch with larger `U_i(k)`.

### 5.4 Pair-mode decision logic

### Mode A: `current_utility`

Logic:
- build the local pair at earliest divergence,
- compute `U_a(k)` and `U_b(k)`,
- if the utility gap is large enough, prefer the branch with larger utility.

This is the baseline mode and intentionally preserves the old behavior.

### Mode B: `correctness_priority`

Logic:
- if final correctness differs, always prefer the finally correct branch,
- if final correctness is the same, fall back to the `current_utility` rule.

In pseudocode:

```text
if y_a != y_b:
    prefer the branch with y = 1
else:
    use utility orientation
```

This mode tests whether mixed-correctness orientation was the main problem.

### Mode C: `strict_purified`

Logic:
- keep only pairs with `y_a != y_b`,
- require a non-weak local divergence,
- drop both-correct pairs,
- drop both-wrong pairs,
- drop same-correctness pairs completely.

In pseudocode:

```text
if weak_divergence:
    drop
elif y_a == y_b:
    drop
else:
    prefer the branch with y = 1
```

This is the cleanest pair-purity test.

### Mode D: `semi_purified`

Logic:
- always keep mixed-correctness pairs,
- allow same-correctness pairs only when all of the following hold:
  - non-weak divergence,
  - confidence above threshold,
  - utility margin above threshold,
  - local segment gap above threshold.

In the refinement configs the conservative gates are:
- `semi_purified_min_confidence = 0.82`
- `semi_purified_min_utility_margin = 0.35`
- `semi_purified_min_local_gap = 0.35`

This mode was implemented and pair-built, but not fully run in training/eval because the core diagnosis was already clear from the stricter comparison.

### 5.5 Divergence-quality logic

The pair builder tracks:
- `same_prefix_but_weak_divergence`
- `near_identical_divergence`
- `degenerate_divergence`
- `very_short_divergent_region`

The refinement configs used:
- `min_divergent_chars = 24`
- `max_near_identical_similarity = 0.94`

Similarity is a conservative text-level proxy based on canonicalized divergent segments.

This is not a semantic equivalence model. It is only intended to remove obviously non-instructional local differences.

### 5.6 Confidence computation

Confidence is still a simple interpretable weighted combination of bounded features:

`w = (sum_m gamma_m * f_m) / (sum_m gamma_m)`

The features remain:
- margin sharpness
- local drop contrast
- branch agreement / support mass
- empirical outcome advantage

The code writes these per-pair features into the audits, but the exact feature extractor stays intentionally simple in this phase.

Important practical point:
- confidence is computed after the candidate pair is defined,
- so it can help only if the candidate pair is already reasonably meaningful.

That is why purification mattered so much.

### 5.7 Training objectives used

### Plain Step-DPO

All kept local pairs receive weight `1`.

Loss:

`L_step_dpo = E[ log(1 + exp(-beta * Delta)) ]`

### Confidence-filter-only

Pairs below the confidence threshold are dropped.
Remaining pairs receive weight `1`.

In the strict filter refinement run:
- `confidence_threshold = 0.6`

### Confidence-weighted Step-DPO

All kept pairs are used, with soft confidence weight `w`.

Loss:

`L_cw = E[ w * log(1 + exp(-beta * Delta)) ]`

where:

`Delta = (log pi_theta(z+|x) - log pi_theta(z-|x)) - (log pi_ref(z+|x) - log pi_ref(z-|x))`

The loss family itself did not change in this phase. The refinement changed the training distribution more than the objective family.

## 6. Hyperparameters used

These are the main settings used in the refinement runs.

### 6.1 Shared data and generation setup

- seed: `42`
- dtype: `bf16`
- dataset: `openai/gsm8k`
- train prompts: `100`
- eval prompts: `24`
- process examples: `48`
- prompt sampling seed: `42`
- append "step by step" suffix: `true`
- traces per train prompt: `4`
- shared train traces: `400`

### 6.2 Policy model

- model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- attention setting in config: `flash_attention_2`
- actual behavior on this machine: automatic fallback to the default attention implementation when Flash Attention was unavailable

### 6.3 Verifier settings

Small-verifier path:
- mode: `judge_token`
- model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- labels: `good` / `bad`

Strong-verifier path:
- mode: `process_reward_model`
- model: `Qwen/Qwen2.5-Math-7B-PRM800K`
- quantization: `load_in_4bit = true`
- PRM step token: `<extra_0>`
- positive label index: `1`

### 6.4 Pair-building settings

- `window_H = 2`
- `alpha_local = 0.8`
- `tau_pair = 0.08`
- `min_weight = 0.0`
- `max_pairs_per_prompt = 20`
- `min_divergent_chars = 24`
- `max_near_identical_similarity = 0.94`
- `semi_purified_min_confidence = 0.82`
- `semi_purified_min_utility_margin = 0.35`
- `semi_purified_min_local_gap = 0.35`

### 6.5 Confidence thresholds

- low bucket threshold: `0.33`
- high bucket threshold: `0.66`
- hard-filter baseline threshold: `0.6`

### 6.6 Training settings

- batch size: `2`
- gradient accumulation: `8`
- effective batch size: `16`
- epochs: `2`
- learning rate: `2e-5`
- DPO beta: `0.1`
- reference penalty lambda: `0.0`
- max sequence length: `2048`
- log every: `5`
- save every: `100`
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

### 6.7 Eval settings

- eval batch size: `4`
- max new tokens: `256`

## 7. Runs actually completed

The following runs were completed end-to-end on the same slice:

| Run name | Method | Pair mode | Verifier | Pairs | Train steps | Final accuracy | Process exact | Process coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `step_current` | Step-DPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 |
| `step_correctness` | Step-DPO | `correctness_priority` | small | 549 | 70 | 0.4583 | 1.0000 | 1.0000 |
| `step_strict` | Step-DPO | `strict_purified` | small | 153 | 20 | 0.4583 | 0.2917 | 0.2917 |
| `cw_current` | CW-SPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 |
| `cw_correctness` | CW-SPO | `correctness_priority` | small | 549 | 70 | 0.4167 | 1.0000 | 1.0000 |
| `cw_strict` | CW-SPO | `strict_purified` | small | 153 | 20 | 0.5000 | 0.2917 | 0.2917 |
| `conf_filter_strict` | confidence-filter-only | `strict_purified` | small | 138 | 18 | 0.5000 | 0.2500 | 0.2500 |
| `cw_strict_strongverifier` | CW-SPO | `strict_purified` | strong | 153 | 20 | 0.4583 | 0.2917 | 0.2917 |

## 8. Main technical findings

### 8.1 `current_utility` was the real problem

The key failure was not that confidence was meaningless.

The key failure was that the local pair candidate pool was bad:
- too many both-correct pairs,
- too many both-wrong pairs,
- too many same-correctness local competitions,
- and some mixed-correctness pairs oriented the wrong way.

This made the weighted objective learn from a contaminated target distribution.

### 8.2 Orientation repair helped Step-DPO

Changing from `current_utility` to `correctness_priority`:
- improved Step-DPO from `0.4167` to `0.4583`
- removed the wrong-way mixed-correctness orientations

This means orientation semantics really did matter.

### 8.3 Orientation repair alone did not help CW-SPO

Changing from `cw_current` to `cw_correctness`:
- did not improve final accuracy
- both stayed at `0.4167`

Interpretation:
- confidence weighting still could not overcome the large ambiguous same-correctness pool

### 8.4 Purification was the missing ingredient

Changing from `cw_current` to `cw_strict`:
- reduced pairs from `536` to `153`
- dropped about `74.2%` of candidate local comparisons
- improved final accuracy from `0.4167` to `0.5000`

That is the strongest result of the refinement phase.

### 8.5 Stronger verifier was not the next main lever

Changing from `cw_strict` to `cw_strict_strongverifier`:
- kept the same pure taxonomy
- did not improve pair count
- reduced final accuracy from `0.5000` to `0.4583`

So the strong verifier is still useful diagnostically, but it was not the main reason the earlier method underperformed.

### 8.6 Soft weighting still has not beaten hard filtering

On the purified target:
- `cw_strict = 0.5000`
- `conf_filter_strict = 0.5000`

This means the latest trial supports:
- purified confidence-aware local supervision

but does not yet support:
- soft weighting as clearly superior to hard confidence filtering

## 9. Issues and bottlenecks encountered

### 9.1 Weak-divergence process examples

The strict modes caused process coverage to fall sharply.

This happened because many process examples had only weak or near-identical local divergences and were intentionally rejected.

So the current process metric mixes:
- eligible-example boundary quality
- and pair-builder coverage

This is a metric design limitation, not necessarily a policy-quality regression.

### 9.2 Step segmentation remains heuristic

All local supervision depends on `split_steps()`.

If a generation merges multiple logical moves into one step, then:
- earliest divergence may be mislocalized,
- local window scores may mix multiple decisions,
- and pair quality can still degrade.

This was not fully solved in this phase.

### 9.3 Same-correctness local supervision is still unresolved

`correctness_priority` showed that simply fixing mixed-correctness orientation is not enough.

The unresolved question is:
- can some same-correctness local pairs still be useful if admitted very conservatively?

That is what `semi_purified` is for, but it has not yet been fully run.

### 9.4 Strong verifier cost-benefit is still weak

The stronger verifier path is now implemented and working, but on the purified target it did not beat the small verifier.

So it should not be the default next move.

## 10. Files and artifacts to inspect

Key code:
- `src/cwspo/pipeline/build_pairs.py`
- `src/cwspo/pipeline/diagnostics.py`
- `src/cwspo/evaluation/process_eval.py`
- `src/cwspo/models/verifier.py`

Key configs:
- `configs/rtx4090_48gb_refine_step_current.yaml`
- `configs/rtx4090_48gb_refine_step_correctness.yaml`
- `configs/rtx4090_48gb_refine_step_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_current.yaml`
- `configs/rtx4090_48gb_refine_cw_correctness.yaml`
- `configs/rtx4090_48gb_refine_cw_strict.yaml`
- `configs/rtx4090_48gb_refine_conf_filter_strict.yaml`
- `configs/rtx4090_48gb_refine_cw_strict_strongverifier.yaml`

Most useful artifacts:
- `outputs/refinement/cw_strict/pair_purity_report.json`
- `outputs/refinement/cw_strict/pair_orientation_audit.md`
- `outputs/refinement/cw_strict/confidence_report.md`
- `outputs/refinement/cw_strict/run_summary.json`
- `outputs/refinement/conf_filter_strict/run_summary.json`
- `outputs/refinement/cw_strict_strongverifier/run_summary.json`

## 11. Recommended interpretation of the latest trial

The latest trial should be read as:

- engineering success: yes
- pair-quality success: yes
- method success: partial but meaningful
- research claim strength: improved, but still narrower than the original ambition

The most defensible current statement is:

> Confidence-aware step preference optimization is only competitive once the local pair target is purified. Pair purification matters more than stronger verifier scoring, and soft weighting still has not clearly beaten hard filtering on the purified target.

## 12. Recommended next step

The next best experiment is:

1. keep `strict_purified` as the default local pair mode,
2. rerun:
   - `step_strict`
   - `conf_filter_strict`
   - `cw_strict`
   across multiple seeds,
3. split process reporting into:
   - eligible-example exact score
   - total coverage
4. only then test whether `semi_purified` can recover pair count without reintroducing the old contamination.

What should not happen yet:
- no 7B policy scaling
- no larger dataset push
- no learned uncertainty head
- no new objective family before the purified-target comparison is stable across seeds

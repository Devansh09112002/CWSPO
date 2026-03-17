# Results Summary

## Scoreboard

| Method | Dataset Size | Num Traces | Num Pairs | Train Steps | Final Accuracy | Process Earliest-Error Exact | Process Coverage | Mean Confidence | High-Confidence Pair Accuracy | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPO | train=100, eval=24, process=48 | 400 | 160 | 20 | 0.5000 | n/a | n/a | n/a | n/a | answer-level baseline; process eval is not applicable because no local-boundary pairs are produced |
| Step-DPO | train=100, eval=24, process=48 | 400 | 536 | 68 | 0.4583 | 0.8958 | 0.8958 | 0.6928 | 0.9145 | small verifier; local pairs; uniform training weights |
| Confidence-Filter-Only | train=100, eval=24, process=48 | 400 | 458 | 58 | 0.4167 | 0.8958 | 0.8958 | 0.7150 | 0.9145 | small verifier; threshold=0.6; dropped 78 of 536 raw pairs |
| Confidence-Weighted Step-DPO | train=100, eval=24, process=48 | 400 | 536 | 68 | 0.4167 | 0.8958 | 0.8958 | 0.6928 | 0.9145 | small verifier; soft weights; tied hard filtering on final accuracy and below plain Step-DPO |
| Confidence-Weighted Step-DPO + Strong Verifier | train=100, eval=24, process=48 | 400 | 488 | 62 | 0.3333 | 1.0000 | 1.0000 | 0.6862 | 0.9706 | `Qwen/Qwen2.5-Math-7B-PRM800K` PRM backend; pair-quality proxies improved, downstream final accuracy worsened |

## Interpretation Notes

- `High-Confidence Pair Accuracy` here means preferred-branch final correctness among decisive high-confidence pairs.
- `Process Earliest-Error Exact` and `Process Coverage` come from `process_eval.json`, which is currently an offline fixed-trace boundary-detection diagnostic rather than a trained-policy process benchmark.
- The per-run machine-readable summaries live in `outputs/real_small/*/run_summary.json`.

## Confidence Diagnostics

- Small-verifier Step-DPO and CW-SPO had strong confidence/correctness correlation (`0.8411`) and high-confidence decisive-pair accuracy (`0.9145`), but still had heavy contamination: about `41.2%` of all local pairs had both branches wrong and about `31.3%` had both branches correct.
- Hard filtering improved decisive-pair accuracy (`0.8629`) and raised mean confidence (`0.7150`) by removing `78` low-confidence pairs, but the downstream answer accuracy still dropped relative to plain Step-DPO.
- The stronger verifier improved pair-quality proxies further: high-confidence pair accuracy rose to `0.9706`, confidence/correctness correlation rose to `0.8820`, and both-branches-wrong contamination fell to `0.3893`.
- Cleaner pair-quality proxies did not translate to better final-answer performance in this run. The strong-verifier CW-SPO run had the weakest held-out answer accuracy (`0.3333`) in the matrix.

## Honest Critique

- Engineering success: yes. This repo is beyond smoke-test status. It now runs a real-small seeded GSM8K slice, writes auditable diagnostics, supports resume, supports four baselines plus a strong-verifier comparison, and completed every requested run honestly on the actual 4090 path.
- Method evidence: mixed to negative on this slice. The experiments are real, but they do not currently support the claim that confidence-weighted Step-DPO is the best training path.

### Q1. Does confidence-weighted Step-DPO beat plain Step-DPO on a non-trivial math subset?

- No on held-out final accuracy in this real-small run.
- Step-DPO: `0.4583`
- Confidence-weighted Step-DPO: `0.4167`

### Q2. Do confidence scores correlate with step-pair reliability?

- Yes, partially.
- High-confidence decisive pairs are much cleaner than the full pair pool, and the confidence/correctness correlations are high (`0.68` to `0.88` depending on run).
- But confidence is not solving the core data-quality problem by itself because many supposedly useful local pairs are still ambiguous or contaminated.

### Q3. Does a stronger offline verifier improve pair quality and downstream performance?

- Pair-quality proxies: yes.
- Downstream final-answer performance: no in this run.
- The strong verifier found a cleaner-looking pair set (`488` pairs instead of `536`, `0.9706` high-confidence pair accuracy), but the resulting CW-SPO model fell to `0.3333` final accuracy.

### Q4. Is soft weighting better than hard filtering?

- No evidence of that here.
- Hard filtering and soft weighting tied on final accuracy (`0.4167` vs `0.4167`) and both were worse than plain Step-DPO.
- Soft weighting preserved more pairs, but that did not produce a measurable gain on this slice.

### Q5. Is the process metric improving in a way that supports the paper hypothesis?

- Only partially, and with an important caveat.
- The strong verifier raised the boundary-detection metric from `0.8958` to `1.0`, which says the offline verifier is very good at locating the earliest error on the fixed process examples.
- But this process metric does not depend on the trained policy adapter, and on the strong-verifier process set it still preferred the wrong branch while selecting the right boundary. So the metric is useful for boundary detection, but it is not yet strong evidence that training improved reasoning behavior.

## Bottom Line

- The repo is now a real experiment system, not just a smoke test.
- Confidence appears meaningful as a reliability signal, but not yet sufficient as a training signal.
- The pair data is plausible enough to study, but still too noisy and ambiguous to trust blindly.
- The strongest baseline on held-out final accuracy in this matrix is currently answer-level DPO (`0.5000`), followed by plain Step-DPO (`0.4583`).
- Confidence-weighted Step-DPO is not winning yet.
- The stronger verifier helps pair cleanliness more than it helps downstream training.

## Biggest Remaining Bottleneck

- Pair orientation and pair purity, especially for local pairs.
- The current utility/confidence recipe still allows too many both-wrong and both-correct pairs, and the strong verifier can improve boundary detection without reliably improving local preference direction for training.

## Pair Refinement Matrix

| Method | Pair Mode | Verifier | Num Pairs | Train Steps | Final Accuracy | Process Earliest-Error Exact | Process Coverage | Mean Confidence | High-Confidence Pair Accuracy | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step-DPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 | 0.6928 | 0.9145 | `20.0%` strictly instructional, `72.6%` ambiguous, `7.5%` mixed-correctness misoriented |
| Step-DPO | `correctness_priority` | small | 549 | 70 | 0.4583 | 1.0000 | 1.0000 | 0.6992 | 1.0000 | fixed wrong-way mixed-correctness orientation, but still kept `389` same-correctness utility-oriented pairs |
| Step-DPO | `strict_purified` | small | 153 | 20 | 0.4583 | 0.2917 | 0.2917 | 0.7577 | 1.0000 | pure `correct_vs_incorrect` set; dropped `74.2%` of candidates |
| Confidence-Weighted Step-DPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 | 0.6928 | 0.9145 | same noisy target as Step-DPO current baseline |
| Confidence-Weighted Step-DPO | `correctness_priority` | small | 549 | 70 | 0.4167 | 1.0000 | 1.0000 | 0.6992 | 1.0000 | orientation fix alone did not improve CW-SPO |
| Confidence-Weighted Step-DPO | `strict_purified` | small | 153 | 20 | 0.5000 | 0.2917 | 0.2917 | 0.7577 | 1.0000 | best local-pair CW-SPO run; matched answer-level DPO on this slice |
| Confidence-Filter-Only | `strict_purified` | small | 138 | 18 | 0.5000 | 0.2500 | 0.2500 | 0.7774 | 1.0000 | hard filtering tied purified CW-SPO after dropping `15` low-confidence strict pairs |
| Confidence-Weighted Step-DPO | `strict_purified` | strong | 153 | 20 | 0.4583 | 0.2917 | 0.2917 | 0.7563 | 1.0000 | same pure taxonomy, but stronger verifier did not improve held-out accuracy |

## Was Pair Purification The Missing Ingredient?

- Yes for pair quality. `current_utility` kept a heavily contaminated set with `168` both-correct, `221` both-wrong, and `40` mixed-correctness pairs oriented against final correctness.
- Yes for Step-DPO. Moving from `current_utility` to either `correctness_priority` or `strict_purified` improved final accuracy from `0.4167` to `0.4583`.
- Yes, even more clearly for CW-SPO. `cw_current` and `cw_correctness` both stayed at `0.4167`, but `cw_strict` rose to `0.5000`.
- Not yet for soft weighting over hard filtering. `conf_filter_strict` tied `cw_strict` at `0.5000`, so confidence-aware selection helps on the purified set, but the advantage of soft weighting over simple filtering remains unproven.
- No for the stronger verifier as the next main lever. `cw_strict_strongverifier` fell back to `0.4583`, so purification helped more than rescoring.
- Only cautiously for the process metric. Strict modes cut process coverage to about `0.25-0.29` because the evaluator could only score `12-14` eligible strict pairs on the `48`-example process set. That is a coverage artifact, not direct evidence that the trained policy got worse.

## Honest Updated Verdict

- Engineering success: yes. The repo now supports auditable pair-mode refinement runs with purity reports, orientation audits, and resumable outputs under `outputs/refinement/*`.
- Pair-quality success: yes. The refinement phase directly validated the contamination diagnosis and produced a mathematically cleaner target.
- Method success: partial but real. Purified local pairs improved the main CW-SPO path from `0.4167` to `0.5000`, but hard filtering tied soft weighting on the purified target.
- Research evidence: promising enough to continue, but not enough to claim that confidence-weighted Step-DPO itself is already the uniquely best mechanism.

## Updated Next Experiment

- Keep the `1.5B` policy fixed.
- Treat `strict_purified` as the default local-pair path.
- Re-run `step_strict`, `conf_filter_strict`, and `cw_strict` across multiple seeds on the same `100 / 24 / 48` slice.
- Report process exact on eligible examples separately from overall coverage.
- Defer `semi_purified`, larger datasets, and any 7B-policy scaling until the purified-target comparison is stable.

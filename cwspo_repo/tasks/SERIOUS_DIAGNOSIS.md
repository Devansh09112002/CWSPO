# Serious Diagnosis: CW-SPO Failure And Repair

This document replaces the earlier vague story with the stricter claim that is actually supported by the repository evidence:

> the primary bottleneck was target quality, not the existence of confidence itself.

The checked-in real-small and refinement runs support that claim much more strongly than the older claim that soft confidence weighting alone should win.

## Scope

This phase stays within the original project boundary:
- math reasoning only
- offline verifier / PRM scoring only
- training-time preference optimization only
- `Qwen/Qwen2.5-Math-1.5B-Instruct` as the primary policy
- no learned uncertainty head
- no 7B-policy scaling yet

## Notation

For prompt `x`, the generator samples traces `tau_i = [s_{i,0}, ..., s_{i,T_i-1}]`.

For each step prefix, the frozen verifier returns:

`q_{i,t} in [0, 1]`

The code normalizes scores within each prompt group:

`z_{i,t} = (q_{i,t} - mu_x) / (sigma_x + 1e-6)`

For a pair of traces `a, b`, let `k` be the first divergent step index. The local segment score is:

`R_i(k) = mean(z_{i,t} for t in [k, min(T_i, k + H)))`

The original local utility was:

`U_i(k) = alpha_local * R_i(k) + (1 - alpha_local) * y_i`

where `y_i in {0, 1}` is final-answer correctness.

The current weighted DPO objective is:

`L = E[w * L_dpo]`

with optional trust penalty:

`L = E[w * L_dpo + lambda_ref * (1 - w) * L_ref]`

where `L_ref` is a reference-closeness penalty on token log-probs.

The process metric is an offline fixed-trace boundary diagnostic:

`M_process = earliest_error_exact over a fixed labeled process set`

It is useful for auditing verifier and pair-builder behavior, but it is not a direct learned-policy metric.

## Core Empirical Facts

From the original real-small matrix:
- answer-level DPO reached `0.5000` final accuracy
- Step-DPO reached `0.4583`
- confidence-filter-only and confidence-weighted Step-DPO both reached `0.4167`
- the stronger verifier improved pair-quality proxies while the downstream answer metric worsened

From the refinement phase:
- `current_utility` local pairs were only `20.0%` strictly instructional
- `72.6%` of kept local pairs were ambiguous same-correctness pairs
- `7.5%` of mixed-correctness pairs were oriented against final correctness
- `correctness_priority` repaired mixed-correctness orientation and improved Step-DPO to `0.4583`
- `strict_purified` dropped `74.2%` of candidates but produced a fully mixed-correctness, correctness-oriented pair set
- `cw_strict` reached `0.5000`, tying answer-level DPO
- `conf_filter_strict` also reached `0.5000`, so soft weighting still has not clearly beaten hard filtering

Those results imply that purification changed the training story more than confidence reweighting did.

## Failure Decomposition

### 1. Target-quality problems

This is the main failure class.

The old objective asked one scalar `U_i(k)` to simultaneously answer:
- is the local continuation plausible?
- is the full trace finally correct?
- is the divergence instructionally meaningful?

Those are different questions. Compressing them into one number causes two predictable mistakes:
- finally correct traces can win local comparisons even when the local continuation after `k` is not the cleaner move
- finally wrong traces can still win if local verifier evidence is noisy enough

That is why `current_utility` admitted:
- both-correct pairs that were often just stylistic alternatives
- both-wrong pairs that were often arbitrary bad-vs-bad contests
- mixed-correctness pairs with the wrong orientation

DPO-style training is strongest when preferred and rejected examples express a meaningful behavioral contrast. A contaminated pair pool violates that assumption at the data level before optimization even begins.

### 2. Confidence-use problems

Confidence was not useless. It was just too downstream.

The repo already showed that confidence correlated with pair cleanliness:
- high-confidence decisive pairs were much better than the full pool
- confidence/correctness correlations were strong

But weighting happens after candidate-pair definition. If the pair itself is non-instructional, then:

`small gradient on bad pair` is still `bad supervision`

That is why the right sequence is:
1. pair admissibility
2. then gradient weighting

This is also why `cw_strict` and `conf_filter_strict` tied: once the admitted target is already very clean, the remaining question is not whether confidence exists, but what mathematically distinctive job it should do beyond filtering.

### 3. Verifier problems

The stronger verifier improved the wrong layer of the stack.

It improved:
- boundary-detection exactness
- high-confidence pair accuracy
- pair-quality proxy cleanliness

But the downstream policy metric did not improve accordingly.

This means verifier improvement alone does not solve the scientific problem. The missing piece was not just better scores, but better semantics for when a local pair should exist at all.

### 4. Evaluation problems

The main downstream metric should remain final-answer accuracy.

The process metric is still valuable, but only as:
- verifier audit
- pair-builder audit
- boundary-localization audit

It is not a trained-policy benchmark, because the current process evaluation runs on fixed labeled traces and does not depend on policy adaptation. Treating it as direct evidence of reasoning improvement was a category error.

### 5. Optimization and training problems

Optimization is not the leading failure mode, but there are two real training issues:

First, when the pair set is small and highly filtered, training variance increases and multi-seed evaluation becomes necessary.

Second, the current use of `lambda_ref` was mostly dormant in the real-small runs because `lambda_ref = 0.0`. That left the method with only two confidence behaviors:
- drop low-confidence pairs
- shrink their DPO weight

The next objective question is therefore:

> on already-purified targets, should low-confidence examples merely count less, or should they also stay closer to the reference model?

## Why `current_utility` was mathematically flawed

`U_i(k)` was attractive because it mixed local and final signals, but it introduced a hidden assumption:

> local preference can be represented as a convex interpolation of short-horizon verifier score and final correctness.

That assumption is too strong.

Local preference and final correctness are correlated but not equivalent. A scalar merge loses structure that matters:
- the divergence might be real or fake
- the local branch might be semantically meaningful or just formatting
- the traces may share final correctness but differ in pedagogical value

The refinement matrix showed that once final correctness was allowed to dominate orientation only where it should, and same-correctness pairs were removed or screened, downstream behavior improved immediately.

## Why `strict_purified` helped

`strict_purified` did three important things at once:
- removed same-correctness ambiguity
- removed wrong-way mixed-correctness orientation
- rejected weak local boundaries

That made the pair set smaller, but far more faithful to the intended supervision type:

`correct local move leading away from incorrect local move`

The fact that `cw_strict` rose to `0.5000` while `cw_current` stayed at `0.4167` is the clearest evidence that target repair, not scoring complexity, was the decisive intervention.

## Why `semi_purified` is the real frontier now

`strict_purified` is the trusted baseline, but it is intentionally high-recall-negative:
- it throws away most candidate pairs
- it refuses to express any same-correctness supervision

That is scientifically safe, but incomplete.

The unresolved question is not whether all same-correctness pairs are useful. They are not.

The unresolved question is:

> when does a same-correctness pair become instructionally admissible?

The two meaningful subcases are different:

### Both-correct pairs

These are admissible only if there is strong evidence that one branch is locally cleaner:
- the divergence is semantically meaningful
- the local score gap is large enough
- confidence is high enough
- branch-support evidence suggests the preferred local move is more reliable

Otherwise they are just stylistic variation and should be dropped as ambiguous.

### Both-wrong pairs

These should default to rejection.

They are admissible only under a much stricter claim:

> one branch delays or softens the error enough to be instructionally better than the other

That requires stronger evidence than the both-correct case. In the repaired builder, both-wrong pairs need not only confidence and local gap, but also evidence consistent with delayed collapse rather than arbitrary bad-vs-bad ordering.

## Implementation Implications

The repaired repository should therefore do the following:

1. keep `strict_purified` as the trusted default local target
2. make `semi_purified` reason-coded and pattern-specific
3. use confidence in two stages:
   - stage A: same-correctness admissibility
   - stage B: training weight after admission
4. expose explicit reason codes for every kept and dropped pair
5. write per-run target-quality summaries, not just loss and accuracy summaries

## Recommended Next Matrix

Priority order:
1. `step_strict_multi_seed`
2. `conf_filter_strict_multi_seed`
3. `cw_strict_multi_seed`
4. `cw_strict_lambda_ref`
5. `step_semi`
6. `cw_semi`
7. optional `cw_semi_lambda_ref`

The scientific logic is:
- stabilize the purified baseline first
- then probe the objective with `lambda_ref`
- then reopen same-correctness supervision under improved `semi_purified`

## Bottom Line

The original hypothesis was too specific in the wrong place.

The repository evidence does not currently justify:
- “confidence weighting alone is the main win”

The repository evidence does justify:
- “pair contamination and pair admissibility were the main bottleneck”
- “purified local preference learning is competitive once the target is repaired”
- “same-correctness supervision is now the main unresolved frontier”

## Post-repair full rerun addendum

The full rerun matrix changes the practical conclusion in an important way.

### What the repaired execution phase actually showed

- The repaired strict target is slightly smaller than the earlier strict target:
  - `step_strict`: `153 -> 148`
  - `cw_strict`: `153 -> 148`
  - `conf_filter_strict`: `138 -> 133`
- Those missing pairs were not arbitrary. They are explained by the repaired divergence and admissibility logic:
  - `dropped_unstable_boundary = 40`
  - `dropped_trivial_segment_difference = 23`
  - `dropped_near_identical = 28`
  - plus the same strict same-correctness rejections as before.

### Downstream outcome after the repair

- `step_strict` is now the best repaired local-pair run at `0.5417` final accuracy.
- `step_strict` therefore beats:
  - answer-level DPO at `0.5000`
  - the earlier strict Step-DPO rerun at `0.4583`
- The confidence-aware runs did not improve correspondingly:
  - `conf_filter_strict = 0.4583`
  - `cw_strict = 0.4583`
  - `cw_strict_lambda_ref = 0.4583`
  - `cw_semi = 0.4167`

### What happened to `semi_purified`

- On this checked-in slice, repaired `semi_purified` still admitted no same-correctness kept pairs.
- The saved run summaries show:
  - `both_correct = 0`
  - `both_wrong = 0`
  - `same_correctness_fraction = 0.0`
- The semi builder is therefore behaving honestly and conservatively, but it is not yet recovering extra supervision on this slice.
- Its dominant drop reasons are:
  - `dropped_same_correctness_low_confidence = 326`
  - `dropped_utility_margin_below_tau_pair = 30`
  - `dropped_both_wrong_uninformative = 2`

### Important variance caveat

- I compared the post-repair pair JSONLs after projecting onto the actual training-relevant fields.
- `step_strict` and `step_semi` are identical as training examples.
- `cw_strict` and `cw_semi` are also identical as weighted training examples.
- Yet the final accuracies differ slightly across those strict/semi pairs of runs.
- That means the remaining single-seed metric differences are not evidence that semi is better or worse on different supervision content.
- They are evidence that this tiny purified setup is still variance-sensitive enough that repeated runs or multiple seeds are required for strong claims.

### Updated scientific conclusion

- The repair was still necessary.
- It improved the best local-pair training path.
- But the repaired matrix does not support the stronger claim that confidence weighting is the mechanism currently delivering the win.
- The strongest supported claim is now narrower:

> on this slice, a repaired purified target helps, and plain Step-DPO on that target is currently the best-performing local-pair training path.

### Updated next matrix

1. repeat `step_strict` with multiple seeds or repeated trainings on the same repaired pair file
2. repeat `cw_strict` and `conf_filter_strict` the same way
3. keep `cw_strict_lambda_ref` in the matrix, but only as a repeated-run objective probe
4. do not spend more budget on `semi_purified` on this slice unless it actually begins admitting same-correctness pairs

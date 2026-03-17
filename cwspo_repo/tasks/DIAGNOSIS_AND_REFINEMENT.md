# Diagnosis And Refinement

## Current Diagnosis

This phase starts from an uncomfortable but useful result: the repository is now an engineering success, but the current confidence-weighted local-pair method is not yet a research success.

What the completed real-small runs established:

- The end-to-end pipeline is real, reproducible, and no longer a smoke test.
- Answer-level DPO is currently the strongest held-out final-answer baseline on the real-small slice.
- Plain Step-DPO beats the current confidence-weighted Step-DPO variant.
- Confidence is measuring something real as a diagnostic.
- The stronger verifier improves pair-quality proxies and offline boundary detection.
- The stronger verifier does not improve downstream held-out final-answer accuracy in the current setup.

The working hypothesis for this refinement phase is therefore:

1. The main bottleneck is contaminated local supervision targets.
2. The current local orientation rule is too willing to trust noisy utility differences.
3. Confidence helps rank pair cleanliness, but it cannot rescue a systematically mis-oriented or semantically weak target.
4. Stronger verifier scores help detect process differences, but their benefit is diluted if the pair set itself is still dominated by both-wrong, both-correct, or cosmetically divergent examples.

## What The Current Method Actually Optimizes

For a prompt \(x\), each sampled trace \(i\) is split into steps \((z_{i,0}, z_{i,1}, \dots)\). The verifier scores each growing prefix, giving step scores \(s_{i,t}\).

The current local pair builder normalizes scores within a prompt:

\[
\tilde{s}_{i,t} = \frac{s_{i,t} - \mu_x}{\sigma_x + \epsilon}
\]

where \(\mu_x\) and \(\sigma_x\) are computed over all step scores from all traces for the same prompt.

At the first divergence index \(k\) between two traces \(a\) and \(b\), the builder computes a local segment score:

\[
R_i(k) = \frac{1}{|S_i(k)|} \sum_{t \in S_i(k)} \tilde{s}_{i,t},
\quad
S_i(k) = \{k, k+1, \dots, \min(T_i, k+H-1)\}
\]

The current scalar utility is:

\[
U_i(k) = \alpha_{\text{local}} R_i(k) + (1-\alpha_{\text{local}}) y_i
\]

where \(y_i \in \{0,1\}\) is final correctness.

The pair is kept only if:

\[
|U_a(k) - U_b(k)| \ge \tau_{\text{pair}}
\]

and the preferred branch is whichever has the larger utility.

The current confidence score is a weighted combination of bounded heuristics:

\[
w = \frac{\sum_m \gamma_m f_m}{\sum_m \gamma_m}
\]

with the current features:

- \(f_{\text{margin}}\): utility-margin sharpness
- \(f_{\text{sharp}}\): drop contrast around the divergence step
- \(f_{\text{agree}}\): support mass on the preferred branch at the same step
- \(f_{\text{out}}\): empirical outcome advantage of the preferred branch family

Training then uses weighted Step-DPO:

\[
\mathcal{L}_{\text{wStepDPO}}
=
\mathbb{E}_{(x, z^+, z^-)}
\left[
w(x,z^+,z^-)\;
\log\left(1 + \exp\left(-\beta \Delta\right)\right)
\right]
\]

where

\[
\Delta =
\left(
\log \pi_\theta(z^+|x) - \log \pi_\theta(z^-|x)
\right)
-
\left(
\log \pi_{\text{ref}}(z^+|x) - \log \pi_{\text{ref}}(z^-|x)
\right).
\]

In the confidence-filter-only baseline, low-confidence pairs are dropped and the remaining pairs receive weight \(1\). In plain Step-DPO, every local pair receives weight \(1\).

## Why The Current Utility Rule May Be Flawed

The current utility rule makes a strong assumption:

> a short local verifier advantage, mixed with final correctness, is enough to define a meaningful training preference at the first lexical divergence.

That assumption is often false in the current data.

### Failure 1: lexical divergence is not the same as semantic divergence

The current builder uses first canonical step mismatch as the boundary. In the audits, many high-confidence pairs diverge at \(k=0\) only because one trace says "Let's break this down" and the other says "We need to follow these steps". Those are not different reasoning moves in a useful sense, but the builder still treats them as local preferences.

This breaks the implicit assumption that:

\[
z^+_{k:k+H} \succ z^-_{k:k+H}
\]

means "the preferred local continuation is more mathematically correct."

In many current pairs it only means "the wording or formatting differed first."

### Failure 2: the mixed utility obscures orientation semantics

The current utility

\[
U_i(k) = \alpha R_i(k) + (1-\alpha)y_i
\]

combines a local verifier score and final correctness into a single scalar. That makes orientation convenient, but it erases the distinction between:

- correctness-driven orientation
- utility-only tie-breaking
- semantically ambiguous local superiority

When \(y_a = y_b\), the orientation is effectively driven by local verifier noise. When both branches are wrong, a higher \(U_i(k)\) still creates a preferred branch, even though neither branch may be instructional. When both are correct, the pair may not teach a meaningful correction at all.

### Failure 3: confidence is conditioned on the same noisy candidate set

The current confidence features are useful as ranking signals, but they are applied after the pair candidate has already been defined by the current boundary and utility assumptions.

If the underlying pair is bad, high confidence only says:

> the system is confident about a possibly irrelevant local distinction.

That is consistent with the observed audits:

- high-confidence buckets have good preferred-branch correctness among decisive pairs,
- but still contain a large both-wrong fraction,
- and still include many non-instructional \(k=0\) formatting divergences.

### Failure 4: stronger verifier can improve proxies without improving the target

The stronger offline verifier improves boundary-related proxies and high-confidence pair cleanliness. That is compatible with the observed rise in offline process detection quality.

But if the pair set still includes:

- both-wrong local competitions,
- both-correct style competitions,
- weak or cosmetic divergences,
- and same-correctness pairs oriented by noisy local utility,

then better scoring alone does not guarantee a better learning signal. The verifier becomes better at measuring differences inside a contaminated candidate pool.

## Likely Harmful Pair Types

The current audits suggest the following pair types are most harmful:

### Both wrong

Both branches end in incorrect answers. The current method may still orient them via local utility. That can train the policy toward a "less bad" branch without evidence that the local continuation is truly corrective or transferable.

### Both correct

Both branches end correctly. If the divergence is stylistic, the pair teaches formatting preference rather than error recovery. If the divergence is substantive, the local preference may still be underspecified without an explicit objective for efficiency or proof quality.

### Same-correctness pairs with weak divergence

These are especially dangerous because the model receives a directional training signal where the two local continuations are nearly equivalent or cosmetically different.

### Incorrectly oriented mixed-correctness pairs

If one branch is finally correct and the other is finally incorrect, the preferred branch should almost always be the correct one unless there is strong evidence that the local segment itself is not the error-bearing difference. The current scalar utility does not enforce that.

## Current Assumptions That May Be Invalid

The current implementation implicitly assumes:

1. First lexical divergence approximates earliest semantic divergence.
2. A short verifier window around the boundary is stable enough to define a local preference.
3. Same-correctness pairs remain useful under a utility-based orientation rule.
4. Confidence weighting can downweight contamination enough to preserve the benefit of the whole pair set.

The audits and results suggest all four assumptions are at least partially invalid on the current slice.

## Specific Technical Critiques

### Local utility definition

The current scalar mixture makes final correctness and local score compete inside one number. That is mathematically tidy but diagnostically opaque. It prevents us from asking the more important scientific question:

> when final correctness disagrees with local utility, which source of supervision should dominate orientation?

The refinement phase should make that choice explicit rather than implicit.

### Tie-breaking logic

Current ties are broken by a global threshold on \(|U_a-U_b|\). This is not enough because:

- a same-correctness pair with a moderate utility gap may still be semantically meaningless,
- a mixed-correctness pair with a small utility gap may still be highly instructional,
- and the threshold does not account for divergence quality.

### Divergence detection

`first_divergence()` is faithful to the literal step strings, but it is not robust to generic openers, equivalent paraphrases, or formatting-only changes. The method is earliest-divergence oriented, which is still correct for the project, but the current boundary detector is too permissive about what counts as a meaningful local branch.

### Step segmentation assumptions

The pipeline depends heavily on `split_steps()` producing semantically aligned steps. When the model emits long paragraphs or merged steps, the first divergence is less meaningful and the local segment window can mix multiple logical moves.

### Correctness use in pair orientation

Final correctness is currently only one term in the scalar utility. That is too weak for the case that matters most:

\[
y_a \neq y_b
\]

If one branch solves the problem and the other fails, final correctness should usually take precedence in the orientation rule.

### Both-correct / both-wrong handling

The current method treats these as ordinary candidate pairs if the utility gap is large enough. That is likely the biggest source of contamination.

### Confidence feature interactions

The confidence features are interpretable, which is good. But because they mix margin, branch support, and outcome advantage, they can become high even when the pair is pedagogically weak. In particular, strong support and margin do not imply the divergent segment is instructional.

## Refined Pair-Construction Modes To Test

The point of this phase is to test whether target quality is the bottleneck, not to speculate abstractly. The implementation therefore introduces explicit pair modes.

### Mode A: `current_utility`

This is the baseline. It preserves the current local utility rule:

\[
\text{prefer } a \iff U_a(k) \ge U_b(k)
\]

with the same thresholding logic.

This mode exists to keep a true comparison point.

### Mode B: `correctness_priority`

This mode makes correctness semantics explicit.

If:

\[
y_a \neq y_b
\]

then the correct branch is preferred. Local utility is used only when:

\[
y_a = y_b
\]

This isolates whether the main problem is orientation noise on mixed-correctness pairs.

### Mode C: `strict_purified`

This mode keeps only strictly instructional mixed-correctness pairs:

\[
y_a \neq y_b
\]

and drops:

- both-correct pairs
- both-wrong pairs

This is the cleanest test of the pair-purity hypothesis.

### Mode D: `semi_purified`

This mode always keeps mixed-correctness pairs and allows same-correctness pairs only when a conservative local criterion is met. The criterion should require:

- non-weak divergence,
- strong utility margin,
- strong local score gap,
- and high confidence.

This tests whether a limited amount of same-correctness local supervision is still helpful when tightly filtered.

## Current And Refined Objectives

### Current weighted Step-DPO

\[
\mathcal{L}_{\text{current}}
=
\mathbb{E}[w \cdot \ell_{\text{DPO}}]
\]

with \(w\) defined on the current candidate set.

### Essential refinement now: purified-pair weighted Step-DPO

Keep the same weighted Step-DPO loss family, but redefine the training distribution by improving pair construction:

\[
\mathcal{L}_{\text{purified}}
=
\mathbb{E}_{(x,z^+,z^-) \sim \mathcal{D}_{\text{purified}}}
[w \cdot \ell_{\text{DPO}}]
\]

where \(\mathcal{D}_{\text{purified}}\) removes non-instructional or weakly justified local pairs.

This is the essential change for this phase.

### Optional later: low-confidence trust penalty

A conservative future extension is to keep the same purified pair set and add a small low-confidence trust penalty or confidence-aware regularization. That is optional later, not essential now.

This is premature until we know whether purified targets alone already help.

## Main Experimental Questions For This Phase

1. If we change only orientation, does Step-DPO improve?
2. If we purify the pair set, does Step-DPO improve?
3. If we purify the pair set, does confidence-weighted Step-DPO become competitive?
4. Does the stronger verifier become useful only after purification?

## What Would Count As Success

Engineering success in this phase:

- new pair modes are implemented cleanly,
- pair taxonomy and purity diagnostics are written,
- orientation audit files are inspectable,
- the focused refinement matrix completes.

Pair-quality success:

- strict purification materially reduces both-wrong and both-correct fractions,
- orientation reasons become more correctness-driven and less utility-only,
- weak divergence rates drop in refined modes.

Method success:

- purified Step-DPO improves over current Step-DPO,
- and ideally purified CW-SPO improves over current CW-SPO.

Research evidence:

- if purified CW-SPO becomes competitive or better, confidence weighting remains a promising main direction;
- if purified Step-DPO improves but CW-SPO still does not, then confidence weighting is not yet justified as the main benefit.

## Refinement Matrix Results

| Run | Method | Pair mode | Verifier | Pairs | Train steps | Final accuracy | Process exact | Process coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Step-DPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 |
| 2 | Step-DPO | `correctness_priority` | small | 549 | 70 | 0.4583 | 1.0000 | 1.0000 |
| 3 | Step-DPO | `strict_purified` | small | 153 | 20 | 0.4583 | 0.2917 | 0.2917 |
| 4 | CW-SPO | `current_utility` | small | 536 | 68 | 0.4167 | 0.8958 | 0.8958 |
| 5 | CW-SPO | `correctness_priority` | small | 549 | 70 | 0.4167 | 1.0000 | 1.0000 |
| 6 | CW-SPO | `strict_purified` | small | 153 | 20 | 0.5000 | 0.2917 | 0.2917 |
| 7 | confidence-filter-only | `strict_purified` | small | 138 | 18 | 0.5000 | 0.2500 | 0.2500 |
| 9 | CW-SPO | `strict_purified` | strong | 153 | 20 | 0.4583 | 0.2917 | 0.2917 |

`semi_purified` was implemented and pair-built, but full train/eval was intentionally deferred after the core matrix plus the strict hard-filter tiebreak had already answered the highest-value questions.

## What The Results Mean

### 1. The current local target was genuinely contaminated

On the real-small slice, `current_utility` kept `536` local pairs, but only `107` were clean `correct_vs_incorrect` orientations and `40` were the reverse `incorrect_vs_correct` case. Among the kept pairs:

- `168` were `both_correct`
- `221` were `both_wrong`
- `389 / 536 = 72.6%` were same-correctness utility-oriented
- `40 / 536 = 7.5%` were mixed-correctness pairs oriented against final correctness

This directly validates the starting diagnosis. The current scalar utility was not merely noisy; it was encoding an assumption that many same-correctness and even wrong-way mixed-correctness local preferences were pedagogically useful.

### 2. Correctness-aware orientation helped Step-DPO, but not CW-SPO

Switching to `correctness_priority` removed the `incorrect_vs_correct` failure mode completely and lifted Step-DPO from `0.4167` to `0.4583`.

But `correctness_priority` still kept the same large ambiguous mass:

- `168` both-correct pairs
- `221` both-wrong pairs
- `389` same-correctness utility-oriented pairs

That is why CW-SPO did not improve there. `cw_correctness` stayed at `0.4167`, exactly matching `cw_current`.

This is the clearest evidence that orientation repair alone was not enough for confidence weighting. Confidence could not rescue a target pool still dominated by same-correctness local competitions.

### 3. Strict purification was the missing ingredient

`strict_purified` kept only `153` mixed-correctness pairs, all oriented by correctness. That means:

- `fraction_strictly_instructional_pairs = 1.0`
- `fraction_ambiguous_pairs = 0.0`
- `fraction_misoriented_mixed_correctness_pairs = 0.0`
- `fraction_dropped_by_purification = 0.7417`

Despite discarding about `74%` of candidate local comparisons, Step-DPO kept its `0.4583` gain and CW-SPO improved to `0.5000`.

This is the main result of the phase:

\[
\text{CW-SPO}_{\text{strict}} > \text{CW-SPO}_{\text{current}}
\]

with

\[
0.5000 > 0.4167.
\]

So the most justified interpretation is:

> the central bottleneck was pair contamination and orientation semantics, not the existence of confidence weighting itself.

### 4. Confidence-aware training became competitive only after purification

On the noisy target:

- `cw_current = 0.4167`
- `cw_correctness = 0.4167`

On the purified target:

- `cw_strict = 0.5000`
- `confidence_filter_only + strict_purified = 0.5000`

That means confidence became useful only once the pair set was already instructional, but soft weighting did not yet beat hard filtering. The honest reading is:

- confidence-aware selection matters more than it did before,
- but the specific benefit of *soft* confidence weighting over simple filtering remains unproven.

### 5. Stronger verifier still was not the main missing piece

The purified strong-verifier rerun held the same pure pair taxonomy:

- `153` correct-vs-incorrect pairs
- `0` ambiguous kept pairs
- `1.0` high-confidence pair accuracy

But downstream final accuracy fell from `0.5000` to `0.4583`.

So the strong verifier improved neither pair count nor held-out answer accuracy after purification. In this phase, it remained a useful diagnostic backend, not the best downstream training choice.

### 6. The process metric still needs careful interpretation

The strict modes dropped process exact and coverage from around `0.9-1.0` down to about `0.25-0.29`. This does **not** mean the purified policy became worse at reasoning step by step. It means the current offline process metric only evaluates examples for which a usable local pair is produced, and strict purification rejected many process examples as weak-divergence or near-identical:

- process set raw candidate pairs: `48`
- process set strict-purified kept pairs: `14`

So the current process artifact is mixing two effects:

1. boundary prediction quality on eligible examples
2. coverage loss from refusing weak local divergences

That metric remains useful, but only if read together with coverage.

## Recommended Updated Research Direction

Primary decision: **Decision A**

Continue the project with purified local pairs as the new main path.

Why this is the most defensible choice:

- purified Step-DPO improved over `current_utility`,
- purified CW-SPO became competitive and matched the best held-out answer baseline on this slice,
- the improvement came from a mathematically cleaner target definition rather than from scaling or hand-waving,
- and the strong verifier was not the missing ingredient.

What this decision does **not** imply:

- it does not prove soft weighting is already the best confidence mechanism,
- it does not prove the strong verifier is useless in general,
- and it does not prove the offline process metric is now aligned with learned-policy improvement.

The updated research stance should be:

1. `strict_purified` becomes the default local-pair research path.
2. `current_utility` stays only as a historical baseline.
3. `correctness_priority` stays as an intermediate diagnostic mode.
4. The strong verifier stays optional, not default, until it helps downstream accuracy on the purified target.

The next experiment should be:

1. rerun `step_strict`, `conf_filter_strict`, and `cw_strict` on multiple seeds on the same `100 / 24 / 48` slice;
2. report process exact on eligible examples separately from process coverage;
3. only then revisit `semi_purified` as a controlled attempt to recover some pair count without reintroducing the old contamination.

What should *not* happen next:

- no 7B policy scaling,
- no bigger dataset push,
- no new learned confidence head,
- and no new objective family before the purified-target comparison is stable across seeds.

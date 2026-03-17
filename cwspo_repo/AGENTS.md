# AGENTS.md

This repository is designed to be run by a code agent or the Codex extension.

## Goal
Implement and iterate on **Confidence-Weighted Step Preference Optimization (CW-SPO)** for mathematical reasoning.

## Core invariant
The project hypothesis must stay the same across minimal and strong runs:

> Noisy step-level supervision should be used selectively and softly, not equally.

## Allowed simplifications
- Start with smaller policy models before scaling up.
- Use offline verifier scoring; do not keep policy and verifier resident together unless necessary.
- Start with simple confidence features before adding learned calibrators.

## Non-goals for the first milestone
- Training a new PRM from scratch.
- Full RLHF/GRPO/PPO loops.
- Multimodal extensions.
- General-domain reasoning outside math.

## Expected pipeline
1. Generate traces.
2. Score traces with verifier/PRM.
3. Build local step preference pairs.
4. Train weighted Step-DPO.
5. Evaluate final-answer and process metrics.

## Coding rules
- Prefer small, composable functions.
- Keep JSONL schemas stable.
- Do not silently change field names used by scripts.
- Add tests for any change to pair construction or weighting logic.
- Preserve offline artifacts under `outputs/` for reproducibility.

## First things to inspect if something breaks
- Step splitting in `src/cwspo/utils/steps.py`
- Verifier backend in `src/cwspo/models/verifier.py`
- Pair construction in `src/cwspo/pipeline/build_pairs.py`
- Loss in `src/cwspo/training/losses.py`

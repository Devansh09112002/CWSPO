# Codex / Agent workflow

## Recommended order

```bash
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
cp .env.example .env
# edit configs/rtx4090_48gb_minimal.yaml as needed
python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml
```

## Expected artifacts
- `traces.jsonl`
- `scored.jsonl`
- `pairs.jsonl`
- `train_metrics.json`
- `final_eval.json`
- `process_eval.json`
- `checkpoints/final/`

## First debugging checklist
1. Inspect 20 generated traces.
2. Inspect 20 scored traces and verify step-score trends look sane.
3. Inspect 20 built pairs and manually validate preferred/dispreferred segments.
4. Only then launch training.

## Recommended first milestone
- Run the minimal 4090 config on a small math subset.
- Compare Step-DPO vs confidence-weighted Step-DPO.
- Keep the policy small and the verifier offline.

#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/minimal.yaml}
python scripts/run_generate.py --config "$CONFIG"
python scripts/run_score.py --config "$CONFIG"
python scripts/run_pairs.py --config "$CONFIG"
python scripts/run_eval_process.py --config "$CONFIG" --ground-truth examples/sample_processbench_like.jsonl

echo "Smoke test completed. Inspect outputs in the configured output_dir."

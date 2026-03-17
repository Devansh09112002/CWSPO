#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/rtx4090_48gb_strong.yaml}
python scripts/run_pipeline.py --config "$CONFIG" --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl

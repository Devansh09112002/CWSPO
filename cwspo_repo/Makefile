CONFIG ?= configs/minimal.yaml
GT ?= examples/sample_processbench_like.jsonl
FLAGS ?=

generate:
	python scripts/run_generate.py --config $(CONFIG)

score:
	python scripts/run_score.py --config $(CONFIG)

pairs:
	python scripts/run_pairs.py --config $(CONFIG)

train:
	python scripts/run_train.py --config $(CONFIG)

eval-final:
	python scripts/run_eval_final.py --config $(CONFIG)

eval-process:
	python scripts/run_eval_process.py --config $(CONFIG) --ground-truth $(GT)

pipeline:
	python scripts/run_pipeline.py --config $(CONFIG) $(FLAGS)

smoke:
	bash scripts/smoke_test.sh $(CONFIG)

bootstrap:
	bash scripts/bootstrap_venv.sh

pipeline-4090:
	python scripts/run_pipeline.py --config configs/rtx4090_48gb_minimal.yaml --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl

pipeline-4090-strong:
	python scripts/run_pipeline.py --config configs/rtx4090_48gb_strong.yaml --with-train --with-final-eval --with-process-eval --ground-truth examples/sample_processbench_like.jsonl

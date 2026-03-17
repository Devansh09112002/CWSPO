from cwspo.evaluation.process_eval import (
    evaluate_process_predictions,
    predict_earliest_error_steps,
    render_process_failure_report,
)
from cwspo.schemas import PairRecord, ProcessGroundTruthRecord


def test_predict_earliest_error_uses_confidence_before_weight():
    pairs = [
        PairRecord(
            id="ex1",
            prompt="Q",
            prefix_steps=[],
            preferred_steps=["good"],
            dispreferred_steps=["bad"],
            weight=1.0,
            confidence=0.4,
            features={"confidence": 0.4},
            meta={"k": 1},
        ),
        PairRecord(
            id="ex1",
            prompt="Q",
            prefix_steps=[],
            preferred_steps=["good"],
            dispreferred_steps=["bad"],
            weight=0.5,
            confidence=0.8,
            features={"confidence": 0.8},
            meta={"k": 2},
        ),
    ]
    preds = predict_earliest_error_steps(pairs)
    assert preds["ex1"]["pred"] == 2


def test_evaluate_process_predictions_handles_missing_and_near_miss():
    predictions = {
        "ex1": {"pred": 2, "score": 0.9, "num_candidates": 2},
        "ex2": {"pred": 0, "score": 0.7, "num_candidates": 1},
    }
    ground_truth = [
        ProcessGroundTruthRecord(id="ex1", gold_earliest_error_step=2, prompt="Q1"),
        ProcessGroundTruthRecord(id="ex2", gold_earliest_error_step=1, prompt="Q2"),
        ProcessGroundTruthRecord(id="ex3", gold_earliest_error_step=3, prompt="Q3"),
    ]

    summary = evaluate_process_predictions(predictions, ground_truth)
    assert summary["earliest_error_exact"] == 1 / 3
    assert summary["earliest_error_near_miss"] == 2 / 3
    assert summary["coverage"] == 2 / 3
    assert summary["num_evaluated_examples"] == 3
    assert summary["rows"][2]["pred"] is None


def test_render_process_failure_report_includes_confidence_buckets():
    summary = {
        "earliest_error_exact": 0.5,
        "earliest_error_near_miss": 1.0,
        "coverage": 1.0,
        "process_confidence_analysis": {
            "bucket_summary": {
                "low": {"count": 1, "mean_confidence": 0.2, "process_boundary_accuracy": 0.0},
                "medium": {"count": 2, "mean_confidence": 0.5, "process_boundary_accuracy": 0.5},
                "high": {"count": 3, "mean_confidence": 0.8, "process_boundary_accuracy": 1.0},
            }
        },
        "rows": [],
    }

    report = render_process_failure_report(summary)
    assert "## Confidence Buckets" in report
    assert "### High" in report
    assert "Process-boundary accuracy" in report

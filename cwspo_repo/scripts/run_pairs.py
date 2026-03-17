import argparse

from cwspo.config import ensure_dirs, load_config
from cwspo.pipeline.build_pairs import build_pair_artifacts
from cwspo.pipeline.diagnostics import (
    analyze_pairs,
    write_confidence_artifacts,
    write_pair_audits,
    write_pair_orientation_audit,
    write_pair_purity_report,
)
from cwspo.schemas import ProcessGroundTruthRecord, ScoredTraceRecord
from cwspo.utils.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    scored = read_jsonl(cfg.paths.scored_file, ScoredTraceRecord)
    artifacts = build_pair_artifacts(cfg, scored)
    write_jsonl(cfg.paths.pairs_file, artifacts.pairs)

    process_gt = None
    if cfg.paths.process_ground_truth_file:
        process_gt = read_jsonl(cfg.paths.process_ground_truth_file, ProcessGroundTruthRecord)

    analysis = analyze_pairs(
        artifacts.pairs,
        low_threshold=cfg.confidence.low_threshold,
        high_threshold=cfg.confidence.high_threshold,
        histogram_bins=cfg.diagnostics.histogram_bins,
        num_raw_pairs=artifacts.num_raw_pairs,
        num_dropped_by_confidence=artifacts.num_dropped_by_confidence,
        method_name=artifacts.method_name,
        pair_mode=artifacts.pair_mode,
        pair_purity_report=artifacts.pair_purity_report,
        process_ground_truth=process_gt,
    )
    write_confidence_artifacts(
        analysis,
        json_path=cfg.paths.confidence_analysis_file,
        report_path=cfg.paths.confidence_report_file,
    )
    write_pair_audits(
        artifacts.pairs,
        low_threshold=cfg.confidence.low_threshold,
        high_threshold=cfg.confidence.high_threshold,
        sample_count=cfg.diagnostics.num_audit_samples_per_bucket,
        seed=cfg.diagnostics.pair_audit_seed,
        low_path=cfg.paths.pair_audit_low_file,
        mid_path=cfg.paths.pair_audit_mid_file,
        high_path=cfg.paths.pair_audit_high_file,
    )
    write_pair_purity_report(
        artifacts.pair_purity_report,
        json_path=cfg.paths.pair_purity_report_file,
    )
    write_pair_orientation_audit(
        artifacts.orientation_audit_rows,
        path=cfg.paths.pair_orientation_audit_file,
        sample_count=cfg.diagnostics.num_orientation_audit_samples,
        seed=cfg.diagnostics.pair_audit_seed,
    )
    print(
        f"Wrote {len(artifacts.pairs)} pairs to {cfg.paths.pairs_file} "
        f"(raw={artifacts.num_raw_pairs}, dropped={artifacts.num_dropped_by_confidence})"
    )


if __name__ == "__main__":
    main()

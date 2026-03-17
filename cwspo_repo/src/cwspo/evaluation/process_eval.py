from __future__ import annotations

from collections import defaultdict

from cwspo.schemas import PairRecord, ProcessGroundTruthRecord


def evaluate_process(pairs: list[PairRecord], ground_truth: list[ProcessGroundTruthRecord]) -> dict:
    gt_map = {g.id: g.gold_earliest_error_step for g in ground_truth}
    pred_steps: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for p in pairs:
        k = int(p.meta.get("k", -1))
        pred_steps[p.id].append((k, float(p.weight)))

    exact = 0
    covered = 0
    rows = []
    for ex_id, gold_k in gt_map.items():
        cands = pred_steps.get(ex_id, [])
        if not cands:
            rows.append({"id": ex_id, "gold": gold_k, "pred": None, "correct": False})
            continue
        pred_k = sorted(cands, key=lambda x: x[1], reverse=True)[0][0]
        ok = pred_k == gold_k
        exact += int(ok)
        covered += 1
        rows.append({"id": ex_id, "gold": gold_k, "pred": pred_k, "correct": ok})

    return {
        "earliest_error_exact": exact / max(1, covered),
        "coverage": covered / max(1, len(gt_map)),
        "rows": rows,
    }

"""Evaluate V17 as a fixed-K residual ranker without the failed count head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate import (
    aggregate_predictions,
    bootstrap_lower,
    change_summary,
    exact_count_mask,
    fixed_k_mask,
    lock_maps,
    mask_from_counts,
    priority_adjusted_scores,
    test_constraints,
    tp,
    write_submission,
)
from v17_data import GraphStore


BETAS = (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)


def fixed_fold_deltas(mask, base_mask, train):
    output = []
    for fold in range(5):
        order_indices = np.flatnonzero(train.folds == fold)
        rows = []
        for order_index in order_indices:
            start = int(train.order_ptr[order_index])
            stop = int(train.order_ptr[order_index + 1])
            rows.extend(range(start, stop))
        rows = np.asarray(rows, dtype=np.int64)
        output.append(tp(mask[rows], train.labels[rows]) - tp(base_mask[rows], train.labels[rows]))
    return output


def blend(base_scores, model_scores, beta):
    return base_scores + beta * (model_scores - base_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    metadata = train.metadata
    predictions = aggregate_predictions(config, train, test, args.predictions)

    target_train = round(config["target_test_predictions"] / len(test) * len(train))
    base_mask = exact_count_mask(train.v11, train.order_ptr, target_train, config["max_rootcauses"])
    base_tp = tp(base_mask, train.labels)
    records = []
    for beta in BETAS:
        scores = blend(train.v11, predictions["oof_scores"], beta)
        mask = fixed_k_mask(scores, train.order_ptr, base_mask)
        seed_deltas = []
        for seed_scores in predictions["seed_oof_scores"]:
            seed_mask = fixed_k_mask(blend(train.v11, seed_scores, beta), train.order_ptr, base_mask)
            seed_deltas.append(tp(seed_mask, train.labels) - base_tp)
        records.append(
            {
                "beta": beta,
                "tp_delta": tp(mask, train.labels) - base_tp,
                "seed_tp_deltas": seed_deltas,
                "fold_tp_deltas": fixed_fold_deltas(mask, base_mask, train),
                "bootstrap_95_lower": bootstrap_lower(mask, base_mask, train),
                "mask": mask,
            }
        )
    best = max(
        records,
        key=lambda item: (
            min(item["seed_tp_deltas"]),
            item["tp_delta"],
            min(item["fold_tp_deltas"]),
            -item["beta"],
        ),
    )

    champion_mask = test.arrays["test_champion_mask"].astype(bool)
    _, _, forced_in, forced_out = test_constraints(test, metadata)
    test_scores = blend(test.v11, predictions["test_scores"], best["beta"])
    test_scores, _ = priority_adjusted_scores(test_scores, test, metadata, champion_mask)
    rank_mask = fixed_k_mask(test_scores, test.order_ptr, champion_mask, forced_in, forced_out)
    per_seed_rank = []
    for seed_scores in predictions["seed_test_scores"]:
        scores = blend(test.v11, seed_scores, best["beta"])
        scores, _ = priority_adjusted_scores(scores, test, metadata, champion_mask)
        per_seed_rank.append(
            fixed_k_mask(scores, test.order_ptr, champion_mask, forced_in, forced_out)
        )
    safe_mask = champion_mask.copy()
    for order_index in range(len(test)):
        start = int(test.order_ptr[order_index])
        stop = int(test.order_ptr[order_index + 1])
        if all(np.array_equal(mask[start:stop], rank_mask[start:stop]) for mask in per_seed_rank):
            safe_mask[start:stop] = rank_mask[start:stop]

    changes, template_share = change_summary(rank_mask, champion_mask, test, metadata)
    safe_changes, safe_template_share = change_summary(safe_mask, champion_mask, test, metadata)
    gates = config["oof_gate"]
    passed = (
        best["tp_delta"] >= gates["joint_min_tp_delta"]
        and min(best["seed_tp_deltas"]) >= gates["seed_min_tp_delta"]
        and min(best["fold_tp_deltas"]) >= gates["fold_min_tp_delta"]
        and best["bootstrap_95_lower"] > gates["bootstrap_lower_bound"]
        and gates["test_min_changed_orders"] <= len(changes) <= gates["test_max_changed_orders"]
        and template_share <= gates["max_template_share"]
    )
    report = {
        "passed": passed,
        "selected_beta": best["beta"],
        "baseline_tp": base_tp,
        "checks": {
            "rank_tp_delta": best["tp_delta"],
            "seed_tp_deltas": best["seed_tp_deltas"],
            "fold_tp_deltas": best["fold_tp_deltas"],
            "bootstrap_95_lower": best["bootstrap_95_lower"],
            "test_changed_orders": len(changes),
            "test_max_template_share": template_share,
            "safe_changed_orders": len(safe_changes),
            "safe_max_template_share": safe_template_share,
        },
        "required": gates,
        "beta_scan": [
            {key: value for key, value in item.items() if key != "mask"} for item in records
        ],
        "champion_sha256": metadata["champion_sha256"],
    }
    (args.output / "rank_only_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if passed:
        write_submission(
            args.output / "v17_1_hgt_rank_only_p1059.csv",
            rank_mask,
            test,
            metadata,
            metadata["champion_sha256"],
            "v17_1_rank_only",
            report,
        )
        write_submission(
            args.output / "v17_1_hgt_safe_core_p1059.csv",
            safe_mask,
            test,
            metadata,
            metadata["champion_sha256"],
            "v17_1_safe_core",
            report,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

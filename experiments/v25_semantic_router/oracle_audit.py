"""Audit whether the V25 candidate action space can support the target gain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from actions import exact_dp, generate_actions
from v25_common import Bundle, exact_count_mask, read_json, write_json


def load_semantic(directory, seed, folds, train_rows, test_rows, train_orders, test_orders):
    oof = np.zeros(train_rows, dtype=np.float32)
    counts = np.zeros((train_orders, 8), dtype=np.float32)
    tests, test_counts = [], []
    for fold in range(5):
        data = np.load(directory / f"semantic_seed_{seed}_fold_{fold}.npz")
        oof[data["validation_rows"]] = data["validation_scores"]
        counts[data["validation_orders"]] = data["validation_counts"]
        tests.append(data["test_scores"])
        test_counts.append(data["test_counts"])
    return oof, counts, np.mean(tests, axis=0), np.mean(test_counts, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    config = read_json(args.config)
    bundle = Bundle.load(args.data_root)
    labels = bundle.arrays["train_labels"]
    ptr = bundle.ptr("train")
    base = exact_count_mask(bundle.arrays["train_v11"], ptr, config["target_train_predictions"])
    semantic, count_prob, _, _ = load_semantic(
        args.semantic_dir, args.seed, bundle.arrays["train_folds"], len(labels),
        len(bundle.arrays["test_v11"]), len(bundle.records["train"]), len(bundle.records["test"]),
    )
    retrieval_data = np.load(args.retrieval_dir / f"retrieval_seed_{args.seed}.npz")
    retrieval_probability = np.nan_to_num(retrieval_data["oof_probability"], nan=semantic)
    experts = np.column_stack(
        [
            bundle.arrays["train_v11"],
            np.nan_to_num(bundle.arrays["train_v13"], nan=bundle.arrays["train_v11"]),
            np.nan_to_num(bundle.arrays["train_v19"], nan=bundle.arrays["train_v11"]),
            semantic,
            retrieval_probability,
        ]
    ).astype(np.float32)
    retrieval = {
        "probability": retrieval_probability,
        "support": retrieval_data["oof_support"],
        "consistency": retrieval_data["oof_consistency"],
        "similarity": retrieval_data["oof_similarity"],
    }
    groups = []
    for order, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        groups.append(
            generate_actions(
                order, int(start), int(stop), base, experts, count_prob, retrieval,
                config["router"]["candidate_depth"], config["router"]["max_action_depth"], labels,
            )
        )
    gain, actions = exact_dp(groups, [[item.gain for item in group] for group in groups], 0)
    selected_adds = {row for action in actions for row in action.add if labels[row] == 1}
    missed = set(np.flatnonzero((labels == 1) & ~base))
    coverage = len(selected_adds & missed) / max(len(missed), 1)
    baseline_tp = int(labels[base].sum())
    gate = config["oracle_gate"]
    passed = gain >= gate["minimum_tp_delta"] and coverage >= gate["minimum_fn_coverage"]
    report = {
        "version": config["version"],
        "status": "passed_oracle_gate" if passed else "failed_oracle_gate",
        "baseline_tp": baseline_tp,
        "oracle_tp": baseline_tp + int(gain),
        "oracle_tp_delta": int(gain),
        "baseline_false_negatives": len(missed),
        "recovered_false_negatives": len(selected_adds & missed),
        "false_negative_coverage": coverage,
        "changed_orders": sum(action.name != "keep" for action in actions),
        "requirements": gate,
        "decision": "train_router" if passed else "terminate_v25",
        "submission_generated": False,
    }
    write_json(args.output, report)
    print(report)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

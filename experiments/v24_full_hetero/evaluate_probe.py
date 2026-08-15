"""Evaluate the first honest V24 seed and decide whether full training may continue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_fold import exact_count_mask
from v24_data import GraphStore


def fixed_k_mask(scores, ptr, base_mask, max_roots=8):
    selected = np.zeros(len(scores), dtype=bool)
    for index in range(len(ptr) - 1):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        count = int(base_mask[start:stop].sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[: min(count, max_roots)]
        selected[start + ranked] = True
    return selected


def per_order_delta(mask, base_mask, labels, ptr):
    values = np.zeros(len(ptr) - 1, dtype=np.int16)
    for index in range(len(values)):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        values[index] = int(labels[start:stop][mask[start:stop]].sum()) - int(
            labels[start:stop][base_mask[start:stop]].sum()
        )
    return values


def bootstrap_lower(deltas, iterations=20000):
    rng = np.random.default_rng(20260804)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_probe.json")
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    target = round(config["target_test_predictions"] / len(test) * len(train))
    base_mask = exact_count_mask(train.v11, train.alarm_ptr, target, config["max_rootcauses"])
    scores = np.zeros(len(train.v11), dtype=np.float32)
    gates = np.zeros(len(train.v11), dtype=np.float32)
    completed = []
    for fold in range(config["folds"]):
        path = args.predictions / f"seed_{args.seed}_fold_{fold}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        scores[data["validation_rows"]] = data["validation_scores"]
        gates[data["validation_rows"]] = data["validation_gates"]
        completed.append(fold)
    mask = fixed_k_mask(scores, train.alarm_ptr, base_mask, config["max_rootcauses"])
    deltas = per_order_delta(mask, base_mask, train.labels, train.alarm_ptr)
    fold_deltas = [int(deltas[train.folds == fold].sum()) for fold in range(config["folds"])]
    total = int(deltas.sum())
    gate = config["probe_gate"]
    passed = bool(
        total >= gate["min_total_tp_delta"]
        and sum(value > 0 for value in fold_deltas) >= gate["min_improved_folds"]
        and min(fold_deltas) >= gate["min_fold_tp_delta"]
    )
    changed_orders = int(
        sum(
            not np.array_equal(
                mask[int(train.alarm_ptr[i]) : int(train.alarm_ptr[i + 1])],
                base_mask[int(train.alarm_ptr[i]) : int(train.alarm_ptr[i + 1])],
            )
            for i in range(len(train))
        )
    )
    report = {
        "version": config["version"],
        "status": "passed_probe_gate" if passed else "failed_probe_gate",
        "seed": args.seed,
        "completed_folds": completed,
        "baseline_tp": int(train.labels[base_mask].sum()),
        "rank_tp_delta": total,
        "fold_tp_deltas": fold_deltas,
        "improved_folds": sum(value > 0 for value in fold_deltas),
        "bootstrap_95_lower": bootstrap_lower(deltas),
        "changed_orders": changed_orders,
        "mean_gate": float(gates.mean()),
        "required": gate,
        "decision": "run_remaining_seeds" if passed else "terminate_v24",
        "submission_generated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

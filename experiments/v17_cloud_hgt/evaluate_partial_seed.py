"""Evaluate completed honest-OOF folds for one seed without requiring all seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate import exact_count_mask, fixed_k_mask, tp
from v17_data import GraphStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    target = round(config["target_test_predictions"] / len(test) * len(train))
    base = exact_count_mask(train.v11, train.order_ptr, target, config["max_rootcauses"])
    fold_deltas = []
    completed = []
    for fold in range(config["folds"]):
        path = args.predictions / f"seed_{args.seed}_fold_{fold}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        rows = data["validation_rows"]
        scores = train.v11.copy()
        scores[rows] = data["validation_scores"]
        candidate = fixed_k_mask(scores, train.order_ptr, base)
        delta = tp(candidate[rows], train.labels[rows]) - tp(base[rows], train.labels[rows])
        fold_deltas.append(int(delta))
        completed.append(fold)
    print(json.dumps({"seed": args.seed, "completed": completed, "fold_deltas": fold_deltas, "total_delta": sum(fold_deltas)}))


if __name__ == "__main__":
    main()

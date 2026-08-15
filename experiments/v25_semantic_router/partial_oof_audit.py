"""Audit completed semantic folds before the full V25 probe is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_common import Bundle, exact_count_mask


def per_order_delta(mask, base, labels, ptr, orders):
    values = []
    for order in orders:
        start, stop = int(ptr[order]), int(ptr[order + 1])
        values.append(
            int(labels[start:stop][mask[start:stop]].sum())
            - int(labels[start:stop][base[start:stop]].sum())
        )
    return np.asarray(values, dtype=np.int16)


def bootstrap_lower(values, iterations=20000, seed=20260805):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def pairwise_accuracy(scores, labels, ptr, orders):
    correct = 0.0
    pairs = 0
    for order in orders:
        start, stop = int(ptr[order]), int(ptr[order + 1])
        positive = scores[start:stop][labels[start:stop] == 1]
        negative = scores[start:stop][labels[start:stop] == 0]
        if not len(positive) or not len(negative):
            continue
        differences = positive[:, None] - negative[None, :]
        correct += float((differences > 0).sum() + 0.5 * (differences == 0).sum())
        pairs += int(differences.size)
    return float(correct / pairs) if pairs else None, pairs


def rank_only_mask(scores, base, ptr, orders):
    selected = np.zeros(len(scores), dtype=bool)
    for order in orders:
        start, stop = int(ptr[order]), int(ptr[order + 1])
        count = int(base[start:stop].sum())
        ranking = np.argsort(-scores[start:stop], kind="stable")[:count]
        selected[start + ranking] = True
    return selected


def subset_exact_count_mask(scores, ptr, orders, target_count, max_roots=8):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order in orders:
        start, stop = int(ptr[order]), int(ptr[order + 1])
        ranking = np.argsort(-scores[start:stop], kind="stable")[:max_roots]
        selected[start + ranking[0]] = True
        optional.extend((start + ranking[1:]).tolist())
    remaining = target_count - int(selected.sum())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    bundle = Bundle.load(args.data_root)
    arrays = bundle.arrays
    labels = arrays["train_labels"].astype(np.int8)
    ptr = bundle.ptr("train")
    base = exact_count_mask(arrays["train_v11"], ptr, 3169)
    semantic = np.full(len(labels), np.nan, dtype=np.float32)
    completed_orders = []
    fold_reports = []
    test_scores = []
    count_probabilities = {}

    for fold in args.folds:
        path = args.semantic_dir / f"semantic_seed_{args.seed}_fold_{fold}.npz"
        with np.load(path, allow_pickle=False) as archive:
            rows = archive["validation_rows"].astype(np.int64)
            orders = archive["validation_orders"].astype(np.int64)
            scores = archive["validation_scores"].astype(np.float32)
            semantic[rows] = scores
            completed_orders.extend(orders.tolist())
            test_scores.append(archive["test_scores"].astype(np.float32))
            count_probabilities[fold] = archive["validation_counts"].astype(np.float32)

        rank_mask = rank_only_mask(semantic, base, ptr, orders)
        delta = per_order_delta(rank_mask, base, labels, ptr, orders)
        fold_rows = bundle.rows_for_orders("train", orders)
        fold_reports.append(
            {
                "fold": int(fold),
                "orders": int(len(orders)),
                "candidates": int(len(fold_rows)),
                "baseline_predictions": int(base[fold_rows].sum()),
                "baseline_tp": int(labels[fold_rows][base[fold_rows]].sum()),
                "rank_only_tp": int(labels[fold_rows][rank_mask[fold_rows]].sum()),
                "rank_only_delta": int(delta.sum()),
                "orders_improved": int((delta > 0).sum()),
                "orders_harmed": int((delta < 0).sum()),
                "orders_unchanged": int((delta == 0).sum()),
            }
        )

    orders = np.asarray(sorted(completed_orders), dtype=np.int64)
    rows = bundle.rows_for_orders("train", orders)
    if np.isnan(semantic[rows]).any():
        raise ValueError("completed fold scores do not cover their validation rows")

    rank_mask = rank_only_mask(semantic, base, ptr, orders)
    rank_delta = per_order_delta(rank_mask, base, labels, ptr, orders)
    target_count = int(base[rows].sum())
    realloc_mask = subset_exact_count_mask(semantic, ptr, orders, target_count)
    realloc_delta = per_order_delta(realloc_mask, base, labels, ptr, orders)
    pairwise, pairs = pairwise_accuracy(semantic, labels, ptr, orders)

    predicted_count_mask = np.zeros(len(labels), dtype=bool)
    count_errors = []
    for fold, report in zip(args.folds, fold_reports):
        fold_orders = np.flatnonzero(arrays["train_folds"] == fold)
        probabilities = count_probabilities[fold]
        if len(fold_orders) != len(probabilities):
            raise ValueError((fold, len(fold_orders), len(probabilities)))
        predicted_counts = probabilities.argmax(axis=1) + 1
        for order, count in zip(fold_orders, predicted_counts):
            start, stop = int(ptr[order]), int(ptr[order + 1])
            count = min(int(count), stop - start, 8)
            ranking = np.argsort(-semantic[start:stop], kind="stable")[:count]
            predicted_count_mask[start + ranking] = True
            count_errors.append(abs(count - int(labels[start:stop].sum())))
        fold_rows = bundle.rows_for_orders("train", fold_orders)
        report["count_head_predictions"] = int(predicted_count_mask[fold_rows].sum())
        report["count_head_tp"] = int(labels[fold_rows][predicted_count_mask[fold_rows]].sum())

    score_correlation = float(np.corrcoef(test_scores[0], test_scores[1])[0, 1])
    baseline_tp = int(labels[rows][base[rows]].sum())
    report = {
        "seed": args.seed,
        "completed_folds": args.folds,
        "orders": int(len(orders)),
        "candidates": int(len(rows)),
        "baseline_predictions": target_count,
        "baseline_tp": baseline_tp,
        "folds": fold_reports,
        "rank_only": {
            "tp": int(labels[rows][rank_mask[rows]].sum()),
            "tp_delta": int(rank_delta.sum()),
            "orders_improved": int((rank_delta > 0).sum()),
            "orders_harmed": int((rank_delta < 0).sum()),
            "bootstrap_95_lower_delta": bootstrap_lower(rank_delta),
        },
        "subset_exact_count": {
            "tp": int(labels[rows][realloc_mask[rows]].sum()),
            "tp_delta": int(realloc_delta.sum()),
            "orders_improved": int((realloc_delta > 0).sum()),
            "orders_harmed": int((realloc_delta < 0).sum()),
            "bootstrap_95_lower_delta": bootstrap_lower(realloc_delta),
        },
        "semantic_pairwise_accuracy": pairwise,
        "semantic_pair_count": pairs,
        "count_head": {
            "predictions": int(predicted_count_mask[rows].sum()),
            "tp": int(labels[rows][predicted_count_mask[rows]].sum()),
            "mean_absolute_count_error": float(np.mean(count_errors)),
        },
        "test_score_correlation_fold_0_1": score_correlation,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

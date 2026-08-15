"""Nested audit of an order-level root-cause count corrector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import v19_nested_count_audit as audit


ROOT = Path(r"D:\zgyidong")
V16_DIR = ROOT / "experiments" / "v16"
RIDGE_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)
BLEND_WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 0.5, 11))


def order_features(node_features, v11_scores, graph_scores, slices, base_mask):
    rows = []
    for order_slice in slices:
        x = node_features[order_slice]
        v11 = v11_scores[order_slice]
        graph = graph_scores[order_slice]
        base_k = int(base_mask[order_slice].sum())

        def distribution(values):
            ranked = np.sort(values)[::-1]
            padded = np.pad(ranked[:8], (0, max(0, 8 - len(ranked))), constant_values=0)
            boundary = padded[min(base_k, 8) - 1]
            next_value = ranked[base_k] if base_k < len(ranked) else 0.0
            return np.concatenate(
                [
                    padded,
                    np.asarray(
                        [
                            values.mean(),
                            values.std(),
                            values.max(),
                            boundary,
                            next_value,
                            boundary - next_value,
                        ]
                    ),
                ]
            )

        structural = np.concatenate(
            [
                x.mean(axis=0),
                x.std(axis=0),
                x.max(axis=0),
            ]
        )
        rows.append(
            np.concatenate(
                [
                    np.asarray([len(v11), base_k], dtype=np.float64),
                    distribution(v11),
                    distribution(graph),
                    structural,
                ]
            )
        )
    return np.asarray(rows, dtype=np.float64)


def ridge_predict(train_x, train_y, validation_x, alpha):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_z = (train_x - mean) / scale
    validation_z = (validation_x - mean) / scale
    train_z = np.column_stack([np.ones(len(train_z)), train_z])
    validation_z = np.column_stack([np.ones(len(validation_z)), validation_z])
    penalty = np.eye(train_z.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train_z.T @ train_z + penalty, train_z.T @ train_y)
    return validation_z @ coefficients


def exact_count_from_prediction(predicted_k, target_count, slices):
    counts = np.ones(len(slices), dtype=np.int16)
    candidates = []
    for order_index, order_slice in enumerate(slices):
        maximum = min(audit.MAX_ROOTCAUSES, order_slice.stop - order_slice.start)
        for current_k in range(1, maximum):
            # Reduction in squared error from increasing current_k to current_k + 1.
            utility = 2.0 * predicted_k[order_index] - 2.0 * current_k - 1.0
            candidates.append((utility, order_index, current_k + 1))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, order_index, new_count in candidates[: target_count - len(slices)]:
        counts[order_index] = max(counts[order_index], new_count)
    if int(counts.sum()) != target_count:
        raise ValueError((int(counts.sum()), target_count))
    return counts


def mask_from_counts(scores, slices, counts):
    mask = np.zeros(len(scores), dtype=bool)
    for order_index, order_slice in enumerate(slices):
        ranked = np.argsort(-scores[order_slice], kind="stable")[: counts[order_index]]
        mask[order_slice.start + ranked] = True
    return mask


def main():
    train_orders = audit.load_orders(audit.TRAIN_DIR, True)
    test_orders = audit.load_orders(audit.TEST_DIR, False)
    slices = audit.build_slices(train_orders)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in train_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    true_k = np.asarray([len(order["roots"]) for order in train_orders], dtype=np.float64)
    folds, group_count, fold_sizes = audit.grouped_folds(train_orders)
    v11_scores = 0.25 * np.load(audit.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        audit.V11_DIR / "v11_oof_meta.npy"
    )
    graph_scores = np.load(V16_DIR / "v16_graph_time_oof.npy")
    node_features = np.load(V16_DIR / "v16_train_features.npy", mmap_mode="r")[:, :97]
    target_count = round(audit.TARGET_TEST_COUNT / len(test_orders) * len(train_orders))
    base_mask = audit.exact_count_mask(v11_scores, slices, target_count)
    features = order_features(node_features, v11_scores, graph_scores, slices, base_mask)

    predictions = {}
    all_orders = np.arange(len(train_orders))
    for alpha in RIDGE_VALUES:
        predicted = np.zeros(len(train_orders), dtype=np.float64)
        for heldout in range(audit.N_FOLDS):
            train_indices = all_orders[folds != heldout]
            validation_indices = all_orders[folds == heldout]
            predicted[validation_indices] = ridge_predict(
                features[train_indices], true_k[train_indices], features[validation_indices], alpha
            )
        predictions[alpha] = np.clip(predicted, 1.0, audit.MAX_ROOTCAUSES)

    candidates = []
    masks = {}
    for alpha, predicted in predictions.items():
        counts = exact_count_from_prediction(predicted, target_count, slices)
        for weight in BLEND_WEIGHTS:
            scores = (1.0 - weight) * v11_scores + weight * graph_scores
            mask = mask_from_counts(scores, slices, counts)
            stats = audit.mask_stats(mask, base_mask, labels, slices, folds)
            key = (alpha, weight)
            masks[key] = mask
            candidates.append({"alpha": alpha, "weight": weight, **stats})

    nested_mask = base_mask.copy()
    choices = []
    for heldout in range(audit.N_FOLDS):
        training_folds = [fold for fold in range(audit.N_FOLDS) if fold != heldout]
        best = max(
            candidates,
            key=lambda item: (
                sum(item["fold_deltas"][fold] for fold in training_folds),
                min(item["fold_deltas"][fold] for fold in training_folds),
                -item["weight"],
                -item["alpha"],
            ),
        )
        chosen = masks[(best["alpha"], best["weight"])]
        for order_index in np.flatnonzero(folds == heldout):
            nested_mask[slices[order_index]] = chosen[slices[order_index]]
        choices.append(
            {
                "heldout_fold": heldout,
                "alpha": best["alpha"],
                "weight": best["weight"],
                "training_delta": int(
                    sum(best["fold_deltas"][fold] for fold in training_folds)
                ),
                "heldout_delta": int(best["fold_deltas"][heldout]),
            }
        )

    nested_stats = audit.mask_stats(nested_mask, base_mask, labels, slices, folds)
    order_deltas = audit.per_order_delta(nested_mask, base_mask, labels, slices)
    best_optimistic = max(
        candidates,
        key=lambda item: (item["tp_delta"], min(item["fold_deltas"]), -item["weight"]),
    )
    bootstrap_lower = audit.bootstrap_lower(order_deltas)
    report = {
        "method": "nested template-grouped ridge count correction",
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "target_predictions": target_count,
        "feature_dimension": int(features.shape[1]),
        "optimistic_best": best_optimistic,
        "nested_choices": choices,
        "nested": {**nested_stats, "bootstrap_95_lower": bootstrap_lower},
        "passed_mean_gate": bool(
            nested_stats["tp_delta"] >= 30
            and min(nested_stats["fold_deltas"]) >= -2
            and bootstrap_lower > 0
        ),
        "passed_significant_gate": False,
        "all_candidates": candidates,
    }
    output = ROOT / "experiments" / "v20_nested_order_count.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_candidates"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

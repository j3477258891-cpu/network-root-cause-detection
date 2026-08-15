"""Nested audit of paired +1/-1 boundary-node transfers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import v19_nested_count_audit as audit


ROOT = Path(r"D:\zgyidong")
V16_DIR = ROOT / "experiments" / "v16"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
TRANSFER_COUNTS = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100)


def node_features(raw, v11, graph, slices, base_mask):
    result = np.zeros((len(v11), raw.shape[1] + 10), dtype=np.float64)
    for order_slice in slices:
        local_v11 = v11[order_slice]
        local_graph = graph[order_slice]
        order_size = len(local_v11)
        rank_v11 = np.empty(order_size, dtype=np.float64)
        rank_graph = np.empty(order_size, dtype=np.float64)
        rank_v11[np.argsort(-local_v11, kind="stable")] = np.arange(order_size)
        rank_graph[np.argsort(-local_graph, kind="stable")] = np.arange(order_size)
        base_k = int(base_mask[order_slice].sum())
        context = np.column_stack(
            [
                local_v11,
                local_graph,
                rank_v11 / max(order_size - 1, 1),
                rank_graph / max(order_size - 1, 1),
                np.full(order_size, order_size / 32.0),
                np.full(order_size, base_k / 8.0),
                np.full(order_size, local_v11.mean()),
                np.full(order_size, local_v11.std()),
                np.full(order_size, local_graph.mean()),
                np.full(order_size, local_graph.std()),
            ]
        )
        result[order_slice] = np.column_stack([raw[order_slice], context])
    return result


def weighted_ridge_predict(train_x, train_y, validation_x, alpha):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (train_x - mean) / scale
    validation = (validation_x - mean) / scale
    x = np.column_stack([np.ones(len(x)), x])
    validation = np.column_stack([np.ones(len(validation)), validation])
    positive = max(int(train_y.sum()), 1)
    negative = max(len(train_y) - positive, 1)
    weights = np.where(train_y == 1, len(train_y) / (2 * positive), len(train_y) / (2 * negative))
    weighted_x = x * weights[:, None]
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ weighted_x + penalty, weighted_x.T @ train_y)
    return np.clip(validation @ coefficients, 0.0, 1.0)


def boundary_candidates(scores, predictions, slices, base_mask):
    additions = []
    removals = []
    for order_index, order_slice in enumerate(slices):
        ranked = np.argsort(-scores[order_slice], kind="stable")
        base_k = int(base_mask[order_slice].sum())
        if base_k < min(audit.MAX_ROOTCAUSES, len(ranked)):
            row = order_slice.start + ranked[base_k]
            additions.append((float(predictions[row]), order_index, row))
        if base_k > 1:
            row = order_slice.start + ranked[base_k - 1]
            removals.append((float(predictions[row]), order_index, row))
    return additions, removals


def transfer_mask(base_mask, additions, removals, count):
    mask = base_mask.copy()
    chosen_orders = set()
    selected_additions = []
    for item in sorted(additions, key=lambda value: (-value[0], value[1])):
        if item[1] in chosen_orders:
            continue
        selected_additions.append(item)
        chosen_orders.add(item[1])
        if len(selected_additions) == count:
            break
    selected_removals = []
    for item in sorted(removals, key=lambda value: (value[0], value[1])):
        if item[1] in chosen_orders:
            continue
        selected_removals.append(item)
        chosen_orders.add(item[1])
        if len(selected_removals) == count:
            break
    if len(selected_additions) != count or len(selected_removals) != count:
        raise ValueError((count, len(selected_additions), len(selected_removals)))
    mask[[item[2] for item in selected_additions]] = True
    mask[[item[2] for item in selected_removals]] = False
    return mask


def main():
    orders = audit.load_orders(audit.TRAIN_DIR, True)
    test_orders = audit.load_orders(audit.TEST_DIR, False)
    slices = audit.build_slices(orders)
    labels = np.asarray(
        [int(node["@rid"] in order["roots"]) for order in orders for node in order["alarms"]],
        dtype=np.int8,
    )
    folds, group_count, fold_sizes = audit.grouped_folds(orders)
    row_folds = np.concatenate(
        [np.full(len(order["alarms"]), folds[index], dtype=np.int8) for index, order in enumerate(orders)]
    )
    v11 = 0.25 * np.load(audit.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        audit.V11_DIR / "v11_oof_meta.npy"
    )
    graph = np.load(V16_DIR / "v16_graph_time_oof.npy")
    target_count = round(audit.TARGET_TEST_COUNT / len(test_orders) * len(orders))
    base_mask = audit.exact_count_mask(v11, slices, target_count)
    raw = np.load(V16_DIR / "v16_train_features.npy", mmap_mode="r")[:, :97]
    features = node_features(raw, v11, graph, slices, base_mask)

    candidates = []
    masks = {}
    for alpha in ALPHAS:
        predictions = np.zeros(len(labels), dtype=np.float64)
        for heldout in range(audit.N_FOLDS):
            train_rows = row_folds != heldout
            validation_rows = row_folds == heldout
            predictions[validation_rows] = weighted_ridge_predict(
                features[train_rows], labels[train_rows], features[validation_rows], alpha
            )
        additions, removals = boundary_candidates(v11, predictions, slices, base_mask)
        for count in TRANSFER_COUNTS:
            mask = transfer_mask(base_mask, additions, removals, count)
            stats = audit.mask_stats(mask, base_mask, labels, slices, folds)
            masks[(alpha, count)] = mask
            candidates.append({"alpha": alpha, "transfer_count": count, **stats})

    nested_mask = base_mask.copy()
    choices = []
    for heldout in range(audit.N_FOLDS):
        training_folds = [fold for fold in range(audit.N_FOLDS) if fold != heldout]
        best = max(
            candidates,
            key=lambda item: (
                sum(item["fold_deltas"][fold] for fold in training_folds),
                min(item["fold_deltas"][fold] for fold in training_folds),
                -item["transfer_count"],
                -item["alpha"],
            ),
        )
        chosen = masks[(best["alpha"], best["transfer_count"])]
        for order_index in np.flatnonzero(folds == heldout):
            nested_mask[slices[order_index]] = chosen[slices[order_index]]
        choices.append(
            {
                "heldout_fold": heldout,
                "alpha": best["alpha"],
                "transfer_count": best["transfer_count"],
                "training_delta": int(sum(best["fold_deltas"][fold] for fold in training_folds)),
                "heldout_delta": int(best["fold_deltas"][heldout]),
            }
        )

    nested_stats = audit.mask_stats(nested_mask, base_mask, labels, slices, folds)
    order_deltas = audit.per_order_delta(nested_mask, base_mask, labels, slices)
    lower = audit.bootstrap_lower(order_deltas)
    optimistic = max(
        candidates,
        key=lambda item: (item["tp_delta"], min(item["fold_deltas"]), -item["transfer_count"]),
    )
    report = {
        "method": "nested paired boundary correctness transfer",
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "target_predictions": target_count,
        "feature_dimension": int(features.shape[1]),
        "optimistic_best": optimistic,
        "nested_choices": choices,
        "nested": {**nested_stats, "bootstrap_95_lower": lower},
        "passed_mean_gate": bool(
            nested_stats["tp_delta"] >= 30
            and min(nested_stats["fold_deltas"]) >= -2
            and lower > 0
        ),
        "passed_significant_gate": False,
        "all_candidates": candidates,
    }
    output = ROOT / "experiments" / "v21_nested_boundary_transfer.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "all_candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Leakage-resistant audit of V16 graph+time count redistribution."""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
V16_DIR = ROOT / "experiments" / "v16"
V19_CLOUD_DIR = ROOT / "experiments" / "v19_cloud"
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
V11_DIR = ROOT / "codexgz" / "v11"
N_FOLDS = 5
TARGET_TEST_COUNT = 1059
MAX_ROOTCAUSES = 8


WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 0.5, 21))
COUNT_CAPS = (0, 1, 2, 7)
SEEDS = (20260803, 20260817, 20260831)


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return value or ""


def load_orders(base_dir, with_labels):
    orders = []
    for directory in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        order_id = directory.name
        topology = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        alarms = [
            node for node in topology.get("nodes", []) if node.get("@class") == "Alarm"
        ]
        roots = set()
        if with_labels:
            root_data = json.loads(
                (directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
            )
            roots = {node["@rid"] for node in root_data.get("rootcause", [])}
        orders.append(
            {
                "id": order_id,
                "topology": topology,
                "alarms": alarms,
                "roots": roots,
            }
        )
    return orders


def order_signature(order):
    title_counts = Counter(scalar(node.get("title")) for node in order["alarms"])
    target_titles = sorted(
        scalar(node.get("title"))
        for node in order["alarms"]
        if node.get("label") == "TargetAlarm"
    )
    return tuple(sorted(title_counts.items())), tuple(target_titles), len(order["alarms"])


def grouped_folds(orders):
    groups = defaultdict(list)
    for index, order in enumerate(orders):
        groups[order_signature(order)].append(index)
    fold_sizes = [0] * N_FOLDS
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ranked:
        fold = min(range(N_FOLDS), key=lambda value: (fold_sizes[value], value))
        folds[indices] = fold
        fold_sizes[fold] += len(indices)
    return folds, len(groups), fold_sizes


def exact_count_mask(scores, slices, target_count):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order_slice in slices:
        ranked = np.argsort(-scores[order_slice], kind="stable")[:MAX_ROOTCAUSES]
        selected[order_slice.start + ranked[0]] = True
        optional.extend(order_slice.start + ranked[1:])
    remaining = max(0, target_count - int(selected.sum()))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def build_slices(orders):
    slices = []
    cursor = 0
    for order in orders:
        stop = cursor + len(order["alarms"])
        slices.append(slice(cursor, stop))
        cursor = stop
    return slices


def capped_count_mask(scores, slices, base_mask, target_count, cap):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order_slice in slices:
        base_k = int(base_mask[order_slice].sum())
        minimum = max(1, base_k - cap)
        maximum = min(MAX_ROOTCAUSES, order_slice.stop - order_slice.start, base_k + cap)
        ranked = np.argsort(-scores[order_slice], kind="stable")
        selected[order_slice.start + ranked[:minimum]] = True
        optional.extend((order_slice.start + ranked[minimum:maximum]).tolist())
    remaining = target_count - int(selected.sum())
    if remaining < 0 or remaining > len(optional):
        raise ValueError((cap, remaining, len(optional)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def foldwise_capped_count_mask(scores, slices, base_mask, folds, cap):
    """Redistribute counts only within each independently trained OOF fold."""
    selected = np.zeros(len(scores), dtype=bool)
    for fold in range(N_FOLDS):
        order_indices = np.flatnonzero(folds == fold)
        fold_slices = [slices[index] for index in order_indices]
        fold_target = int(sum(base_mask[order_slice].sum() for order_slice in fold_slices))
        fold_mask = capped_count_mask(scores, fold_slices, base_mask, fold_target, cap)
        for order_slice in fold_slices:
            selected[order_slice] = fold_mask[order_slice]
    if int(selected.sum()) != int(base_mask.sum()):
        raise ValueError((int(selected.sum()), int(base_mask.sum())))
    return selected


def per_order_delta(mask, base_mask, labels, slices):
    values = np.zeros(len(slices), dtype=np.int16)
    for index, order_slice in enumerate(slices):
        values[index] = int(np.sum(mask[order_slice] & (labels[order_slice] == 1))) - int(
            np.sum(base_mask[order_slice] & (labels[order_slice] == 1))
        )
    return values


def mask_stats(mask, base_mask, labels, slices, folds):
    deltas = per_order_delta(mask, base_mask, labels, slices)
    fold_deltas = [int(deltas[folds == fold].sum()) for fold in range(N_FOLDS)]
    changed = []
    count_changes = []
    for index, order_slice in enumerate(slices):
        if not np.array_equal(mask[order_slice], base_mask[order_slice]):
            changed.append(index)
        count_changes.append(int(mask[order_slice].sum() - base_mask[order_slice].sum()))
    return {
        "tp_delta": int(deltas.sum()),
        "fold_deltas": fold_deltas,
        "changed_orders": len(changed),
        "count_increased": int(sum(value > 0 for value in count_changes)),
        "count_decreased": int(sum(value < 0 for value in count_changes)),
        "max_abs_count_change": int(max(abs(value) for value in count_changes)),
    }


def bootstrap_lower(deltas, iterations=20000):
    rng = np.random.default_rng(20260803)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def main():
    train_orders = load_orders(TRAIN_DIR, True)
    test_orders = load_orders(TEST_DIR, False)
    slices = build_slices(train_orders)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in train_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    folds, group_count, fold_sizes = grouped_folds(train_orders)
    v11_scores = 0.25 * np.load(V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        V11_DIR / "v11_oof_meta.npy"
    )
    graph_mean = np.load(V19_CLOUD_DIR / "v19_graph_time_oof_mean.npy")
    graph_seeds = [
        np.load(V19_CLOUD_DIR / f"v19_graph_time_oof_seed_{seed}.npy") for seed in SEEDS
    ]
    target_count = round(TARGET_TEST_COUNT / len(test_orders) * len(train_orders))
    base_mask = exact_count_mask(v11_scores, slices, target_count)
    base_tp = int(np.sum(base_mask & (labels == 1)))

    candidates = []
    masks = {}
    for cap in COUNT_CAPS:
        for weight in WEIGHTS:
            scores = (1.0 - weight) * v11_scores + weight * graph_mean
            mask = foldwise_capped_count_mask(scores, slices, base_mask, folds, cap)
            key = (cap, weight)
            masks[key] = mask
            stats = mask_stats(mask, base_mask, labels, slices, folds)
            candidates.append({"cap": cap, "weight": weight, **stats})

    nested_mask = base_mask.copy()
    selected = []
    for heldout in range(N_FOLDS):
        train_folds = [fold for fold in range(N_FOLDS) if fold != heldout]
        best = max(
            candidates,
            key=lambda item: (
                sum(item["fold_deltas"][fold] for fold in train_folds),
                min(item["fold_deltas"][fold] for fold in train_folds),
                -item["cap"],
                -item["weight"],
            ),
        )
        order_indices = np.flatnonzero(folds == heldout)
        chosen_mask = masks[(best["cap"], best["weight"])]
        for order_index in order_indices:
            nested_mask[slices[order_index]] = chosen_mask[slices[order_index]]
        selected.append(
            {
                "heldout_fold": heldout,
                "cap": best["cap"],
                "weight": best["weight"],
                "train_fold_tp_delta": int(
                    sum(best["fold_deltas"][fold] for fold in train_folds)
                ),
                "heldout_tp_delta": int(best["fold_deltas"][heldout]),
            }
        )

    nested_stats = mask_stats(nested_mask, base_mask, labels, slices, folds)
    nested_order_deltas = per_order_delta(nested_mask, base_mask, labels, slices)
    seed_results = []
    for seed, graph_scores in zip(SEEDS, graph_seeds):
        seed_mask = base_mask.copy()
        for choice in selected:
            heldout = choice["heldout_fold"]
            scores = (1.0 - choice["weight"]) * v11_scores + choice["weight"] * graph_scores
            choice_mask = foldwise_capped_count_mask(
                scores, slices, base_mask, folds, choice["cap"]
            )
            for order_index in np.flatnonzero(folds == heldout):
                seed_mask[slices[order_index]] = choice_mask[slices[order_index]]
        seed_results.append(
            {"seed": seed, **mask_stats(seed_mask, base_mask, labels, slices, folds)}
        )
    global_best = max(
        candidates,
        key=lambda item: (
            item["tp_delta"],
            min(item["fold_deltas"]),
            -item["cap"],
            -item["weight"],
        ),
    )
    report = {
        "method": "five-fold nested weight/count-cap selection with per-fold score calibration",
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "baseline_tp": base_tp,
        "target_predictions": target_count,
        "weights": WEIGHTS,
        "count_caps": COUNT_CAPS,
        "global_best_optimistic": global_best,
        "nested_choices": selected,
        "nested": {
            **nested_stats,
            "bootstrap_95_lower": bootstrap_lower(nested_order_deltas),
        },
        "seed_results_using_nested_choices": seed_results,
        "passed_mean_gate": bool(
            nested_stats["tp_delta"] >= 30
            and min(nested_stats["fold_deltas"]) >= -2
            and bootstrap_lower(nested_order_deltas) > 0
        ),
        "passed_significant_gate": bool(
            nested_stats["tp_delta"] >= 30
            and min(item["tp_delta"] for item in seed_results) >= 24
            and min(nested_stats["fold_deltas"]) >= -2
            and bootstrap_lower(nested_order_deltas) > 0
        ),
        "all_candidates": candidates,
    }
    output = ROOT / "experiments" / "v19_nested_count_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "all_candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

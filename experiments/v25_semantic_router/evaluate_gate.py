"""Ensemble router actions, run V25 gates, and persist final masks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from actions import ACTION_NAMES, Action, apply_actions, exact_dp
from oracle_audit import load_semantic
from v25_common import Bundle, bootstrap_lower, exact_count_mask, fixed_k_mask, read_json, write_json


def action_map(archive, prefix):
    rows = archive[f"{prefix}_action_order"]
    names = archive[f"{prefix}_action_name"]
    adds = archive[f"{prefix}_action_add"]
    removes = archive[f"{prefix}_action_remove"]
    utilities = archive[f"{prefix}_action_utility"]
    output = {}
    for order, name, add, remove, utility in zip(rows, names, adds, removes, utilities):
        add = tuple(int(value) for value in add if value >= 0)
        remove = tuple(int(value) for value in remove if value >= 0)
        key = (int(order), int(name), add, remove)
        output[key] = float(utility)
    return output


def ensemble_actions(archives, prefix, order_count):
    mappings = [action_map(archive, prefix) for archive in archives]
    common = set(mappings[0]).intersection(*(set(value) for value in mappings[1:]))
    groups = [[] for _ in range(order_count)]
    utilities = [[] for _ in range(order_count)]
    for key in sorted(common):
        order, name, add, remove = key
        action = Action(
            order=order,
            name=ACTION_NAMES[name],
            add=add,
            remove=remove,
            delta=len(add) - len(remove),
            features=np.zeros(0, dtype=np.float32),
        )
        groups[order].append(action)
        utilities[order].append(float(np.mean([mapping[key] for mapping in mappings])))
    for order, group in enumerate(groups):
        if not any(action.name == "keep" for action in group):
            raise ValueError(f"seed action intersection lost keep for order {order}")
    _, chosen = exact_dp(groups, utilities, 0)
    return chosen, groups, utilities


def per_order_delta(mask, base, labels, ptr):
    output = np.zeros(len(ptr) - 1, dtype=np.int16)
    for index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        output[index] = int(labels[start:stop][mask[start:stop]].sum()) - int(
            labels[start:stop][base[start:stop]].sum()
        )
    return output


def distribution_check(bundle, semantic_dir, retrieval_dir, seeds, chosen, test_mask):
    train_semantics, test_semantics, train_retrieval, test_retrieval = [], [], [], []
    for seed in seeds:
        semantic, _, test, _ = load_semantic(
            semantic_dir, seed, bundle.arrays["train_folds"], len(bundle.arrays["train_labels"]),
            len(bundle.arrays["test_v11"]), len(bundle.records["train"]), len(bundle.records["test"]),
        )
        retrieval = np.load(retrieval_dir / f"retrieval_seed_{seed}.npz")
        train_semantics.append(semantic)
        test_semantics.append(test)
        train_retrieval.append(np.isfinite(retrieval["oof_probability"]).astype(float))
        test_retrieval.append(np.isfinite(retrieval["test_probability"]).astype(float))
    train_values = {
        "semantic_confidence": np.mean(train_semantics, axis=0),
        "retrieval_coverage": np.mean(train_retrieval, axis=0),
    }
    test_values = {
        "semantic_confidence": np.mean(test_semantics, axis=0),
        "retrieval_coverage": np.mean(test_retrieval, axis=0),
    }
    checks, passed = {}, True
    for name in train_values:
        low, high = np.quantile(train_values[name], [0.025, 0.975])
        test_median = float(np.median(test_values[name]))
        outlier_share = float(np.mean((test_values[name] < low) | (test_values[name] > high)))
        ok = bool(low <= test_median <= high and outlier_share <= 0.10)
        checks[name] = {
            "oof_95_interval": [float(low), float(high)],
            "test_median": test_median,
            "test_outlier_share": outlier_share,
            "passed": ok,
        }
        passed &= ok
    train_base = exact_count_mask(bundle.arrays["train_v11"], bundle.ptr("train"), 3169)
    train_counts = np.asarray(
        [train_base[int(a) : int(b)].sum() for a, b in zip(bundle.ptr("train")[:-1], bundle.ptr("train")[1:])]
    )
    test_counts = np.asarray(
        [test_mask[int(a) : int(b)].sum() for a, b in zip(bundle.ptr("test")[:-1], bundle.ptr("test")[1:])]
    )
    checks["root_count"] = {
        "train_95_interval": [float(np.quantile(train_counts, 0.025)), float(np.quantile(train_counts, 0.975))],
        "test_95_interval": [float(np.quantile(test_counts, 0.025)), float(np.quantile(test_counts, 0.975))],
        "passed": bool(test_counts.min() >= 1 and test_counts.max() <= 8),
    }
    passed &= checks["root_count"]["passed"]
    return {"passed": bool(passed), "checks": checks}


def chosen_arrays(chosen):
    return {
        "order": np.asarray([item.order for item in chosen], dtype=np.int32),
        "name": np.asarray([ACTION_NAMES.index(item.name) for item in chosen], dtype=np.int8),
        "add": np.asarray([list(item.add) + [-1] * (2 - len(item.add)) for item in chosen], dtype=np.int32),
        "remove": np.asarray([list(item.remove) + [-1] * (2 - len(item.remove)) for item in chosen], dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--mode", choices=("probe", "final"), default="final")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = read_json(args.config)
    seeds = args.seeds or (config["seeds"][:1] if args.mode == "probe" else config["seeds"])
    bundle = Bundle.load(args.data_root)
    archives = [np.load(args.router_dir / f"router_seed_{seed}.npz") for seed in seeds]
    labels = bundle.arrays["train_labels"]
    train_base = exact_count_mask(
        bundle.arrays["train_v11"], bundle.ptr("train"), config["target_train_predictions"]
    )
    test_base = bundle.arrays["test_champion_mask"].astype(bool)
    oof_chosen, _, _ = ensemble_actions(archives, "oof", len(bundle.records["train"]))
    test_chosen, _, _ = ensemble_actions(archives, "test", len(bundle.records["test"]))
    oof_mask = apply_actions(train_base, oof_chosen)
    test_mask = apply_actions(test_base, test_chosen)
    deltas = per_order_delta(oof_mask, train_base, labels, bundle.ptr("train"))
    baseline_tp = int(labels[train_base].sum())
    ensemble_tp = int(labels[oof_mask].sum())
    folds = bundle.arrays["train_folds"]
    fold_deltas = [int(deltas[folds == fold].sum()) for fold in range(config["folds"])]
    seed_results = []
    for seed, archive in zip(seeds, archives):
        mask = archive["oof_mask"].astype(bool)
        seed_results.append({"seed": seed, "tp": int(labels[mask].sum()), "tp_delta": int(labels[mask].sum()) - baseline_tp})
    ordered = np.argsort([order["fault_time"] for order in bundle.records["train"]])
    late = ordered[int(len(ordered) * 0.8) :]
    stress = {
        "primary_template_folds": fold_deltas,
        "station_group_folds": [
            int(deltas[bundle.arrays["train_station_folds"] == fold].sum()) for fold in range(config["folds"])
        ],
        "legacy_template_folds": [
            int(deltas[bundle.arrays["train_legacy_folds"] == fold].sum()) for fold in range(config["folds"])
        ],
        "latest_20_percent_delta": int(deltas[late].sum()),
    }
    distribution = distribution_check(
        bundle, args.semantic_dir, args.retrieval_dir, seeds, test_chosen, test_mask
    )
    bootstrap = bootstrap_lower(deltas)
    if args.mode == "probe":
        gate = config["probe_gate"]
        passed = ensemble_tp - baseline_tp >= gate["minimum_tp_delta"] and min(fold_deltas) >= gate["minimum_fold_delta"]
    else:
        gate = config["final_gate"]
        passed = bool(
            ensemble_tp >= gate["minimum_mean_tp"]
            and ensemble_tp - baseline_tp >= gate["minimum_mean_tp_delta"]
            and min(item["tp_delta"] for item in seed_results) >= gate["minimum_seed_tp_delta"]
            and min(fold_deltas) >= gate["minimum_fold_delta"]
            and bootstrap >= gate["minimum_bootstrap_lower"]
            and min(stress["legacy_template_folds"]) >= 0
            and min(stress["station_group_folds"]) >= 0
            and stress["latest_20_percent_delta"] >= 0
            and distribution["passed"]
        )
    # Rank-only diagnostic keeps the champion count of every order.
    semantic_tests = []
    for seed in seeds:
        _, _, test_semantic, _ = load_semantic(
            args.semantic_dir, seed, folds, len(labels), len(bundle.arrays["test_v11"]),
            len(bundle.records["train"]), len(bundle.records["test"]),
        )
        semantic_tests.append(test_semantic)
    champion_counts = [
        int(test_base[int(a) : int(b)].sum()) for a, b in zip(bundle.ptr("test")[:-1], bundle.ptr("test")[1:])
    ]
    rank_only_mask = fixed_k_mask(np.mean(semantic_tests, axis=0), bundle.ptr("test"), champion_counts)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = chosen_arrays(test_chosen)
    np.savez_compressed(
        args.output / "v25_final_masks.npz",
        oof_mask=oof_mask,
        test_mask=test_mask,
        rank_only_mask=rank_only_mask,
        **{f"chosen_{key}": value for key, value in selected.items()},
    )
    report = {
        "version": config["version"],
        "mode": args.mode,
        "status": "gate_passed" if passed else "gate_failed",
        "baseline_tp": baseline_tp,
        "ensemble_tp": ensemble_tp,
        "ensemble_tp_delta": ensemble_tp - baseline_tp,
        "fold_deltas": fold_deltas,
        "bootstrap_95_lower": bootstrap,
        "seed_results": seed_results,
        "stress_tests": stress,
        "distribution_shift": distribution,
        "oof_changed_orders": sum(item.name != "keep" for item in oof_chosen),
        "test_changed_orders": sum(item.name != "keep" for item in test_chosen),
        "test_predictions": int(test_mask.sum()),
        "requirements": gate,
        "submission_generated": False,
    }
    write_json(args.output / "v25_probe_report.json", report)
    print(report)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

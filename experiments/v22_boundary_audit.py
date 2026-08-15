"""Corrected cap-specific audit for persisted V22 OOF probabilities."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT / "experiments"))

import v19_nested_count_audit as audit  # noqa: E402
import v22_action_dp as v22  # noqa: E402


def fixed_cap_decode(
    cap, add_p, remove_p, add_rows, remove_rows, base_mask, labels, slices, folds
):
    stitched = base_mask.copy()
    choices = []
    actions_all = []
    for heldout in range(v22.N_FOLDS):
        training_folds = [fold for fold in range(v22.N_FOLDS) if fold != heldout]
        candidates = []
        for penalty in v22.PENALTIES:
            deltas = []
            for fold in training_folds:
                indices = np.flatnonzero(folds == fold)
                actions = v22.exact_dp(
                    indices, add_p, remove_p, add_rows, remove_rows, cap, penalty
                )
                mask = v22.apply_actions(base_mask, actions)
                deltas.append(
                    v22.action_stats(base_mask, mask, labels, slices, indices)
                )
            candidates.append(
                {
                    "penalty": penalty,
                    "train_delta": int(sum(deltas)),
                    "train_worst": int(min(deltas)),
                }
            )
        choice = max(
            candidates,
            key=lambda item: (
                item["train_delta"],
                item["train_worst"],
                item["penalty"],
            ),
        )
        indices = np.flatnonzero(folds == heldout)
        actions = v22.exact_dp(
            indices,
            add_p,
            remove_p,
            add_rows,
            remove_rows,
            cap,
            choice["penalty"],
        )
        fold_mask = v22.apply_actions(base_mask, actions)
        for order_index in indices:
            stitched[slices[int(order_index)]] = fold_mask[slices[int(order_index)]]
        heldout_delta = v22.action_stats(
            base_mask, fold_mask, labels, slices, indices
        )
        choices.append(
            {"heldout": heldout, **choice, "heldout_delta": int(heldout_delta)}
        )
        actions_all.extend(item for item in actions if item["name"] != "keep")
    order_deltas = audit.per_order_delta(stitched, base_mask, labels, slices)
    return {
        "tp_delta": int(order_deltas.sum()),
        "fold_deltas": [
            int(order_deltas[folds == fold].sum()) for fold in range(v22.N_FOLDS)
        ],
        "changed_orders": len(actions_all),
        "bootstrap_95_lower": audit.bootstrap_lower(order_deltas),
        "choices": choices,
    }


def main():
    output_dir = ROOT / "experiments" / "v22"
    orders = audit.load_orders(audit.TRAIN_DIR, True)
    slices = v22.build_slices(orders)
    folds, group_count, fold_sizes = audit.grouped_folds(orders)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    raw = np.load(v22.V16_DIR / "v16_train_features.npy", mmap_mode="r")[:, :97]
    v11 = 0.25 * np.load(v22.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        v22.V11_DIR / "v11_oof_meta.npy"
    )
    graph = np.load(v22.V19_DIR / "v19_graph_time_oof_mean.npy")
    base_mask = audit.exact_count_mask(v11, slices, v22.TARGET_TRAIN_COUNT)
    tables = v22.build_boundary_table(
        orders, slices, raw, v11, graph, base_mask, labels=labels
    )
    _, add_rows = v22.matrix_by_order(
        tables["add"], np.zeros(len(tables["add"]["y"])), len(orders)
    )
    _, remove_rows = v22.matrix_by_order(
        tables["remove"], np.zeros(len(tables["remove"]["y"])), len(orders)
    )

    seed_probabilities = []
    for seed in v22.SEEDS:
        seed_probabilities.append(
            {
                "seed": seed,
                "add": np.load(output_dir / f"v22_add_oof_seed_{seed}.npy"),
                "remove": np.load(output_dir / f"v22_remove_oof_seed_{seed}.npy"),
            }
        )
    mean_add = v22.nanmean_matrices([item["add"] for item in seed_probabilities])
    mean_remove = v22.nanmean_matrices(
        [item["remove"] for item in seed_probabilities]
    )

    cap_results = {}
    for cap in v22.CAPS:
        seeds = []
        for item in seed_probabilities:
            seeds.append(
                {
                    "seed": item["seed"],
                    **fixed_cap_decode(
                        cap,
                        item["add"],
                        item["remove"],
                        add_rows,
                        remove_rows,
                        base_mask,
                        labels,
                        slices,
                        folds,
                    ),
                }
            )
        mean = fixed_cap_decode(
            cap,
            mean_add,
            mean_remove,
            add_rows,
            remove_rows,
            base_mask,
            labels,
            slices,
            folds,
        )
        cap_results[f"cap{cap}"] = {"mean": mean, "seeds": seeds}

    add_depth1 = tables["add"]["y"][tables["add"]["depths"] == 1]
    remove_depth1 = tables["remove"]["y"][tables["remove"]["depths"] == 1]
    oracle_pairs = min(int(add_depth1.sum()), int((remove_depth1 == 0).sum()))
    report = {
        "method": "corrected V22 cap-specific audit on exact-count baseline",
        "baseline": {
            "tp": int(np.sum(base_mask & (labels == 1))),
            "predictions": int(base_mask.sum()),
            "positives": int(labels.sum()),
        },
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "boundary": {
            "add_all_samples": len(tables["add"]["y"]),
            "add_all_positive_rate": float(tables["add"]["y"].mean()),
            "add_depth1_samples": len(add_depth1),
            "add_depth1_positive_rate": float(add_depth1.mean()),
            "remove_all_samples": len(tables["remove"]["y"]),
            "remove_all_positive_rate": float(tables["remove"]["y"].mean()),
            "remove_depth1_samples": len(remove_depth1),
            "remove_depth1_positive_rate": float(remove_depth1.mean()),
            "cap1_oracle_pair_upper_bound_tp": oracle_pairs,
        },
        "caps": cap_results,
        "gate": {
            "mean_tp_delta_min": 30,
            "each_seed_tp_delta_min": 24,
            "passed": False,
        },
        "status": "NO_SUBMISSION",
    }
    for result in cap_results.values():
        mean = result["mean"]
        if (
            mean["tp_delta"] >= 30
            and min(item["tp_delta"] for item in result["seeds"]) >= 24
            and min(mean["fold_deltas"]) >= -2
            and mean["bootstrap_95_lower"] > 0
        ):
            report["gate"]["passed"] = True
            report["status"] = "GATE_PASSED"
    path = output_dir / "v22_action_audit_corrected.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Cross-fitted selector for applying V17 rank-only corrections order by order."""

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
    priority_adjusted_scores,
    test_constraints,
    tp,
    write_submission,
)
from v17_data import GraphStore


ALPHAS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def build_masks(scores, seed_scores, store, base_mask, forced_in=None, forced_out=None):
    rank = fixed_k_mask(scores, store.order_ptr, base_mask, forced_in, forced_out)
    seeds = [
        fixed_k_mask(item, store.order_ptr, base_mask, forced_in, forced_out)
        for item in seed_scores
    ]
    return rank, seeds


def event_features(store, base_scores, model_scores, seed_scores, base_mask, rank_mask, seed_masks):
    rows, order_indices = [], []
    residual = model_scores - base_scores
    for order_index in range(len(store)):
        start = int(store.order_ptr[order_index])
        stop = int(store.order_ptr[order_index + 1])
        before = base_mask[start:stop]
        after = rank_mask[start:stop]
        if np.array_equal(before, after):
            continue
        removed = np.flatnonzero(before & ~after)
        added = np.flatnonzero(after & ~before)
        if not len(removed) or len(removed) != len(added):
            continue
        absolute_removed = start + removed
        absolute_added = start + added
        model_margin = float(model_scores[absolute_added].mean() - model_scores[absolute_removed].mean())
        residual_margin = float(residual[absolute_added].mean() - residual[absolute_removed].mean())
        v11_cost = float(base_scores[absolute_removed].mean() - base_scores[absolute_added].mean())
        seed_margins = np.asarray(
            [float(item[absolute_added].mean() - item[absolute_removed].mean()) for item in seed_scores],
            dtype=np.float64,
        )
        agreement = []
        exact_agreement = 0
        for seed_mask in seed_masks:
            local = seed_mask[start:stop]
            agreement.append(float(np.mean(local == after)))
            exact_agreement += int(np.array_equal(local, after))
        rows.append(
            [
                float(len(added)),
                float(before.sum()),
                model_margin,
                residual_margin,
                v11_cost,
                float(seed_margins.mean()),
                float(seed_margins.min()),
                float(seed_margins.std()),
                float(np.mean(agreement)),
                float(exact_agreement) / len(seed_masks),
            ]
        )
        order_indices.append(order_index)
    return np.asarray(rows, dtype=np.float64), np.asarray(order_indices, dtype=np.int64)


def order_deltas(mask, base_mask, store):
    values = np.zeros(len(store), dtype=np.int16)
    for index in range(len(store)):
        start = int(store.order_ptr[index])
        stop = int(store.order_ptr[index + 1])
        values[index] = tp(mask[start:stop], store.labels[start:stop]) - tp(
            base_mask[start:stop], store.labels[start:stop]
        )
    return values


def fit_ridge(x, y, alpha):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return mean, scale, weights


def predict_ridge(model, x):
    mean, scale, weights = model
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return design @ weights


def choose_hyperparameters(x, y, folds):
    best = None
    for alpha in ALPHAS:
        predictions = np.zeros(len(x), dtype=np.float64)
        for fold in sorted(set(folds.tolist())):
            train_rows = folds != fold
            valid_rows = folds == fold
            predictions[valid_rows] = predict_ridge(
                fit_ridge(x[train_rows], y[train_rows], alpha), x[valid_rows]
            )
        for threshold in THRESHOLDS:
            selected = predictions > threshold
            fold_deltas = [int(y[(folds == fold) & selected].sum()) for fold in sorted(set(folds.tolist()))]
            record = {
                "alpha": alpha,
                "threshold": threshold,
                "total_delta": int(y[selected].sum()),
                "fold_deltas": fold_deltas,
                "selected": int(selected.sum()),
            }
            key = (min(fold_deltas), record["total_delta"], -record["selected"], -alpha, -abs(threshold))
            if best is None or key > best[0]:
                best = (key, record)
    return best[1]


def apply_orders(base_mask, rank_mask, store, selected_orders):
    output = base_mask.copy()
    for order_index in selected_orders:
        start = int(store.order_ptr[order_index])
        stop = int(store.order_ptr[order_index + 1])
        output[start:stop] = rank_mask[start:stop]
    return output


def fixed_fold_deltas(mask, base_mask, train):
    values = order_deltas(mask, base_mask, train)
    return [int(values[train.folds == fold].sum()) for fold in range(5)]


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

    train_rank, train_seed_masks = build_masks(
        predictions["oof_scores"], predictions["seed_oof_scores"], train, base_mask
    )
    train_x, train_orders = event_features(
        train,
        train.v11,
        predictions["oof_scores"],
        predictions["seed_oof_scores"],
        base_mask,
        train_rank,
        train_seed_masks,
    )
    train_delta_by_order = order_deltas(train_rank, base_mask, train)
    train_y = train_delta_by_order[train_orders].astype(np.float64)
    event_folds = train.folds[train_orders]

    crossfit_predictions = np.zeros(len(train_x), dtype=np.float64)
    outer_choices = []
    for outer_fold in range(5):
        fit_rows = event_folds != outer_fold
        valid_rows = event_folds == outer_fold
        choice = choose_hyperparameters(train_x[fit_rows], train_y[fit_rows], event_folds[fit_rows])
        model = fit_ridge(train_x[fit_rows], train_y[fit_rows], choice["alpha"])
        crossfit_predictions[valid_rows] = predict_ridge(model, train_x[valid_rows])
        outer_choices.append(choice)

    final_choice = choose_hyperparameters(train_x, train_y, event_folds)
    selected_event_rows = crossfit_predictions > np.asarray(
        [outer_choices[int(fold)]["threshold"] for fold in event_folds]
    )
    selected_train_orders = train_orders[selected_event_rows]
    selective_mask = apply_orders(base_mask, train_rank, train, selected_train_orders)
    seed_selective_masks = [
        apply_orders(base_mask, seed_mask, train, selected_train_orders) for seed_mask in train_seed_masks
    ]

    champion_mask = test.arrays["test_champion_mask"].astype(bool)
    _, _, forced_in, forced_out = test_constraints(test, metadata)
    test_scores, _ = priority_adjusted_scores(
        predictions["test_scores"], test, metadata, champion_mask
    )
    adjusted_seed_scores = [
        priority_adjusted_scores(item, test, metadata, champion_mask)[0]
        for item in predictions["seed_test_scores"]
    ]
    test_rank, test_seed_masks = build_masks(
        test_scores, adjusted_seed_scores, test, champion_mask, forced_in, forced_out
    )
    test_x, test_orders = event_features(
        test,
        test.v11,
        test_scores,
        adjusted_seed_scores,
        champion_mask,
        test_rank,
        test_seed_masks,
    )
    final_model = fit_ridge(train_x, train_y, final_choice["alpha"])
    test_predictions = predict_ridge(final_model, test_x)
    selected_test_orders = test_orders[test_predictions > final_choice["threshold"]]
    test_mask = apply_orders(champion_mask, test_rank, test, selected_test_orders)

    changes, template_share = change_summary(test_mask, champion_mask, test, metadata)
    checks = {
        "rank_tp_delta": tp(selective_mask, train.labels) - base_tp,
        "seed_tp_deltas": [tp(item, train.labels) - base_tp for item in seed_selective_masks],
        "fold_tp_deltas": fixed_fold_deltas(selective_mask, base_mask, train),
        "bootstrap_95_lower": bootstrap_lower(selective_mask, base_mask, train),
        "train_changed_orders": int(len(selected_train_orders)),
        "test_changed_orders": len(changes),
        "test_max_template_share": template_share,
    }
    gates = config["oof_gate"]
    passed = (
        checks["rank_tp_delta"] >= gates["joint_min_tp_delta"]
        and min(checks["seed_tp_deltas"]) >= gates["seed_min_tp_delta"]
        and min(checks["fold_tp_deltas"]) >= gates["fold_min_tp_delta"]
        and checks["bootstrap_95_lower"] > gates["bootstrap_lower_bound"]
        and gates["test_min_changed_orders"] <= len(changes) <= gates["test_max_changed_orders"]
        and template_share <= gates["max_template_share"]
    )
    report = {
        "passed": passed,
        "baseline_tp": base_tp,
        "checks": checks,
        "required": gates,
        "final_choice": final_choice,
        "outer_choices": outer_choices,
        "train_events": int(len(train_orders)),
        "test_events": int(len(test_orders)),
        "champion_sha256": metadata["champion_sha256"],
    }
    (args.output / "selective_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if passed:
        write_submission(
            args.output / "v17_2_hgt_selective_p1059.csv",
            test_mask,
            test,
            metadata,
            metadata["champion_sha256"],
            "v17_2_selective",
            report,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

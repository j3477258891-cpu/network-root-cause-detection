"""Train one seed of the fold-honest residual action router."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from actions import ACTION_NAMES, apply_actions, exact_dp, generate_actions
from oracle_audit import load_semantic
from v25_common import Bundle, exact_count_mask, load_locks, read_json, seed_everything, write_json


def expert_data(bundle, semantic_dir, retrieval_dir, seed):
    semantic, train_counts, test_semantic, test_counts = load_semantic(
        semantic_dir,
        seed,
        bundle.arrays["train_folds"],
        len(bundle.arrays["train_labels"]),
        len(bundle.arrays["test_v11"]),
        len(bundle.records["train"]),
        len(bundle.records["test"]),
    )
    retrieval = np.load(retrieval_dir / f"retrieval_seed_{seed}.npz")
    train_retrieval = np.nan_to_num(retrieval["oof_probability"], nan=semantic)
    test_retrieval = np.nan_to_num(retrieval["test_probability"], nan=test_semantic)
    train_experts = np.column_stack(
        [
            bundle.arrays["train_v11"],
            np.nan_to_num(bundle.arrays["train_v13"], nan=bundle.arrays["train_v11"]),
            np.nan_to_num(bundle.arrays["train_v19"], nan=bundle.arrays["train_v11"]),
            semantic,
            train_retrieval,
        ]
    ).astype(np.float32)
    test_experts = np.column_stack(
        [
            bundle.arrays["test_v11"],
            np.nan_to_num(bundle.arrays["test_v13"], nan=bundle.arrays["test_v11"]),
            np.nan_to_num(bundle.arrays["test_v19"], nan=bundle.arrays["test_v11"]),
            test_semantic,
            test_retrieval,
        ]
    ).astype(np.float32)
    train_fields = {
        "probability": train_retrieval,
        "support": retrieval["oof_support"],
        "consistency": retrieval["oof_consistency"],
        "similarity": retrieval["oof_similarity"],
    }
    test_fields = {
        "probability": test_retrieval,
        "support": retrieval["test_support"],
        "consistency": retrieval["test_consistency"],
        "similarity": retrieval["test_similarity"],
    }
    return train_experts, train_counts, train_fields, test_experts, test_counts, test_fields


def lock_rows(bundle, locks):
    result = {"in": set(), "out": set()}
    ptr = bundle.ptr("test")
    for order_index, order in enumerate(bundle.records["test"]):
        start = int(ptr[order_index])
        for local, alarm in enumerate(order["alarms"]):
            key = (order["order_id"], alarm["rid"])
            for name in result:
                if key in locks[name]:
                    result[name].add(start + local)
    return result


def generate_all(bundle, split, base, experts, counts, retrieval, config, labels=None, locks=None):
    ptr = bundle.ptr(split)
    allowed_add = allowed_remove = None
    if locks is not None:
        all_rows = set(range(int(ptr[-1])))
        allowed_add = all_rows - locks["out"]
        allowed_remove = all_rows - locks["in"]
    return [
        generate_actions(
            order,
            int(start),
            int(stop),
            base,
            experts,
            counts,
            retrieval,
            config["candidate_depth"],
            config["max_action_depth"],
            labels,
            allowed_add,
            allowed_remove,
        )
        for order, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:]))
    ]


def fit_router(actions, order_mask, config, seed):
    rows = [action for group in actions for action in group if order_mask[action.order]]
    x = np.stack([action.features for action in rows])
    y = np.asarray([action.gain for action in rows], dtype=np.float32)
    extra = ExtraTreesRegressor(
        n_estimators=config["extra_trees"],
        max_depth=16,
        min_samples_leaf=3,
        max_features=0.75,
        n_jobs=int(os.environ.get("V25_ROUTER_JOBS", "-1")),
        random_state=seed,
    )
    hist = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=config["hist_iterations"],
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed + 101,
    )
    extra.fit(x, y)
    hist.fit(x, y)
    return extra, hist


def predict_router(models, groups, config):
    extra, hist = models
    output = []
    for group in groups:
        x = np.stack([action.features for action in group])
        values = (
            config["blend_extra_trees"] * extra.predict(x)
            + config["blend_hist"] * hist.predict(x)
        )
        output.append(values)
    return output


def serialize(groups, utilities):
    order, name, add, remove, utility = [], [], [], [], []
    for group, scores in zip(groups, utilities):
        for action, score in zip(group, scores):
            order.append(action.order)
            name.append(ACTION_NAMES.index(action.name))
            add.append(list(action.add) + [-1] * (2 - len(action.add)))
            remove.append(list(action.remove) + [-1] * (2 - len(action.remove)))
            utility.append(score)
    return {
        "order": np.asarray(order, dtype=np.int32),
        "name": np.asarray(name, dtype=np.int8),
        "add": np.asarray(add, dtype=np.int32),
        "remove": np.asarray(remove, dtype=np.int32),
        "utility": np.asarray(utility, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--locks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    full_config = read_json(args.config)
    config = full_config["router"]
    seed_everything(args.seed)
    bundle = Bundle.load(args.data_root)
    train_experts, train_counts, train_retrieval, test_experts, test_counts, test_retrieval = expert_data(
        bundle, args.semantic_dir, args.retrieval_dir, args.seed
    )
    train_base = exact_count_mask(
        bundle.arrays["train_v11"], bundle.ptr("train"), full_config["target_train_predictions"]
    )
    test_base = bundle.arrays["test_champion_mask"].astype(bool)
    locks = lock_rows(bundle, load_locks(args.locks))
    train_actions = generate_all(
        bundle, "train", train_base, train_experts, train_counts, train_retrieval,
        config, labels=bundle.arrays["train_labels"],
    )
    test_actions = generate_all(
        bundle, "test", test_base, test_experts, test_counts, test_retrieval,
        config, locks=locks,
    )
    folds = bundle.arrays["train_folds"]
    oof_utilities = [np.zeros(len(group), dtype=np.float32) for group in train_actions]
    fold_deltas = []
    labels = bundle.arrays["train_labels"]
    for fold in range(full_config["folds"]):
        models = fit_router(train_actions, folds != fold, config, args.seed + fold * 31)
        indices = np.flatnonzero(folds == fold)
        groups = [train_actions[index] for index in indices]
        values = predict_router(models, groups, config)
        for index, predicted in zip(indices, values):
            oof_utilities[int(index)] = predicted
        _, chosen = exact_dp(groups, values, 0)
        fold_mask = apply_actions(train_base, chosen)
        rows = bundle.rows_for_orders("train", indices)
        fold_deltas.append(int(labels[rows][fold_mask[rows]].sum() - labels[rows][train_base[rows]].sum()))
    _, oof_chosen = exact_dp(train_actions, oof_utilities, 0)
    oof_mask = apply_actions(train_base, oof_chosen)
    models = fit_router(train_actions, np.ones(len(train_actions), dtype=bool), config, args.seed + 1009)
    test_utilities = predict_router(models, test_actions, config)
    _, test_chosen = exact_dp(test_actions, test_utilities, 0)
    test_mask = apply_actions(test_base, test_chosen)
    args.output.mkdir(parents=True, exist_ok=True)
    train_serialized = serialize(train_actions, oof_utilities)
    test_serialized = serialize(test_actions, test_utilities)
    np.savez_compressed(
        args.output / f"router_seed_{args.seed}.npz",
        oof_mask=oof_mask,
        test_mask=test_mask,
        **{f"oof_action_{key}": value for key, value in train_serialized.items()},
        **{f"test_action_{key}": value for key, value in test_serialized.items()},
    )
    baseline_tp = int(labels[train_base].sum())
    report = {
        "seed": args.seed,
        "baseline_tp": baseline_tp,
        "oof_tp": int(labels[oof_mask].sum()),
        "oof_tp_delta": int(labels[oof_mask].sum()) - baseline_tp,
        "fold_deltas": fold_deltas,
        "oof_changed_orders": sum(action.name != "keep" for action in oof_chosen),
        "test_changed_orders": sum(action.name != "keep" for action in test_chosen),
        "test_predictions": int(test_mask.sum()),
    }
    write_json(args.output / f"router_seed_{args.seed}.json", report)
    print(report)


if __name__ == "__main__":
    main()

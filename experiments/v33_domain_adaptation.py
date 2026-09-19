"""Diagnose domain shift and train importance-weighted root-cause models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from v30_meta_stack import exact_count_mask, feature_matrix


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
OUT = ROOT / "experiments/v33_domain_adaptation"
TARGET_TRAIN_P = 3103
SEED = 20260820
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def domain_probabilities(x_train: np.ndarray, x_test: np.ndarray, train_ptr, test_ptr):
    x = np.vstack([x_train, x_test])
    target = np.r_[np.zeros(len(x_train), dtype=np.int8),
                   np.ones(len(x_test), dtype=np.int8)]
    train_groups = np.repeat(np.arange(len(train_ptr) - 1), np.diff(train_ptr))
    test_groups = np.repeat(
        np.arange(len(test_ptr) - 1) + len(train_ptr) - 1, np.diff(test_ptr)
    )
    groups = np.r_[train_groups, test_groups]
    probability = np.zeros(len(x), dtype=np.float32)
    fold_aucs = []
    for fold, (fit_rows, valid_rows) in enumerate(
        GroupKFold(n_splits=5).split(x, target, groups)
    ):
        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=SEED + fold,
        )
        model.fit(x[fit_rows], target[fit_rows])
        probability[valid_rows] = model.predict_proba(x[valid_rows])[:, 1]
        fold_aucs.append(float(roc_auc_score(target[valid_rows], probability[valid_rows])))
    return probability[:len(x_train)], probability[len(x_train):], fold_aucs


def importance_weights(probability: np.ndarray):
    probability = np.clip(probability.astype(np.float64), 0.01, 0.99)
    ratio = probability / (1.0 - probability)
    ratio = np.clip(ratio, 0.05, 20.0)
    return (ratio / ratio.mean()).astype(np.float32)


def root_model(alpha: float, fold: int):
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED + 100 + fold + round(alpha * 1000),
    )


def crossfit_root(x_train, labels, x_test, order_rows, folds, weights, alpha):
    oof = np.zeros(len(labels), dtype=np.float32)
    test_parts = []
    weighted = np.power(weights, alpha).astype(np.float32)
    weighted /= weighted.mean()
    for fold in range(5):
        fit_rows = folds[order_rows] != fold
        valid_rows = ~fit_rows
        model = root_model(alpha, fold)
        model.fit(x_train[fit_rows], labels[fit_rows], sample_weight=weighted[fit_rows])
        oof[valid_rows] = model.predict_proba(x_train[valid_rows])[:, 1]
        test_parts.append(model.predict_proba(x_test)[:, 1])
    return oof, np.mean(test_parts, axis=0).astype(np.float32)


def fixed_order_reorder(scores, base, ptr):
    proposed = np.zeros(len(scores), dtype=bool)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        count = int(base[start:stop].sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[:count]
        proposed[start + ranked] = True
    return proposed


def evaluate(scores, arrays, labels, base, domain_train):
    ptr = arrays["train_alarm_ptr"]
    mask = exact_count_mask(scores, ptr, TARGET_TRAIN_P)
    base_tp = int((base & labels).sum())
    tp = int((mask & labels).sum())
    f1 = 2.0 * tp / (int(labels.sum()) + TARGET_TRAIN_P)

    order_domain = np.asarray([
        float(domain_train[int(start):int(stop)].mean())
        for start, stop in zip(ptr[:-1], ptr[1:])
    ])
    testlike_threshold = float(np.quantile(order_domain, 0.75))
    testlike_orders = order_domain >= testlike_threshold
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    fixed = fixed_order_reorder(scores, base, ptr)
    fixed_delta = (fixed & labels).astype(np.int8) - (base & labels).astype(np.int8)
    testlike_delta = int(fixed_delta[testlike_orders[order_rows]].sum())
    other_delta = int(fixed_delta[~testlike_orders[order_rows]].sum())

    station = arrays["train_station_folds"].astype(np.int8)
    fold_deltas = []
    for fold in range(5):
        rows = station[order_rows] == fold
        fold_deltas.append(int(((mask & labels).astype(np.int8)
                                - (base & labels).astype(np.int8))[rows].sum()))
    return {
        "tp": tp,
        "delta_tp": tp - base_tp,
        "f1": f1,
        "station_fold_delta_tp": fold_deltas,
        "nonnegative_station_folds": int(sum(value >= 0 for value in fold_deltas)),
        "fixed_count_testlike_quartile_delta_tp": testlike_delta,
        "fixed_count_other_orders_delta_tp": other_delta,
        "testlike_threshold": testlike_threshold,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    x_train, _ = feature_matrix(arrays, "train")
    x_test, _ = feature_matrix(arrays, "test")
    labels = arrays["train_labels"].astype(bool)
    ptr = arrays["train_alarm_ptr"]
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    base = exact_count_mask(arrays["train_v11"], ptr, TARGET_TRAIN_P)

    domain_train, domain_test, domain_aucs = domain_probabilities(
        x_train, x_test, ptr, arrays["test_alarm_ptr"]
    )
    weights = importance_weights(domain_train)
    np.save(OUT / "domain_train_probability.npy", domain_train)
    np.save(OUT / "domain_test_probability.npy", domain_test)
    np.save(OUT / "importance_weights.npy", weights)

    results = {}
    for split_name, fold_key in (
        ("station", "train_station_folds"),
        ("template", "train_folds"),
    ):
        folds = arrays[fold_key].astype(np.int8)
        for alpha in ALPHAS:
            key = f"{split_name}_alpha_{alpha:g}"
            oof_path = OUT / f"{key}_oof.npy"
            test_path = OUT / f"{key}_test.npy"
            if oof_path.exists() and test_path.exists():
                print(f"loading {key}", flush=True)
                oof, test = np.load(oof_path), np.load(test_path)
            else:
                print(f"training {key}", flush=True)
                oof, test = crossfit_root(
                    x_train, labels.astype(np.int8), x_test, order_rows,
                    folds, weights, alpha
                )
                np.save(oof_path, oof)
                np.save(test_path, test)
            results[key] = evaluate(oof, arrays, labels, base, domain_train)

    eligible = [
        (key, value) for key, value in results.items()
        if value["nonnegative_station_folds"] >= 4
        and value["fixed_count_testlike_quartile_delta_tp"] >= 0
    ]
    eligible.sort(key=lambda item: (
        item[1]["tp"], item[1]["fixed_count_testlike_quartile_delta_tp"]
    ), reverse=True)
    report = {
        "version": "v33-domain-adaptation-1",
        "target_train_predictions": TARGET_TRAIN_P,
        "train_positive_count": int(labels.sum()),
        "base_tp": int((base & labels).sum()),
        "base_f1": 2.0 * int((base & labels).sum()) / (int(labels.sum()) + TARGET_TRAIN_P),
        "domain_classifier": {
            "fold_auc": domain_aucs,
            "mean_auc": float(np.mean(domain_aucs)),
            "train_probability_quantiles": {
                str(q): float(np.quantile(domain_train, q))
                for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
            },
            "weight_quantiles": {
                str(q): float(np.quantile(weights, q))
                for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
            },
        },
        "results": results,
        "eligible_models": [key for key, _ in eligible],
        "best_eligible": eligible[0][0] if eligible else None,
        "gate": {
            "requires_f1": 0.94,
            "passed": bool(eligible and eligible[0][1]["f1"] >= 0.94),
            "reason": "full candidate is withheld unless robust OOF reaches 0.94",
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

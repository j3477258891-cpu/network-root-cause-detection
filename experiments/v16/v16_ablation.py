"""OOF ablation for independent V16 feature groups."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).parent))
import v16_structure_signal as v16


FEATURE_GROUPS = {
    "graph_only": slice(0, 72),
    "time_only": slice(72, 97),
    "addinfo_only": slice(97, 252),
    "graph_time": slice(0, 97),
}


def main():
    train_orders = v16.load_orders(v16.TRAIN_DIR, True)
    test_orders = v16.load_orders(v16.TEST_DIR, False)
    train_x = np.load(v16.OUTPUT_DIR / "v16_train_features.npy")
    train_slices = []
    cursor = 0
    labels = []
    for order in train_orders:
        next_cursor = cursor + len(order["alarms"])
        train_slices.append(slice(cursor, next_cursor))
        cursor = next_cursor
        labels.extend(int(node["@rid"] in order["roots"]) for node in order["alarms"])
    labels = np.asarray(labels, dtype=np.int8)
    folds, _, fold_sizes = v16.grouped_folds(train_orders)
    all_orders = np.arange(len(train_orders))
    v11_scores = 0.25 * np.load(v16.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        v16.V11_DIR / "v11_oof_meta.npy"
    )
    target_count = round(v16.TARGET_TEST_COUNT / len(test_orders) * len(train_orders))
    base_mask = v16.exact_count_mask(v11_scores, train_slices, target_count)
    base_metrics = v16.metrics(base_mask, labels)
    report = {"fold_sizes": fold_sizes, "base": base_metrics, "groups": {}}

    for name, feature_slice in FEATURE_GROUPS.items():
        features = train_x[:, feature_slice]
        per_seed = [np.zeros(len(labels), dtype=np.float64) for _ in v16.SEEDS]
        for heldout in range(v16.N_FOLDS):
            train_order_indices = all_orders[folds != heldout]
            validation_order_indices = all_orders[folds == heldout]
            train_rows = v16.rows_for_orders(train_slices, train_order_indices)
            validation_rows = v16.rows_for_orders(train_slices, validation_order_indices)
            for seed_index, seed in enumerate(v16.SEEDS):
                model = ExtraTreesClassifier(
                    n_estimators=350,
                    max_depth=16,
                    min_samples_leaf=3,
                    max_features=0.75,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed + heldout,
                )
                model.fit(features[train_rows], labels[train_rows])
                per_seed[seed_index][validation_rows] = model.predict_proba(
                    features[validation_rows]
                )[:, 1]
        scores = np.mean(per_seed, axis=0)
        np.save(v16.OUTPUT_DIR / f"v16_{name}_oof.npy", scores)
        structure_mask = v16.exact_count_mask(scores, train_slices, target_count)
        fixed_mask = v16.fixed_k_mask(scores, train_slices, base_mask)
        scan = []
        for weight in np.linspace(0.05, 0.5, 10):
            blend = (1.0 - weight) * v11_scores + weight * scores
            global_mask = v16.exact_count_mask(blend, train_slices, target_count)
            fixed_blend = v16.fixed_k_mask(blend, train_slices, base_mask)
            scan.append(
                {
                    "weight": float(weight),
                    "global": v16.metrics(global_mask, labels),
                    "fixed_k": v16.metrics(fixed_blend, labels),
                }
            )
        catalog = v16.swap_catalog(
            train_orders,
            train_slices,
            scores,
            per_seed,
            base_mask,
            labels=labels,
        )
        order_index = {order["id"]: index for index, order in enumerate(train_orders)}
        for item in catalog:
            item["order_index"] = order_index[item["order_id"]]
        swap_validation = v16.top_candidate_validation(
            catalog, folds, "seed_min_margin", counts=(1, 2, 3, 5)
        )
        report["groups"][name] = {
            "feature_dim": int(features.shape[1]),
            "correlation": {
                "pearson": float(np.corrcoef(v11_scores, scores)[0, 1]),
                "spearman": float(
                    np.corrcoef(v16.rank_values(v11_scores), v16.rank_values(scores))[0, 1]
                ),
            },
            "standalone": v16.metrics(structure_mask, labels),
            "fixed_v11_k": v16.metrics(fixed_mask, labels),
            "weight_scan": scan,
            "raw_swap_validation": swap_validation,
        }
        print(name, json.dumps(report["groups"][name], ensure_ascii=False), flush=True)

    (v16.OUTPUT_DIR / "v16_ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

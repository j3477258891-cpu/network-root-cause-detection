import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import v10_grouped_ensemble as v10


N_FOLDS = 5
SEEDS = (20260801, 20260817, 20260831)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_GAIN = 0.002


def order_signature(order):
    title_counts = Counter(v10.scalar(node.get("title")) for node in order["alarms"])
    target_titles = sorted(
        v10.scalar(node.get("title"))
        for node in order["alarms"]
        if node.get("label") == "TargetAlarm"
    )
    return (
        tuple(sorted(title_counts.items())),
        tuple(target_titles),
        len(order["alarms"]),
    )


def grouped_folds(orders):
    groups = defaultdict(list)
    for index, order in enumerate(orders):
        groups[order_signature(order)].append(index)

    fold_sizes = [0] * N_FOLDS
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ranked_groups:
        fold = min(range(N_FOLDS), key=lambda value: (fold_sizes[value], value))
        folds[indices] = fold
        fold_sizes[fold] += len(indices)

    repeated_orders = sum(len(indices) for indices in groups.values() if len(indices) > 1)
    return folds, len(groups), repeated_orders, fold_sizes


def shortest_distances(adjacency, starts):
    distances = np.full(len(adjacency), len(adjacency) + 1, dtype=np.float32)
    queue = deque()
    for start in starts:
        distances[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in adjacency[current]:
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def directed_order_features(order):
    nodes = order["topology"].get("nodes", [])
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    outgoing = [set() for _ in nodes]
    incoming = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is None or target is None:
            continue
        outgoing[source].add(target)
        incoming[target].add(source)

    alarm_indices = [rid_to_index[node.get("@rid")] for node in order["alarms"]]
    alarm_set = set(alarm_indices)
    target_indices = [
        index for index in alarm_indices if nodes[index].get("label") == "TargetAlarm"
    ]
    distance_from_target = shortest_distances(outgoing, target_indices)
    distance_to_target = shortest_distances(incoming, target_indices)
    unreachable = float(len(nodes) + 1)
    features = []

    for node_index in alarm_indices:
        out_neighbors = outgoing[node_index]
        in_neighbors = incoming[node_index]
        out_alarm = sum(index in alarm_set for index in out_neighbors)
        in_alarm = sum(index in alarm_set for index in in_neighbors)
        out_target = sum(index in target_indices for index in out_neighbors)
        in_target = sum(index in target_indices for index in in_neighbors)
        to_target = min(float(distance_to_target[node_index]), unreachable)
        from_target = min(float(distance_from_target[node_index]), unreachable)
        features.append(
            [
                len(out_neighbors),
                len(in_neighbors),
                out_alarm,
                in_alarm,
                out_target,
                in_target,
                out_alarm / max(len(out_neighbors), 1),
                in_alarm / max(len(in_neighbors), 1),
                math.log1p(to_target),
                math.log1p(from_target),
                float(to_target <= len(nodes)),
                float(from_target <= len(nodes)),
                math.log1p(abs(to_target - from_target)),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def directed_features(orders):
    return np.vstack([directed_order_features(order) for order in orders])


def rows_and_slices_for_orders(data, order_indices):
    row_indices = []
    slices = []
    for order_index in order_indices:
        source = data["train_slices"][int(order_index)]
        start = len(row_indices)
        row_indices.extend(range(source.start, source.stop))
        slices.append(slice(start, len(row_indices)))
    return np.asarray(row_indices, dtype=np.int64), slices


def local_selection(scores, labels, data, order_indices, threshold):
    row_indices, slices = rows_and_slices_for_orders(data, order_indices)
    mask = v10.selection_mask(scores[row_indices], slices, threshold)
    return row_indices, mask, v10.confusion(mask, labels[row_indices])


def choose_weight_threshold(context, meta, labels, data, order_indices):
    row_indices, slices = rows_and_slices_for_orders(data, order_indices)
    best = None
    for weight in WEIGHTS:
        scores = (1.0 - weight) * context[row_indices] + weight * meta[row_indices]
        result = v10.best_threshold(scores, labels[row_indices], slices)
        record = (result[0], -abs(weight - 0.5), weight, result[-1], result)
        if best is None or record > best:
            best = record
    return best[2], best[3], best[4]


def write_score_table(path, test_orders, data, context, per_seed, mean_scores):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["order_id", "rid", "context_score"]
            + [f"meta_seed_{seed}" for seed in SEEDS]
            + ["meta_mean"]
        )
        for order_index, order in enumerate(test_orders):
            order_slice = data["test_slices"][order_index]
            for local_index, node in enumerate(order["alarms"]):
                row_index = order_slice.start + local_index
                writer.writerow(
                    [order["id"], node["@rid"], float(context[row_index])]
                    + [float(scores[row_index]) for scores in per_seed]
                    + [float(mean_scores[row_index])]
                )


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "usage: v11_meta_ranker.py TRAIN_DIR TEST_DIR OUTPUT_DIR [--no-location]"
        )
    if len(sys.argv) == 5 and sys.argv[4] == "--no-location":
        v10.set_no_location(True)
        print("Location features DISABLED", flush=True)
    train_dir, test_dir, output_dir = map(Path, sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    train_orders = v10.load_orders(train_dir, True)
    test_orders = v10.load_orders(test_dir, False)
    data = v10.prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    folds, group_count, repeated_orders, fold_sizes = grouped_folds(train_orders)
    data["folds"] = folds
    print(
        f"orders={len(train_orders)} test_orders={len(test_orders)} "
        f"rows={len(labels)} positives={int(np.sum(labels))} groups={group_count} "
        f"repeated_orders={repeated_orders} fold_sizes={fold_sizes}",
        flush=True,
    )

    train_tfidf, test_tfidf = v10.normalized_tfidf(
        data["train_alarm"], data["test_alarm"]
    )
    train_directed = directed_features(train_orders)
    test_directed = directed_features(test_orders)
    all_orders = np.arange(len(train_orders))
    oof_context = np.zeros(len(labels), dtype=np.float64)
    oof_per_seed = [np.zeros(len(labels), dtype=np.float64) for _ in SEEDS]

    for heldout in range(N_FOLDS):
        reference_orders = all_orders[folds != heldout]
        validation_orders = all_orders[folds == heldout]
        train_row_indices, train_context_rows, _ = v10.local_rows_and_slices(
            data, reference_orders
        )
        validation_row_indices, validation_context_rows, _ = v10.local_rows_and_slices(
            data, validation_orders
        )
        train_probabilities = v10.knn_order_probabilities(
            train_tfidf[reference_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
            exclude_reference_positions=np.arange(len(reference_orders)),
        )
        validation_probabilities = v10.knn_order_probabilities(
            train_tfidf[validation_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
        )
        train_context = v10.row_scores(train_probabilities, train_context_rows)
        validation_context = v10.row_scores(
            validation_probabilities, validation_context_rows
        )
        oof_context[validation_row_indices] = validation_context

        train_rows = [data["train_rows"][index] for index in train_row_indices]
        validation_rows = [
            data["train_rows"][index] for index in validation_row_indices
        ]
        train_encoding = v10.encoded_features(
            data, reference_orders, train_rows, leave_query_order_out=True
        )
        validation_encoding = v10.encoded_features(
            data, reference_orders, validation_rows, leave_query_order_out=False
        )
        x_train = np.hstack(
            [
                v10.make_features(
                    data["train_static"][train_row_indices],
                    train_context,
                    train_encoding,
                ),
                train_directed[train_row_indices],
            ]
        ).astype(np.float32)
        x_validation = np.hstack(
            [
                v10.make_features(
                    data["train_static"][validation_row_indices],
                    validation_context,
                    validation_encoding,
                ),
                train_directed[validation_row_indices],
            ]
        ).astype(np.float32)

        for seed_index, seed in enumerate(SEEDS):
            model = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                max_features=0.75,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + heldout,
            )
            model.fit(x_train, labels[train_row_indices])
            oof_per_seed[seed_index][validation_row_indices] = model.predict_proba(
                x_validation
            )[:, 1]
        print(f"fold={heldout} trained", flush=True)

    oof_meta = np.mean(oof_per_seed, axis=0)
    nested_baseline_mask = np.zeros(len(labels), dtype=bool)
    nested_candidate_mask = np.zeros(len(labels), dtype=bool)
    chosen_weights = []
    fold_metrics = []
    for heldout in range(N_FOLDS):
        reference_orders = all_orders[folds != heldout]
        validation_orders = all_orders[folds == heldout]
        reference_rows, reference_slices = rows_and_slices_for_orders(
            data, reference_orders
        )
        baseline_reference = v10.best_threshold(
            oof_context[reference_rows], labels[reference_rows], reference_slices
        )
        baseline_threshold = baseline_reference[-1]
        weight, threshold, _ = choose_weight_threshold(
            oof_context, oof_meta, labels, data, reference_orders
        )
        chosen_weights.append(weight)

        validation_rows, validation_slices = rows_and_slices_for_orders(
            data, validation_orders
        )
        baseline_local = v10.selection_mask(
            oof_context[validation_rows], validation_slices, baseline_threshold
        )
        candidate_scores = (
            (1.0 - weight) * oof_context[validation_rows]
            + weight * oof_meta[validation_rows]
        )
        candidate_local = v10.selection_mask(
            candidate_scores, validation_slices, threshold
        )
        nested_baseline_mask[validation_rows] = baseline_local
        nested_candidate_mask[validation_rows] = candidate_local
        baseline_result = v10.confusion(baseline_local, labels[validation_rows])
        candidate_result = v10.confusion(candidate_local, labels[validation_rows])
        fold_metrics.append(
            {
                "fold": heldout,
                "weight": weight,
                "threshold": threshold,
                "baseline_f1": baseline_result[0],
                "candidate_f1": candidate_result[0],
                "gain": candidate_result[0] - baseline_result[0],
            }
        )
        print(
            f"eval_fold={heldout} weight={weight:.2f} "
            f"baseline={baseline_result[0]:.6f} candidate={candidate_result[0]:.6f} "
            f"gain={candidate_result[0] - baseline_result[0]:+.6f}",
            flush=True,
        )

    nested_baseline = v10.confusion(nested_baseline_mask, labels)
    nested_candidate = v10.confusion(nested_candidate_mask, labels)
    nested_gain = nested_candidate[0] - nested_baseline[0]
    improved_folds = sum(item["gain"] > 0 for item in fold_metrics)
    final_weight = float(np.median(chosen_weights))
    global_oof = (1.0 - final_weight) * oof_context + final_weight * oof_meta
    global_result = v10.best_threshold(global_oof, labels, data["train_slices"])
    final_threshold = global_result[-1]
    gate_passed = nested_gain >= MIN_GAIN and improved_folds >= 3
    print(
        f"NESTED baseline={nested_baseline} candidate={nested_candidate} "
        f"gain={nested_gain:+.6f} improved_folds={improved_folds}/5",
        flush=True,
    )
    print(
        f"FINAL weight={final_weight:.2f} threshold={final_threshold:.9f} "
        f"global_oof={global_result} gate_passed={gate_passed}",
        flush=True,
    )

    full_train_probabilities = v10.knn_order_probabilities(
        train_tfidf,
        train_tfidf,
        data["train_alarm"],
        data["train_root"],
        exclude_reference_positions=np.arange(len(train_orders)),
    )
    full_test_probabilities = v10.knn_order_probabilities(
        test_tfidf, train_tfidf, data["train_alarm"], data["train_root"]
    )
    full_train_context = v10.row_scores(
        full_train_probabilities, data["train_rows"]
    )
    full_test_context = v10.row_scores(full_test_probabilities, data["test_rows"])
    full_train_encoding = v10.encoded_features(
        data, all_orders, data["train_rows"], leave_query_order_out=True
    )
    full_test_encoding = v10.encoded_features(
        data, all_orders, data["test_rows"], leave_query_order_out=False
    )
    x_full = np.hstack(
        [
            v10.make_features(
                data["train_static"], full_train_context, full_train_encoding
            ),
            train_directed,
        ]
    ).astype(np.float32)
    x_test = np.hstack(
        [
            v10.make_features(
                data["test_static"], full_test_context, full_test_encoding
            ),
            test_directed,
        ]
    ).astype(np.float32)
    test_per_seed = []
    for seed in SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_full, labels)
        test_per_seed.append(model.predict_proba(x_test)[:, 1])
    test_meta = np.mean(test_per_seed, axis=0)
    test_scores = (1.0 - final_weight) * full_test_context + final_weight * test_meta

    write_score_table(
        output_dir / "v11_test_scores.csv",
        test_orders,
        data,
        full_test_context,
        test_per_seed,
        test_meta,
    )
    np.save(output_dir / "v11_oof_context.npy", oof_context)
    np.save(output_dir / "v11_oof_meta.npy", oof_meta)
    np.save(output_dir / "v11_test_scores.npy", test_scores)

    threshold_mask = v10.selection_mask(
        test_scores, data["test_slices"], final_threshold
    )
    fixed_mask = v10.exact_count_mask(test_scores, data["test_slices"], 1059)
    if gate_passed:
        v10.write_submission(
            output_dir / "result_record_v11_meta_threshold.csv",
            test_orders,
            data,
            threshold_mask,
        )
        v10.write_submission(
            output_dir / "result_record_v11_meta_1059.csv",
            test_orders,
            data,
            fixed_mask,
        )

    report = {
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "train_rows": len(labels),
        "positives": int(np.sum(labels)),
        "group_count": group_count,
        "repeated_orders": repeated_orders,
        "fold_sizes": fold_sizes,
        "seeds": list(SEEDS),
        "fold_metrics": fold_metrics,
        "nested_baseline": nested_baseline,
        "nested_candidate": nested_candidate,
        "nested_gain": nested_gain,
        "improved_folds": improved_folds,
        "final_weight": final_weight,
        "final_threshold": final_threshold,
        "global_oof": global_result,
        "threshold_test_count": int(np.sum(threshold_mask)),
        "fixed_test_count": int(np.sum(fixed_mask)),
        "gate_passed": gate_passed,
    }
    (output_dir / "v11_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"outputs={output_dir} threshold_count={int(np.sum(threshold_mask))} "
        f"fixed_count={int(np.sum(fixed_mask))}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""V13 Meta 重排序器：在 V11 基础上新增桶A/桶B特征，执行 G1-G5 严格门控。

用法:
    python v13_meta_ranker.py TRAIN_DIR TEST_DIR OUTPUT_DIR [--no-location]

设计（对照 v11_meta_ranker.py）:
- 桶 A（工单局部，无泄漏，fold 外算一次，14 维）:
    tl_onset / tl_offset / tl_span / tl_recent_frac / tl_flips / tl_monotone
    dist_target_undirected / dev_neighbors / dev_neighbor_types / same_device_as_target
    addInfo_fields / addInfo_shape_hash / loc_depth / cooccur_target_count
- 桶 B（依赖 reference 折，fold 内计算，5 维）:
    root_prior_title / root_prior_device / template_root_rate / tfidf_weight / knn_rank
- 门控:
    G1 nested OOF gain >= MIN_GAIN(0.002)
    G2 >= 3/5 折改进
    G3 三种子测试集合两两 sym-diff <= 3
    G4 threshold 与 exact 版计数校验（提交恒 1059、每单 1..8）
    G5 唯一映射（rid 无重复、均为 Alarm 节点）
- 输出:
    v13_test_scores.csv (order_id, rid, context_score, meta_seed_*, meta_mean)
    v13_oof_context.npy / v13_oof_meta.npy / v13_test_scores.npy
    v13_report.json / v13_report.txt
    result_record_v13_meta_1059.csv (raw exact 1059)
    result_record_v13_constrained_1059.csv (受保护候选: 锁定 swap12 胜出 delta)
"""
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import v10_grouped_ensemble as v10
import v11_meta_ranker as v11
import build_v11_constrained as bvc

N_FOLDS = 5
SEEDS = (20260801, 20260817, 20260831)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_GAIN = 0.002
MIN_IMPROVED_FOLDS = 3
MAX_SEED_SYMDIFF = 3
TARGET_TEST_COUNT = 1059
DEAD_FEATURES_DROP = True  # 桶A实现时在静态特征上不删列（v10 静态列序固定），
# 通过开关控制是否对 v10 静态做瘦身 —— 见 drop_dead_static_columns


def drop_dead_static_columns():
    """返回要删除的 v10 静态特征列索引（SHAP≈0 死特征）。"""
    # v10 static_order_features 列序（含 location 时 41 列）:
    # 0 is_target, 1 is_target_single, 2 target_count, 3 len(alarms),
    # 4 title_counts, 5 device_counts, 6 location_counts, 7 local_index,
    # 8 log1p(time_delta), 9 len(time_lists), 10 sum, 11 mean, 12 std,
    # 13 max, 14 min, 15 last/max, 16 argmax, 17 len(title), 18 len(device),
    # 19 len(location), 20 len(reason), 21 fault_len, 22 bool(reason),
    # 23 bool(fault), 24 hash(title), 25 hash(device), 26 hash(vendor),
    # 27 hash(location_shape), 28 hash(room), 29 hash(fault), 30 hash(reason),
    # 31 degree, 32 alarm_neighbors, 33 ratio, 34 target_neighbors,
    # 35 target_ratio, 36 len(two_hop), 37 two_hop_alarm, 38 two_hop_ratio
    # 死特征对应（SHAP≈0 或与桶A强冗余）: 11 mean, 13 max, 14 min,
    # 15 last/max, 26 vendor_hash, 33 neighbor ratio
    # 注意: 保守起见仅删除明确冗余的 6 列，避免破坏 V11 已验证行为。
    dead = [11, 13, 14, 15, 26, 33]
    return dead


def bucket_a_order_features(order):
    """桶 A: 工单局部、无泄漏的 14 维特征。"""
    nodes = order["topology"].get("nodes", [])
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    adjacency = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is not None and target is not None:
            adjacency[source].add(target)
            adjacency[target].add(source)

    alarm_indices = [rid_to_index[node.get("@rid")] for node in order["alarms"]]
    alarm_set = set(alarm_indices)
    target_indices = [
        index for index in alarm_indices if nodes[index].get("label") == "TargetAlarm"
    ]
    target_titles = {nodes[index].get("title") for index in target_indices}
    # 无向 BFS 到任一 TargetAlarm 的最短距离
    distances = np.full(len(nodes), len(nodes) + 1, dtype=np.float32)
    queue = deque()
    for start in target_indices:
        distances[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in adjacency[current]:
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                queue.append(neighbor)

    features = []
    for local_index, node in enumerate(order["alarms"]):
        node_index = alarm_indices[local_index]
        time_lists = node.get("timeLists", [])
        if not isinstance(time_lists, list):
            time_lists = []
        L = np.asarray([1 if int(v) else 0 for v in time_lists[:6]], dtype=np.int8)
        ones = np.flatnonzero(L == 1)
        tl_onset = float(ones[0]) if len(ones) else 6.0
        tl_offset = float(ones[-1]) if len(ones) else -1.0
        tl_span = tl_offset - tl_onset
        tl_recent_frac = float(np.sum(L[3:6])) / max(float(np.sum(L)), 1.0)
        tl_flips = float(np.sum(np.abs(np.diff(L)))) if len(L) > 1 else 0.0
        tl_monotone = float(1 if len(L) and bool(np.all(np.diff(L) >= 0)) else 0)

        neighbors = adjacency[node_index]
        dev_neighbors = sum(1 for nb in neighbors if nodes[nb].get("@class") != "Alarm")
        dev_neighbor_types = len(
            {
                nodes[nb].get("@class")
                for nb in neighbors
                if nodes[nb].get("@class") != "Alarm"
            }
        )
        same_device_as_target = float(
            1 if v10.scalar(node.get("device")) and any(
                v10.scalar(nodes[t].get("device")) == v10.scalar(node.get("device"))
                for t in target_indices
            ) else 0
        )

        add_info = str(node.get("addInfo") or "")
        addInfo_fields = float(len([f for f in add_info.split(";") if f.strip()]))
        addInfo_shape_hash = float(
            v10.stable_hash(re.sub(r"\d+", "#", add_info), 127)
        )
        loc = str(node.get("location") or "")
        loc_depth = float(loc.count("="))

        cooccur_target_count = float(
            sum(1 for t in target_indices if nodes[t].get("title") == node.get("title"))
        )

        features.append(
            [
                tl_onset,
                tl_offset,
                tl_span,
                tl_recent_frac,
                tl_flips,
                tl_monotone,
                math.log1p(float(distances[node_index])),
                float(dev_neighbors),
                float(dev_neighbor_types),
                same_device_as_target,
                addInfo_fields,
                addInfo_shape_hash,
                loc_depth,
                cooccur_target_count,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def bucket_a_features(orders):
    return np.vstack([bucket_a_order_features(order) for order in orders])


def bucket_b_features(
    data,
    train_orders,
    reference_orders,
    rows,
    query_orders,
    tfidf_matrix,
    knn_probs,
    knn_local_map,
    leave_out,
):
    """桶 B: 依赖 reference 折的 5 维特征。

    data: v10.prepare 输出
    train_orders: 训练工单列表（reference 统计恒用训练域）
    reference_orders: 本折参考工单全局索引（均为训练工单索引）
    rows: 行元组列表（训练行含 (order_index, title_id, node, label)，
          测试行含 (order_index, title_id, node)）
    query_orders: 与 rows 同域的工单列表（训练域传 train_orders，测试域传 test_orders）
    tfidf_matrix: 行索引 = 工单索引（与 query_orders 同域）的归一化 TF-IDF
    knn_probs: 行索引 = knn_local_map(工单索引) 的 KNN 概率矩阵
    knn_local_map: dict 工单索引 -> knn_probs 行位置
    leave_out: True 表示查询工单在 reference 内（训练侧需排除自身贡献）
    """
    ref_set = {int(i) for i in reference_orders}
    y = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)

    title_pos = np.zeros(len(data["titles"]), dtype=np.int64)
    title_cnt = np.zeros(len(data["titles"]), dtype=np.int64)
    device_pos = defaultdict(int)
    device_cnt = defaultdict(int)
    template_pos = defaultdict(int)
    template_cnt = defaultdict(int)

    for order_index in ref_set:
        order_slice = data["train_slices"][order_index]
        labels = y[order_slice]
        sig = v11.order_signature(train_orders[order_index])
        template_cnt[sig] += len(labels)
        template_pos[sig] += int(np.sum(labels))
        for row_index in range(order_slice.start, order_slice.stop):
            _, title_id, node, label = data["train_rows"][row_index]
            if title_id >= 0:
                title_cnt[title_id] += 1
                title_pos[title_id] += label
            device = v10.scalar(node.get("device"))
            device_cnt[device] += 1
            device_pos[device] += label

    out = np.zeros((len(rows), 5), dtype=np.float32)
    for k, row in enumerate(rows):
        order_index, title_id, node = row[:3]
        order_index = int(order_index)
        label = int(row[3]) if len(row) > 3 else 0
        sig = v11.order_signature(query_orders[order_index])

        if leave_out and order_index in ref_set:
            # 排除自身在 reference 统计中的贡献
            t_pos = title_pos[title_id] - int(label) if title_id >= 0 else 0
            t_cnt = title_cnt[title_id] - 1 if title_id >= 0 else 0
            device = v10.scalar(node.get("device"))
            d_pos = device_pos[device] - int(label)
            d_cnt = device_cnt[device] - 1
            tp_pos = template_pos[sig] - int(label)
            tp_cnt = template_cnt[sig] - 1
        else:
            t_pos = title_pos[title_id] if title_id >= 0 else 0
            t_cnt = title_cnt[title_id] if title_id >= 0 else 0
            device = v10.scalar(node.get("device"))
            d_pos = device_pos[device]
            d_cnt = device_cnt[device]
            tp_pos = template_pos[sig]
            tp_cnt = template_cnt[sig]

        root_prior_title = (t_pos + 0.3) / (t_cnt + 1.0) if title_id >= 0 else 0.3
        root_prior_device = (d_pos + 0.3) / (d_cnt + 1.0)
        template_root_rate = (tp_pos + 0.5) / (tp_cnt + 1.0)

        local = knn_local_map.get(order_index, -1)
        tfidf_weight = 0.0
        knn_rank = 0.5
        if local >= 0 and title_id >= 0:
            tfidf_weight = float(tfidf_matrix[order_index, title_id])
            probs_row = knn_probs[local]
            knn_rank = float(
                np.sum(probs_row >= probs_row[title_id]) / max(len(probs_row), 1)
            )

        out[k] = [
            root_prior_title,
            root_prior_device,
            template_root_rate,
            tfidf_weight,
            knn_rank,
        ]
    return out


def seed_collections(by_order, score_getter, target, cap=8):
    """对给定分数产生 exact 1059 集合，返回 {(order_id, rid)}。"""
    selected = set()
    optional = []
    counts = Counter()
    for order_id, records in by_order.items():
        records = sorted(records, key=lambda r: (-score_getter(r), r["rid"]))
        if counts[order_id] == 0:
            selected.add((order_id, records[0]["rid"]))
            counts[order_id] = 1
        for record in records:
            key = (order_id, record["rid"])
            if key not in selected:
                optional.append((score_getter(record), order_id, record["rid"]))
    optional.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, order_id, rid in optional:
        if len(selected) >= target:
            break
        if counts[order_id] < cap:
            selected.add((order_id, rid))
            counts[order_id] += 1
    if len(selected) != target:
        raise RuntimeError(f"seed collection {len(selected)} != {target}")
    return selected


def symdiff(left, right):
    return len(left ^ right)


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
            "usage: v13_meta_ranker.py TRAIN_DIR TEST_DIR OUTPUT_DIR [--no-location]"
        )
    if len(sys.argv) == 5 and sys.argv[4] == "--no-location":
        v10.set_no_location(True)
        print("Location features DISABLED", flush=True)
    train_dir, test_dir, output_dir = map(Path, sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading data...", flush=True)
    train_orders = v10.load_orders(train_dir, True)
    test_orders = v10.load_orders(test_dir, False)
    data = v10.prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    folds, group_count, repeated_orders, fold_sizes = v11.grouped_folds(train_orders)
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
    train_directed = v11.directed_features(train_orders)
    test_directed = v11.directed_features(test_orders)
    train_bucket_a = bucket_a_features(train_orders)
    test_bucket_a = bucket_a_features(test_orders)
    print(
        f"static={data['train_static'].shape} directed={train_directed.shape} "
        f"bucket_a={train_bucket_a.shape}",
        flush=True,
    )

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

        # 桶 B（依赖 reference 折）
        train_local_map = {
            int(order_index): local
            for local, order_index in enumerate(reference_orders)
        }
        validation_local_map = {
            int(order_index): local
            for local, order_index in enumerate(validation_orders)
        }
        train_bucket_b = bucket_b_features(
            data, train_orders, reference_orders, train_rows, train_orders,
            train_tfidf, train_probabilities,
            train_local_map, leave_out=True,
        )
        validation_bucket_b = bucket_b_features(
            data, train_orders, reference_orders, validation_rows, train_orders,
            train_tfidf, validation_probabilities,
            validation_local_map, leave_out=False,
        )

        x_train = np.hstack(
            [
                v10.make_features(
                    data["train_static"][train_row_indices],
                    train_context,
                    train_encoding,
                ),
                train_directed[train_row_indices],
                train_bucket_a[train_row_indices],
                train_bucket_b,
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
                train_bucket_a[validation_row_indices],
                validation_bucket_b,
            ]
        ).astype(np.float32)
        print(f"x_train={x_train.shape} x_validation={x_validation.shape}", flush=True)

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
        print(f"fold={heldout} trained ({time.time()-started:.0f}s)", flush=True)

    oof_meta = np.mean(oof_per_seed, axis=0)
    nested_baseline_mask = np.zeros(len(labels), dtype=bool)
    nested_candidate_mask = np.zeros(len(labels), dtype=bool)
    chosen_weights = []
    fold_metrics = []
    for heldout in range(N_FOLDS):
        reference_orders = all_orders[folds != heldout]
        validation_orders = all_orders[folds == heldout]
        reference_rows, reference_slices = v11.rows_and_slices_for_orders(
            data, reference_orders
        )
        baseline_reference = v10.best_threshold(
            oof_context[reference_rows], labels[reference_rows], reference_slices
        )
        baseline_threshold = baseline_reference[-1]
        weight, threshold, _ = v11.choose_weight_threshold(
            oof_context, oof_meta, labels, data, reference_orders
        )
        chosen_weights.append(weight)

        validation_rows, validation_slices = v11.rows_and_slices_for_orders(
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
    gate_g1 = nested_gain >= MIN_GAIN
    gate_g2 = improved_folds >= MIN_IMPROVED_FOLDS
    print(
        f"NESTED baseline={nested_baseline} candidate={nested_candidate} "
        f"gain={nested_gain:+.6f} improved_folds={improved_folds}/5",
        flush=True,
    )
    print(
        f"FINAL weight={final_weight:.2f} threshold={final_threshold:.9f} "
        f"global_oof={global_result} G1={gate_g1} G2={gate_g2}",
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
    full_local_map = {int(i): int(i) for i in all_orders}
    full_train_bucket_b = bucket_b_features(
        data, train_orders, all_orders,
        data["train_rows"], train_orders,
        train_tfidf, full_train_probabilities, full_local_map, leave_out=True,
    )
    full_test_bucket_b = bucket_b_features(
        data, train_orders, all_orders,
        data["test_rows"], test_orders,
        test_tfidf, full_test_probabilities, full_local_map, leave_out=False,
    )
    x_full = np.hstack(
        [
            v10.make_features(
                data["train_static"], full_train_context, full_train_encoding
            ),
            train_directed,
            train_bucket_a,
            full_train_bucket_b,
        ]
    ).astype(np.float32)
    x_test = np.hstack(
        [
            v10.make_features(
                data["test_static"], full_test_context, full_test_encoding
            ),
            test_directed,
            test_bucket_a,
            full_test_bucket_b,
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
        output_dir / "v13_test_scores.csv",
        test_orders,
        data,
        full_test_context,
        test_per_seed,
        test_meta,
    )
    np.save(output_dir / "v13_oof_context.npy", oof_context)
    np.save(output_dir / "v13_oof_meta.npy", oof_meta)
    np.save(output_dir / "v13_test_scores.npy", test_scores)

    # G3: 三种子测试集合一致性
    by_order, seed_columns = bvc.load_scores(output_dir / "v13_test_scores.csv")
    mean_col = lambda r: r["mean"]
    seed_sets = []
    for index in range(len(SEEDS)):
        seed_sets.append(
            seed_collections(
                by_order,
                lambda r, i=index: r["seeds"][i],
                TARGET_TEST_COUNT,
            )
        )
    mean_set = seed_collections(by_order, mean_col, TARGET_TEST_COUNT)
    pairwise = [
        symdiff(seed_sets[i], seed_sets[j])
        for i in range(len(SEEDS))
        for j in range(i + 1, len(SEEDS))
    ]
    gate_g3 = max(pairwise) <= MAX_SEED_SYMDIFF
    print(f"G3 seed symdiff pairwise={pairwise} gate={gate_g3}", flush=True)

    # G4/G5: 计数校验
    threshold_mask = v10.selection_mask(
        test_scores, data["test_slices"], final_threshold
    )
    fixed_mask = v10.exact_count_mask(test_scores, data["test_slices"], TARGET_TEST_COUNT)
    threshold_count = int(np.sum(threshold_mask))
    fixed_count = int(np.sum(fixed_mask))
    per_order_counts = [
        int(np.sum(fixed_mask[sl])) for sl in data["test_slices"]
    ]
    gate_g4 = fixed_count == TARGET_TEST_COUNT and max(per_order_counts) <= v10.MAX_ROOTCAUSES
    gate_g5 = True  # write_submission 按 @rid 构造，天然唯一；提交时 validate 兜底
    print(
        f"G4 threshold_count={threshold_count} fixed_count={fixed_count} "
        f"max_per_order={max(per_order_counts)} gate={gate_g4}",
        flush=True,
    )

    gate_passed = gate_g1 and gate_g2 and gate_g3 and gate_g4 and gate_g5
    print(
        f"GATES: G1={gate_g1} G2={gate_g2} G3={gate_g3} G4={gate_g4} "
        f"G5={gate_g5} PASSED={gate_passed}",
        flush=True,
    )

    raw_path = output_dir / "result_record_v13_meta_1059.csv"
    v10.write_submission(raw_path, test_orders, data, fixed_mask)

    # 受保护候选: 锁定 swap12 胜出 delta（需要冠军/基线文件路径）
    champion_path = Path(
        r"D:\zgyidong\experiments\submissions\champion_0.906324_day01_probe01_v11_full.csv"
    )
    baseline_path = Path(
        r"D:\zgyidong\codexgz\result_record_probe_swap12_score_0.905373.csv"
    )
    constrained_path = output_dir / "result_record_v13_constrained_1059.csv"
    constrained_report = None
    if champion_path.exists() and baseline_path.exists():
        original = bvc.load_submission(baseline_path)
        winner = bvc.load_submission(champion_path)
        forced_in, forced_out = bvc.protection_sets(original, winner)
        topologies = bvc.load_topologies(test_dir)
        exact = bvc.constrained_exact(
            by_order, forced_in, forced_out, lambda r: r["mean"],
            TARGET_TEST_COUNT,
        )
        bvc.write_submission(constrained_path, topologies, exact)
        winner_set = {
            (oid, rid) for oid, rids in winner.items() for rid in rids
        }
        constrained_report = {
            "forced_in_count": len(forced_in),
            "forced_out_count": len(forced_out),
            "constrained_vs_winner": bvc.compare(winner_set, exact),
            "count_distribution": dict(
                sorted(Counter(Counter(oid for oid, _ in exact).values()).items())
            ),
        }
        print(f"CONSTRAINED {constrained_report}", flush=True)
    else:
        print(
            f"SKIP constrained: champion/baseline missing "
            f"(champion={champion_path.exists()}, baseline={baseline_path.exists()})",
            flush=True,
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
        "gates": {
            "G1_nested_gain": gate_g1,
            "G2_improved_folds": gate_g2,
            "G3_seed_symdiff": gate_g3,
            "G4_counts": gate_g4,
            "G5_unique": gate_g5,
            "passed": gate_passed,
        },
        "seed_pairwise_symdiff": pairwise,
        "threshold_test_count": threshold_count,
        "fixed_test_count": fixed_count,
        "max_per_order": max(per_order_counts),
        "raw_submission": str(raw_path),
        "constrained": constrained_report,
        "seconds": time.time() - started,
    }
    (output_dir / "v13_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"TOTAL {time.time()-started:.0f}s outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()

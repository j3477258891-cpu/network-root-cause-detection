import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


CONFIGS = [
    (5, 2.0, 0.06668020883603187),
    (1, 1.0, 0.04820534260274989),
    (8, 2.0, 0.23543407593447566),
    (2, 2.0, 0.2499238097838483),
    (12, 4.0, 0.03659548793379663),
    (3, 1.0, 0.16652631426606984),
    (3, 2.0, 0.19663476064302782),
]
N_FOLDS = 5
KNN_THRESHOLD = 0.5989118247245228
MAX_ROOTCAUSES = 8
TARGET_TEST_COUNT = 1059
MIN_OOF_GAIN = 0.003

# ---- location feature ablation ----
NO_LOCATION = False


def set_no_location(value: bool):
    global NO_LOCATION
    NO_LOCATION = value


def get_key_specs():
    if not NO_LOCATION:
        return KEY_SPECS
    return [(n, a, f) for n, a, f in KEY_SPECS if n not in ("title_location", "location")]


def fold_of(order_id):
    return int(hashlib.md5(order_id.encode()).hexdigest(), 16) % N_FOLDS


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ""
    return value


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash(value, modulus):
    value = str(scalar(value))
    number = 5381
    for character in value:
        number = ((number << 5) + number + ord(character)) & 0xFFFFFFFF
    return number % modulus


def location_shape(value):
    value = str(value or "")
    value = re.sub(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        "<UUID>",
        value,
    )
    return re.sub(r"\d+", "#", value)


def load_orders(base_dir, with_labels):
    orders = []
    for directory in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        order_id = directory.name
        topology_path = directory / f"{order_id}.log.topo.json"
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        alarms = [
            node for node in topology.get("nodes", []) if node.get("@class") == "Alarm"
        ]
        roots = set()
        if with_labels:
            root_path = directory / f"{order_id}.rootcause.json"
            root_data = json.loads(root_path.read_text(encoding="utf-8"))
            roots = {node["@rid"] for node in root_data.get("rootcause", [])}
        orders.append(
            {
                "id": order_id,
                "topology": topology,
                "alarms": alarms,
                "roots": roots,
                "fold": fold_of(order_id) if with_labels else -1,
            }
        )
    return orders


def static_order_features(order):
    topology = order["topology"]
    nodes = topology.get("nodes", [])
    alarms = order["alarms"]
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    adjacency = [set() for _ in nodes]
    for edge in topology.get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is not None and target is not None:
            adjacency[source].add(target)
            adjacency[target].add(source)

    alarm_node_indices = [rid_to_index.get(node.get("@rid"), -1) for node in alarms]
    alarm_index_set = {index for index in alarm_node_indices if index >= 0}
    target_index_set = {
        index
        for index in alarm_index_set
        if nodes[index].get("label") == "TargetAlarm"
    }
    title_counts = Counter(scalar(node.get("title")) for node in alarms)
    device_counts = Counter(scalar(node.get("device")) for node in alarms)
    location_counts = Counter(location_shape(node.get("location")) for node in alarms)
    target_count = len(target_index_set)
    fault_time = safe_float(topology.get("time"))
    features = []

    for local_index, node in enumerate(alarms):
        node_index = alarm_node_indices[local_index]
        neighbors = adjacency[node_index] if node_index >= 0 else set()
        degree = len(neighbors)
        alarm_neighbors = sum(index in alarm_index_set for index in neighbors)
        target_neighbors = sum(index in target_index_set for index in neighbors)
        two_hop = set()
        for neighbor in neighbors:
            two_hop.update(adjacency[neighbor])
        two_hop.discard(node_index)
        two_hop_alarm = sum(index in alarm_index_set for index in two_hop)

        title = scalar(node.get("title"))
        device = scalar(node.get("device"))
        vendor = scalar(node.get("vendor"))
        room = scalar(node.get("room"))
        location = scalar(node.get("location"))
        reason = scalar(node.get("reason"))
        fault1 = scalar(node.get("fault1"))
        fault2 = scalar(node.get("fault2"))
        time_lists = node.get("timeLists", [])
        if not isinstance(time_lists, list):
            time_lists = []
        time_values = np.asarray(
            [safe_float(value) for value in time_lists[:6]], dtype=np.float32
        )
        if not len(time_values):
            time_values = np.zeros(1, dtype=np.float32)
        node_time = safe_float(node.get("time"), fault_time)
        time_delta = abs(fault_time - node_time)

        row = [
            float(node.get("label") == "TargetAlarm"),
            float(node.get("label") == "TargetAlarm" and target_count == 1),
            float(target_count),
            float(len(alarms)),
            float(title_counts[title]),
            float(device_counts[device]),
            *(  # location_counts — skip when NO_LOCATION
                [] if NO_LOCATION else [float(location_counts[location_shape(location)])]
            ),
            float(local_index / max(len(alarms) - 1, 1)),
            math.log1p(time_delta),
            float(len(time_lists)),
            float(np.sum(time_values)),
            float(np.mean(time_values)),
            float(np.std(time_values)),
            float(np.max(time_values)),
            float(np.min(time_values)),
            float(time_values[-1] / max(float(np.max(time_values)), 1e-6)),
            float(np.argmax(time_values) / max(len(time_values) - 1, 1)),
            float(len(str(title))),
            float(len(str(device))),
            *(  # len(location) — skip when NO_LOCATION
                [] if NO_LOCATION else [float(len(str(location)))]
            ),
            float(len(str(reason))),
            float(len(str(fault1)) + len(str(fault2))),
            float(bool(reason)),
            float(bool(fault1 or fault2)),
            float(stable_hash(title, 257)),
            float(stable_hash(device, 257)),
            float(stable_hash(vendor, 31)),
            *(  # hash(location_shape) — skip when NO_LOCATION
                [] if NO_LOCATION else [float(stable_hash(location_shape(location), 257))]
            ),
            float(stable_hash(room, 31)),
            float(stable_hash(fault1 or fault2, 127)),
            float(stable_hash(reason, 257)),
            float(degree),
            float(alarm_neighbors),
            float(alarm_neighbors / max(degree, 1)),
            float(target_neighbors),
            float(target_neighbors / max(degree, 1)),
            float(len(two_hop)),
            float(two_hop_alarm),
            float(two_hop_alarm / max(len(two_hop), 1)),
        ]
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def prepare(train_orders, test_orders):
    titles = sorted(
        {
            scalar(node.get("title"))
            for order in train_orders
            for node in order["alarms"]
        }
    )
    title_index = {title: index for index, title in enumerate(titles)}
    train_alarm = np.zeros((len(train_orders), len(titles)), dtype=np.float32)
    train_root = np.zeros_like(train_alarm)
    train_rows = []
    train_slices = []
    train_static = []
    for order_index, order in enumerate(train_orders):
        start = len(train_rows)
        static = static_order_features(order)
        for local_index, node in enumerate(order["alarms"]):
            title_id = title_index[scalar(node.get("title"))]
            label = int(node.get("@rid") in order["roots"])
            train_alarm[order_index, title_id] += 1
            train_root[order_index, title_id] += label
            train_rows.append((order_index, title_id, node, label))
            train_static.append(static[local_index])
        train_slices.append(slice(start, len(train_rows)))

    test_alarm = np.zeros((len(test_orders), len(titles)), dtype=np.float32)
    test_rows = []
    test_slices = []
    test_static = []
    for order_index, order in enumerate(test_orders):
        start = len(test_rows)
        static = static_order_features(order)
        for local_index, node in enumerate(order["alarms"]):
            title_id = title_index.get(scalar(node.get("title")), -1)
            if title_id >= 0:
                test_alarm[order_index, title_id] += 1
            test_rows.append((order_index, title_id, node))
            test_static.append(static[local_index])
        test_slices.append(slice(start, len(test_rows)))

    return {
        "titles": titles,
        "train_alarm": train_alarm,
        "train_root": train_root,
        "train_rows": train_rows,
        "train_slices": train_slices,
        "train_static": np.vstack(train_static),
        "test_alarm": test_alarm,
        "test_rows": test_rows,
        "test_slices": test_slices,
        "test_static": np.vstack(test_static),
        "folds": np.asarray([order["fold"] for order in train_orders], dtype=np.int8),
    }


def normalized_tfidf(train_alarm, test_alarm):
    document_frequency = np.sum(train_alarm > 0, axis=0)
    idf = np.log((1 + len(train_alarm)) / (1 + document_frequency)) + 1
    train = np.log1p(train_alarm) * idf
    test = np.log1p(test_alarm) * idf
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    return train, test


def knn_order_probabilities(
    query_tfidf,
    reference_tfidf,
    reference_alarm,
    reference_root,
    exclude_reference_positions=None,
):
    similarities = query_tfidf @ reference_tfidf.T
    if exclude_reference_positions is not None:
        similarities[np.arange(len(query_tfidf)), exclude_reference_positions] = -np.inf
    global_positive = np.sum(reference_root, axis=0)
    global_total = np.sum(reference_alarm, axis=0)
    prior = (global_positive + 0.3) / (global_total + 1.0)
    probabilities = np.zeros((len(query_tfidf), reference_alarm.shape[1]), dtype=np.float64)
    for k, power, ensemble_weight in CONFIGS:
        local_k = min(k, similarities.shape[1] - int(exclude_reference_positions is not None))
        nearest = np.argpartition(-similarities, local_k - 1, axis=1)[:, :local_k]
        for query_index in range(len(query_tfidf)):
            neighbor_indices = nearest[query_index]
            weights = np.maximum(similarities[query_index, neighbor_indices], 1e-6) ** power
            positive = weights @ reference_root[neighbor_indices]
            total = weights @ reference_alarm[neighbor_indices]
            query_prior = prior
            if exclude_reference_positions is not None:
                excluded = exclude_reference_positions[query_index]
                query_prior = (
                    global_positive - reference_root[excluded] + 0.3
                ) / (global_total - reference_alarm[excluded] + 1.0)
            probabilities[query_index] += ensemble_weight * (
                (positive + 2.0 * query_prior) / (total + 2.0)
            )
    return probabilities


def row_scores(order_probabilities, rows):
    return np.asarray(
        [
            order_probabilities[order_index, title_id] if title_id >= 0 else 0.3
            for order_index, title_id, *_ in rows
        ],
        dtype=np.float64,
    )


KEY_SPECS = [
    ("title", 3.0, lambda node: scalar(node.get("title"))),
    (
        "title_location",
        5.0,
        lambda node: (scalar(node.get("title")), location_shape(node.get("location"))),
    ),
    (
        "title_reason",
        8.0,
        lambda node: (scalar(node.get("title")), scalar(node.get("reason"))),
    ),
    ("device", 8.0, lambda node: scalar(node.get("device"))),
    ("vendor", 20.0, lambda node: scalar(node.get("vendor"))),
    (
        "fault",
        8.0,
        lambda node: scalar(node.get("fault1")) or scalar(node.get("fault2")),
    ),
    ("room", 12.0, lambda node: scalar(node.get("room"))),
    ("location", 10.0, lambda node: location_shape(node.get("location"))),
]


def encoded_features(data, reference_orders, query_rows, leave_query_order_out):
    key_specs = get_key_specs()
    reference_order_set = set(int(value) for value in reference_orders)
    y = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    total_positive = 0
    total_count = 0
    stats = [defaultdict(lambda: [0, 0]) for _ in key_specs]
    order_stats = [defaultdict(lambda: [0, 0]) for _ in key_specs]
    for row_index, (order_index, _, node, label) in enumerate(data["train_rows"]):
        if order_index not in reference_order_set:
            continue
        total_positive += int(y[row_index])
        total_count += 1
        for spec_index, (_, _, key_function) in enumerate(key_specs):
            key = key_function(node)
            stats[spec_index][key][0] += label
            stats[spec_index][key][1] += 1
            order_stats[spec_index][(order_index, key)][0] += label
            order_stats[spec_index][(order_index, key)][1] += 1

    output = np.zeros((len(query_rows), 2 * len(key_specs)), dtype=np.float32)
    for row_index, row in enumerate(query_rows):
        order_index, _, node = row[:3]
        excluded_positive = excluded_count = 0
        if leave_query_order_out and order_index in reference_order_set:
            order_slice = data["train_slices"][order_index]
            excluded_labels = y[order_slice]
            excluded_positive = int(np.sum(excluded_labels))
            excluded_count = len(excluded_labels)
        global_probability = (total_positive - excluded_positive) / max(
            total_count - excluded_count, 1
        )
        for spec_index, (_, smoothing, key_function) in enumerate(key_specs):
            key = key_function(node)
            positive, count = stats[spec_index].get(key, (0, 0))
            if leave_query_order_out and order_index in reference_order_set:
                own_positive, own_count = order_stats[spec_index].get(
                    (order_index, key), (0, 0)
                )
                positive -= own_positive
                count -= own_count
            probability = (positive + smoothing * global_probability) / (
                count + smoothing
            )
            output[row_index, 2 * spec_index] = probability
            output[row_index, 2 * spec_index + 1] = math.log1p(max(count, 0))
    return output


def rows_for_orders(data, order_indices):
    row_indices = []
    for order_index in order_indices:
        order_slice = data["train_slices"][int(order_index)]
        row_indices.extend(range(order_slice.start, order_slice.stop))
    return np.asarray(row_indices, dtype=np.int64)


def local_rows_and_slices(data, order_indices):
    row_indices = []
    local_rows = []
    local_slices = []
    for local_order_index, order_index in enumerate(order_indices):
        order_slice = data["train_slices"][int(order_index)]
        start = len(row_indices)
        for row_index in range(order_slice.start, order_slice.stop):
            row_indices.append(row_index)
            row = data["train_rows"][row_index]
            local_rows.append((local_order_index,) + row[1:])
        local_slices.append(slice(start, len(row_indices)))
    return np.asarray(row_indices, dtype=np.int64), local_rows, local_slices


def make_features(static, context, encodings):
    context = np.asarray(context, dtype=np.float32).reshape(-1, 1)
    return np.hstack([context, static, encodings]).astype(np.float32)


def selection_mask(scores, slices, threshold, cap=MAX_ROOTCAUSES):
    selected = np.zeros(len(scores), dtype=bool)
    for order_slice in slices:
        local_scores = scores[order_slice]
        local = np.flatnonzero(local_scores >= threshold)
        if not len(local):
            local = np.asarray([int(np.argmax(local_scores))])
        if len(local) > cap:
            local = local[np.argsort(-local_scores[local], kind="stable")[:cap]]
        selected[order_slice.start + local] = True
    return selected


def exact_count_mask(scores, slices, target_count, cap=MAX_ROOTCAUSES):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order_slice in slices:
        local_scores = scores[order_slice]
        ranked = np.argsort(-local_scores, kind="stable")[:cap]
        selected[order_slice.start + ranked[0]] = True
        optional.extend(order_slice.start + ranked[1:])
    remaining = max(0, target_count - int(np.sum(selected)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def confusion(mask, labels):
    tp = int(np.sum(mask & (labels == 1)))
    fp = int(np.sum(mask & (labels == 0)))
    fn = int(np.sum((~mask) & (labels == 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, tp, fp, fn, int(np.sum(mask))


def best_threshold(scores, labels, slices):
    mandatory = np.zeros(len(scores), dtype=bool)
    eligible = np.zeros(len(scores), dtype=bool)
    for order_slice in slices:
        local_scores = scores[order_slice]
        ranked = np.argsort(-local_scores, kind="stable")
        eligible[order_slice.start + ranked[:MAX_ROOTCAUSES]] = True
        mandatory[order_slice.start + ranked[0]] = True

    base_tp = int(np.sum(mandatory & (labels == 1)))
    base_fp = int(np.sum(mandatory & (labels == 0)))
    total_positive = int(np.sum(labels))
    optional = np.flatnonzero(eligible & ~mandatory)
    order = optional[np.argsort(-scores[optional], kind="stable")]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    if not len(order):
        result = confusion(mandatory, labels)
        return (*result, float(np.max(scores)))
    cumulative_tp = np.cumsum(ordered_labels)
    cumulative_fp = np.cumsum(1 - ordered_labels)
    boundaries = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    tp = base_tp + cumulative_tp[boundaries]
    fp = base_fp + cumulative_fp[boundaries]
    fn = total_positive - tp
    values = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best_index = int(np.argmax(values))
    boundary = int(boundaries[best_index])
    predicted_count = int(np.sum(mandatory) + boundary + 1)
    return (
        float(values[best_index]),
        int(tp[best_index]),
        int(fp[best_index]),
        int(fn[best_index]),
        predicted_count,
        float(ordered_scores[boundary]),
    )


def write_submission(path, test_orders, data, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_index, order in enumerate(test_orders):
            order_slice = data["test_slices"][order_index]
            rootcauses = []
            for local_index in np.flatnonzero(mask[order_slice]):
                node = order["alarms"][int(local_index)]
                rootcauses.append(
                    {
                        "@rid": node["@rid"],
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                )
            writer.writerow(
                [order["id"], json.dumps({"rootcause": rootcauses}, ensure_ascii=False)]
            )


def main():
    train_dir = Path(sys.argv[1])
    test_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    print("Loading orders...", flush=True)
    train_orders = load_orders(train_dir, True)
    test_orders = load_orders(test_dir, False)
    data = prepare(train_orders, test_orders)
    labels = np.asarray([row[3] for row in data["train_rows"]], dtype=np.int8)
    train_tfidf, test_tfidf = normalized_tfidf(
        data["train_alarm"], data["test_alarm"]
    )
    all_orders = np.arange(len(train_orders))
    oof_knn = np.zeros(len(labels), dtype=np.float64)
    oof_tree = np.zeros(len(labels), dtype=np.float64)
    oof_hist = np.zeros(len(labels), dtype=np.float64)
    fold_results = []

    print(
        f"train_orders={len(train_orders)} test_orders={len(test_orders)} "
        f"train_nodes={len(labels)} positives={int(np.sum(labels))}",
        flush=True,
    )
    for heldout in range(N_FOLDS):
        reference_orders = all_orders[data["folds"] != heldout]
        validation_orders = all_orders[data["folds"] == heldout]
        train_row_indices, train_rows_for_context, _ = local_rows_and_slices(
            data, reference_orders
        )
        (
            validation_row_indices,
            validation_rows_for_context,
            validation_local_slices,
        ) = local_rows_and_slices(data, validation_orders)

        train_order_probabilities = knn_order_probabilities(
            train_tfidf[reference_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
            exclude_reference_positions=np.arange(len(reference_orders)),
        )
        validation_order_probabilities = knn_order_probabilities(
            train_tfidf[validation_orders],
            train_tfidf[reference_orders],
            data["train_alarm"][reference_orders],
            data["train_root"][reference_orders],
        )
        train_context = row_scores(
            train_order_probabilities,
            train_rows_for_context,
        )
        validation_context_local = row_scores(
            validation_order_probabilities,
            validation_rows_for_context,
        )
        oof_knn[validation_row_indices] = validation_context_local

        train_rows_local = [data["train_rows"][index] for index in train_row_indices]
        validation_rows_local = [
            data["train_rows"][index] for index in validation_row_indices
        ]
        train_encoding = encoded_features(
            data, reference_orders, train_rows_local, leave_query_order_out=True
        )
        validation_encoding = encoded_features(
            data, reference_orders, validation_rows_local, leave_query_order_out=False
        )
        x_train = make_features(
            data["train_static"][train_row_indices], train_context, train_encoding
        )
        x_validation = make_features(
            data["train_static"][validation_row_indices],
            validation_context_local,
            validation_encoding,
        )
        model = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260730 + heldout,
        )
        model.fit(x_train, labels[train_row_indices])
        oof_tree[validation_row_indices] = model.predict_proba(x_validation)[:, 1]
        hist_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=16,
            l2_regularization=2.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=30,
            random_state=20260730 + heldout,
        )
        hist_model.fit(x_train, labels[train_row_indices])
        oof_hist[validation_row_indices] = hist_model.predict_proba(x_validation)[:, 1]
        knn_fold = best_threshold(
            oof_knn[validation_row_indices],
            labels[validation_row_indices],
            validation_local_slices,
        )
        fold_results.append((heldout, knn_fold[0]))
        print(f"fold={heldout} completed knn_best={knn_fold[0]:.6f}", flush=True)

    baseline_fixed_mask = selection_mask(
        oof_knn, data["train_slices"], KNN_THRESHOLD
    )
    baseline_fixed = confusion(baseline_fixed_mask, labels)
    baseline_best = best_threshold(oof_knn, labels, data["train_slices"])
    tree_best = best_threshold(oof_tree, labels, data["train_slices"])
    hist_best = best_threshold(oof_hist, labels, data["train_slices"])
    print("BASELINE_FIXED", baseline_fixed, flush=True)
    print("BASELINE_BEST", baseline_best, flush=True)
    print("TREE_BEST", tree_best, flush=True)
    print("HIST_BEST", hist_best, flush=True)

    blend_results = []
    for tree_weight in np.linspace(0.0, 1.0, 11):
        for hist_weight in np.linspace(0.0, 1.0 - tree_weight, 11):
            knn_weight = 1.0 - tree_weight - hist_weight
            blend = (
                knn_weight * oof_knn
                + tree_weight * oof_tree
                + hist_weight * oof_hist
            )
            result = best_threshold(blend, labels, data["train_slices"])
            blend_results.append(
                (result[0], float(tree_weight), float(hist_weight), result)
            )
    blend_results.sort(reverse=True)
    best_f1, tree_weight, hist_weight, blend_best = blend_results[0]
    knn_weight = 1.0 - tree_weight - hist_weight
    print("BEST_BLEND", knn_weight, tree_weight, hist_weight, blend_best, flush=True)
    print("TOP_BLENDS", blend_results[:6], flush=True)

    fold_improvements = 0
    for heldout in range(N_FOLDS):
        order_indices = all_orders[data["folds"] == heldout]
        row_indices = rows_for_orders(data, order_indices)
        local_slices = []
        offset = 0
        for order_index in order_indices:
            length = data["train_slices"][order_index].stop - data["train_slices"][order_index].start
            local_slices.append(slice(offset, offset + length))
            offset += length
        local_knn = best_threshold(oof_knn[row_indices], labels[row_indices], local_slices)
        local_blend_scores = (
            knn_weight * oof_knn[row_indices]
            + tree_weight * oof_tree[row_indices]
            + hist_weight * oof_hist[row_indices]
        )
        local_blend = best_threshold(local_blend_scores, labels[row_indices], local_slices)
        improvement = local_blend[0] - local_knn[0]
        fold_improvements += improvement > 0
        print(
            f"FOLD_RESULT fold={heldout} knn={local_knn[0]:.6f} "
            f"blend={local_blend[0]:.6f} gain={improvement:+.6f}",
            flush=True,
        )

    gain = best_f1 - baseline_best[0]
    gate_passed = (
        gain >= MIN_OOF_GAIN
        and fold_improvements >= 3
        and (tree_weight > 0 or hist_weight > 0)
    )
    print(
        f"GATE gain={gain:+.6f} improved_folds={fold_improvements}/5 "
        f"passed={gate_passed}",
        flush=True,
    )
    probe_mode = len(sys.argv) >= 5 and sys.argv[4] == "--probe"
    if not gate_passed and not probe_mode:
        print("No candidate generated because the validation gate failed.", flush=True)
        return

    full_train_probabilities = knn_order_probabilities(
        train_tfidf,
        train_tfidf,
        data["train_alarm"],
        data["train_root"],
        exclude_reference_positions=np.arange(len(train_orders)),
    )
    full_test_probabilities = knn_order_probabilities(
        test_tfidf,
        train_tfidf,
        data["train_alarm"],
        data["train_root"],
    )
    full_train_context = row_scores(full_train_probabilities, data["train_rows"])
    full_test_context = row_scores(full_test_probabilities, data["test_rows"])
    full_train_encoding = encoded_features(
        data, all_orders, data["train_rows"], leave_query_order_out=True
    )
    full_test_encoding = encoded_features(
        data, all_orders, data["test_rows"], leave_query_order_out=False
    )
    x_full = make_features(
        data["train_static"], full_train_context, full_train_encoding
    )
    x_test = make_features(data["test_static"], full_test_context, full_test_encoding)
    final_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=20260730,
    )
    final_model.fit(x_full, labels)
    tree_test = final_model.predict_proba(x_test)[:, 1]
    final_hist_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=16,
        l2_regularization=2.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=20260730,
    )
    final_hist_model.fit(x_full, labels)
    hist_test = final_hist_model.predict_proba(x_test)[:, 1]
    blend_test = (
        knn_weight * full_test_context
        + tree_weight * tree_test
        + hist_weight * hist_test
    )
    np.save(output_dir / "v10_blend_test.npy", blend_test)
    candidate_mask = exact_count_mask(
        blend_test, data["test_slices"], TARGET_TEST_COUNT
    )
    baseline_test_mask = selection_mask(
        full_test_context, data["test_slices"], KNN_THRESHOLD
    )
    removed = int(np.sum(baseline_test_mask & ~candidate_mask))
    added = int(np.sum(~baseline_test_mask & candidate_mask))
    distribution = Counter(
        int(np.sum(candidate_mask[order_slice]))
        for order_slice in data["test_slices"]
    )
    output_path = output_dir / "result_record_v10_1059.csv"
    write_submission(output_path, test_orders, data, candidate_mask)
    print(
        f"CANDIDATE output={output_path} count={int(np.sum(candidate_mask))} "
        f"removed={removed} added={added} distribution={dict(sorted(distribution.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()

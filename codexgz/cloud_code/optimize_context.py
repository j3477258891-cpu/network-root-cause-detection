import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def fold_of(order_id):
    return int(hashlib.md5(order_id.encode()).hexdigest(), 16) % 5


def location_shape(value):
    value = value or ""
    value = re.sub(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", "<UUID>", value)
    return re.sub(r"\d+", "#", value)


train_dir = Path(sys.argv[1])
orders = []
all_titles = set()
for directory in sorted(path for path in train_dir.iterdir() if path.is_dir()):
    order_id = directory.name
    with (directory / f"{order_id}.log.topo.json").open("r", encoding="utf-8") as handle:
        topo = json.load(handle)
    with (directory / f"{order_id}.rootcause.json").open("r", encoding="utf-8") as handle:
        root_ids = {node["@rid"] for node in json.load(handle)["rootcause"]}
    alarms = [node for node in topo["nodes"] if node.get("@class") == "Alarm"]
    all_titles.update(node.get("title", "") or "" for node in alarms)
    orders.append((order_id, alarms, root_ids))

titles = sorted(all_titles)
title_index = {title: index for index, title in enumerate(titles)}
n_orders = len(orders)
n_titles = len(titles)
alarm_counts = np.zeros((n_orders, n_titles), dtype=np.float32)
root_counts = np.zeros((n_orders, n_titles), dtype=np.float32)
folds = np.zeros(n_orders, dtype=np.int8)

rows = []
order_slices = []
for order_index, (order_id, alarms, root_ids) in enumerate(orders):
    folds[order_index] = fold_of(order_id)
    start = len(rows)
    for node in alarms:
        title_id = title_index[node.get("title", "") or ""]
        alarm_counts[order_index, title_id] += 1
        is_root = int(node["@rid"] in root_ids)
        root_counts[order_index, title_id] += is_root
        rows.append((order_index, title_id, node, is_root))
    order_slices.append(slice(start, len(rows)))

y = np.asarray([row[3] for row in rows], dtype=np.int8)
document_frequency = np.sum(alarm_counts > 0, axis=0)
idf = np.log((1 + n_orders) / (1 + document_frequency)) + 1
tfidf = np.log1p(alarm_counts) * idf
tfidf /= np.maximum(np.linalg.norm(tfidf, axis=1, keepdims=True), 1e-12)


def make_context_feature(k, power):
    output = np.zeros(len(rows), dtype=np.float64)
    for heldout in range(5):
        query_indices = np.flatnonzero(folds == heldout)
        train_indices = np.flatnonzero(folds != heldout)
        similarities = tfidf[query_indices] @ tfidf[train_indices].T
        k_actual = min(k, len(train_indices))
        nearest_local = np.argpartition(-similarities, k_actual - 1, axis=1)[:, :k_actual]
        for query_local, query_index in enumerate(query_indices):
            neighbor_indices = train_indices[nearest_local[query_local]]
            weights = np.maximum(similarities[query_local, nearest_local[query_local]], 1e-6) ** power
            positive = weights @ root_counts[neighbor_indices]
            total = weights @ alarm_counts[neighbor_indices]
            global_positive = np.sum(root_counts[train_indices], axis=0)
            global_total = np.sum(alarm_counts[train_indices], axis=0)
            prior = (global_positive + 0.3) / (global_total + 1.0)
            probabilities = (positive + 2.0 * prior) / (total + 2.0)
            order_slice = order_slices[query_index]
            for row_index in range(order_slice.start, order_slice.stop):
                output[row_index] = probabilities[rows[row_index][1]]
    return output


def make_signature_feature():
    output = np.zeros(len(rows), dtype=np.float64)
    for heldout in range(5):
        counts = defaultdict(lambda: [0, 0])
        title_counts = defaultdict(lambda: [0, 0])
        train_labels = []
        for order_index, title_id, node, label in rows:
            if folds[order_index] == heldout:
                continue
            title = titles[title_id]
            title_counts[title][0] += label
            title_counts[title][1] += 1
            key = (title, location_shape(node.get("location")))
            counts[key][0] += label
            counts[key][1] += 1
            train_labels.append(label)
        global_probability = sum(train_labels) / len(train_labels)
        for row_index, (order_index, title_id, node, _) in enumerate(rows):
            if folds[order_index] != heldout:
                continue
            title = titles[title_id]
            title_pos, title_total = title_counts.get(title, (0, 0))
            title_probability = (title_pos + 5 * global_probability) / (title_total + 5)
            pos, total = counts.get((title, location_shape(node.get("location"))), (0, 0))
            output[row_index] = (pos + 5 * title_probability) / (total + 5)
    return output


def best_threshold(scores):
    mandatory = np.zeros(len(scores), dtype=bool)
    for order_slice in order_slices:
        mandatory[order_slice.start + int(np.argmax(scores[order_slice]))] = True
    base_tp = int(np.sum(mandatory & (y == 1)))
    base_fp = int(np.sum(mandatory & (y == 0)))
    total_positive = int(np.sum(y))
    optional = np.flatnonzero(~mandatory)
    order = optional[np.argsort(-scores[optional], kind="stable")]
    ordered_scores = scores[order]
    ordered_y = y[order]
    cumulative_tp = np.cumsum(ordered_y)
    cumulative_fp = np.cumsum(1 - ordered_y)
    boundaries = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    tp = base_tp + cumulative_tp[boundaries]
    fp = base_fp + cumulative_fp[boundaries]
    fn = total_positive - tp
    values = 2 * tp / (2 * tp + fp + fn)
    index = int(np.argmax(values))
    boundary = int(boundaries[index])
    return float(values[index]), int(tp[index]), int(fp[index]), int(fn[index]), int(np.sum(mandatory) + boundary + 1), float(ordered_scores[boundary])


features = {"signature": make_signature_feature()}
for k in (1, 2, 3, 5, 8, 12, 20, 35, 60):
    for power in (1.0, 2.0, 4.0):
        features[f"knn_{k}_p{power:g}"] = make_context_feature(k, power)

print(f"orders={n_orders} titles={n_titles} rows={len(rows)}")
print("INDIVIDUAL")
ranked = []
for name, scores in features.items():
    result = best_threshold(scores)
    ranked.append((result[0], name, result))
for item in sorted(ranked, reverse=True):
    print(item[1], item[2])

names = list(features)
matrix = np.column_stack([features[name] for name in names])
top_indices = [names.index(item[1]) for item in sorted(ranked, reverse=True)[:12]]
rng = np.random.default_rng(20260729)
best = None
for _ in range(4000):
    count = int(rng.integers(2, min(7, len(top_indices)) + 1))
    selected = rng.choice(top_indices, size=count, replace=False)
    weights = rng.dirichlet(np.ones(count))
    scores = matrix[:, selected] @ weights
    result = best_threshold(scores)
    record = (result[0], selected.tolist(), weights.tolist(), result)
    if best is None or record[0] > best[0]:
        best = record

print("BEST")
print("score", best[0])
print("features", [names[index] for index in best[1]])
print("weights", best[2])
print("metrics", best[3])


def evaluate_fixed(scores, threshold, cap=None):
    predicted = np.zeros(len(scores), dtype=bool)
    for order_slice in order_slices:
        local_scores = scores[order_slice]
        selected = np.flatnonzero(local_scores >= threshold)
        if not len(selected):
            selected = np.asarray([int(np.argmax(local_scores))])
        if cap is not None and len(selected) > cap:
            selected = selected[np.argsort(-local_scores[selected], kind="stable")[:cap]]
        predicted[order_slice.start + selected] = True
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn), tp, fp, fn, int(np.sum(predicted))


best_scores = matrix[:, best[1]] @ np.asarray(best[2])
best_threshold_value = best[3][-1]
print("CAP_AND_TIEBREAK")
for signature_weight in (0.0, 0.01, 0.03, 0.05):
    adjusted = (1 - signature_weight) * best_scores + signature_weight * features["signature"]
    for cap in (None, 11, 8, 6):
        print(signature_weight, cap, evaluate_fixed(adjusted, best_threshold_value, cap))

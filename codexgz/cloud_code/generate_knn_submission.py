import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


TRAIN_DIR = Path(sys.argv[1])
TEST_DIR = Path(sys.argv[2])
OUTPUT_PATH = Path(sys.argv[3])

CONFIGS = [
    (5, 2.0, 0.06668020883603187),
    (1, 1.0, 0.04820534260274989),
    (8, 2.0, 0.23543407593447566),
    (2, 2.0, 0.2499238097838483),
    (12, 4.0, 0.03659548793379663),
    (3, 1.0, 0.16652631426606984),
    (3, 2.0, 0.19663476064302782),
]
THRESHOLD = 0.5989118247245228
SIGNATURE_WEIGHT = 0.01
MAX_ROOTCAUSES = 8


def location_shape(value):
    value = value or ""
    value = re.sub(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", "<UUID>", value)
    return re.sub(r"\d+", "#", value)


def load_orders(base_dir, with_labels):
    orders = []
    for directory in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        order_id = directory.name
        with (directory / f"{order_id}.log.topo.json").open("r", encoding="utf-8") as handle:
            topology = json.load(handle)
        alarms = [node for node in topology["nodes"] if node.get("@class") == "Alarm"]
        roots = set()
        if with_labels:
            with (directory / f"{order_id}.rootcause.json").open("r", encoding="utf-8") as handle:
                roots = {node["@rid"] for node in json.load(handle)["rootcause"]}
        orders.append((order_id, alarms, roots))
    return orders


train_orders = load_orders(TRAIN_DIR, True)
test_orders = load_orders(TEST_DIR, False)
titles = sorted({node.get("title", "") or "" for _, alarms, _ in train_orders for node in alarms})
title_index = {title: index for index, title in enumerate(titles)}
n_train = len(train_orders)
n_titles = len(titles)

train_alarm_counts = np.zeros((n_train, n_titles), dtype=np.float32)
train_root_counts = np.zeros((n_train, n_titles), dtype=np.float32)
for order_index, (_, alarms, roots) in enumerate(train_orders):
    for node in alarms:
        title_id = title_index[node.get("title", "") or ""]
        train_alarm_counts[order_index, title_id] += 1
        train_root_counts[order_index, title_id] += node["@rid"] in roots

test_alarm_counts = np.zeros((len(test_orders), n_titles), dtype=np.float32)
for order_index, (_, alarms, _) in enumerate(test_orders):
    for node in alarms:
        title = node.get("title", "") or ""
        if title in title_index:
            test_alarm_counts[order_index, title_index[title]] += 1

document_frequency = np.sum(train_alarm_counts > 0, axis=0)
idf = np.log((1 + n_train) / (1 + document_frequency)) + 1
train_tfidf = np.log1p(train_alarm_counts) * idf
test_tfidf = np.log1p(test_alarm_counts) * idf
train_tfidf /= np.maximum(np.linalg.norm(train_tfidf, axis=1, keepdims=True), 1e-12)
test_tfidf /= np.maximum(np.linalg.norm(test_tfidf, axis=1, keepdims=True), 1e-12)
similarities = test_tfidf @ train_tfidf.T

global_positive = np.sum(train_root_counts, axis=0)
global_total = np.sum(train_alarm_counts, axis=0)
prior = (global_positive + 0.3) / (global_total + 1.0)
ensemble_probabilities = np.zeros((len(test_orders), n_titles), dtype=np.float64)

for k, power, ensemble_weight in CONFIGS:
    nearest = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
    for test_index in range(len(test_orders)):
        neighbor_indices = nearest[test_index]
        weights = np.maximum(similarities[test_index, neighbor_indices], 1e-6) ** power
        positive = weights @ train_root_counts[neighbor_indices]
        total = weights @ train_alarm_counts[neighbor_indices]
        probabilities = (positive + 2.0 * prior) / (total + 2.0)
        ensemble_probabilities[test_index] += ensemble_weight * probabilities

title_stats = defaultdict(lambda: [0, 0])
signature_stats = defaultdict(lambda: [0, 0])
total_labels = 0
total_nodes = 0
for _, alarms, roots in train_orders:
    for node in alarms:
        label = int(node["@rid"] in roots)
        title = node.get("title", "") or ""
        title_stats[title][0] += label
        title_stats[title][1] += 1
        signature_stats[(title, location_shape(node.get("location")))][0] += label
        signature_stats[(title, location_shape(node.get("location")))][1] += 1
        total_labels += label
        total_nodes += 1
global_probability = total_labels / total_nodes

predictions = {}
for test_index, (order_id, alarms, _) in enumerate(test_orders):
    node_scores = []
    for node_index, node in enumerate(alarms):
        title = node.get("title", "") or ""
        context_score = ensemble_probabilities[test_index, title_index[title]] if title in title_index else 0.3
        title_positive, title_total = title_stats.get(title, (0, 0))
        title_probability = (title_positive + 5 * global_probability) / (title_total + 5)
        positive, total = signature_stats.get((title, location_shape(node.get("location"))), (0, 0))
        signature_score = (positive + 5 * title_probability) / (total + 5)
        score = (1 - SIGNATURE_WEIGHT) * context_score + SIGNATURE_WEIGHT * signature_score
        node_scores.append((float(score), node_index))
    selected = [node_index for score, node_index in node_scores if score >= THRESHOLD]
    if not selected:
        selected = [max(node_scores)[1]]
    if len(selected) > MAX_ROOTCAUSES:
        selected = sorted(selected, key=lambda index: node_scores[index][0], reverse=True)[:MAX_ROOTCAUSES]
    rootcauses = []
    for node_index in selected:
        node = alarms[node_index]
        rootcauses.append({
            "@rid": node["@rid"],
            "title": node.get("title", ""),
            "location": node.get("location", ""),
            "reason": node.get("reason", ""),
        })
    predictions[order_id] = {"rootcause": rootcauses}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["order_id", "output"])
    for order_id in sorted(predictions):
        writer.writerow([order_id, json.dumps(predictions[order_id], ensure_ascii=False)])

distribution = defaultdict(int)
for payload in predictions.values():
    distribution[len(payload["rootcause"])] += 1
print(f"train_orders={len(train_orders)} test_orders={len(test_orders)}")
print(f"predicted_nodes={sum(k * v for k, v in distribution.items())}")
print(f"rootcause_count_distribution={dict(sorted(distribution.items()))}")
print(f"output={OUTPUT_PATH}")

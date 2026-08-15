"""V16 structure-only root-cause model and equal-swap probe generator.

The model intentionally excludes title, location and vendor values.  It uses
directed graph roles, PageRank, temporal propagation and parsed addInfo fields.
The online probes preserve the champion's per-order K and all V11 locks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


ROOT = Path(r"D:\zgyidong")
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
OUTPUT_DIR = ROOT / "experiments" / "v16"
CHAMPION = ROOT / "experiments" / "submissions" / "champion_0.906324_day01_probe01_v11_full.csv"
V11_DIR = ROOT / "codexgz" / "v11"
LOCK_REPORT = V11_DIR / "v11_constrained_report.json"
EXCLUSIONS = ROOT / "experiments" / "v15" / "v15b_exclusions.json"

N_FOLDS = 5
SEEDS = (20260803, 20260817, 20260831)
TARGET_TEST_COUNT = 1059
MAX_ROOTCAUSES = 8
PROBE_SIZE = 5

NODE_CLASSES = (
    "Alarm",
    "RRU",
    "Cell",
    "BaseStation",
    "TransDevice",
    "TransBoard",
    "Board",
    "BBU",
    "Room",
    "RiPort",
    "NTransPort",
    "TransCircuit",
    "TransCircuitRoute",
    "UTransZPort",
    "UTransAPort",
)
CLASS_INDEX = {name: index for index, name in enumerate(NODE_CLASSES)}
ADDINFO_FIELDS = ("DeviceType", "BoardType", "Cause", "Radio", "deployment", "LDNHead")
VOCAB_LIMITS = {
    "DeviceType": 32,
    "BoardType": 64,
    "Cause": 48,
    "Radio": 20,
    "deployment": 12,
    "LDNHead": 32,
}


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return value or ""


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_orders(base_dir: Path, with_labels: bool):
    orders = []
    for directory in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        order_id = directory.name
        topology = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        alarms = [
            node for node in topology.get("nodes", []) if node.get("@class") == "Alarm"
        ]
        roots = set()
        if with_labels:
            root_data = json.loads(
                (directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
            )
            roots = {node["@rid"] for node in root_data.get("rootcause", [])}
        orders.append(
            {
                "id": order_id,
                "topology": topology,
                "alarms": alarms,
                "roots": roots,
            }
        )
    return orders


def order_signature(order):
    # Used only to prevent template leakage between OOF folds, never as a feature.
    title_counts = Counter(scalar(node.get("title")) for node in order["alarms"])
    target_titles = sorted(
        scalar(node.get("title"))
        for node in order["alarms"]
        if node.get("label") == "TargetAlarm"
    )
    return tuple(sorted(title_counts.items())), tuple(target_titles), len(order["alarms"])


def grouped_folds(orders):
    groups = defaultdict(list)
    for index, order in enumerate(orders):
        groups[order_signature(order)].append(index)
    fold_sizes = [0] * N_FOLDS
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ranked:
        fold = min(range(N_FOLDS), key=lambda value: (fold_sizes[value], value))
        folds[indices] = fold
        fold_sizes[fold] += len(indices)
    return folds, len(groups), fold_sizes


ADDINFO_PATTERN = re.compile(r"(?:^|;)([A-Za-z][A-Za-z0-9_-]*):([^;]*)", re.I)


def parse_addinfo(text):
    text = str(text or "")
    values = {}
    for match in ADDINFO_PATTERN.finditer(text):
        key = match.group(1).strip().lower()
        values.setdefault(key, match.group(2).strip())
    ldn = values.get("ldn", "")
    ldn_head = ""
    if ldn:
        first = ldn.split(",", 1)[0]
        ldn_head = first.split("=", 1)[0].strip()
    return {
        "DeviceType": values.get("devicetype", "").upper(),
        "BoardType": values.get("boardtype", "").upper(),
        "Cause": values.get("cause", ""),
        "Radio": values.get("radio", "").upper(),
        "deployment": values.get("deployment", "").upper(),
        "LDNHead": ldn_head.upper(),
        "Aid": values.get("aid", ""),
        "DeviceId": values.get("deviceid", ""),
        "raw": text,
        "key_count": len(values),
    }


def build_vocab(train_orders):
    counters = {name: Counter() for name in ADDINFO_FIELDS}
    for order in train_orders:
        for node in order["alarms"]:
            parsed = parse_addinfo(node.get("addInfo"))
            for name in ADDINFO_FIELDS:
                value = parsed[name]
                if value:
                    counters[name][value] += 1
    vocab = {}
    for name in ADDINFO_FIELDS:
        values = [value for value, _ in counters[name].most_common(VOCAB_LIMITS[name])]
        vocab[name] = {value: index for index, value in enumerate(values)}
    return vocab


def class_vector(indices, nodes):
    output = np.zeros(len(NODE_CLASSES), dtype=np.float32)
    for index in indices:
        class_name = nodes[index].get("@class")
        if class_name in CLASS_INDEX:
            output[CLASS_INDEX[class_name]] += 1.0
    return output


def pagerank(adjacency, iterations=20, damping=0.85):
    count = len(adjacency)
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    scores = np.full(count, 1.0 / count, dtype=np.float64)
    for _ in range(iterations):
        updated = np.full(count, (1.0 - damping) / count, dtype=np.float64)
        dangling = 0.0
        for source, neighbors in enumerate(adjacency):
            if neighbors:
                share = damping * scores[source] / len(neighbors)
                for target in neighbors:
                    updated[target] += share
            else:
                dangling += damping * scores[source] / count
        updated += dangling
        scores = updated
    return scores.astype(np.float32)


def shortest_distances(adjacency, starts):
    unreachable = len(adjacency) + 1
    distances = np.full(len(adjacency), unreachable, dtype=np.int32)
    queue = deque()
    for start in starts:
        distances[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if distances[neighbor] > distances[current] + 1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def temporal_features(node, fault_time, order_times, title_group_size):
    values = node.get("timeLists", [])
    if not isinstance(values, list):
        values = []
    timeline = np.zeros(6, dtype=np.float32)
    for index, value in enumerate(values[:6]):
        timeline[index] = safe_float(value)
    binary = timeline > 0
    transitions = int(np.sum(binary[1:] != binary[:-1]))
    segments = int(binary[0]) + int(np.sum((~binary[:-1]) & binary[1:]))

    def longest(target):
        best = current = 0
        for value in binary:
            if bool(value) == target:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    ones = np.flatnonzero(binary)
    first_one = float(ones[0] / 5.0) if len(ones) else 1.2
    last_one = float(ones[-1] / 5.0) if len(ones) else -0.2
    recency = float(np.dot(timeline, np.arange(1, 7)) / max(np.sum(timeline), 1.0))
    slope = float(np.polyfit(np.arange(6), timeline, 1)[0])
    node_time = safe_float(node.get("time"), fault_time)
    sorted_times = sorted(order_times)
    rank = sorted_times.index(node_time) / max(len(sorted_times) - 1, 1)
    min_time = min(order_times) if order_times else node_time
    max_time = max(order_times) if order_times else node_time
    span = max(max_time - min_time, 1.0)
    return np.asarray(
        timeline.tolist()
        + [
            float(np.sum(timeline)),
            float(np.mean(timeline)),
            float(np.std(timeline)),
            float(np.min(timeline)),
            float(np.max(timeline)),
            float(transitions),
            float(segments),
            float(longest(True)),
            float(longest(False)),
            first_one,
            last_one,
            recency,
            slope,
            math.log1p(abs(fault_time - node_time)),
            rank,
            (node_time - min_time) / span,
            (max_time - node_time) / span,
            float(sum(value == node_time for value in order_times)),
            float(title_group_size),
        ],
        dtype=np.float32,
    )


def addinfo_features(parsed, vocab, order_aids):
    raw = parsed["raw"]
    digits = re.findall(r"\d+", raw)
    aid = safe_float(parsed["Aid"], 0.0)
    device_id = safe_float(parsed["DeviceId"], 0.0)
    aid_values = sorted(order_aids)
    aid_rank = aid_values.index(aid) / max(len(aid_values) - 1, 1) if aid and aid in aid_values else -1.0
    numeric = [
        float(bool(raw)),
        math.log1p(len(raw)),
        float(raw.count(";")),
        float(raw.count(",")),
        float(parsed["key_count"]),
        float(len(digits)),
        float(sum(character.isdigit() for character in raw) / max(len(raw), 1)),
        float("PING" in raw.upper()),
        float("LOSS" in raw.upper() or "% loss" in raw.lower()),
        math.log1p(abs(aid)),
        aid_rank,
        math.log1p(abs(device_id)),
        float(parsed["Cause"].isdigit()),
        math.log1p(safe_float(parsed["Cause"], 0.0)),
    ]
    one_hot = []
    for name in ADDINFO_FIELDS:
        vector = np.zeros(len(vocab[name]) + 2, dtype=np.float32)
        value = parsed[name]
        if not value:
            vector[-2] = 1.0
        elif value in vocab[name]:
            vector[vocab[name][value]] = 1.0
        else:
            vector[-1] = 1.0
        one_hot.extend(vector.tolist())
    return np.asarray(numeric + one_hot, dtype=np.float32)


def order_features(order, vocab):
    nodes = order["topology"].get("nodes", [])
    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    outgoing = [set() for _ in nodes]
    incoming = [set() for _ in nodes]
    undirected = [set() for _ in nodes]
    for edge in order["topology"].get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is None or target is None:
            continue
        outgoing[source].add(target)
        incoming[target].add(source)
        undirected[source].add(target)
        undirected[target].add(source)
    alarm_indices = [rid_to_index[node["@rid"]] for node in order["alarms"]]
    alarm_set = set(alarm_indices)
    target_set = {
        index for index in alarm_indices if nodes[index].get("label") == "TargetAlarm"
    }
    from_target = shortest_distances(outgoing, target_set)
    to_target = shortest_distances(incoming, target_set)
    undirected_target = shortest_distances(undirected, target_set)
    directed_pr = pagerank(outgoing)
    undirected_pr = pagerank(undirected)
    alarm_pr_values = sorted(float(directed_pr[index]) for index in alarm_indices)
    title_counts = Counter(scalar(node.get("title")) for node in order["alarms"])
    fault_time = safe_float(order["topology"].get("time"), 0.0)
    order_times = [safe_float(node.get("time"), fault_time) for node in order["alarms"]]
    parsed_infos = [parse_addinfo(node.get("addInfo")) for node in order["alarms"]]
    order_aids = sorted(
        value
        for parsed in parsed_infos
        for value in [safe_float(parsed["Aid"], 0.0)]
        if value
    )
    rows = []
    for local_index, node in enumerate(order["alarms"]):
        node_index = alarm_indices[local_index]
        out = outgoing[node_index]
        inc = incoming[node_index]
        und = undirected[node_index]
        two_hop = set()
        for neighbor in und:
            two_hop.update(undirected[neighbor])
        two_hop.discard(node_index)
        three_hop = set(two_hop)
        for neighbor in list(two_hop):
            three_hop.update(undirected[neighbor])
        three_hop.discard(node_index)
        unreachable = len(nodes) + 1
        pr_value = float(directed_pr[node_index])
        pr_rank = sum(value < pr_value for value in alarm_pr_values) / max(len(alarm_pr_values) - 1, 1)
        graph = [
            float(node.get("label") == "TargetAlarm"),
            float(len(target_set)),
            float(len(alarm_indices)),
            float(len(nodes)),
            float(len(out)),
            float(len(inc)),
            float(len(und)),
            float(sum(index in alarm_set for index in out)),
            float(sum(index in alarm_set for index in inc)),
            float(sum(index in target_set for index in out)),
            float(sum(index in target_set for index in inc)),
            float(sum(index in alarm_set for index in und)) / max(len(und), 1),
            float(sum(index in target_set for index in und)) / max(len(und), 1),
            math.log1p(min(int(from_target[node_index]), unreachable)),
            math.log1p(min(int(to_target[node_index]), unreachable)),
            math.log1p(min(int(undirected_target[node_index]), unreachable)),
            float(from_target[node_index] <= len(nodes)),
            float(to_target[node_index] <= len(nodes)),
            float(directed_pr[node_index]),
            float(undirected_pr[node_index]),
            pr_rank,
            float(len(two_hop)),
            float(sum(index in alarm_set for index in two_hop)),
            float(sum(index in target_set for index in two_hop)),
            float(len(three_hop)),
            float(sum(index in alarm_set for index in three_hop)),
            float(sum(index in target_set for index in three_hop)),
        ]
        graph.extend(class_vector(out, nodes).tolist())
        graph.extend(class_vector(inc, nodes).tolist())
        graph.extend(class_vector(two_hop, nodes).tolist())
        time = temporal_features(
            node,
            fault_time,
            order_times,
            title_counts[scalar(node.get("title"))],
        )
        addinfo = addinfo_features(parsed_infos[local_index], vocab, order_aids)
        rows.append(np.concatenate([np.asarray(graph, dtype=np.float32), time, addinfo]))
    return np.vstack(rows)


def feature_matrix(orders, vocab):
    matrices = []
    slices = []
    cursor = 0
    for index, order in enumerate(orders):
        matrix = order_features(order, vocab)
        matrices.append(matrix)
        slices.append(slice(cursor, cursor + len(matrix)))
        cursor += len(matrix)
        if (index + 1) % 200 == 0:
            print(f"features {index + 1}/{len(orders)}", flush=True)
    return np.vstack(matrices), slices


def rows_for_orders(slices, order_indices):
    return np.concatenate(
        [np.arange(slices[int(index)].start, slices[int(index)].stop) for index in order_indices]
    )


def local_slices(orders, order_indices):
    output = []
    cursor = 0
    for index in order_indices:
        length = len(orders[int(index)]["alarms"])
        output.append(slice(cursor, cursor + length))
        cursor += length
    return output


def exact_count_mask(scores, slices, target_count):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order_slice in slices:
        local_scores = scores[order_slice]
        ranked = np.argsort(-local_scores, kind="stable")[:MAX_ROOTCAUSES]
        selected[order_slice.start + ranked[0]] = True
        optional.extend(order_slice.start + ranked[1:])
    remaining = max(0, target_count - int(np.sum(selected)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def fixed_k_mask(scores, slices, reference_mask):
    selected = np.zeros(len(scores), dtype=bool)
    for order_slice in slices:
        count = int(np.sum(reference_mask[order_slice]))
        ranked = np.argsort(-scores[order_slice], kind="stable")[:count]
        selected[order_slice.start + ranked] = True
    return selected


def metrics(mask, labels):
    tp = int(np.sum(mask & (labels == 1)))
    fp = int(np.sum(mask & (labels == 0)))
    fn = int(np.sum((~mask) & (labels == 1)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"f1": f1, "tp": tp, "fp": fp, "fn": fn, "predictions": int(np.sum(mask))}


def rank_values(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def read_submission(path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["order_id", "output"]
        for row in reader:
            rows[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return rows


def validate_submission(path, test_orders):
    rows = read_submission(path)
    assert set(rows) == {order["id"] for order in test_orders}
    total = 0
    distribution = Counter()
    for order in test_orders:
        order_id = order["id"]
        rootcauses = rows[order_id]
        assert 1 <= len(rootcauses) <= MAX_ROOTCAUSES, (order_id, len(rootcauses))
        nodes = {node["@rid"]: node for node in order["topology"].get("nodes", [])}
        seen = set()
        for item in rootcauses:
            assert list(item) == ["@rid", "title", "location", "reason"]
            rid = item["@rid"]
            assert rid not in seen and rid in nodes
            seen.add(rid)
            source = nodes[rid]
            for field in ("title", "location", "reason"):
                assert item[field] == source.get(field, ""), (order_id, rid, field)
        total += len(rootcauses)
        distribution[len(rootcauses)] += 1
    assert len(rows) == 546 and total == TARGET_TEST_COUNT
    return {"orders": len(rows), "predictions": total, "distribution": dict(sorted(distribution.items()))}


def write_probe(path, champion_rows, test_by_id, swaps):
    rows = {order_id: list(items) for order_id, items in champion_rows.items()}
    for swap in swaps:
        order_id = swap["order_id"]
        removed = swap["removed_rid"]
        added = swap["added_rid"]
        before_count = len(rows[order_id])
        rows[order_id] = [item for item in rows[order_id] if item["@rid"] != removed]
        source = {
            node["@rid"]: node for node in test_by_id[order_id]["alarms"]
        }[added]
        rows[order_id].append(
            {
                "@rid": source["@rid"],
                "title": source.get("title", ""),
                "location": source.get("location", ""),
                "reason": source.get("reason", ""),
            }
        )
        assert len(rows[order_id]) == before_count
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in sorted(rows):
            writer.writerow(
                [order_id, json.dumps({"rootcause": rows[order_id]}, ensure_ascii=False)]
            )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "source_champion": str(CHAMPION),
        "delta_p": 0,
        "swaps": swaps,
        "sha256": sha,
    }
    path.with_suffix(".diff.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    path.with_suffix(".sha256").write_text(f"{sha}  {path.name}\n", encoding="ascii")


def swap_catalog(orders, slices, scores, per_seed, selected_mask, labels=None, excluded=None):
    excluded = excluded or set()
    output = []
    for order_index, order in enumerate(orders):
        if order["id"] in excluded:
            continue
        order_slice = slices[order_index]
        selected_local = np.flatnonzero(selected_mask[order_slice])
        unselected_local = np.flatnonzero(~selected_mask[order_slice])
        if not len(selected_local) or not len(unselected_local):
            continue
        local_scores = scores[order_slice]
        removed_local = int(selected_local[np.argmin(local_scores[selected_local])])
        added_local = int(unselected_local[np.argmax(local_scores[unselected_local])])
        removed_row = order_slice.start + removed_local
        added_row = order_slice.start + added_local
        seed_margins = [float(seed[added_row] - seed[removed_row]) for seed in per_seed]
        record = {
            "order_id": order["id"],
            "removed_rid": order["alarms"][removed_local]["@rid"],
            "added_rid": order["alarms"][added_local]["@rid"],
            "removed_score": float(scores[removed_row]),
            "added_score": float(scores[added_row]),
            "margin": float(scores[added_row] - scores[removed_row]),
            "seed_margins": seed_margins,
            "seed_min_margin": min(seed_margins),
            "high_conflict": bool(scores[added_row] >= 0.7 and scores[removed_row] <= 0.4),
        }
        if labels is not None:
            record["label_delta"] = int(labels[added_row] - labels[removed_row])
        output.append(record)
    return sorted(
        output,
        key=lambda item: (
            item["high_conflict"],
            item["seed_min_margin"],
            item["margin"],
            item["order_id"],
        ),
        reverse=True,
    )


def swap_feature_matrix(catalog, orders, slices, node_features, structure_scores, structure_per_seed, base_scores, selected_mask):
    order_index = {order["id"]: index for index, order in enumerate(orders)}
    matrices = []
    candidate_order_indices = []
    for item in catalog:
        oi = order_index[item["order_id"]]
        order = orders[oi]
        local_by_rid = {node["@rid"]: index for index, node in enumerate(order["alarms"])}
        removed_local = local_by_rid[item["removed_rid"]]
        added_local = local_by_rid[item["added_rid"]]
        removed_row = slices[oi].start + removed_local
        added_row = slices[oi].start + added_local
        raw = [
            float(structure_scores[added_row]),
            float(structure_scores[removed_row]),
            float(structure_scores[added_row] - structure_scores[removed_row]),
            float(base_scores[added_row]),
            float(base_scores[removed_row]),
            float(base_scores[added_row] - base_scores[removed_row]),
            float(np.sum(selected_mask[slices[oi]])),
            float(len(order["alarms"])),
        ]
        for seed_scores in structure_per_seed:
            raw.extend(
                [
                    float(seed_scores[added_row]),
                    float(seed_scores[removed_row]),
                    float(seed_scores[added_row] - seed_scores[removed_row]),
                ]
            )
        added_features = node_features[added_row]
        removed_features = node_features[removed_row]
        matrices.append(
            np.concatenate(
                [
                    np.asarray(raw, dtype=np.float32),
                    added_features,
                    removed_features,
                    added_features - removed_features,
                ]
            )
        )
        candidate_order_indices.append(oi)
    return np.vstack(matrices), np.asarray(candidate_order_indices, dtype=np.int32)


def cross_validated_swap_meta(features, targets, candidate_folds):
    per_seed = [np.zeros(len(targets), dtype=np.float64) for _ in SEEDS]
    for heldout in range(N_FOLDS):
        train_rows = np.flatnonzero(candidate_folds != heldout)
        validation_rows = np.flatnonzero(candidate_folds == heldout)
        for seed_index, seed in enumerate(SEEDS):
            model = ExtraTreesRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=8,
                max_features=0.6,
                n_jobs=-1,
                random_state=seed + heldout,
            )
            model.fit(features[train_rows], targets[train_rows])
            per_seed[seed_index][validation_rows] = model.predict(features[validation_rows])
    return per_seed


def top_candidate_validation(catalog, folds, score_key, counts=(5, 10, 20)):
    output = {}
    for count in counts:
        fold_deltas = []
        for fold in range(N_FOLDS):
            candidates = sorted(
                [item for item in catalog if folds[item["order_index"]] == fold],
                key=lambda item: (item[score_key], item["seed_min_margin"], item["margin"]),
                reverse=True,
            )[:count]
            fold_deltas.append(sum(item["label_delta"] for item in candidates))
        output[str(count)] = {
            "fold_deltas": fold_deltas,
            "net_delta": sum(fold_deltas),
        }
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading orders", flush=True)
    train_orders = load_orders(TRAIN_DIR, True)
    test_orders = load_orders(TEST_DIR, False)
    vocab = build_vocab(train_orders)
    (OUTPUT_DIR / "addinfo_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("building structure-only features", flush=True)
    train_x, train_slices = feature_matrix(train_orders, vocab)
    test_x, test_slices = feature_matrix(test_orders, vocab)
    np.save(OUTPUT_DIR / "v16_train_features.npy", train_x)
    np.save(OUTPUT_DIR / "v16_test_features.npy", test_x)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in train_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    folds, group_count, fold_sizes = grouped_folds(train_orders)
    all_orders = np.arange(len(train_orders))
    oof_per_seed = [np.zeros(len(labels), dtype=np.float64) for _ in SEEDS]
    for heldout in range(N_FOLDS):
        train_order_indices = all_orders[folds != heldout]
        validation_order_indices = all_orders[folds == heldout]
        train_rows = rows_for_orders(train_slices, train_order_indices)
        validation_rows = rows_for_orders(train_slices, validation_order_indices)
        for seed_index, seed in enumerate(SEEDS):
            model = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=18,
                min_samples_leaf=3,
                max_features=0.75,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + heldout,
            )
            model.fit(train_x[train_rows], labels[train_rows])
            oof_per_seed[seed_index][validation_rows] = model.predict_proba(
                train_x[validation_rows]
            )[:, 1]
        print(f"oof fold {heldout} complete", flush=True)
    oof_scores = np.mean(oof_per_seed, axis=0)
    np.save(OUTPUT_DIR / "v16_oof_scores.npy", oof_scores)
    for seed, scores in zip(SEEDS, oof_per_seed):
        np.save(OUTPUT_DIR / f"v16_oof_seed_{seed}.npy", scores)

    v11_oof = 0.25 * np.load(V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        V11_DIR / "v11_oof_meta.npy"
    )
    target_train_count = round(TARGET_TEST_COUNT / len(test_orders) * len(train_orders))
    base_mask = exact_count_mask(v11_oof, train_slices, target_train_count)
    structure_mask = exact_count_mask(oof_scores, train_slices, target_train_count)
    fixed_structure_mask = fixed_k_mask(oof_scores, train_slices, base_mask)
    weight_scan = []
    for weight in np.linspace(0.0, 0.6, 13):
        blended = (1.0 - weight) * v11_oof + weight * oof_scores
        global_mask = exact_count_mask(blended, train_slices, target_train_count)
        fixed_mask = fixed_k_mask(blended, train_slices, base_mask)
        weight_scan.append(
            {
                "weight": float(weight),
                "global": metrics(global_mask, labels),
                "fixed_k": metrics(fixed_mask, labels),
            }
        )
    pearson = float(np.corrcoef(v11_oof, oof_scores)[0, 1])
    spearman = float(np.corrcoef(rank_values(v11_oof), rank_values(oof_scores))[0, 1])

    oof_catalog = swap_catalog(
        train_orders,
        train_slices,
        oof_scores,
        oof_per_seed,
        base_mask,
        labels=labels,
    )
    order_index_by_id = {order["id"]: index for index, order in enumerate(train_orders)}
    for item in oof_catalog:
        item["order_index"] = order_index_by_id[item["order_id"]]
    top_swap_validation = top_candidate_validation(oof_catalog, folds, "seed_min_margin")
    oof_swap_x, oof_swap_order_indices = swap_feature_matrix(
        oof_catalog,
        train_orders,
        train_slices,
        train_x,
        oof_scores,
        oof_per_seed,
        v11_oof,
        base_mask,
    )
    oof_swap_y = np.asarray([item["label_delta"] for item in oof_catalog], dtype=np.float32)
    meta_per_seed = cross_validated_swap_meta(
        oof_swap_x, oof_swap_y, folds[oof_swap_order_indices]
    )
    meta_scores = np.mean(meta_per_seed, axis=0)
    for index, item in enumerate(oof_catalog):
        item["meta_mean"] = float(meta_scores[index])
        item["meta_seed_min"] = float(min(scores[index] for scores in meta_per_seed))
        item["meta_seed_scores"] = [float(scores[index]) for scores in meta_per_seed]
    meta_swap_validation = top_candidate_validation(oof_catalog, folds, "meta_seed_min")

    report = {
        "feature_policy": {
            "excluded": ["title value", "location value", "vendor value"],
            "included": ["directed graph role", "PageRank", "timeLists", "time rank", "parsed addInfo", "same-title group size"],
        },
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "train_rows": len(labels),
        "feature_dim": int(train_x.shape[1]),
        "positive_labels": int(np.sum(labels)),
        "group_count": group_count,
        "fold_sizes": fold_sizes,
        "target_train_count": target_train_count,
        "correlation": {"pearson": pearson, "spearman": spearman},
        "v11_exact": metrics(base_mask, labels),
        "structure_exact": metrics(structure_mask, labels),
        "structure_fixed_v11_k": metrics(fixed_structure_mask, labels),
        "weight_scan": weight_scan,
        "raw_margin_swap_validation": top_swap_validation,
        "meta_swap_validation": meta_swap_validation,
        "swap_label_distribution": dict(
            sorted(Counter(int(value) for value in oof_swap_y).items())
        ),
    }
    (OUTPUT_DIR / "v16_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "v16_oof_swap_catalog.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_catalog[0]))
        writer.writeheader()
        writer.writerows(oof_catalog)

    print("training full structure model", flush=True)
    test_per_seed = []
    for seed in SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_x, labels)
        test_per_seed.append(model.predict_proba(test_x)[:, 1])
    test_scores = np.mean(test_per_seed, axis=0)
    np.save(OUTPUT_DIR / "v16_test_scores.npy", test_scores)
    for seed, scores in zip(SEEDS, test_per_seed):
        np.save(OUTPUT_DIR / f"v16_test_seed_{seed}.npy", scores)

    champion_rows = read_submission(CHAMPION)
    champion_mask = np.zeros(len(test_scores), dtype=bool)
    for order_index, order in enumerate(test_orders):
        selected = {item["@rid"] for item in champion_rows[order["id"]]}
        for local_index, node in enumerate(order["alarms"]):
            champion_mask[test_slices[order_index].start + local_index] = node["@rid"] in selected
    exclusion_data = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    excluded_orders = set(exclusion_data["total"])
    lock_data = json.loads(LOCK_REPORT.read_text(encoding="utf-8"))
    excluded_orders.update(
        item["order_id"] for item in lock_data["forced_in"] + lock_data["forced_out"]
    )
    test_catalog = swap_catalog(
        test_orders,
        test_slices,
        test_scores,
        test_per_seed,
        champion_mask,
        excluded=excluded_orders,
    )
    v11_test_scores = np.load(V11_DIR / "v11_test_scores.npy")
    test_swap_x, _ = swap_feature_matrix(
        test_catalog,
        test_orders,
        test_slices,
        test_x,
        test_scores,
        test_per_seed,
        v11_test_scores,
        champion_mask,
    )
    test_meta_per_seed = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=10,
            min_samples_leaf=8,
            max_features=0.6,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(oof_swap_x, oof_swap_y)
        test_meta_per_seed.append(model.predict(test_swap_x))
    test_meta_scores = np.mean(test_meta_per_seed, axis=0)
    for index, item in enumerate(test_catalog):
        item["meta_mean"] = float(test_meta_scores[index])
        item["meta_seed_min"] = float(
            min(scores[index] for scores in test_meta_per_seed)
        )
        item["meta_seed_scores"] = [
            float(scores[index]) for scores in test_meta_per_seed
        ]
    test_catalog = sorted(
        test_catalog,
        key=lambda item: (
            item["meta_seed_min"],
            item["meta_mean"],
            item["high_conflict"],
            item["seed_min_margin"],
        ),
        reverse=True,
    )
    stable_catalog = [
        item for item in test_catalog if item["seed_min_margin"] > 0
    ]
    with (OUTPUT_DIR / "v16_test_swap_catalog.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_catalog[0]))
        writer.writeheader()
        writer.writerows(test_catalog)

    probe_a_swaps = stable_catalog[:PROBE_SIZE]
    probe_b_swaps = stable_catalog[PROBE_SIZE : 2 * PROBE_SIZE]
    if len(probe_a_swaps) < PROBE_SIZE or len(probe_b_swaps) < PROBE_SIZE:
        raise RuntimeError(
            f"not enough stable equal-swap candidates: {len(stable_catalog)}"
        )
    test_by_id = {order["id"]: order for order in test_orders}
    probe_paths = [
        OUTPUT_DIR / "v16_probeA_meta_structure_equal_swap5_p1059.csv",
        OUTPUT_DIR / "v16_probeB_meta_structure_equal_swap5_p1059.csv",
    ]
    for path, swaps in zip(probe_paths, (probe_a_swaps, probe_b_swaps)):
        write_probe(path, champion_rows, test_by_id, swaps)
        validation = validate_submission(path, test_orders)
        metadata_path = path.with_suffix(".diff.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["validation"] = validation
        metadata["excluded_order_count"] = len(excluded_orders)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    with (OUTPUT_DIR / "v16_test_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["order_id", "rid", "structure_mean"]
            + [f"structure_seed_{seed}" for seed in SEEDS]
            + ["champion_selected"]
        )
        for order_index, order in enumerate(test_orders):
            order_slice = test_slices[order_index]
            for local_index, node in enumerate(order["alarms"]):
                row_index = order_slice.start + local_index
                writer.writerow(
                    [order["id"], node["@rid"], float(test_scores[row_index])]
                    + [float(scores[row_index]) for scores in test_per_seed]
                    + [int(champion_mask[row_index])]
                )
    summary = {
        "probe_a": str(probe_paths[0]),
        "probe_b": str(probe_paths[1]),
        "stable_candidates": len(stable_catalog),
        "high_conflict_candidates": sum(item["high_conflict"] for item in stable_catalog),
        "top_a_margins": [item["margin"] for item in probe_a_swaps],
        "top_b_margins": [item["margin"] for item in probe_b_swaps],
        "top_a_meta": [item["meta_mean"] for item in probe_a_swaps],
        "top_b_meta": [item["meta_mean"] for item in probe_b_swaps],
    }
    (OUTPUT_DIR / "v16_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

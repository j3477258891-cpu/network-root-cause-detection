import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


SEEDS = (20260801, 20260817, 20260831)
MIN_SUPPORT = 2
MIN_CONFIDENCE = 0.90


def scalar(value):
    if isinstance(value, list):
        return tuple(value)
    return "" if value is None else value


def location_shape(value):
    value = str(value or "")
    value = re.sub(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        "<UUID>",
        value,
    )
    return re.sub(r"\d+", "#", value)


def stable_digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:24]


def shortest_distances(adjacency, starts, node_count):
    unreachable = node_count + 1
    distance = [unreachable] * node_count
    queue = deque()
    for start in starts:
        distance[start] = 0
        queue.append(start)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if distance[neighbor] > distance[current] + 1:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def prepare_order(directory, with_labels):
    order_id = directory.name
    topology = json.loads(
        (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
    )
    nodes = topology.get("nodes", [])
    alarms = [node for node in nodes if node.get("@class") == "Alarm"]
    roots = set()
    if with_labels:
        root_data = json.loads(
            (directory / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
        )
        roots = {node["@rid"] for node in root_data.get("rootcause", [])}

    rid_to_index = {node.get("@rid"): index for index, node in enumerate(nodes)}
    incoming = [set() for _ in nodes]
    outgoing = [set() for _ in nodes]
    for edge in topology.get("edges", []):
        source = rid_to_index.get(edge.get("in"))
        target = rid_to_index.get(edge.get("out"))
        if source is None or target is None:
            continue
        outgoing[source].add(target)
        incoming[target].add(source)

    base_tokens = []
    for node in nodes:
        base_tokens.append(
            (
                scalar(node.get("@class")),
                scalar(node.get("label")),
                scalar(node.get("title")),
                location_shape(node.get("location")),
                scalar(node.get("device")),
            )
        )
    colors = [stable_digest(token) for token in base_tokens]
    structural_colors = [
        stable_digest((token[0], token[1], token[2])) for token in base_tokens
    ]
    for _ in range(3):
        colors = [
            stable_digest(
                (
                    colors[index],
                    tuple(sorted(colors[value] for value in incoming[index])),
                    tuple(sorted(colors[value] for value in outgoing[index])),
                )
            )
            for index in range(len(nodes))
        ]
        structural_colors = [
            stable_digest(
                (
                    structural_colors[index],
                    tuple(
                        sorted(structural_colors[value] for value in incoming[index])
                    ),
                    tuple(
                        sorted(structural_colors[value] for value in outgoing[index])
                    ),
                )
            )
            for index in range(len(nodes))
        ]

    target_indices = [
        index
        for index, node in enumerate(nodes)
        if node.get("@class") == "Alarm" and node.get("label") == "TargetAlarm"
    ]
    to_target = shortest_distances(outgoing, target_indices, len(nodes))
    from_target = shortest_distances(incoming, target_indices, len(nodes))
    fingerprints = {}
    for node in alarms:
        index = rid_to_index[node["@rid"]]
        incoming_alarm_titles = tuple(
            sorted(
                scalar(nodes[value].get("title"))
                for value in incoming[index]
                if nodes[value].get("@class") == "Alarm"
            )
        )
        outgoing_alarm_titles = tuple(
            sorted(
                scalar(nodes[value].get("title"))
                for value in outgoing[index]
                if nodes[value].get("@class") == "Alarm"
            )
        )
        strict = (
            scalar(node.get("title")),
            scalar(node.get("label")),
            location_shape(node.get("location")),
            colors[index],
            to_target[index],
            from_target[index],
        )
        medium = (
            scalar(node.get("title")),
            scalar(node.get("label")),
            location_shape(node.get("location")),
            len(incoming[index]),
            len(outgoing[index]),
            to_target[index],
            from_target[index],
            incoming_alarm_titles,
            outgoing_alarm_titles,
        )
        structural = (
            scalar(node.get("title")),
            scalar(node.get("label")),
            structural_colors[index],
            to_target[index],
            from_target[index],
            incoming_alarm_titles,
            outgoing_alarm_titles,
        )
        fingerprints[node["@rid"]] = {
            "strict": strict,
            "medium": medium,
            "structural": structural,
        }

    title_counts = Counter(scalar(node.get("title")) for node in alarms)
    target_titles = sorted(
        scalar(node.get("title"))
        for node in alarms
        if node.get("label") == "TargetAlarm"
    )
    signature = (tuple(sorted(title_counts.items())), tuple(target_titles), len(alarms))
    return {
        "id": order_id,
        "alarms": alarms,
        "roots": roots,
        "signature": signature,
        "fingerprints": fingerprints,
    }


def load_orders(base_dir, with_labels):
    return [
        prepare_order(directory, with_labels)
        for directory in sorted(path for path in Path(base_dir).iterdir() if path.is_dir())
    ]


def load_submission(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return result


def fingerprint_index(order, tier):
    result = defaultdict(list)
    for node in order["alarms"]:
        result[order["fingerprints"][node["@rid"]][tier]].append(node["@rid"])
    return result


def labels_for_key(references, tier, key):
    labels = []
    for reference in references:
        matches = fingerprint_index(reference, tier).get(key, [])
        if len(matches) == 1:
            labels.append(int(matches[0] in reference["roots"]))
    return labels


def seeded_probabilities(labels):
    probabilities = []
    for seed in SEEDS:
        rng = random.Random(seed)
        sample = [labels[rng.randrange(len(labels))] for _ in labels]
        probabilities.append(sum(sample) / len(sample))
    return probabilities


def node_vote(query, rid, references):
    for tier in ("strict", "medium", "structural"):
        key = query["fingerprints"][rid][tier]
        if len(fingerprint_index(query, tier).get(key, [])) != 1:
            continue
        labels = labels_for_key(references, tier, key)
        if len(labels) < MIN_SUPPORT:
            continue
        probability = sum(labels) / len(labels)
        seed_probabilities = seeded_probabilities(labels)
        positive = probability >= MIN_CONFIDENCE and all(
            value >= MIN_CONFIDENCE for value in seed_probabilities
        )
        negative = probability <= 1.0 - MIN_CONFIDENCE and all(
            value <= 1.0 - MIN_CONFIDENCE for value in seed_probabilities
        )
        return {
            "tier": tier,
            "support": len(labels),
            "probability": probability,
            "seed_probabilities": seed_probabilities,
            "positive": positive,
            "negative": negative,
        }
    return None


def stable_k(references):
    values = [len(order["roots"]) for order in references]
    winner, count = Counter(values).most_common(1)[0]
    if count / len(values) < MIN_CONFIDENCE:
        return None
    seed_values = []
    for seed in SEEDS:
        rng = random.Random(seed)
        sample = [values[rng.randrange(len(values))] for _ in values]
        seed_values.append(Counter(sample).most_common(1)[0][0])
    return winner if all(value == winner for value in seed_values) else None


def canonical_key(node):
    return (
        scalar(node.get("title")),
        scalar(node.get("label")),
        location_shape(node.get("location")),
    )


def canonical_template_prediction(query, references):
    if len(references) < MIN_SUPPORT:
        return None
    root_patterns = [
        tuple(
            sorted(
                Counter(
                    canonical_key(node)
                    for node in reference["alarms"]
                    if node["@rid"] in reference["roots"]
                ).items()
            )
        )
        for reference in references
    ]
    expected_pattern, pattern_support = Counter(root_patterns).most_common(1)[0]
    if pattern_support / len(root_patterns) < MIN_CONFIDENCE:
        return None
    for seed in SEEDS:
        rng = random.Random(seed)
        sample = [root_patterns[rng.randrange(len(root_patterns))] for _ in root_patterns]
        if Counter(sample).most_common(1)[0][0] != expected_pattern:
            return None

    expected = Counter(dict(expected_pattern))
    query_nodes = defaultdict(list)
    for node in query["alarms"]:
        query_nodes[canonical_key(node)].append(node["@rid"])
    selected = set()
    for key, count in expected.items():
        matches = query_nodes.get(key, [])
        if len(matches) != count:
            return None
        selected.update(matches)

    votes = {}
    for key, rids in query_nodes.items():
        labels = []
        for reference in references:
            matching = [
                node
                for node in reference["alarms"]
                if canonical_key(node) == key
            ]
            if not matching:
                continue
            root_count = sum(node["@rid"] in reference["roots"] for node in matching)
            labels.extend([1] * root_count)
            labels.extend([0] * (len(matching) - root_count))
        if len(labels) < MIN_SUPPORT:
            continue
        probability = sum(labels) / len(labels)
        seed_probabilities = seeded_probabilities(labels)
        vote = {
            "tier": "canonical",
            "support": len(labels),
            "probability": probability,
            "seed_probabilities": seed_probabilities,
            "positive": probability >= MIN_CONFIDENCE and all(
                value >= MIN_CONFIDENCE for value in seed_probabilities
            ),
            "negative": probability <= 1.0 - MIN_CONFIDENCE and all(
                value <= 1.0 - MIN_CONFIDENCE for value in seed_probabilities
            ),
        }
        for rid in rids:
            votes[rid] = vote
    if any(not votes.get(rid, {}).get("positive", False) for rid in selected):
        return None
    return {
        "selected": selected,
        "negative": {rid for rid, vote in votes.items() if vote["negative"]},
        "expected_k": sum(expected.values()),
        "votes": votes,
    }


def template_prediction(query, references):
    expected_k = stable_k(references)
    if expected_k is None:
        return None
    votes = {}
    for node in query["alarms"]:
        vote = node_vote(query, node["@rid"], references)
        if vote is not None:
            votes[node["@rid"]] = vote
    positives = [rid for rid, vote in votes.items() if vote["positive"]]
    negatives = [rid for rid, vote in votes.items() if vote["negative"]]
    if len(positives) != expected_k:
        return None
    return {
        "selected": set(positives),
        "negative": set(negatives),
        "expected_k": expected_k,
        "votes": votes,
    }


def loo_validation(groups, predictor=template_prediction):
    tp = fp = fn = eligible_orders = 0
    exact_orders = 0
    fold_results = {seed: {"tp": 0, "fp": 0, "fn": 0} for seed in SEEDS}
    for references in groups.values():
        if len(references) < 3:
            continue
        for index, query in enumerate(references):
            peers = references[:index] + references[index + 1 :]
            prediction = predictor(query, peers)
            if prediction is None:
                continue
            eligible_orders += 1
            selected = prediction["selected"]
            roots = query["roots"]
            local_tp = len(selected & roots)
            tp += local_tp
            fp += len(selected - roots)
            fn += len(roots - selected)
            exact_orders += selected == roots
            for seed in SEEDS:
                fold = fold_results[seed]
                fold["tp"] += local_tp
                fold["fp"] += len(selected - roots)
                fold["fn"] += len(roots - selected)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    for value in fold_results.values():
        value["f1"] = 2 * value["tp"] / max(2 * value["tp"] + value["fp"] + value["fn"], 1)
    return {
        "eligible_orders": eligible_orders,
        "exact_orders": exact_orders,
        "exact_rate": exact_orders / max(eligible_orders, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": f1,
        "seed_metrics": fold_results,
    }


def load_v11_scores(path):
    scores = {}
    by_order = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        seed_columns = [name for name in reader.fieldnames if name.startswith("meta_seed_")]
        for row in reader:
            context = float(row["context_score"])
            seed_scores = [0.25 * context + 0.75 * float(row[name]) for name in seed_columns]
            mean = 0.25 * context + 0.75 * float(row["meta_mean"])
            record = {"rid": row["rid"], "mean": mean, "seeds": seed_scores}
            scores[(row["order_id"], row["rid"])] = record
            by_order[row["order_id"]].append(record)
    return scores, by_order, seed_columns


def exact_selection(by_order, score_getter, forced_in, forced_out, target=1059):
    selected = set(forced_in)
    counts = Counter(order_id for order_id, _ in selected)
    optional = []
    for order_id, records in by_order.items():
        eligible = [r for r in records if (order_id, r["rid"]) not in forced_out]
        eligible.sort(key=lambda r: (-score_getter(r), r["rid"]))
        if counts[order_id] == 0:
            selected.add((order_id, eligible[0]["rid"]))
            counts[order_id] = 1
        for record in eligible:
            key = (order_id, record["rid"])
            if key not in selected:
                optional.append((score_getter(record), order_id, record["rid"]))
    optional.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, order_id, rid in optional:
        if len(selected) == target:
            break
        if counts[order_id] < 8:
            selected.add((order_id, rid))
            counts[order_id] += 1
    return selected


def pair_changes(source, removals, additions, priority, metadata):
    removals = list(removals)
    additions = list(additions)
    units = []
    by_removed_order = defaultdict(list)
    by_added_order = defaultdict(list)
    for item in removals:
        by_removed_order[item[0]].append(item)
    for item in additions:
        by_added_order[item[0]].append(item)
    for order_id in sorted(set(by_removed_order) & set(by_added_order)):
        while by_removed_order[order_id] and by_added_order[order_id]:
            removal = by_removed_order[order_id].pop(0)
            addition = by_added_order[order_id].pop(0)
            removals.remove(removal)
            additions.remove(addition)
            units.append((removal, addition))
    units.extend(zip(sorted(removals), sorted(additions)))
    result = []
    for index, (removal, addition) in enumerate(units, 1):
        result.append(
            {
                "id": f"{source}_{index:03d}",
                "source": source,
                "priority": priority,
                "remove": {"order_id": removal[0], "rid": removal[1]},
                "add": {"order_id": addition[0], "rid": addition[1]},
                "evidence": metadata.get((removal, addition), {}),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--v11", required=True)
    parser.add_argument("--v11-scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_orders = load_orders(args.train_dir, True)
    test_orders = load_orders(args.test_dir, False)
    groups = defaultdict(list)
    for order in train_orders:
        groups[order["signature"]].append(order)
    loo = loo_validation(groups)
    canonical_loo = loo_validation(groups, predictor=canonical_template_prediction)

    original = load_submission(args.original)
    champion = load_submission(args.champion)
    v11 = load_submission(args.v11)
    champion_set = {(oid, rid) for oid, rids in champion.items() for rid in rids}
    v11_set = {(oid, rid) for oid, rids in v11.items() for rid in rids}
    original_set = {(oid, rid) for oid, rids in original.items() for rid in rids}
    forced_in = champion_set - original_set
    forced_out = original_set - champion_set

    template_additions = []
    template_removals = []
    template_evidence = {}
    matched_orders = eligible_template_orders = 0
    for order in test_orders:
        references = groups.get(order["signature"], [])
        if not references:
            continue
        matched_orders += 1
        prediction = template_prediction(order, references)
        if prediction is None:
            prediction = canonical_template_prediction(order, references)
        if prediction is None:
            continue
        current = champion[order["id"]]
        additions = sorted(prediction["selected"] - current)
        removals = sorted((current - prediction["selected"]) & prediction["negative"])
        pair_count = min(len(additions), len(removals))
        if not pair_count:
            continue
        eligible_template_orders += 1
        for removed_rid, added_rid in zip(removals[:pair_count], additions[:pair_count]):
            removal = (order["id"], removed_rid)
            addition = (order["id"], added_rid)
            if removal in forced_in or addition in forced_out:
                continue
            template_removals.append(removal)
            template_additions.append(addition)
            template_evidence[(removal, addition)] = {
                "remove_vote": prediction["votes"].get(removed_rid),
                "add_vote": prediction["votes"].get(added_rid),
                "train_template_support": len(references),
            }

    scores, by_order, seed_columns = load_v11_scores(args.v11_scores)
    seed_selections = [
        exact_selection(by_order, lambda r, i=index: r["seeds"][i], forced_in, forced_out)
        for index in range(len(seed_columns))
    ]
    stable_v11_additions = sorted(
        key for key in (v11_set - champion_set) if all(key in value for value in seed_selections)
    )
    stable_v11_removals = sorted(
        key for key in (champion_set - v11_set) if all(key not in value for value in seed_selections)
    )
    v11_metadata = {}
    for removal in stable_v11_removals:
        for addition in stable_v11_additions:
            if removal[0] == addition[0]:
                v11_metadata[(removal, addition)] = {
                    "remove_score": scores[removal]["mean"],
                    "add_score": scores[addition]["mean"],
                    "seed_columns": seed_columns,
                    "remove_seed_scores": scores[removal]["seeds"],
                    "add_seed_scores": scores[addition]["seeds"],
                }

    units = pair_changes(
        "template", template_removals, template_additions, 1, template_evidence
    )
    units.extend(
        pair_changes(
            "v11_stable", stable_v11_removals, stable_v11_additions, 2, v11_metadata
        )
    )
    units.sort(key=lambda unit: (unit["priority"], unit["id"]))

    protected_violations = []
    for unit in units:
        removal = (unit["remove"]["order_id"], unit["remove"]["rid"])
        addition = (unit["add"]["order_id"], unit["add"]["rid"])
        if removal in forced_in or addition in forced_out:
            protected_violations.append(unit["id"])
    if protected_violations:
        raise RuntimeError(f"protected swap12 violations: {protected_violations}")

    output = {
        "version": 1,
        "anchored_to": str(Path(args.champion)),
        "matched_test_orders": matched_orders,
        "eligible_template_orders": eligible_template_orders,
        "template_loo": loo,
        "canonical_template_loo": canonical_loo,
        "forced_in_count": len(forced_in),
        "forced_out_count": len(forced_out),
        "v11_full_additions": len(v11_set - champion_set),
        "v11_full_removals": len(champion_set - v11_set),
        "v11_stable_additions": len(stable_v11_additions),
        "v11_stable_removals": len(stable_v11_removals),
        "template_units": sum(unit["source"] == "template" for unit in units),
        "v11_units": sum(unit["source"] == "v11_stable" for unit in units),
        "units": units,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in output.items() if key != "units"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

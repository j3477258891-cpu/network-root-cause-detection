import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_submission(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return result


def load_scores(path):
    scores = {}
    by_order = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        seed_columns = [name for name in reader.fieldnames if name.startswith("meta_seed_")]
        for row in reader:
            key = (row["order_id"], row["rid"])
            context = float(row["context_score"])
            seed_values = [float(row[name]) for name in seed_columns]
            meta_mean = float(row["meta_mean"])
            record = {
                "context": context,
                "seed_values": seed_values,
                "meta_mean": meta_mean,
                "final": 0.25 * context + 0.75 * meta_mean,
                "seed_finals": [0.25 * context + 0.75 * value for value in seed_values],
            }
            scores[key] = record
            by_order[row["order_id"]].append((row["rid"], record))
    return scores, by_order, seed_columns


def exact_count_selection(by_order, score_key, target=1059, cap=8):
    selected = set()
    optional = []
    for order_id, nodes in by_order.items():
        ranked = sorted(nodes, key=lambda item: (-score_key(item[1]), item[0]))[:cap]
        selected.add((order_id, ranked[0][0]))
        optional.extend(
            (score_key(record), order_id, rid) for rid, record in ranked[1:]
        )
    optional.sort(key=lambda item: (-item[0], item[1], item[2]))
    remaining = target - len(selected)
    selected.update((order_id, rid) for _, order_id, rid in optional[:remaining])
    return selected


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: analyze_v11_candidate.py HIGH CANDIDATE SCORES OUTPUT_JSON"
        )
    high_path, candidate_path, score_path, output_path = map(Path, sys.argv[1:5])
    high = load_submission(high_path)
    candidate = load_submission(candidate_path)
    scores, by_order, seed_columns = load_scores(score_path)

    removed = []
    added = []
    changed_orders = []
    for order_id in sorted(high):
        local_removed = sorted(high[order_id] - candidate[order_id])
        local_added = sorted(candidate[order_id] - high[order_id])
        if local_removed or local_added:
            changed_orders.append(
                {
                    "order_id": order_id,
                    "before": len(high[order_id]),
                    "after": len(candidate[order_id]),
                    "removed": local_removed,
                    "added": local_added,
                }
            )
        for rid in local_removed:
            removed.append((order_id, rid))
        for rid in local_added:
            added.append((order_id, rid))

    average_selection = exact_count_selection(by_order, lambda record: record["final"])
    candidate_selection = {
        (order_id, rid) for order_id, rids in candidate.items() for rid in rids
    }
    reconstruction_difference = average_selection ^ candidate_selection
    seed_stability = []
    for seed_index, column in enumerate(seed_columns):
        selection = exact_count_selection(
            by_order, lambda record, index=seed_index: record["seed_finals"][index]
        )
        symmetric_difference = average_selection ^ selection
        seed_stability.append(
            {
                "seed_column": column,
                "different_nodes": len(symmetric_difference),
                "different_orders": len({order_id for order_id, _ in symmetric_difference}),
                "agreement": 1.0 - len(symmetric_difference) / (2 * len(average_selection)),
            }
        )

    def describe(key):
        order_id, rid = key
        record = scores[key]
        return {
            "order_id": order_id,
            "rid": rid,
            "context": record["context"],
            "meta_mean": record["meta_mean"],
            "final": record["final"],
            "seed_finals": record["seed_finals"],
            "seed_range": max(record["seed_finals"]) - min(record["seed_finals"]),
        }

    report = {
        "high": str(high_path),
        "candidate": str(candidate_path),
        "orders": len(high),
        "predictions": sum(map(len, candidate.values())),
        "changed_orders": changed_orders,
        "changed_order_count": len(changed_orders),
        "removed": [describe(key) for key in removed],
        "added": [describe(key) for key in added],
        "removed_count": len(removed),
        "added_count": len(added),
        "changed_order_histogram": dict(
            sorted(Counter(item["before"] for item in changed_orders).items())
        ),
        "seed_stability": seed_stability,
        "score_reconstruction_difference": len(reconstruction_difference),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

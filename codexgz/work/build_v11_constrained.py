import csv
import hashlib
import json
import platform
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
    by_order = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        seed_columns = [name for name in reader.fieldnames if name.startswith("meta_seed_")]
        for row in reader:
            context = float(row["context_score"])
            seeds = [float(row[name]) for name in seed_columns]
            record = {
                "rid": row["rid"],
                "mean": 0.25 * context + 0.75 * float(row["meta_mean"]),
                "seeds": [0.25 * context + 0.75 * value for value in seeds],
            }
            by_order[row["order_id"]].append(record)
    return by_order, seed_columns


def protection_sets(original, winner):
    forced_in = set()
    forced_out = set()
    for order_id in winner:
        forced_in.update((order_id, rid) for rid in winner[order_id] - original[order_id])
        forced_out.update((order_id, rid) for rid in original[order_id] - winner[order_id])
    return forced_in, forced_out


def constrained_exact(by_order, forced_in, forced_out, score_getter, target, cap=8):
    selected = set(forced_in)
    optional = []
    counts = Counter(order_id for order_id, _ in selected)

    for order_id, records in by_order.items():
        eligible = [r for r in records if (order_id, r["rid"]) not in forced_out]
        eligible.sort(key=lambda r: (-score_getter(r), r["rid"]))
        if not eligible:
            raise RuntimeError(f"no eligible alarm nodes for {order_id}")
        if counts[order_id] == 0:
            selected.add((order_id, eligible[0]["rid"]))
            counts[order_id] = 1
        for record in eligible:
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
        raise RuntimeError(f"selected {len(selected)} nodes, expected {target}")
    return selected


def constrained_threshold(by_order, forced_in, forced_out, score_getter, threshold, cap=8):
    selected = set(forced_in)
    for order_id, records in by_order.items():
        eligible = [r for r in records if (order_id, r["rid"]) not in forced_out]
        eligible.sort(key=lambda r: (-score_getter(r), r["rid"]))
        wanted = [r for r in eligible if score_getter(r) >= threshold]
        forced_local = {rid for oid, rid in forced_in if oid == order_id}
        local = list(forced_local)
        for record in wanted:
            if record["rid"] not in local and len(local) < cap:
                local.append(record["rid"])
        if not local:
            local.append(eligible[0]["rid"])
        selected.update((order_id, rid) for rid in local)
    return selected


def load_topologies(test_dir):
    result = {}
    for directory in sorted(path for path in Path(test_dir).iterdir() if path.is_dir()):
        order_id = directory.name
        topo_path = directory / f"{order_id}.log.topo.json"
        topo = json.loads(topo_path.read_text(encoding="utf-8"))
        result[order_id] = [node for node in topo["nodes"] if node.get("@class") == "Alarm"]
    return result


def write_submission(path, topologies, selected):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id, nodes in topologies.items():
            local = {rid for oid, rid in selected if oid == order_id}
            rootcauses = [
                {
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                }
                for node in nodes
                if node["@rid"] in local
            ]
            writer.writerow([order_id, json.dumps({"rootcause": rootcauses}, ensure_ascii=False)])


def compare(left, right):
    difference = left ^ right
    return {
        "different_nodes": len(difference),
        "different_orders": len({order_id for order_id, _ in difference}),
        "removed": len(left - right),
        "added": len(right - left),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 8:
        raise SystemExit(
            "usage: build_v11_constrained.py ORIGINAL WINNER FULL_V11 SCORES "
            "TEST_DIR OUTPUT_DIR THRESHOLD"
        )
    original_path, winner_path, full_v11_path, scores_path, test_dir, output_dir = map(
        Path, sys.argv[1:7]
    )
    threshold = float(sys.argv[7])
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_submission(original_path)
    winner = load_submission(winner_path)
    full_v11 = load_submission(full_v11_path)
    winner_set = {(oid, rid) for oid, rids in winner.items() for rid in rids}
    full_v11_set = {(oid, rid) for oid, rids in full_v11.items() for rid in rids}
    forced_in, forced_out = protection_sets(original, winner)
    by_order, seed_columns = load_scores(scores_path)
    topologies = load_topologies(test_dir)

    exact = constrained_exact(by_order, forced_in, forced_out, lambda r: r["mean"], 1059)
    thresholded = constrained_threshold(
        by_order, forced_in, forced_out, lambda r: r["mean"], threshold
    )
    exact_path = output_dir / "result_record_v11_constrained_1059.csv"
    threshold_path = output_dir / "result_record_v11_constrained_threshold.csv"
    write_submission(exact_path, topologies, exact)
    write_submission(threshold_path, topologies, thresholded)

    seed_stability = []
    for index, column in enumerate(seed_columns):
        seed_exact = constrained_exact(
            by_order, forced_in, forced_out, lambda r, i=index: r["seeds"][i], 1059
        )
        seed_threshold = constrained_threshold(
            by_order, forced_in, forced_out, lambda r, i=index: r["seeds"][i], threshold
        )
        seed_stability.append(
            {
                "seed_column": column,
                "exact_vs_mean": compare(exact, seed_exact),
                "threshold_vs_mean": compare(thresholded, seed_threshold),
                "threshold_count": len(seed_threshold),
            }
        )

    report = {
        "method": "V11 scores with the empirically winning swap12 delta locked",
        "threshold": threshold,
        "forced_in_count": len(forced_in),
        "forced_out_count": len(forced_out),
        "forced_in": [{"order_id": oid, "rid": rid} for oid, rid in sorted(forced_in)],
        "forced_out": [{"order_id": oid, "rid": rid} for oid, rid in sorted(forced_out)],
        "exact_count": len(exact),
        "threshold_count": len(thresholded),
        "exact_vs_winner": compare(winner_set, exact),
        "exact_vs_full_v11": compare(full_v11_set, exact),
        "threshold_vs_winner": compare(winner_set, thresholded),
        "threshold_vs_full_v11": compare(full_v11_set, thresholded),
        "seed_stability": seed_stability,
        "count_distribution_exact": dict(
            sorted(Counter(Counter(oid for oid, _ in exact).values()).items())
        ),
        "count_distribution_threshold": dict(
            sorted(Counter(Counter(oid for oid, _ in thresholded).values()).items())
        ),
        "python": platform.python_version(),
        "inputs": {
            str(path): sha256(path)
            for path in (original_path, winner_path, full_v11_path, scores_path)
        },
        "outputs": {
            str(exact_path): sha256(exact_path),
            str(threshold_path): sha256(threshold_path),
        },
    }
    report_path = output_dir / "v11_constrained_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

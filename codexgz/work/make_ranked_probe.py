import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_submission(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = json.loads(row["output"])["rootcause"]
    return result


def write_submission(path, orders, selected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id, alarms in orders:
            rootcauses = []
            for node in alarms:
                if node["@rid"] not in selected[order_id]:
                    continue
                rootcauses.append(
                    {
                        "@rid": node["@rid"],
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                )
            writer.writerow(
                [order_id, json.dumps({"rootcause": rootcauses}, ensure_ascii=False)]
            )


def main():
    if len(sys.argv) not in (7, 8):
        raise SystemExit(
            "usage: make_ranked_probe.py BASELINE MODEL_CANDIDATE SCORES "
            "TEST_DIR OUTPUT SWAPS [PROTECT_REFERENCE]"
        )
    baseline_path, candidate_path, scores_path, test_dir, output_path = map(
        Path, sys.argv[1:6]
    )
    swap_count = int(sys.argv[6])
    protect_reference = load_submission(sys.argv[7]) if len(sys.argv) == 8 else None

    baseline = load_submission(baseline_path)
    candidate = load_submission(candidate_path)
    scores = np.load(scores_path)
    selected = {
        order_id: {node["@rid"] for node in nodes}
        for order_id, nodes in baseline.items()
    }
    protected = {}
    if protect_reference is not None:
        for order_id, nodes in baseline.items():
            current_ids = {node["@rid"] for node in nodes}
            reference_ids = {
                node["@rid"] for node in protect_reference[order_id]
            }
            protected[order_id] = current_ids ^ reference_ids
    orders = []
    additions = []
    removals = []
    flat_index = 0

    for directory in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        order_id = directory.name
        topology = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        alarms = [node for node in topology["nodes"] if node.get("@class") == "Alarm"]
        orders.append((order_id, alarms))
        baseline_ids = selected[order_id]
        candidate_ids = {node["@rid"] for node in candidate[order_id]}
        node_scores = {
            node["@rid"]: float(scores[flat_index + index])
            for index, node in enumerate(alarms)
        }
        flat_index += len(alarms)
        for rid in candidate_ids - baseline_ids:
            if len(baseline_ids) < 8 and rid not in protected.get(order_id, set()):
                additions.append((node_scores[rid], order_id, rid))
        for rid in baseline_ids - candidate_ids:
            if len(baseline_ids) > 1 and rid not in protected.get(order_id, set()):
                removals.append((node_scores[rid], order_id, rid))

    additions.sort(reverse=True)
    removals.sort()
    if swap_count > min(len(additions), len(removals)):
        raise RuntimeError(
            f"requested {swap_count} swaps, available "
            f"{min(len(additions), len(removals))}"
        )

    for _, order_id, rid in removals[:swap_count]:
        selected[order_id].remove(rid)
    for _, order_id, rid in additions[:swap_count]:
        selected[order_id].add(rid)

    write_submission(output_path, orders, selected)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "orders": len(selected),
                "predictions": sum(map(len, selected.values())),
                "swaps": swap_count,
                "removed": removals[:swap_count],
                "added": additions[:swap_count],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

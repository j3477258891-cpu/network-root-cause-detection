"""Audit the complete per-order heterogeneous topology before V24 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(root: Path, with_labels: bool) -> dict:
    node_classes: Counter[str] = Counter()
    edge_classes: Counter[str] = Counter()
    node_keys: Counter[str] = Counter()
    edge_keys: Counter[str] = Counter()
    rid_orders: dict[str, str] = {}
    repeated_rids: list[dict[str, str]] = []
    repeated_rid_count = 0
    totals = Counter()
    maxima = Counter()
    order_sizes = []

    directories = sorted(path for path in root.iterdir() if path.is_dir())
    for index, directory in enumerate(directories):
        order_id = directory.name
        topology_path = directory / f"{order_id}.log.topo.json"
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        nodes = topology.get("nodes", [])
        edges = topology.get("edges", [])
        local_rids = {node.get("@rid") for node in nodes if node.get("@rid")}
        alarms = [node for node in nodes if node.get("@class") == "Alarm"]
        missing_endpoints = sum(
            edge.get("in") not in local_rids or edge.get("out") not in local_rids
            for edge in edges
        )
        self_loops = sum(edge.get("in") == edge.get("out") for edge in edges)

        for node in nodes:
            node_classes[str(node.get("@class") or "<missing>")] += 1
            node_keys.update(node.keys())
            rid = node.get("@rid")
            if rid:
                previous = rid_orders.setdefault(rid, order_id)
                if previous != order_id:
                    repeated_rid_count += 1
                    if len(repeated_rids) < 100:
                        repeated_rids.append(
                            {"rid": rid, "first_order": previous, "second_order": order_id}
                        )
        for edge in edges:
            edge_classes[str(edge.get("@class") or edge.get("label") or "<missing>")] += 1
            edge_keys.update(edge.keys())

        roots = []
        if with_labels:
            root_path = directory / f"{order_id}.rootcause.json"
            root_data = json.loads(root_path.read_text(encoding="utf-8"))
            roots = root_data.get("rootcause", [])
            root_rids = {node.get("@rid") for node in roots}
            totals["unique_roots"] += len(root_rids)
            totals["duplicate_root_entries"] += len(roots) - len(root_rids)
            if not root_rids <= {node.get("@rid") for node in alarms}:
                totals["root_not_alarm_orders"] += 1

        totals.update(
            orders=1,
            nodes=len(nodes),
            edges=len(edges),
            alarms=len(alarms),
            roots=len(roots),
            missing_endpoints=missing_endpoints,
            self_loops=self_loops,
        )
        maxima["nodes"] = max(maxima["nodes"], len(nodes))
        maxima["edges"] = max(maxima["edges"], len(edges))
        maxima["alarms"] = max(maxima["alarms"], len(alarms))
        order_sizes.append(
            {
                "order_id": order_id,
                "nodes": len(nodes),
                "edges": len(edges),
                "alarms": len(alarms),
            }
        )
        if (index + 1) % 250 == 0:
            print(f"audit {root.name}: {index + 1}/{len(directories)}", flush=True)

    return {
        "root": str(root),
        "totals": dict(totals),
        "maxima": dict(maxima),
        "node_classes": dict(node_classes.most_common()),
        "edge_classes": dict(edge_classes.most_common()),
        "node_keys": dict(node_keys.most_common()),
        "edge_keys": dict(edge_keys.most_common()),
        "cross_order_rid_collisions": repeated_rid_count,
        "cross_order_rid_collision_examples": repeated_rids,
        "largest_orders": sorted(
            order_sizes, key=lambda item: (item["nodes"], item["edges"]), reverse=True
        )[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "version": "v24-full-hetero-audit-1",
        "design": "disjoint complete per-order heterogeneous graphs",
        "train": audit_split(args.train, True),
        "test": audit_split(args.test, False),
    }
    report["combined"] = {
        key: report["train"]["totals"].get(key, 0)
        + report["test"]["totals"].get(key, 0)
        for key in sorted(set(report["train"]["totals"]) | set(report["test"]["totals"]))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), **report["combined"]}, indent=2))


if __name__ == "__main__":
    main()

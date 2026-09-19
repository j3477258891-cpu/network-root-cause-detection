"""Independent structural validation for V29 generated probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
DEFAULT_ROOT = ROOT / "experiments/v29_domain_ranker"
CHAMPION = ROOT / "experiments/v28_ten_day_campaign/submissions/day04_verified_checkpoint.csv"
HISTORY = (
    ROOT / "experiments/v28_ten_day_campaign/state.json",
    ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
)
EXPECTED_ORDERS = 546
MAX_ROOTS = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_touched_orders(paths: tuple[Path, ...]) -> set[str]:
    touched = set()
    for path in paths:
        if not path.exists():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        for batch in state.get("batches", []):
            for action in batch.get("actions", []):
                touched.add(action["order_id"])
    return touched


def validate_csv(path: Path):
    order_ids = []
    predictions = 0
    roots_by_order = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError((path, reader.fieldnames))
        for row in reader:
            order_ids.append(row["order_id"])
            roots = json.loads(row["output"])["rootcause"]
            if not 1 <= len(roots) <= MAX_ROOTS:
                raise ValueError((path, row["order_id"], len(roots)))
            rids = [node["@rid"] for node in roots]
            if len(rids) != len(set(rids)):
                raise ValueError((path, row["order_id"], "duplicate rid"))
            predictions += len(roots)
            roots_by_order[row["order_id"]] = set(rids)
    if len(order_ids) != EXPECTED_ORDERS or len(order_ids) != len(set(order_ids)):
        raise ValueError((path, len(order_ids), len(set(order_ids))))
    return predictions, roots_by_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = json.loads((args.root / "reports/v29_report.json").read_text(encoding="utf-8"))
    baseline_predictions, champion_roots = validate_csv(CHAMPION)
    if baseline_predictions != report["baseline"]["predictions"]:
        raise ValueError(("baseline predictions", baseline_predictions))
    if sha256(CHAMPION) != report["baseline"]["sha256"]:
        raise ValueError("baseline sha256 mismatch")
    touched_orders = load_touched_orders(HISTORY)
    results = {}
    all_orders = set()
    for name, probe in report["probes"].items():
        path = Path(probe["path"])
        predictions, probe_roots = validate_csv(path)
        if predictions != probe["predictions"]:
            raise ValueError((name, predictions, probe["predictions"]))
        if sha256(path) != probe["sha256"]:
            raise ValueError((name, "sha256 mismatch"))
        action_orders = [action["order_id"] for action in probe["actions"]]
        if len(action_orders) != len(set(action_orders)):
            raise ValueError((name, "action order conflict"))
        historical_overlap = touched_orders.intersection(action_orders)
        if historical_overlap:
            raise ValueError((name, "historical order conflict", sorted(historical_overlap)))
        actions_by_order = {action["order_id"]: action for action in probe["actions"]}
        for order_id, baseline_rids in champion_roots.items():
            expected = set(baseline_rids)
            action = actions_by_order.get(order_id)
            if action:
                remove_rids = set(action["remove_rids"])
                add_rids = set(action["add_rids"])
                if not remove_rids.issubset(expected) or add_rids.intersection(expected):
                    raise ValueError((name, order_id, "invalid action against champion"))
                if remove_rids.intersection(add_rids):
                    raise ValueError((name, order_id, "action rid conflict"))
                expected.difference_update(remove_rids)
                expected.update(add_rids)
            if probe_roots[order_id] != expected:
                raise ValueError((name, order_id, "probe differs from declared actions"))
        if name != "combined":
            overlap = all_orders.intersection(action_orders)
            if overlap:
                raise ValueError((name, "cross-probe conflict", sorted(overlap)))
            all_orders.update(action_orders)
        results[name] = {
            "path": str(path),
            "predictions": predictions,
            "actions": len(action_orders),
            "historical_overlap": 0,
            "sha256": probe["sha256"],
        }
    print(json.dumps({"passed": True, "probes": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

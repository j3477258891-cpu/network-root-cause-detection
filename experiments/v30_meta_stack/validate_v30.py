"""Independent structural and provenance validation for V30 probes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
EXPERIMENT = ROOT / "experiments/v30_meta_stack"
CHAMPION = ROOT / "experiments/v29_domain_ranker/submissions/day05_delete_verified.csv"
ONLINE = ROOT / "experiments/v29_domain_ranker/reports/online_results.json"
V29_MANIFESTS = ROOT / "experiments/v29_domain_ranker/manifests"
HISTORY = (
    ROOT / "experiments/v28_ten_day_campaign/state.json",
    ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
)
EXPECTED_ORDERS = 546
MAX_ROOTS = 8
TRUE_ROOTS = 1044


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_submission(path: Path):
    roots = {}
    predictions = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError((path, reader.fieldnames))
        for row in reader:
            nodes = json.loads(row["output"])["rootcause"]
            rids = [node["@rid"] for node in nodes]
            if not 1 <= len(rids) <= MAX_ROOTS or len(rids) != len(set(rids)):
                raise ValueError((path, row["order_id"], len(rids)))
            roots[row["order_id"]] = set(rids)
            predictions += len(rids)
    if len(roots) != EXPECTED_ORDERS:
        raise ValueError((path, len(roots)))
    return predictions, roots


def independently_touched_orders() -> set[str]:
    touched = set()
    for path in HISTORY:
        state = json.loads(path.read_text(encoding="utf-8"))
        for batch in state.get("batches", []):
            for action in batch.get("actions", []):
                touched.add(action["order_id"])
    online = json.loads(ONLINE.read_text(encoding="utf-8"))
    for result in online.get("verified", []):
        manifest_path = V29_MANIFESTS / f"{result['probe_id']}.json"
        if not manifest_path.exists():
            raise ValueError(("missing verified manifest", manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for action in manifest.get("actions", []):
            touched.add(action["order_id"])
    return touched


def assert_tp_inference_regression() -> None:
    # Historical public scores are rounded to six decimals; these checks lock
    # the exact prediction-count convention used by all probe manifests.
    cases = ((0.906324, 1059, 953), (0.905373, 1059, 952))
    for score, predictions, expected_tp in cases:
        inferred = round(score * (TRUE_ROOTS + predictions) / 2)
        if inferred != expected_tp:
            raise ValueError(("tp inference regression", score, predictions, inferred, expected_tp))


def main():
    assert_tp_inference_regression()
    report = json.loads((EXPERIMENT / "v30_cross_order_report.json").read_text(encoding="utf-8"))
    baseline_predictions, baseline_roots = load_submission(CHAMPION)
    if baseline_predictions != report["baseline_predictions"]:
        raise ValueError(("baseline", baseline_predictions))
    excluded_orders = independently_touched_orders()
    if len(excluded_orders) != report["excluded_orders"]:
        raise ValueError(("excluded order count", len(excluded_orders), report["excluded_orders"]))
    results = {}
    for name, probe in report["probes"].items():
        path = Path(probe["path"])
        predictions, roots = load_submission(path)
        if predictions != probe["predictions"] or sha256(path) != probe["sha256"]:
            raise ValueError((name, "count or hash mismatch"))
        action_orders = [action["order_id"] for action in probe["actions"]]
        if len(action_orders) != len(set(action_orders)):
            raise ValueError((name, "duplicate action order"))
        overlap = set(action_orders).intersection(excluded_orders)
        if overlap:
            raise ValueError((name, "historical/V29 order overlap", sorted(overlap)))
        margins = [float(action["evidence"]["pair_margin"]) for action in probe["actions"]]
        if not margins or min(margins) <= 0:
            raise ValueError((name, "non-positive pair margin", min(margins, default=None)))
        actions = {action["order_id"]: action for action in probe["actions"]}
        pair_counts = {}
        for action in probe["actions"]:
            pair_id = action["evidence"]["pair_id"]
            pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        if set(pair_counts.values()) != {2} or len(pair_counts) != probe["pair_count"]:
            raise ValueError((name, "incomplete pairs", pair_counts))
        for order_id, base in baseline_roots.items():
            expected = set(base)
            if order_id in actions:
                action = actions[order_id]
                remove = set(action["remove_rids"])
                add = set(action["add_rids"])
                if not remove.issubset(expected) or add.intersection(expected):
                    raise ValueError((name, order_id, "invalid action"))
                expected.difference_update(remove)
                expected.update(add)
            if roots[order_id] != expected:
                raise ValueError((name, order_id, "undeclared difference"))
        results[name] = {
            "pairs": probe["pair_count"],
            "actions": len(probe["actions"]),
            "predictions": predictions,
            "sha256": probe["sha256"],
        }
    print(
        json.dumps(
            {"passed": True, "excluded_orders": len(excluded_orders), "probes": results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

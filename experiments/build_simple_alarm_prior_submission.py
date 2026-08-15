"""Build a submit-ready CSV from simple alarm-field priors.

This is intentionally independent from the previous model stack: it learns
smoothed root-cause rates from training alarm title/reason/label fields, scores
test alarms inside each order, then selects a globally fixed number of alarms.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
OUT_DIR = ROOT / "experiments" / "submissions"
OUT_CSV = OUT_DIR / "result_record_simple_alarm_prior_1059.csv"
OUT_REPORT = OUT_DIR / "result_record_simple_alarm_prior_1059.report.json"

TARGET_TEST_COUNT = 1059
MAX_ROOTS_PER_ORDER = 8


def alarm_nodes(topo: dict) -> list[dict]:
    return [
        node
        for node in topo.get("nodes", [])
        if node.get("@class") == "Alarm"
        or "title" in node
        or "reason" in node
        or node.get("label")
    ]


def load_topo(split_dir: Path, order_id: str) -> dict:
    path = split_dir / order_id / f"{order_id}.log.topo.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_orders() -> list[dict]:
    orders = []
    for order_dir in sorted(path for path in TRAIN_DIR.iterdir() if path.is_dir()):
        order_id = order_dir.name
        topo = load_topo(TRAIN_DIR, order_id)
        roots = json.loads(
            (order_dir / f"{order_id}.rootcause.json").read_text(encoding="utf-8")
        )["rootcause"]
        root_ids = {item["@rid"] for item in roots}
        orders.append(
            {
                "id": order_id,
                "alarms": alarm_nodes(topo),
                "root_ids": root_ids,
            }
        )
    return orders


def feature_keys(alarm: dict) -> list[tuple[str, str, str]]:
    title = alarm.get("title", "")
    reason = str(alarm.get("reason", ""))
    label = alarm.get("label", "")
    return [
        ("title_reason", title, reason),
        ("title", title, ""),
        ("reason", reason, ""),
        ("label", label, ""),
    ]


def build_priors(train_orders: list[dict]) -> tuple[Counter, Counter, float]:
    totals: Counter = Counter()
    positives: Counter = Counter()
    alarm_count = 0
    root_count = 0
    for order in train_orders:
        root_ids = order["root_ids"]
        for alarm in order["alarms"]:
            alarm_count += 1
            is_root = alarm["@rid"] in root_ids
            root_count += int(is_root)
            for key in feature_keys(alarm):
                totals[key] += 1
                positives[key] += int(is_root)
    return totals, positives, root_count / alarm_count


def smoothed_rate(
    key: tuple[str, str, str], totals: Counter, positives: Counter, prior: float, alpha: float
) -> float:
    return (positives[key] + alpha * prior) / (totals[key] + alpha)


def score_alarm(alarm: dict, totals: Counter, positives: Counter, prior: float) -> float:
    title = alarm.get("title", "")
    reason = str(alarm.get("reason", ""))
    label = alarm.get("label", "")
    title_reason = smoothed_rate(("title_reason", title, reason), totals, positives, prior, 2.0)
    title_rate = smoothed_rate(("title", title, ""), totals, positives, prior, 3.0)
    reason_rate = smoothed_rate(("reason", reason, ""), totals, positives, prior, 3.0)
    label_rate = smoothed_rate(("label", label, ""), totals, positives, prior, 10.0)

    score = 0.55 * title_reason + 0.15 * title_rate + 0.25 * reason_rate + 0.05 * label_rate
    if label == "TargetAlarm":
        score += 0.005
    time_lists = alarm.get("timeLists") or []
    if isinstance(time_lists, list):
        score += min(0.015, 0.001 * sum(value for value in time_lists if isinstance(value, (int, float))))
    return score


def choose_counts(order_scores: list[list[float]], target_count: int) -> list[int]:
    counts = [1 for _ in order_scores]
    candidates = []
    for order_index, scores in enumerate(order_scores):
        ranked = sorted(scores, reverse=True)
        maximum = min(MAX_ROOTS_PER_ORDER, len(ranked))
        for count in range(2, maximum + 1):
            candidates.append((ranked[count - 1], order_index, count))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    remaining = target_count - len(order_scores)
    if remaining < 0:
        raise ValueError("target_count must be at least the number of test orders")
    for _, order_index, count in candidates[:remaining]:
        counts[order_index] = max(counts[order_index], count)
    return counts


def node_payload(alarm: dict) -> dict:
    return {
        "@rid": alarm["@rid"],
        "title": alarm.get("title", ""),
        "location": alarm.get("location", ""),
        "reason": alarm.get("reason", ""),
    }


def validate_submission(path: Path) -> dict:
    test_ids = sorted(path.name for path in TEST_DIR.iterdir() if path.is_dir())
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["order_id"] for row in rows] != test_ids:
        raise AssertionError("order_id list does not match sorted test ids")

    count_distribution: Counter = Counter()
    predicted_nodes = 0
    for row in rows:
        payload = json.loads(row["output"])
        rootcauses = payload["rootcause"]
        topo = load_topo(TEST_DIR, row["order_id"])
        nodes = {node["@rid"]: node for node in topo["nodes"]}
        seen = set()
        for prediction in rootcauses:
            if list(prediction) != ["@rid", "title", "location", "reason"]:
                raise AssertionError((row["order_id"], list(prediction)))
            rid = prediction["@rid"]
            if rid in seen or rid not in nodes:
                raise AssertionError((row["order_id"], rid))
            seen.add(rid)
            source = nodes[rid]
            for field in ("title", "location", "reason"):
                if prediction[field] != source.get(field, ""):
                    raise AssertionError((row["order_id"], rid, field))
        count_distribution[len(rootcauses)] += 1
        predicted_nodes += len(rootcauses)
    return {
        "rows": len(rows),
        "predicted_nodes": predicted_nodes,
        "rootcause_count_distribution": dict(sorted(count_distribution.items())),
    }


def main() -> None:
    train_orders = load_train_orders()
    totals, positives, prior = build_priors(train_orders)

    test_orders = []
    order_scores = []
    for order_dir in sorted(path for path in TEST_DIR.iterdir() if path.is_dir()):
        order_id = order_dir.name
        alarms = alarm_nodes(load_topo(TEST_DIR, order_id))
        if not alarms:
            raise ValueError(f"{order_id} has no alarm candidates")
        scores = [score_alarm(alarm, totals, positives, prior) for alarm in alarms]
        test_orders.append({"id": order_id, "alarms": alarms, "scores": scores})
        order_scores.append(scores)

    counts = choose_counts(order_scores, TARGET_TEST_COUNT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order, count in zip(test_orders, counts):
            ranked = sorted(
                range(len(order["alarms"])),
                key=lambda index: (-order["scores"][index], index),
            )[:count]
            rootcauses = [node_payload(order["alarms"][index]) for index in ranked]
            writer.writerow(
                [
                    order["id"],
                    json.dumps({"rootcause": rootcauses}, ensure_ascii=False),
                ]
            )

    validation = validate_submission(OUT_CSV)
    digest = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    report = {
        "method": "simple smoothed alarm title/reason/label prior",
        "target_test_count": TARGET_TEST_COUNT,
        "max_roots_per_order": MAX_ROOTS_PER_ORDER,
        "train_orders": len(train_orders),
        "test_orders": len(test_orders),
        "train_alarm_prior": prior,
        "validation": validation,
        "sha256": digest,
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

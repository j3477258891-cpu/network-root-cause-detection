"""Audit conservative within-order alarm-group consistency rules.

This script is deliberately an offline audit first. It learns only rules that
are perfectly consistent on training orders and evaluates them on held-out
station folds before producing any test actions.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
MAX_ROOTS = 8
TRAIN_TARGET = 3169


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:MAX_ROOTS]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional_array = np.asarray(optional, dtype=np.int64)
    optional_array = optional_array[np.argsort(-scores[optional_array], kind="stable")]
    optional_array = optional_array[: target - int(selected.sum())]
    selected[optional_array] = True
    return selected


def norm(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", "<UUID>", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def group_key(alarm: dict, row: int, arrays: dict[str, np.ndarray], split: str, level: str):
    source = alarm.get("source", {})
    fields = [norm(source.get("title", alarm.get("title", ""))),
              norm(source.get("reason", alarm.get("reason", "")))]
    if level in ("device", "topology"):
        fields.extend([
            norm(alarm.get("vendor", "")),
            norm(alarm.get("device_type", "")),
            norm(alarm.get("board_type", "")),
            norm(alarm.get("cause", "")),
            norm(alarm.get("radio", "")),
            norm(alarm.get("deployment", "")),
        ])
    if level == "topology":
        fields.extend([
            tuple(int(x) for x in arrays[f"{split}_path_node_type"][row]),
            tuple(int(x) for x in arrays[f"{split}_path_edge_type"][row]),
            int(arrays[f"{split}_path_length"][row]),
        ])
    return tuple(fields)


def order_groups(records, arrays, split: str, ptr: np.ndarray, level: str):
    groups = []
    for order_index, (order, start, stop) in enumerate(zip(records, ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        by_key = defaultdict(list)
        for row, alarm in enumerate(order["alarms"], start):
            by_key[group_key(alarm, row, arrays, split, level)].append(row)
        groups.append((order_index, by_key))
    return groups


def learn_rules(records, arrays, labels, fold_ids, level: str, train_fold: int):
    ptr = arrays["train_alarm_ptr"]
    groups = order_groups(records, arrays, "train", ptr, level)
    observations = defaultdict(list)
    for order_index, by_key in groups:
        if fold_ids[order_index] == train_fold:
            continue
        stations = tuple(sorted(str(x) for x in records[order_index].get("station_ids", [])))
        for key, rows in by_key.items():
            if len(rows) < 2:
                continue
            values = labels[rows].astype(bool)
            pattern = "all" if values.all() else "none" if (~values).all() else "mixed"
            observations[key].append((order_index, stations, len(rows), pattern))
    rules = {}
    for key, obs in observations.items():
        if len(obs) < 5 or len({s for _, stations, _, _ in obs for s in stations}) < 3:
            continue
        patterns = {item[3] for item in obs}
        if patterns == {"all"}:
            rules[key] = "all"
        elif patterns == {"none"}:
            rules[key] = "none"
    return rules


def apply_rules(records, arrays, labels, base, fold_ids, level: str, heldout_fold: int):
    ptr = arrays["train_alarm_ptr"]
    groups = order_groups(records, arrays, "train", ptr, level)
    # Train only on the complementary folds, then apply to held-out orders.
    rules = learn_rules(records, arrays, labels, fold_ids, level, heldout_fold)
    proposed = base.copy()
    changed_orders = set()
    changed_rows = 0
    for order_index, by_key in groups:
        if fold_ids[order_index] != heldout_fold:
            continue
        for key, rows in by_key.items():
            if key not in rules:
                continue
            desired = rules[key] == "all"
            if np.any(proposed[rows] != desired):
                changed_orders.add(order_index)
                changed_rows += int(np.sum(proposed[rows] != desired))
                proposed[rows] = desired
    delta = int((proposed & labels.astype(bool)).sum() - (base & labels.astype(bool)).sum())
    return delta, len(changed_orders), changed_rows, len(rules)


def main():
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["train"]
    labels = arrays["train_labels"].astype(np.int8)
    fold_ids = arrays["train_station_folds"].astype(np.int8)
    base = exact_count_mask(arrays["train_v11"], arrays["train_alarm_ptr"], TRAIN_TARGET)
    base_tp = int((base & labels.astype(bool)).sum())
    report = {"base_tp": base_tp, "levels": {}}
    for level in ("title_reason", "device", "topology"):
        folds = []
        for fold in range(5):
            folds.append(apply_rules(records, arrays, labels, base, fold_ids, level, fold))
        report["levels"][level] = {
            "folds": [
                {"delta_tp": d, "changed_orders": o, "changed_rows": r, "rules": n}
                for d, o, r, n in folds
            ],
            "delta_tp": int(sum(row[0] for row in folds)),
            "changed_orders": int(sum(row[1] for row in folds)),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

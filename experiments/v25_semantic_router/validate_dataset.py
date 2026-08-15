"""Validate V25 alignment, split isolation, and text redaction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from semantic_model import candidate_prompt
from v25_common import Bundle, IP_PATTERN, UUID_PATTERN, write_json


def assert_group_isolation(keys, folds, name):
    seen = defaultdict(set)
    for key, fold in zip(keys, folds):
        seen[repr(key)].add(int(fold))
    leaking = [key for key, values in seen.items() if len(values) > 1]
    if leaking:
        raise ValueError(f"{name} groups cross folds: {len(leaking)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = Bundle.load(args.data_root)
    train = bundle.records["train"]
    test = bundle.records["test"]
    labels = bundle.arrays["train_labels"]
    if sum(len(order["alarms"]) for order in train) != len(labels):
        raise ValueError("train record alignment")
    if sum(len(order["alarms"]) for order in test) != len(bundle.arrays["test_v11"]):
        raise ValueError("test record alignment")
    record_labels = np.asarray(
        [alarm["is_root"] for order in train for alarm in order["alarms"]], dtype=np.int8
    )
    if not np.array_equal(record_labels, labels):
        raise ValueError("record/array labels differ")
    assert_group_isolation(
        [order["signature"] for order in train], bundle.arrays["train_folds"], "template"
    )
    assert_group_isolation(
        [
            ("station", tuple(order["station_ids"]))
            if order["station_ids"]
            else ("no_station", order["signature"])
            for order in train
        ],
        bundle.arrays["train_station_folds"],
        "station",
    )
    prompts = 0
    for order in train + test:
        for alarm in order["alarms"]:
            prompt = candidate_prompt(alarm)
            if IP_PATTERN.search(prompt) or UUID_PATTERN.search(prompt):
                raise ValueError((order["order_id"], alarm["rid"], "unredacted identity"))
            if set(alarm["source"]) != {"title", "location", "reason"}:
                raise ValueError((order["order_id"], alarm["rid"], "source fields"))
            prompts += 1
    report = {
        "version": bundle.metadata["version"],
        "train_orders": len(train),
        "test_orders": len(test),
        "train_alarms": len(labels),
        "test_alarms": len(bundle.arrays["test_v11"]),
        "labels": int(labels.sum()),
        "champion_predictions": int(bundle.arrays["test_champion_mask"].sum()),
        "validated_prompts": prompts,
        "template_fold_sizes": np.bincount(bundle.arrays["train_folds"], minlength=5).tolist(),
        "station_fold_sizes": np.bincount(bundle.arrays["train_station_folds"], minlength=5).tolist(),
        "passed": True,
    }
    if report["champion_predictions"] != 1059:
        raise ValueError(report["champion_predictions"])
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()

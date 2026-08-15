"""Diagnose fixed-K graph+time swaps using OOF labels."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v16_structure_signal as v16


WEIGHT = 0.195


def build_slices(orders):
    output = []
    cursor = 0
    for order in orders:
        output.append(slice(cursor, cursor + len(order["alarms"])))
        cursor += len(order["alarms"])
    return output


def summarize(records):
    if not records:
        return {"count": 0, "net_tp_delta": 0, "positive": 0, "zero": 0, "negative": 0}
    deltas = [item["label_delta"] for item in records]
    return {
        "count": len(records),
        "net_tp_delta": int(sum(deltas)),
        "positive": sum(delta > 0 for delta in deltas),
        "zero": sum(delta == 0 for delta in deltas),
        "negative": sum(delta < 0 for delta in deltas),
        "mean_label_delta": float(np.mean(deltas)),
    }


def main():
    orders = v16.load_orders(v16.TRAIN_DIR, True)
    slices = build_slices(orders)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    v11_scores = 0.25 * np.load(v16.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        v16.V11_DIR / "v11_oof_meta.npy"
    )
    graph_scores = np.load(v16.OUTPUT_DIR / "v16_graph_time_oof.npy")
    blend_scores = (1.0 - WEIGHT) * v11_scores + WEIGHT * graph_scores
    target_count = round(v16.TARGET_TEST_COUNT / 546 * len(orders))
    base_mask = v16.exact_count_mask(v11_scores, slices, target_count)
    candidate_mask = v16.fixed_k_mask(blend_scores, slices, base_mask)
    folds, _, _ = v16.grouped_folds(orders)

    records = []
    for order_index, (order, order_slice) in enumerate(zip(orders, slices)):
        before = set(np.flatnonzero(base_mask[order_slice]).tolist())
        after = set(np.flatnonzero(candidate_mask[order_slice]).tolist())
        if before == after:
            continue
        removed = sorted(before - after)
        added = sorted(after - before)
        assert len(removed) == len(added)
        removed_rows = [order_slice.start + index for index in removed]
        added_rows = [order_slice.start + index for index in added]
        records.append(
            {
                "order_id": order["id"],
                "fold": int(folds[order_index]),
                "k": len(before),
                "swap_count": len(removed),
                "label_delta": int(
                    np.sum(labels[added_rows]) - np.sum(labels[removed_rows])
                ),
                "v11_boundary_margin": float(
                    min(v11_scores[added_rows]) - max(v11_scores[removed_rows])
                ),
                "graph_time_boundary_margin": float(
                    min(graph_scores[added_rows]) - max(graph_scores[removed_rows])
                ),
                "blend_boundary_margin": float(
                    min(blend_scores[added_rows]) - max(blend_scores[removed_rows])
                ),
            }
        )

    records.sort(key=lambda item: item["blend_boundary_margin"], reverse=True)
    groups = {
        "all": records,
        "k1": [item for item in records if item["k"] == 1],
        "blend_margin_ge_0_05": [
            item for item in records if item["blend_boundary_margin"] >= 0.05
        ],
        "graph_margin_ge_0_40": [
            item for item in records if item["graph_time_boundary_margin"] >= 0.40
        ],
        "test_stable_profile": [
            item
            for item in records
            if item["k"] == 1
            and item["blend_boundary_margin"] >= 0.07
            and item["graph_time_boundary_margin"] >= 0.40
        ],
    }
    fold_summary = defaultdict(list)
    for item in records:
        fold_summary[str(item["fold"])].append(item)
    report = {
        "weight": WEIGHT,
        "selection": "strict per-order fixed K",
        "summary": {name: summarize(items) for name, items in groups.items()},
        "by_fold": {fold: summarize(items) for fold, items in sorted(fold_summary.items())},
        "records": records,
    }
    output_json = v16.OUTPUT_DIR / "v16_graph_time_swap_diagnostics.json"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_csv = v16.OUTPUT_DIR / "v16_graph_time_oof_fixed_k_swaps.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Audit whether V33 can safely replace V30 on stable order strata.

This is deliberately an audit, not a submission generator.  V33 and V30 are
both station-cross-fitted, so an order-level comparison can be evaluated on
the held-out station.  We use the true per-order count only to isolate ranking
quality; deploying a stratum would still require a separately verified count
policy.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V33 = ROOT / "experiments/v33_domain_adaptation"
OUT = ROOT / "experiments/v69_domain_stratified_router"
MAX_ROOTS = 8


def normalized(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def order_mask(score: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros(len(score), dtype=bool)
    result[np.argsort(-score, kind="stable")[:count]] = True
    return result


def quantile_bin(value: float, edges: np.ndarray) -> str:
    return f"q{int(np.searchsorted(edges, value, side='right'))}"


def records_for_split(payload: dict, split: str) -> list[dict]:
    # The semantic archive uses train/test; tolerate older flat layouts.
    return payload[split] if split in payload else payload[f"{split}_records"]


def order_rows(arrays: dict[str, np.ndarray], split: str, records: list[dict],
               station: np.ndarray | None, domain: np.ndarray,
               v30: np.ndarray, v33: np.ndarray, labels: np.ndarray | None):
    ptr = arrays[f"{split}_alarm_ptr"]
    alarm_x = arrays[f"{split}_alarm_x"]
    order_domain = np.asarray([domain[int(a):int(b)].mean() for a, b in zip(ptr[:-1], ptr[1:])])
    domain_edges = np.quantile(order_domain, [0.2, 0.4, 0.6, 0.8])
    output = []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        record = records[oi]
        alarms = record["alarms"]
        vendor_values = sorted({normalized(a.get("vendor", "")) for a in alarms if normalized(a.get("vendor", ""))})
        title_values = sorted({normalized(a.get("title", "")) for a in alarms if normalized(a.get("title", ""))})
        vendor = vendor_values[0] if len(vendor_values) == 1 else "mixed_or_missing"
        top_title = title_values[0] if len(title_values) == 1 else "mixed_titles"
        v30_rank = np.argsort(-v30[start:stop], kind="stable")
        v33_rank = np.argsort(-v33[start:stop], kind="stable")
        row = {
            "order_index": oi,
            "order_id": record["order_id"],
            "station_fold": None if station is None else int(station[oi]),
            "alarm_count": stop - start,
            "alarm_bin": "le4" if stop - start <= 4 else "5_8" if stop - start <= 8 else "9_16" if stop - start <= 16 else "gt16",
            "domain_mean": float(order_domain[oi]),
            "domain_bin": quantile_bin(float(order_domain[oi]), domain_edges),
            "vendor": vendor,
            "title_class": top_title,
            "v30_v33_top1_same": bool(v30_rank[0] == v33_rank[0]),
            "top1_gap": float(v33[start + v33_rank[0]] - v30[start + v30_rank[0]]),
            "score_disagreement": float(np.abs(v30[start:stop] - v33[start:stop]).mean()),
            "static_mean": float(np.nan_to_num(alarm_x[start:stop], nan=0.0).mean()),
        }
        if labels is not None:
            count = min(MAX_ROOTS, int(labels[start:stop].sum()))
            base = order_mask(v30[start:stop], count)
            proposed = order_mask(v33[start:stop], count)
            row.update({
                "root_count": count,
                "changed": bool(np.any(base != proposed)),
                "v30_tp": int((base & labels[start:stop]).sum()),
                "v33_tp": int((proposed & labels[start:stop]).sum()),
                "delta_tp": int((proposed & labels[start:stop]).sum() - (base & labels[start:stop]).sum()),
            })
        output.append(row)
    return output


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for value, members in groups.items():
        folds = [sum(r["delta_tp"] for r in members if r["station_fold"] == fold) for fold in range(5)]
        changed = [r for r in members if r["changed"]]
        output.append({
            "stratum": key,
            "value": value,
            "orders": len(members),
            "changed_orders": len(changed),
            "delta_tp": int(sum(r["delta_tp"] for r in members)),
            "station_fold_delta_tp": folds,
            "nonnegative_folds": int(sum(delta >= 0 for delta in folds)),
            "strictly_positive_folds": int(sum(delta > 0 for delta in folds)),
        })
    return sorted(output, key=lambda row: (row["strictly_positive_folds"], row["delta_tp"], row["orders"]), reverse=True)


def stable_combinations(rows: list[dict]) -> list[dict]:
    # Only a small, predeclared interaction family.  It avoids post-hoc searches
    # over arbitrary title strings while still checking whether domain shift is
    # concentrated in broad deployment characteristics.
    candidates = []
    for first, second in (("domain_bin", "alarm_bin"), ("domain_bin", "vendor"), ("alarm_bin", "v30_v33_top1_same")):
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            groups[(str(row[first]), str(row[second]))].append(row)
        for values, members in groups.items():
            if len(members) < 20:
                continue
            folds = [sum(r["delta_tp"] for r in members if r["station_fold"] == fold) for fold in range(5)]
            candidates.append({
                "stratum": f"{first}+{second}", "value": list(values), "orders": len(members),
                "changed_orders": sum(r["changed"] for r in members), "delta_tp": int(sum(folds)),
                "station_fold_delta_tp": folds,
                "nonnegative_folds": int(sum(delta >= 0 for delta in folds)),
                "strictly_positive_folds": int(sum(delta > 0 for delta in folds)),
            })
    return sorted(candidates, key=lambda row: (row["strictly_positive_folds"], row["delta_tp"], row["orders"]), reverse=True)


def test_match_count(train_rows: list[dict], test_rows: list[dict], field: str, value: object) -> int:
    value = str(value)
    return sum(str(row[field]) == value for row in test_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        archive_records = json.load(handle)

    train_rows = order_rows(
        arrays, "train", records_for_split(archive_records, "train"),
        arrays["train_station_folds"].astype(np.int8),
        np.load(V33 / "domain_train_probability.npy"),
        np.load(V30 / "station_extra_trees_oof.npy"),
        np.load(V33 / "station_alpha_2_oof.npy"), arrays["train_labels"].astype(bool),
    )
    test_rows = order_rows(
        arrays, "test", records_for_split(archive_records, "test"), None,
        np.load(V33 / "domain_test_probability.npy"),
        np.load(V30 / "station_extra_trees_test.npy"),
        np.load(V33 / "station_alpha_2_test.npy"), None,
    )
    one_way = [entry for key in ("domain_bin", "alarm_bin", "vendor", "v30_v33_top1_same") for entry in summarize(train_rows, key)]
    interactions = stable_combinations(train_rows)
    strict = [entry for entry in one_way + interactions if entry["orders"] >= 20 and entry["changed_orders"] >= 8 and entry["strictly_positive_folds"] == 5]
    for entry in strict:
        if "+" not in entry["stratum"]:
            entry["test_orders"] = test_match_count(train_rows, test_rows, entry["stratum"], entry["value"])
        else:
            left, right = entry["stratum"].split("+")
            first, second = entry["value"]
            entry["test_orders"] = sum(str(row[left]) == first and str(row[right]) == second for row in test_rows)
    report = {
        "version": "v69-domain-stratified-router-audit-1",
        "method": "V33 versus V30 ranking at oracle per-order counts; station-cross-fitted predictions only.",
        "warning": "Oracle counts isolate ranking and cannot be submitted without a count policy.",
        "orders": {"train": len(train_rows), "test": len(test_rows)},
        "overall": {
            "changed_orders": sum(row["changed"] for row in train_rows),
            "delta_tp": int(sum(row["delta_tp"] for row in train_rows)),
            "station_fold_delta_tp": [int(sum(row["delta_tp"] for row in train_rows if row["station_fold"] == fold)) for fold in range(5)],
        },
        "one_way_strata": one_way,
        "interaction_strata": interactions,
        "strict_stable_candidates": strict,
        "gate": {
            "requires_strictly_positive_all_station_folds": True,
            "requires_at_least_20_train_orders": True,
            "requires_at_least_8_changed_train_orders": True,
            "passed": bool(strict),
            "note": "Passing only identifies a ranking stratum; it does not solve count allocation or prove public-board gain.",
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": report["overall"],
        "strict_stable_candidates": strict,
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

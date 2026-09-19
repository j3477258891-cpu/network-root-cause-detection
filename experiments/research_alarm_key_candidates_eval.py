"""Cross-validated audit of ``research_alarm_key_candidates`` deletions.

The companion generator uses repeated alarm keys to suggest test deletions.
This script evaluates that rule on training orders without leaking the held-out
order's labels.  It also checks the suggestions against the current 1,035-root
test baseline.  The train analogue uses the V11 exact-count mask (3,169 train
roots) and reports the resulting TP/F1 after top-k deletions.  A second fold
mode excludes reference orders sharing a station with each held-out order.

Research only: this script never creates a submission CSV and does not treat
OOF estimates as online evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
NPZ = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
CANDIDATE_JSON = ROOT / "experiments/research_alarm_key_candidates.json"
OUT = ROOT / "experiments/research_alarm_key_candidates_eval.json"
V30_REPORT = ROOT / "experiments/v30_meta_stack/v30_report.json"
TARGET_TRAIN = 3169
KINDS = (
    "full_no_location",
    "title_reason_cause_timeline",
    "title_cause_timeline",
    "title_reason",
    "title_label",
    "reason_label",
)
TOP_K = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100)

sys.path.insert(0, str(ROOT / "experiments"))
import research_alarm_key_candidates as keygen  # noqa: E402


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def load_orders() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with gzip.open(DATA, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["train"], payload["test"]


def load_baseline() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with BASE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            payload = json.loads(row["output"])
            result[row["order_id"]] = {node["@rid"] for node in payload["rootcause"]}
    return result


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    """Copy the V11/V30 exact-count selection rule."""
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:8]
        if len(ranked) == 0:
            continue
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional_array = np.asarray(optional, dtype=np.int64)
    optional_array = optional_array[np.argsort(-scores[optional_array], kind="stable")]
    remaining = target - int(selected.sum())
    if not 0 <= remaining <= len(optional_array):
        raise ValueError((target, int(selected.sum()), len(optional_array)))
    selected[optional_array[:remaining]] = True
    return selected


def key_counts(
    orders: list[dict[str, Any]], indices: Iterable[int], kind: str
) -> tuple[dict[tuple, tuple[int, int]], list[dict[tuple, tuple[int, int]]]]:
    """Return aggregate and per-order (positive,count) key statistics."""
    aggregate: defaultdict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    per_order: list[dict[tuple, tuple[int, int]]] = [dict() for _ in orders]
    for index in indices:
        local: defaultdict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        for alarm in orders[index]["alarms"]:
            token = keygen.key(alarm, kind)
            value = int(alarm.get("is_root") or 0)
            aggregate[token][0] += value
            aggregate[token][1] += 1
            local[token][0] += value
            local[token][1] += 1
        per_order[index] = {token: (vals[0], vals[1]) for token, vals in local.items()}
    return {token: (vals[0], vals[1]) for token, vals in aggregate.items()}, per_order


def station_exclusion_index(
    orders: list[dict[str, Any]], reference: list[int]
) -> dict[str, set[int]]:
    by_station: defaultdict[str, set[int]] = defaultdict(set)
    for index in reference:
        for station in orders[index].get("station_ids") or []:
            by_station[str(station)].add(index)
    return by_station


def lookup_stats(
    token: tuple,
    aggregate: dict[tuple, tuple[int, int]],
    per_order: list[dict[tuple, tuple[int, int]]],
    excluded: set[int] | None,
) -> tuple[int, int]:
    positives, count = aggregate.get(token, (0, 0))
    if excluded:
        for index in excluded:
            old_positive, old_count = per_order[index].get(token, (0, 0))
            positives -= old_positive
            count -= old_count
    return positives, count


def make_train_baseline(
    orders: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(NPZ, allow_pickle=False) as archive:
        ptr = archive["train_alarm_ptr"].copy()
        scores = archive["train_v11"].copy()
        labels = archive["train_labels"].astype(np.int8).copy()
    if len(orders) != len(ptr) - 1:
        raise ValueError("semantic record and NPZ train order counts differ")
    selected = exact_count_mask(scores, ptr, TARGET_TRAIN)
    tp = int(np.sum(selected & labels.astype(bool)))
    metadata = {
        "predictions": int(selected.sum()),
        "tp": tp,
        "positive_labels": int(labels.sum()),
        "f1": 2.0 * tp / (int(labels.sum()) + int(selected.sum())),
        "rule": "V11 exact_count_mask(train_v11, target=3169)",
    }
    return ptr, labels, selected, metadata


def deletion_curve(
    selected_rows: list[dict[str, Any]],
    baseline: np.ndarray,
    labels: np.ndarray,
    ptr: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> dict[str, Any]:
    """Rank held-out deletion rows and enforce at least one root/order."""
    ranked = sorted(
        selected_rows,
        key=lambda row: (row["upper90"], -row["support"], row["p"], row["flat_index"]),
    )
    remaining = {
        order_index: int(
            baseline[int(ptr[order_index]) : int(ptr[order_index + 1])].sum()
        )
        for order_index in {row["order_index"] for row in ranked}
    }
    safe_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in ranked:
        if remaining[row["order_index"]] <= 1:
            skipped += 1
            continue
        safe_rows.append(row)
        remaining[row["order_index"]] -= 1

    baseline_tp = int(np.sum(baseline & labels.astype(bool)))
    baseline_p = int(baseline.sum())
    n_positive = int(labels.sum())
    curves: list[dict[str, Any]] = []
    for k in top_k:
        chosen = safe_rows[:k]
        removed_tp = int(sum(row["label"] for row in chosen))
        removed_fp = len(chosen) - removed_tp
        new_p = baseline_p - len(chosen)
        new_tp = baseline_tp - removed_tp
        f1 = 2.0 * new_tp / (n_positive + new_p)
        curves.append(
            {
                "requested_k": k,
                "selected_k": len(chosen),
                "removed_tp": removed_tp,
                "removed_false_positive": removed_fp,
                "delta_tp": -removed_tp,
                "predictions": new_p,
                "tp": new_tp,
                "f1": f1,
                "delta_f1": f1 - 2.0 * baseline_tp / (n_positive + baseline_p),
            }
        )
    return {
        "pool_count": len(ranked),
        "safe_pool_count": len(safe_rows),
        "skipped_for_min_one_root": skipped,
        "curves": curves,
        "top_rows": safe_rows[:20],
    }


def evaluate_train(
    orders: list[dict[str, Any]],
    ptr: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    fold_name: str,
    fold_values: np.ndarray,
    station_disjoint: bool,
    min_support: int,
    upper_threshold: float,
) -> dict[str, Any]:
    all_indices = list(range(len(orders)))
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(set(int(x) for x in fold_values)):
        heldout = [index for index in all_indices if int(fold_values[index]) == fold]
        reference = [index for index in all_indices if int(fold_values[index]) != fold]
        aggregate, per_order = key_counts(orders, reference, fold_name)
        by_station = station_exclusion_index(orders, reference) if station_disjoint else {}
        fold_count = 0
        for order_index in heldout:
            excluded: set[int] = set()
            if station_disjoint:
                for station in orders[order_index].get("station_ids") or []:
                    excluded.update(by_station.get(str(station), set()))
            start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
            baseline_count = int(baseline[start:stop].sum())
            if baseline_count <= 1:
                continue
            for local_index, alarm in enumerate(orders[order_index]["alarms"]):
                flat_index = start + local_index
                if not baseline[flat_index]:
                    continue
                token = keygen.key(alarm, fold_name)
                positives, count = lookup_stats(token, aggregate, per_order, excluded)
                if count < min_support:
                    continue
                p = positives / count
                _, upper = wilson(positives, count)
                if upper > upper_threshold:
                    continue
                selected_rows.append(
                    {
                        "order_index": order_index,
                        "rid": alarm["rid"],
                        "flat_index": flat_index,
                        "support": count,
                        "p": p,
                        "upper90": upper,
                        "label": int(labels[flat_index]),
                        "fold": fold,
                    }
                )
                fold_count += 1
        fold_rows.append({"fold": fold, "heldout_orders": len(heldout), "pool_count": fold_count})
    curve = deletion_curve(selected_rows, baseline, labels, ptr)
    curve.update(
        {
            "fold_name": fold_name,
            "fold_strategy": "station_disjoint" if station_disjoint else "order_grouped",
            "min_support": min_support,
            "upper90_threshold": upper_threshold,
            "folds": fold_rows,
        }
    )
    return curve


def test_candidate_audit(test_orders: list[dict[str, Any]], base: dict[str, set[str]]) -> dict[str, Any]:
    payload = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    deletions = payload.get("deletions", [])
    rows: list[dict[str, Any]] = []
    for candidate in deletions:
        order_id, rid = candidate["order_id"], candidate["rid"]
        roots = base.get(order_id, set())
        row = dict(candidate)
        row["exists_in_test_baseline"] = rid in roots
        row["baseline_root_count"] = len(roots)
        row["safe_individual_delete"] = rid in roots and len(roots) > 1
        rows.append(row)
    rows.sort(key=lambda row: (row.get("upper90", 1.0), -row.get("support", 0), row["order_id"], row["rid"]))
    remaining = Counter({order_id: len(roots) for order_id, roots in base.items()})
    safe_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if not row["exists_in_test_baseline"] or remaining[row["order_id"]] <= 1:
            skipped += 1
            continue
        safe_rows.append(row)
        remaining[row["order_id"]] -= 1
    baseline_tp, baseline_p, positives = 956, 1035, 1044
    curves = []
    for k in TOP_K:
        chosen = safe_rows[:k]
        p = baseline_p - len(chosen)
        # Without online labels, these are the exact all-false/all-true bounds.
        tp_if_all_false = baseline_tp
        tp_if_all_true = baseline_tp - len(chosen)
        curves.append(
            {
                "requested_k": k,
                "selected_k": len(chosen),
                "predictions": p,
                "tp_range_if_labels_unknown": [tp_if_all_true, tp_if_all_false],
                "f1_range_if_labels_unknown": [
                    2.0 * tp_if_all_true / (positives + p),
                    2.0 * tp_if_all_false / (positives + p),
                ],
                "selected_orders": len({row["order_id"] for row in chosen}),
            }
        )
    return {
        "candidate_count": len(rows),
        "exists_count": sum(row["exists_in_test_baseline"] for row in rows),
        "safe_individual_count": sum(row["safe_individual_delete"] for row in rows),
        "greedy_safe_count": len(safe_rows),
        "skipped_by_existence_or_min_one_root": skipped,
        "baseline": {"predictions": baseline_p, "tp": baseline_tp, "positive_roots": positives, "f1": 2.0 * baseline_tp / (positives + baseline_p)},
        "curves": curves,
        "top_rows": safe_rows[:30],
        "all_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--upper90-threshold", type=float, default=0.30)
    args = parser.parse_args()
    train_orders, test_orders = load_orders()
    with np.load(NPZ, allow_pickle=False) as archive:
        folds = archive["train_folds"].copy()
        station_folds = archive["train_station_folds"].copy()
    ptr, labels, baseline, train_baseline = make_train_baseline(train_orders)
    base_test = load_baseline()
    train_results: list[dict[str, Any]] = []
    # Each key representation is evaluated independently.  The default
    # threshold mirrors the candidate generator; callers can rerun with a
    # stricter support/upper bound.
    for kind in KINDS:
        train_results.append(
            evaluate_train(
                train_orders,
                ptr,
                labels,
                baseline,
                kind,
                folds,
                False,
                args.min_support,
                args.upper90_threshold,
            )
        )
        train_results.append(
            evaluate_train(
                train_orders,
                ptr,
                labels,
                baseline,
                kind,
                station_folds,
                True,
                args.min_support,
                args.upper90_threshold,
            )
        )
    v30 = {}
    if V30_REPORT.exists():
        report = json.loads(V30_REPORT.read_text(encoding="utf-8"))
        v30 = {
            "base_tp": report.get("base_tp"),
            "consensus_exact_count_tp": report.get("results", {}).get("consensus", {}).get("tp"),
            "consensus_delta_tp": report.get("results", {}).get("consensus", {}).get("delta_tp"),
            "note": "offline V30 exact-count analogue, not online evidence",
        }
    test_audit = test_candidate_audit(test_orders, base_test)
    nonzero_curves = []
    for result in train_results:
        for curve in result["curves"]:
            if curve["requested_k"] > 0 and curve["selected_k"] > 0:
                nonzero_curves.append(
                    {
                        "f1": curve["f1"],
                        "delta_f1": curve["delta_f1"],
                        "delta_tp": curve["delta_tp"],
                        "k": curve["selected_k"],
                        "kind": result["fold_name"],
                        "fold_strategy": result["fold_strategy"],
                    }
                )
    best_train = max(nonzero_curves, key=lambda row: (row["f1"], row["delta_f1"]))
    output = {
        "version": 1,
        "source_policy": "train leave-out/order-grouped and station-disjoint diagnostics only; no online claims",
        "train_baseline": train_baseline,
        "v30_exact_count_analogue": v30,
        "train_results": train_results,
        "test_candidate_audit": test_audit,
        "screening_recommendation": {
            "decision": "do_not_submit_deletion_pool",
            "best_train_curve": best_train,
            "test_exists_count": test_audit["exists_count"],
            "test_safe_individual_count": test_audit["safe_individual_count"],
            "test_greedy_safe_count": test_audit["greedy_safe_count"],
            "reason": "Cross-validated gains peak at a small exploratory k and then turn negative; test labels remain unknown and no deletion set is equation-safe.",
        },
        "interpretation": [
            "A deletion changes TP by minus the number of true roots removed; F1 can rise only when removed roots are false positives often enough to offset the lower prediction count.",
            "The test candidate audit proves existence and min-one-root structure only; unknown test labels are not filled from train rates.",
            "No submission CSV was generated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    # concise console summary for reproducibility logs
    best = []
    for result in train_results:
        top = max(result["curves"], key=lambda row: row["f1"])
        best.append((top["f1"], result["fold_name"], result["fold_strategy"], top["requested_k"], top["delta_tp"]))
    print(json.dumps({"output": str(args.output), "train_baseline": train_baseline, "best_train_curves": sorted(best, reverse=True)[:12], "test": {k: output["test_candidate_audit"][k] for k in ("candidate_count", "exists_count", "safe_individual_count", "greedy_safe_count")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

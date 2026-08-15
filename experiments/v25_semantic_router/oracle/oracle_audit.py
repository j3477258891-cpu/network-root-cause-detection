"""
V25 Semantic Router — Oracle 审计
计算候选动作集合的理论上限，回答三个问题：
1. 完美路由能达到多少 TP？
2. 覆盖了多少 V11 漏选根因？
3. 每个工单的 oracle 增益分布如何？
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add V10 path
sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10

sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data


MAX_ROOT = 8
MAX_ACTION = 2  # max add/remove/swap per order


def load_baseline():
    """Load V11 baseline: train OOF logits, labels, slices, folds."""
    dataset = v25_data.load_all_data(
        "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
    )
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    folds = dataset["folds"]
    train_v11 = dataset["train_v11"]
    train_orders = dataset["train_orders"]

    target_train = round(1059 / 546 * 1634)
    base_mask = v10.exact_count_mask(train_v11, slices, target_train, MAX_ROOT)

    base_stats = v25_data.confusion(base_mask, labels)
    print(f"V11 Baseline: TP={base_stats[1]} FP={base_stats[2]} FN={base_stats[3]} N={base_stats[4]}")

    # Per-order V11 prediction
    base_per_order = []
    for sl in slices:
        base_per_order.append(base_mask[sl.start:sl.stop])

    return {
        "labels": labels,
        "slices": slices,
        "folds": folds,
        "base_mask": base_mask,
        "base_stats": base_stats,
        "base_per_order": base_per_order,
        "train_orders": train_orders,
        "target_train": target_train,
    }


def oracle_per_order(labels_sl, base_pred_sl, v11_scores_sl):
    """
    Compute oracle for a single work order.

    Given V11 selected nodes and all ground truth labels,
    compute the maximum achievable TP if we can:
    - add up to MAX_ACTION new root causes
    - remove up to MAX_ACTION false selections
    - swap up to MAX_ACTION selections

    Returns:
        current_tp: TP with V11 baseline
        oracle_tp: maximum achievable TP
        best_actions: (add_rids, remove_rids, swap_pairs)
        fn_covered: how many of V11's missed root causes are covered
    """
    n = len(labels_sl)
    base_tp = int(np.sum(base_pred_sl & (labels_sl == 1)))
    base_count = int(np.sum(base_pred_sl))

    # V11 FN: true root causes not selected
    fn_mask = (base_pred_sl == 0) & (labels_sl == 1)
    fn_indices = np.where(fn_mask)[0]
    n_fn = len(fn_indices)

    # V11 FP: false selections
    fp_mask = base_pred_sl & (labels_sl == 0)
    fp_indices = np.where(fp_mask)[0]
    n_fp = len(fp_indices)

    # Can't improve if no FN and no FP
    if n_fn == 0 and n_fp == 0:
        return base_tp, base_tp, {"add": [], "remove": [], "swap": []}, 0

    # Best possible: add all FN, remove all FP, but limited by MAX_ACTION
    fn_sorted = fn_indices[np.argsort(-v11_scores_sl[fn_indices])]  # most likely FN first
    fp_sorted = fp_indices[np.argsort(v11_scores_sl[fp_indices])]   # most likely FP first

    add_count = min(n_fn, MAX_ACTION, MAX_ROOT - base_count)
    rm_count = min(n_fp, MAX_ACTION, base_count - 1)  # keep at least 1

    # Try all combinations of add/remove within budget
    best_tp = base_tp
    best_add = []
    best_rm = []
    best_swap = []

    for na in range(0, min(n_fn, MAX_ACTION) + 1):
        for nr in range(0, min(n_fp, MAX_ACTION) + 1):
            if na + nr > MAX_ACTION:
                continue
            if base_count + na - nr > MAX_ROOT:
                continue
            if base_count + na - nr < 1:
                continue

            new_mask = base_pred_sl.copy()
            if na > 0:
                new_mask[fn_sorted[:na]] = True
            if nr > 0:
                new_mask[fp_sorted[:nr]] = False

            new_tp = int(np.sum(new_mask & (labels_sl == 1)))
            if new_tp > best_tp:
                best_tp = new_tp
                best_add = list(fn_sorted[:na])
                best_rm = list(fp_sorted[:nr])
                best_swap = []

    # Also try swaps (1 swap = 1 add + 1 remove)
    for na in range(0, min(n_fn, MAX_ACTION) + 1):
        for nr in range(0, min(n_fp, MAX_ACTION) + 1):
            ns = na + nr
            if ns > MAX_ACTION:
                continue
            if base_count + na - nr > MAX_ROOT or base_count + na - nr < 1:
                continue

            new_mask = base_pred_sl.copy()
            if na > 0:
                new_mask[fn_sorted[:na]] = True
            if nr > 0:
                new_mask[fp_sorted[:nr]] = False

            new_tp = int(np.sum(new_mask & (labels_sl == 1)))
            if new_tp > best_tp:
                best_tp = new_tp
                best_add = list(fn_sorted[:na])
                best_rm = list(fp_sorted[:nr])
                best_swap = []

    fn_covered = len(set(best_add))
    return base_tp, best_tp, {"add": best_add, "remove": best_rm, "swap": best_swap}, fn_covered


def oracle_audit(baseline):
    """
    Full oracle audit across all training work orders.

    Returns:
        total_oracle_tp_gain: max TP gain across all orders
        fn_coverage: fraction of V11 FNs covered
        per_fold: fold-level breakdown
        order_details: per-order oracle, sorted by gain
    """
    labels = baseline["labels"]
    slices = baseline["slices"]
    base_per_order = baseline["base_per_order"]
    folds = baseline["folds"]
    train_v11 = baseline.get("train_v11")

    total_base_tp = 0
    total_oracle_tp = 0
    total_fn = 0
    total_fn_covered = 0
    per_fold_base = defaultdict(int)
    per_fold_oracle = defaultdict(int)
    order_details = []

    for oi, sl in enumerate(slices):
        start, stop = sl.start, sl.stop
        order_labels = labels[start:stop]
        order_base = base_per_order[oi]
        order_v11 = train_v11[start:stop] if train_v11 is not None else np.zeros_like(
            order_labels, dtype=np.float32
        )

        base_tp, oracle_tp, actions, fn_cov = oracle_per_order(
            order_labels, order_base, order_v11
        )

        total_base_tp += base_tp
        total_oracle_tp += oracle_tp
        total_fn += int(np.sum((order_base == 0) & (order_labels == 1)))
        total_fn_covered += fn_cov

        fold = int(folds[oi])
        per_fold_base[fold] += base_tp
        per_fold_oracle[fold] += oracle_tp

        gain = oracle_tp - base_tp
        order_details.append({
            "order_idx": oi,
            "fold": fold,
            "base_tp": base_tp,
            "oracle_tp": oracle_tp,
            "gain": gain,
            "fn": int(np.sum((order_base == 0) & (order_labels == 1))),
            "fn_covered": fn_cov,
            "add_count": len(actions["add"]),
            "rm_count": len(actions["remove"]),
            "swap_count": len(actions["swap"]),
        })

    # Sort by gain descending
    order_details.sort(key=lambda x: -x["gain"])

    total_gain = total_oracle_tp - total_base_tp
    fn_coverage = total_fn_covered / max(total_fn, 1)

    print(f"\n{'='*60}")
    print(f"ORACLE AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline TP:    {total_base_tp}")
    print(f"  Oracle TP:      {total_oracle_tp}")
    print(f"  Max gain:       {total_gain:+d}")
    print(f"  V11 FN:         {total_fn}")
    print(f"  FN covered:     {total_fn_covered} ({fn_coverage:.2%})")

    print(f"\n  Per-fold:")
    for fold in sorted(per_fold_base):
        gain = per_fold_oracle[fold] - per_fold_base[fold]
        print(f"    Fold {fold}: base={per_fold_base[fold]} → oracle={per_fold_oracle[fold]} (gain={gain:+d})")

    print(f"\n  Top 20 orders by oracle gain:")
    for d in order_details[:20]:
        print(f"    order[{d['order_idx']}] fold={d['fold']} "
              f"base={d['base_tp']}→oracle={d['oracle_tp']} "
              f"gain={d['gain']:+d} fn={d['fn']} fn_cov={d['fn_covered']}")

    # Distribution of gains
    gain_dist = defaultdict(int)
    for d in order_details:
        gain_dist[d["gain"]] += 1
    print(f"\n  Gain distribution:")
    for gain in sorted(gain_dist):
        print(f"    +{gain}: {gain_dist[gain]} orders")

    # Gate check
    oracle_gate = {"min_gain": 170, "min_fn_coverage": 0.80}

    passed = (
        total_gain >= oracle_gate["min_gain"]
        and fn_coverage >= oracle_gate["min_fn_coverage"]
    )
    print(f"\n  ORACLE GATE: {'PASSED' if passed else 'FAILED'}")
    print(f"    Gain {total_gain:+d} >= {oracle_gate['min_gain']}: "
          f"{'YES' if total_gain >= oracle_gate['min_gain'] else 'NO'}")
    print(f"    FN coverage {fn_coverage:.2%} >= {oracle_gate['min_fn_coverage']:.0%}: "
          f"{'YES' if fn_coverage >= oracle_gate['min_fn_coverage'] else 'NO'}")

    return {
        "base_tp": int(total_base_tp),
        "oracle_tp": int(total_oracle_tp),
        "gain": int(total_gain),
        "fn_total": int(total_fn),
        "fn_covered": int(total_fn_covered),
        "fn_coverage": float(fn_coverage),
        "per_fold": {str(k): {"base": v, "oracle": per_fold_oracle[k]} for k, v in per_fold_base.items()},
        "order_details": order_details[:30],  # keep top 30
        "gain_distribution": {str(k): v for k, v in sorted(gain_dist.items())},
        "gate_passed": passed,
    }


def main():
    print("Loading V11 baseline...")
    baseline = load_baseline()

    print("\nRunning oracle audit...")
    result = oracle_audit(baseline)

    # Save
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "v25_oracle_report.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {output_dir / 'v25_oracle_report.json'}")
    return result


if __name__ == "__main__":
    main()

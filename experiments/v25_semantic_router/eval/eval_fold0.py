"""
V25 Semantic Router — 单折探针评估
等云端的 fold-0 logits 产出后，做 fixed-K TP delta 评估和逐工单审计。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

MAX_ROOT = 8


def load_v11_baseline():
    dataset = v25_data.load_all_data(
        "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
    )
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    folds = dataset["folds"]
    train_v11 = dataset["train_v11"]

    target_train = round(1059 / 546 * 1634)
    base_mask = v10.exact_count_mask(train_v11, slices, target_train, MAX_ROOT)

    return {
        "labels": labels,
        "slices": slices,
        "folds": folds,
        "train_v11": train_v11,
        "base_mask": base_mask,
        "target_train": target_train,
        "dataset": dataset,
    }


def evaluate_fold_zero(baseline, fold0_logits, fold=0):
    """
    Evaluate semantic expert's fold-0 OOF logits.

    Args:
        baseline: V11 baseline dict from load_v11_baseline()
        fold0_logits: np.array of shape (n_rows,) — semantic expert OOF for fold 0
        fold: which fold to evaluate (default 0)

    Returns:
        eval_report dict
    """
    labels = baseline["labels"]
    slices = baseline["slices"]
    folds = baseline["folds"]
    base_mask = baseline["base_mask"]
    train_v11 = baseline["train_v11"]
    target_train = baseline["target_train"]

    all_orders = np.arange(len(slices))
    fold_orders = all_orders[folds == fold]

    # Compute fold rows
    fold_rows = np.concatenate([
        np.arange(sl.start, sl.stop) for i, sl in enumerate(slices) if i in fold_orders
    ])

    # Rebuild local slices for fold
    local_slices = []
    offset = 0
    for oi in fold_orders:
        sl = slices[oi]
        n = sl.stop - sl.start
        local_slices.append(slice(offset, offset + n))
        offset += n

    fold_labels = labels[fold_rows]
    fold_base = base_mask[fold_rows]
    fold_v11 = train_v11[fold_rows]

    # Proportional target for fold
    fold_target = round(target_train * len(fold_rows) / len(labels))

    # Fixed-K evaluation
    fold_new = v10.exact_count_mask(fold0_logits, local_slices, fold_target, MAX_ROOT)

    fold_base_tp = int(np.sum(fold_base & (fold_labels == 1)))
    fold_new_tp = int(np.sum(fold_new & (fold_labels == 1)))
    fold_delta = fold_new_tp - fold_base_tp

    # Per-order breakdown
    per_order = []
    for local_oi, oi in enumerate(fold_orders):
        sl = slices[oi]
        local_sl = local_slices[local_oi]
        order_base = fold_base[local_sl]
        order_labels = fold_labels[local_sl]
        order_new = fold_new[local_sl]

        base_tp = int(np.sum(order_base & (order_labels == 1)))
        new_tp = int(np.sum(order_new & (order_labels == 1)))
        delta = new_tp - base_tp

        base_fn = order_labels & ~order_base
        new_fn = order_labels & ~order_new

        per_order.append({
            "global_oi": int(oi),
            "base_tp": base_tp,
            "new_tp": new_tp,
            "delta": int(delta),
            "base_fn_count": int(np.sum(base_fn)),
            "new_fn_count": int(np.sum(new_fn)),
            "v11_fns_fixed": int(np.sum(base_fn & order_new)),
        })

    # Fold-level stats
    print(f"\n{'='*60}")
    print(f"FOLD {fold} EVALUATION")
    print(f"{'='*60}")
    print(f"  Orders:       {len(fold_orders)}")
    print(f"  Rows:         {len(fold_rows)}")
    print(f"  Target count: {fold_target}")
    print(f"  Base TP:      {fold_base_tp}")
    print(f"  New TP:       {fold_new_tp}")
    print(f"  TP Delta:     {fold_delta:+d}")

    # Breakdown by delta
    improved = sum(1 for d in per_order if d["delta"] > 0)
    unchanged = sum(1 for d in per_order if d["delta"] == 0)
    degraded = sum(1 for d in per_order if d["delta"] < 0)
    print(f"\n  Orders improved:  {improved}")
    print(f"  Orders unchanged: {unchanged}")
    print(f"  Orders degraded:  {degraded}")

    # V11 FN fix rate
    total_v11_fn = sum(d["base_fn_count"] for d in per_order)
    v11_fn_fixed = sum(d["v11_fns_fixed"] for d in per_order)
    print(f"\n  V11 FNs:         {total_v11_fn}")
    print(f"  V11 FNs fixed:   {v11_fn_fixed} ({v11_fn_fixed/max(total_v11_fn,1):.2%})")

    # Top improved/degraded orders
    per_order.sort(key=lambda x: -x["delta"])
    print(f"\n  Top 10 improved:")
    for d in per_order[:10]:
        print(f"    order[{d['global_oi']}]: {d['base_tp']}→{d['new_tp']} (+{d['delta']}) "
              f"fn={d['base_fn_count']} fixed={d['v11_fns_fixed']}")

    degraded_orders = [d for d in per_order if d["delta"] < 0]
    print(f"\n  Degraded orders ({len(degraded_orders)}):")
    for d in sorted(degraded_orders, key=lambda x: x["delta"])[:10]:
        print(f"    order[{d['global_oi']}]: {d['base_tp']}→{d['new_tp']} ({d['delta']}) "
              f"fn={d['base_fn_count']}")

    # Decision
    semantic_gate = {
        "min_fold_delta": 0,  # must be non-negative
        "min_fn_fix_rate": 0.15,  # fix at least 15% of V11 FNs
    }

    passes_fold = fold_delta >= semantic_gate["min_fold_delta"]
    fn_rate = v11_fn_fixed / max(total_v11_fn, 1)
    passes_fn = fn_rate >= semantic_gate["min_fn_fix_rate"]

    print(f"\n  SEMANTIC GATE: {'PASS' if (passes_fold and passes_fn) else 'FAIL'}")
    print(f"    Fold TP delta >= 0: {'Y' if passes_fold else 'N'} ({fold_delta:+d})")
    print(f"    FN fix rate >= {semantic_gate['min_fn_fix_rate']:.0%}: "
          f"{'Y' if passes_fn else 'N'} ({fn_rate:.2%})")

    return {
        "fold": fold,
        "fold_delta": int(fold_delta),
        "base_tp": int(fold_base_tp),
        "new_tp": int(fold_new_tp),
        "orders_improved": improved,
        "orders_degraded": degraded,
        "v11_fn_total": int(total_v11_fn),
        "v11_fn_fixed": int(v11_fn_fixed),
        "fn_fix_rate": float(fn_rate),
        "gate_pass_fold": passes_fold,
        "gate_pass_fn": passes_fn,
        "per_order": per_order,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True, help="Path to fold-0 OOF logits .npy")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--output", default="outputs/v25_fold0_eval.json", help="Output path")
    args = parser.parse_args()

    baseline = load_v11_baseline()
    logits = np.load(args.logits).astype(np.float32)

    if len(logits) != baseline["labels"][baseline["folds"] == args.fold].size:
        # Logits may be full-size; subset to fold
        fold_rows = np.zeros(len(baseline["labels"]), dtype=bool)
        all_orders = np.arange(len(baseline["slices"]))
        fold_orders = all_orders[baseline["folds"] == args.fold]
        for oi in fold_orders:
            sl = baseline["slices"][oi]
            fold_rows[sl.start:sl.stop] = True
        logits = logits[fold_rows]
        print(f"Subset logits to fold rows: {len(logits)}")

    result = evaluate_fold_zero(baseline, logits, args.fold)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.output}")
    return result


if __name__ == "__main__":
    main()

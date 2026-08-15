"""
V26 Selective Error Router — Oracle 分解
分析 +184 TP 的构成：哪些 action、模板、排名区间贡献了增益
"""

import sys, json, hashlib
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

MAX_ROOT = 8
MAX_ACTION = 3


def load_data():
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

    return {
        "labels": labels, "slices": slices, "folds": folds,
        "train_v11": train_v11, "base_mask": base_mask,
        "train_orders": train_orders, "target_train": target_train,
    }


def oracle_per_order(order_labels, base_pred, v11_scores):
    """
    Compute best action per order.
    Returns (best_action_type, base_tp, oracle_tp, add_indices, rm_indices).
    action_type: 0=keep, 1=add1, 2=add2, 3=add3, 4=rm1, 5=rm2, 6=swap1, 7=swap2
    """
    n = len(order_labels)
    base_tp = int(np.sum(base_pred & (order_labels == 1)))
    base_count = int(np.sum(base_pred))

    fn_mask = (base_pred == 0) & (order_labels == 1)
    fp_mask = base_pred & (order_labels == 0)
    fn_idx = np.where(fn_mask)[0]
    fp_idx = np.where(fp_mask)[0]

    if len(fn_idx) == 0 and len(fp_idx) == 0:
        return 0, base_tp, base_tp, [], []

    fn_sorted = fn_idx[np.argsort(-v11_scores[fn_idx])]
    fp_sorted = fp_idx[np.argsort(v11_scores[fp_idx])]

    best_action = 0
    best_tp = base_tp
    best_add = []
    best_rm = []

    actions = [
        (0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0),
        (4, 0, 1), (5, 0, 2), (6, 1, 1), (7, 2, 2),
    ]

    for a_id, na, nr in actions:
        if na > len(fn_idx) or nr > len(fp_idx):
            continue
        new_count = base_count + na - nr
        if new_count < 1 or new_count > MAX_ROOT:
            continue

        new_mask = base_pred.copy()
        add_idx = []
        rm_idx = []
        if na:
            new_mask[fn_sorted[:na]] = True
            add_idx = fn_sorted[:na].tolist()
        if nr:
            new_mask[fp_sorted[:nr]] = False
            rm_idx = fp_sorted[:nr].tolist()

        new_tp = int(np.sum(new_mask & (order_labels == 1)))
        if new_tp > best_tp:
            best_tp = new_tp
            best_action = a_id
            best_add = add_idx
            best_rm = rm_idx

    return best_action, base_tp, best_tp, best_add, best_rm


def order_signature(order):
    tc = Counter(v10.scalar(n.get("title")) for n in order["alarms"])
    tt = sorted(v10.scalar(n.get("title")) for n in order["alarms"] if n.get("label") == "TargetAlarm")
    return (tuple(sorted(tc.items())), tuple(tt), len(order["alarms"]))


def decompose_oracle(data):
    """Full oracle decomposition."""
    labels = data["labels"]
    slices = data["slices"]
    folds = data["folds"]
    base_mask = data["base_mask"]
    train_v11 = data["train_v11"]
    orders = data["train_orders"]

    action_names = {0: "keep", 1: "add1", 2: "add2", 3: "add3",
                    4: "rm1", 5: "rm2", 6: "swap1", 7: "swap2"}

    total_base = 0
    total_oracle = 0
    action_counts = Counter()
    action_gains = defaultdict(int)
    per_fold = defaultdict(lambda: {"base": 0, "oracle": 0, "actions": Counter()})
    template_gains = defaultdict(lambda: {"count": 0, "gain": 0})

    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base_mask[sl]
        ov = train_v11[sl]

        action, bt, ot, add, rm = oracle_per_order(ol, ob, ov)
        gain = ot - bt

        total_base += bt
        total_oracle += ot
        action_counts[action] += 1
        if gain > 0:
            action_gains[action] += gain

        f = int(folds[oi])
        per_fold[f]["base"] += bt
        per_fold[f]["oracle"] += ot
        if action != 0:
            per_fold[f]["actions"][action] += 1

        # Template grouping
        sig = order_signature(orders[oi])
        template_gains[sig]["count"] += 1
        template_gains[sig]["gain"] += gain

    print("=" * 60)
    print("ORACLE DECOMPOSITION")
    print("=" * 60)
    print(f"Total base: {total_base}  Oracle: {total_oracle}  Gain: {total_oracle - total_base:+d}")

    print(f"\nAction distribution:")
    for a_id in sorted(action_counts):
        name = action_names[a_id]
        print(f"  {name}: {action_counts[a_id]} orders, total gain={action_gains[a_id]:+d}")

    print(f"\nPer-fold:")
    for f in sorted(per_fold):
        pf = per_fold[f]
        print(f"  Fold {f}: base={pf['base']} oracle={pf['oracle']} gain={pf['oracle']-pf['base']:+d}")
        print(f"    Actions: {dict(pf['actions'])}")

    # Rank interval analysis: where are the FN that get fixed?
    print(f"\nFN rank distribution (V11 score rank among order alarms):")
    fn_ranks = []
    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base_mask[sl]
        ov = train_v11[sl]
        _, bt, ot, add, rm = oracle_per_order(ol, ob, ov)
        if ot > bt:
            for a_idx in add:
                rank = int(np.sum(ov >= ov[a_idx]))
                fn_ranks.append(rank)

    if fn_ranks:
        fn_ranks = np.array(fn_ranks)
        for lo, hi, label in [(1, 3, "top-3"), (4, 8, "rank 4-8"), (9, 20, "rank 9-20"), (21, 99, "rank 21+")]:
            count = np.sum((fn_ranks >= lo) & (fn_ranks <= hi))
            print(f"  {label}: {count} FN ({count/len(fn_ranks):.1%})")

    # Template analysis
    high_gain_templates = [(k, v) for k, v in template_gains.items() if v["gain"] >= 2]
    high_gain_templates.sort(key=lambda x: -x[1]["gain"])
    print(f"\nHigh-gain templates (gain>=2): {len(high_gain_templates)}")
    for sig, stats in high_gain_templates[:10]:
        print(f"  gain={stats['gain']:+d} count={stats['count']}")

    return {
        "base_tp": total_base, "oracle_tp": total_oracle, "gain": total_oracle - total_base,
        "action_counts": {action_names[k]: v for k, v in action_counts.items()},
        "action_gains": {action_names[k]: v for k, v in action_gains.items()},
        "per_fold": {str(k): v for k, v in per_fold.items()},
    }


if __name__ == "__main__":
    data = load_data()
    result = decompose_oracle(data)
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "v26_oracle_decompose.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_dir / 'v26_oracle_decompose.json'}")

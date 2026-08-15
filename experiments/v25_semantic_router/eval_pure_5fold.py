"""5-fold CV OOF evaluation of pure statistical approach."""
import sys, json, hashlib
import numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10


def load_data():
    """Load train orders and build labels/slices manually (no v10.prepare)."""
    train = v10.load_orders(Path("D:/zgyidong/train"), True)
    labels = []
    slices = []
    offset = 0
    for order in train:
        n = len(order["alarms"])
        for alarm in order["alarms"]:
            labels.append(int(alarm["@rid"] in order["roots"]))
        slices.append(slice(offset, offset + n))
        offset += n
    labels = np.array(labels, dtype=np.int8)
    return train, labels, slices


def grouped_folds(orders):
    groups = defaultdict(list)
    for i, o in enumerate(orders):
        tc = Counter(v10.scalar(n.get("title")) for n in o["alarms"])
        tt = sorted(
            v10.scalar(n.get("title"))
            for n in o["alarms"]
            if n.get("label") == "TargetAlarm"
        )
        sig = (tuple(sorted(tc.items())), tuple(tt), len(o["alarms"]))
        groups[sig].append(i)
    fs = [0] * 5
    folds = np.zeros(len(orders), dtype=np.int8)
    ranked = sorted(
        groups.items(),
        key=lambda x: (-len(x[1]), hashlib.sha256(repr(x[0]).encode()).hexdigest()),
    )
    for _, idx in ranked:
        f = min(range(5), key=lambda v: (fs[v], v))
        folds[idx] = f
        fs[f] += len(idx)
    return folds


def graph_distance_prior(order):
    nodes = order["topology"].get("nodes", [])
    alarms = order["alarms"]
    rid_to_idx = {n.get("@rid"): i for i, n in enumerate(nodes)}
    adj = [set() for _ in nodes]
    for e in order["topology"].get("edges", []):
        s, t = rid_to_idx.get(e.get("in")), rid_to_idx.get(e.get("out"))
        if s is not None and t is not None:
            adj[s].add(t)
            adj[t].add(s)
    targets = [
        rid_to_idx.get(a.get("@rid"))
        for a in alarms
        if a.get("label") == "TargetAlarm"
        and rid_to_idx.get(a.get("@rid")) is not None
    ]
    if not targets:
        return np.full(len(alarms), 0.5)
    dist = np.full(len(nodes), len(nodes) + 1, dtype=np.int32)
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)
    while q:
        cur = q.popleft()
        nd = dist[cur] + 1
        for nb in adj[cur]:
            if nd < dist[nb]:
                dist[nb] = nd
                q.append(nb)
    return np.array([
        1.0 / (1.0 + dist[rid_to_idx.get(a.get("@rid"), -1)])
        if rid_to_idx.get(a.get("@rid")) is not None
        else 0.2
        for a in alarms
    ])


def score_oof(test_orders, train_orders):
    pos = Counter()
    tot = Counter()
    cooc = defaultdict(lambda: [0, 0])
    for o in train_orders:
        ts = [str(a.get("title", "")) for a in o["alarms"]]
        rts = {
            str(a.get("title", ""))
            for a in o["alarms"]
            if a["@rid"] in o["roots"]
        }
        for t in ts:
            tot[t] += 1
        for t in rts:
            pos[t] += 1
        for at in ts:
            for bt in ts:
                cooc[(at, bt)][1] += 1
                if at in rts:
                    cooc[(at, bt)][0] += 1

    gm = sum(pos.values()) / max(sum(tot.values()), 1)
    cgm = sum(v[0] for v in cooc.values()) / max(sum(v[1] for v in cooc.values()), 1)

    # Template library
    tlib = defaultdict(Counter)
    for o in train_orders:
        ts = frozenset(str(a.get("title", "")) for a in o["alarms"])
        for a in o["alarms"]:
            if a["@rid"] in o["roots"]:
                tlib[ts][str(a.get("title", ""))] += 1
    tlib_hc = {
        k: v
        for k, v in tlib.items()
        if sum(v.values()) >= 3 and max(v.values()) / sum(v.values()) >= 0.90
    }

    all_scores = []
    for order in test_orders:
        alarms = order["alarms"]
        n = len(alarms)
        ts_list = [str(a.get("title", "")) for a in alarms]
        ts_set = frozenset(ts_list)

        # Template
        tp = np.full(n, gm)
        if ts_set in tlib_hc:
            tc = tlib_hc[ts_set]
            tt = sum(tc.values())
            if tt > 0:
                for i, t in enumerate(ts_list):
                    tp[i] = tc.get(t, 0) / tt

        # Title
        ti = np.array([(pos.get(t, 0) + gm) / (tot.get(t, 0) + 1) for t in ts_list])

        # Cooc
        co = np.zeros(n)
        for i, at in enumerate(ts_list):
            vals = [
                (cooc.get((at, bt), [0, 0])[0] + cgm)
                / (cooc.get((at, bt), [0, 0])[1] + 1)
                for j, bt in enumerate(ts_list)
                if i != j
            ]
            co[i] = np.mean(vals) if vals else cgm

        # Graph
        gr = graph_distance_prior(order)

        final = 0.35 * tp + 0.15 * ti + 0.25 * co + 0.25 * gr
        all_scores.append(final)

    return np.concatenate(all_scores).astype(np.float32)


def main():
    train_orders, labels, slices = load_data()
    folds = grouped_folds(train_orders)
    all_oi = np.arange(len(train_orders))
    target_train = round(1059 / 546 * 1634)

    # V11 baseline
    v11 = (
        0.25 * np.load("D:/zgyidong/codexgz/v11/v11_oof_context.npy")
        + 0.75 * np.load("D:/zgyidong/codexgz/v11/v11_oof_meta.npy")
    )
    base_mask = v10.exact_count_mask(v11, slices, target_train, 8)
    base_tp = int(np.sum(base_mask & (labels == 1)))
    base_f1 = (
        2
        * base_tp
        / (
            2 * base_tp
            + int(np.sum(base_mask & (labels == 0)))
            + int(np.sum((~base_mask) & (labels == 1)))
        )
    )
    print(f"V11 Baseline: TP={base_tp} F1={base_f1:.6f}")

    # 5-fold OOF
    oof_scores = np.zeros(len(labels), dtype=np.float32)
    fold_deltas = []
    for fold in range(5):
        tr_oi = all_oi[folds != fold]
        val_oi = all_oi[folds == fold]
        tr_orders = [train_orders[i] for i in tr_oi]
        val_orders = [train_orders[i] for i in val_oi]

        val_scores = score_oof(val_orders, tr_orders)

        val_rows = []
        for oi in val_oi:
            sl = slices[oi]
            val_rows.extend(range(sl.start, sl.stop))

        local_slices = []
        offset = 0
        for oi in val_oi:
            sl = slices[oi]
            n = sl.stop - sl.start
            local_slices.append(slice(offset, offset + n))
            offset += n

        oof_scores[np.array(val_rows)] = val_scores

        fold_target = round(target_train * len(val_rows) / len(labels))
        fold_mask = v10.exact_count_mask(val_scores, local_slices, fold_target, 8)
        fold_base = base_mask[val_rows]
        fold_labels = labels[val_rows]
        fold_btp = int(np.sum(fold_base & (fold_labels == 1)))
        fold_ntp = int(np.sum(fold_mask & (fold_labels == 1)))
        delta = fold_ntp - fold_btp
        fold_deltas.append(delta)
        print(f"  Fold {fold}: base={fold_btp} new={fold_ntp} delta={delta:+d}")

    total_mask = v10.exact_count_mask(oof_scores, slices, target_train, 8)
    total_tp = int(np.sum(total_mask & (labels == 1)))
    total_delta = total_tp - base_tp
    print(f"\nPure Statistical 5-fold OOF: TP={total_tp} delta={total_delta:+d}")
    print(f"Fold deltas: {fold_deltas}")
    print(f"Improved folds: {sum(d > 0 for d in fold_deltas)}/5")


if __name__ == "__main__":
    main()

"""
V26 aggressive — 用混合排序信号做候选选择
V11 scores × 0.6 + graph_proximity × 0.4 作为 add/remove 准则
"""

import sys, json, csv, hashlib, numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

dataset = v25_data.load_all_data("D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11")
train_orders = dataset["train_orders"]
test_orders = dataset["test_orders"]
test_slices = dataset["data"]["test_slices"]
test_v11 = dataset["test_v11"]

MAX_ROOT = 8


def graph_scores(order):
    """Score each alarm by proximity to TargetAlarm in the topology graph."""
    nodes = order["topology"].get("nodes", [])
    alarms = order["alarms"]
    rid2 = {n.get("@rid"): i for i, n in enumerate(nodes)}

    adj = [set() for _ in nodes]
    for e in order["topology"].get("edges", []):
        s, t = rid2.get(e.get("in")), rid2.get(e.get("out"))
        if s is not None and t is not None:
            adj[s].add(t); adj[t].add(s)

    targets = [rid2.get(a.get("@rid")) for a in alarms
               if a.get("label") == "TargetAlarm" and rid2.get(a.get("@rid")) is not None]

    if not targets:
        return np.full(len(alarms), 0.5)

    dist = np.full(len(nodes), len(nodes) + 1, dtype=np.int32)
    q = deque()
    for t in targets: dist[t] = 0; q.append(t)
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if dist[cur] + 1 < dist[nb]:
                dist[nb] = dist[cur] + 1; q.append(nb)

    return np.array([1.0 / (1.0 + dist[rid2.get(a.get("@rid"), -1)])
                     if rid2.get(a.get("@rid")) is not None else 0.2
                     for a in alarms])


# Pre-compute graph scores for all test orders
print("Computing graph scores...")
all_graph_scores = []
for order in test_orders:
    all_graph_scores.append(graph_scores(order))


# V26 trigger (from earlier run)
# Use the binary probabilities we computed
from sklearn.ensemble import ExtraTreesClassifier
import sys
sys.path.insert(0, "D:/zgyidong/codexgz/work")
labels = dataset["labels"]
slices = dataset["data"]["train_slices"]
train_v11 = dataset["train_v11"]

def oracle_labels(orders, lbls, sls, v11_s):
    ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (2, 2)}
    y_bin = np.zeros(len(orders), dtype=np.int32)
    y_act = np.zeros(len(orders), dtype=np.int32)
    base = v10.exact_count_mask(v11_s, sls, round(1059 / 546 * len(orders)), 8)
    for oi in range(len(orders)):
        sl = sls[oi]; ol = lbls[sl]; ob = base[sl]; ov = v11_s[sl]
        bt = int(np.sum(ob & (ol == 1))); bc = int(np.sum(ob))
        fn = np.where((ob == 0) & (ol == 1))[0]; fp = np.where(ob & (ol == 0))[0]
        if len(fn) == 0 and len(fp) == 0: continue
        fn_s = fn[np.argsort(-ov[fn])]; fp_s = fp[np.argsort(ov[fp])]
        ba, bg = 0, 0
        for a_id, (na, nr) in ACTION_MAP.items():
            if na > len(fn) or nr > len(fp): continue
            if bc + na - nr < 1 or bc + na - nr > 8: continue
            nm = ob.copy()
            if na: nm[fn_s[:na]] = True
            if nr: nm[fp_s[:nr]] = False
            g = int(np.sum(nm & (ol == 1))) - bt
            if g > bg: bg = g; ba = a_id
        if bg > 0: y_bin[oi] = 1; y_act[oi] = ba
    return y_bin, y_act

print("Training trigger model...")
# Simplified features: just V11 boundary gap + graph stats per order
def quick_features(orders, v11_s, sls):
    n = len(orders)
    feats = np.zeros((n, 8), dtype=np.float32)
    for oi in range(n):
        sl = sls[oi]; ov = v11_s[sl]
        ob = v10.exact_count_mask(v11_s, sls, round(1059/546*n), 8)[sl]
        sel = np.where(ob)[0]; unsel = np.where(~ob)[0]
        ss = ov[sel] if len(sel) > 0 else np.array([0.5])
        us = ov[unsel] if len(unsel) > 0 else np.array([0.])
        gap = float(np.min(ss)) - float(np.max(us)) if len(us) > 0 else 1.0
        feats[oi] = [gap,
                     float(np.mean(ov)), float(np.std(ov)), float(np.max(ov)),
                     float(len(sel)), float(len(unsel)),
                     len(sel)/max(len(ov),1),
                     float(len(orders[oi]["alarms"]))]
    return feats

X_tr = quick_features(train_orders, train_v11, slices)
y_bin, _ = oracle_labels(train_orders, labels, slices, train_v11)

sw = np.ones(len(y_bin)); pos = y_bin == 1
if np.sum(pos) > 0: sw[pos] = np.sum(~pos) / np.sum(pos)

SEEDS = [20260801, 20260817, 20260831]
test_bin = np.zeros(len(test_orders))
for seed in SEEDS:
    et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=8, max_features=0.7, n_jobs=-1, random_state=seed)
    et.fit(X_tr, y_bin, sample_weight=sw)
    test_bin += et.predict_proba(quick_features(test_orders, test_v11, test_slices))[:, 1]
test_bin /= len(SEEDS)

# Now apply actions with MIXED scoring (V11 + graph)
print("\nThreshold sweep with mixed scoring:")
test_base = v10.exact_count_mask(test_v11, test_slices, 1059, MAX_ROOT)

for THRESH in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    trig = test_bin >= THRESH

    new_mask = test_base.copy()
    for oi in np.where(trig)[0]:
        sl = test_slices[oi]
        ov = test_v11[sl]
        gs = all_graph_scores[oi]
        # Mixed score
        ms = 0.6 * ov + 0.4 * gs
        ob = new_mask[sl]
        n = len(ov)

        # Simple heuristic: if boundary is tight, swap worst selected for best unselected
        sel = np.where(ob)[0]
        unsel = np.where(~ob)[0]

        if len(sel) == 0 or len(unsel) == 0:
            continue

        # Swap: remove lowest mixed-score selected, add highest mixed-score unselected
        worst_sel = sel[np.argmin(ms[sel])]
        best_unsel = unsel[np.argmax(ms[unsel])]

        # Only swap if best_unsel has higher mixed score than worst_sel
        if ms[best_unsel] > ms[worst_sel]:
            ob[worst_sel] = False
            ob[best_unsel] = True

    # Fix min 1 + exact 1059
    for oi, sl in enumerate(test_slices):
        if np.sum(new_mask[sl]) == 0:
            new_mask[sl.start + int(np.argmax(test_v11[sl]))] = True

    cur = int(np.sum(new_mask))
    if cur < 1059:
        add_list = []
        for oi, sl in enumerate(test_slices):
            ov = test_v11[sl]; ob = new_mask[sl]
            ranked = np.argsort(-ov, kind="stable")
            for c in ranked[~ob[ranked]][:2]: add_list.append((oi, c, ov[c]))
        add_list.sort(key=lambda x: -x[2])
        for oi, c, _ in add_list[:1059 - cur]: new_mask[test_slices[oi].start + c] = True
    elif cur > 1059:
        rm_list = []
        for oi, sl in enumerate(test_slices):
            ov = test_v11[sl]; ob = new_mask[sl]
            sel_idx = np.where(ob)[0]
            if len(sel_idx) > 1:
                for si in sel_idx: rm_list.append((oi, si, ov[si]))
        rm_list.sort(key=lambda x: x[2])
        for oi, si, _ in rm_list[:cur - 1059]: new_mask[test_slices[oi].start + si] = False

    # Count changes
    diff = 0
    total_mod = 0  # total nodes added/removed
    for oi, sl in enumerate(test_slices):
        vp = set(np.where(test_base[sl])[0])
        vm = set(np.where(new_mask[sl])[0])
        if vp != vm:
            diff += 1
            total_mod += len(vp ^ vm)

    print(f"  thresh={THRESH:.2f}: trig={int(np.sum(trig))} changed_orders={diff} "
          f"node_changes={total_mod} total={int(np.sum(new_mask))}")

    # Generate CSV at thresh=0.25
    if abs(THRESH - 0.25) < 0.01:
        out_path = Path("submissions/v26_aggressive_mixed_score_p1059.csv")
        out_path.parent.mkdir(exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["order_id", "output"])
            for oi, order in enumerate(test_orders):
                sl = test_slices[oi]
                rcs = []
                for local_idx in np.flatnonzero(new_mask[sl]):
                    node = order["alarms"][int(local_idx)]
                    rcs.append({"@rid": node["@rid"], "title": node.get("title", ""),
                                "location": node.get("location", ""), "reason": node.get("reason", "")})
                writer.writerow([order["id"], json.dumps({"rootcause": rcs}, ensure_ascii=False)])

        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"\n  Generated: {out_path}")
        print(f"  SHA256: {sha}")
        print(f"  Orders changed: {diff}  Node changes: {total_mod}")

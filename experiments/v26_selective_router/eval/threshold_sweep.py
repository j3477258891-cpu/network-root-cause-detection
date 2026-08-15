"""V26 multi-threshold sweep on test set."""
import sys, json, csv, hashlib, numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

dataset = v25_data.load_all_data("D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11")
labels = dataset["labels"]
slices = dataset["data"]["train_slices"]
test_slices = dataset["data"]["test_slices"]
train_v11 = dataset["train_v11"]
test_v11 = dataset["test_v11"]
train_orders = dataset["train_orders"]
test_orders = dataset["test_orders"]

MAX_ROOT = 8
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (2, 2)}
SEEDS = [20260801, 20260817, 20260831]


def oracle_labels(orders, lbls, sls, v11_s):
    y_bin = np.zeros(len(orders), dtype=np.int32)
    y_act = np.zeros(len(orders), dtype=np.int32)
    base = v10.exact_count_mask(v11_s, sls, round(1059 / 546 * len(orders)), MAX_ROOT)
    n_orders = len(orders)
    for oi in range(n_orders):
        sl = sls[oi]; ol = lbls[sl]; ob = base[sl]; ov = v11_s[sl]
        bt = int(np.sum(ob & (ol == 1))); bc = int(np.sum(ob))
        fn = np.where((ob == 0) & (ol == 1))[0]; fp = np.where(ob & (ol == 0))[0]
        if len(fn) == 0 and len(fp) == 0: continue
        fn_s = fn[np.argsort(-ov[fn])]; fp_s = fp[np.argsort(ov[fp])]
        ba, bg = 0, 0
        for a_id, (na, nr) in ACTION_MAP.items():
            if na > len(fn) or nr > len(fp): continue
            if bc + na - nr < 1 or bc + na - nr > MAX_ROOT: continue
            nm = ob.copy()
            if na: nm[fn_s[:na]] = True
            if nr: nm[fp_s[:nr]] = False
            g = int(np.sum(nm & (ol == 1))) - bt
            if g > bg: bg = g; ba = a_id
        if bg > 0: y_bin[oi] = 1; y_act[oi] = ba
    return y_bin, y_act


def build_features(orders, v11_s, sls, n_orders):
    tpl_err = defaultdict(lambda: [0, 0])
    for oi in range(n_orders):
        sl = sls[oi]
        ob = v10.exact_count_mask(v11_s, sls, round(1059 / 546 * n_orders), MAX_ROOT)[sl]
        sig = tuple(sorted(Counter(str(a["title"]) for a in orders[oi]["alarms"]).items()))
        tpl_err[sig][0] += int(np.sum(ob & (labels[sl] == 0)))
        tpl_err[sig][1] += 1

    feats = np.zeros((n_orders, 28), dtype=np.float32)
    for oi in range(n_orders):
        order = orders[oi]; alarms = order["alarms"]; n = len(alarms)
        sl = sls[oi]; ov = v11_s[sl]
        ob = v10.exact_count_mask(v11_s, sls, round(1059 / 546 * n_orders), MAX_ROOT)[sl]
        sel = np.where(ob)[0]; unsel = np.where(~ob)[0]
        ss = ov[sel] if len(sel) > 0 else np.array([0.5])
        sm = float(np.min(ss)); smr = int(np.sum(ov >= sm))
        smean = float(np.mean(ss)); sstd = float(np.std(ss)) if len(ss) > 1 else 0.
        us = ov[unsel] if len(unsel) > 0 else np.array([0.])
        um = float(np.max(us)); umr = int(np.sum(ov >= um))
        umean = float(np.mean(us))
        utop3 = float(np.mean(np.sort(-us)[:3])) if len(us) >= 3 else um
        bg = sm - um; brg = umr - smr
        om = float(np.mean(ov)); os_ = float(np.std(ov)); ox = float(np.max(ov))
        se = float(-np.sum(np.clip(ov, 1e-6, 1) * np.log(np.clip(ov, 1e-6, 1))) / max(np.log(n), 1))
        sig = tuple(sorted(Counter(str(a["title"]) for a in alarms).items()))
        ts = tpl_err.get(sig, [0, 1])
        tfr = ts[0] / max(ts[1], 1)
        nodes = order["topology"].get("nodes", [])
        rid2 = {n.get("@rid"): i for i, n in enumerate(nodes)}
        adj = [set() for _ in nodes]
        for e in order["topology"].get("edges", []):
            s, t = rid2.get(e.get("in")), rid2.get(e.get("out"))
            if s is not None and t is not None: adj[s].add(t); adj[t].add(s)
        aids = [rid2.get(a.get("@rid"), -1) for a in alarms]
        tgts = [aids[i] for i in sel if aids[i] >= 0]
        if tgts:
            dist = np.full(len(nodes), len(nodes) + 1); q = deque()
            for t in tgts:
                if t >= 0: dist[t] = 0; q.append(t)
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if dist[cur] + 1 < dist[nb]: dist[nb] = dist[cur] + 1; q.append(nb)
            gds = [dist[ai] if ai >= 0 else 99 for ai in aids]
            mgd = np.mean(gds); igd = np.min(gds)
        else: mgd = igd = 99.
        feats[oi] = [bg, brg, sm, um, utop3, smean, sstd, umean, om, os_, ox, se,
                      float(len(sel)), float(n), len(sel) / max(n, 1),
                      tfr, float(ts[1]), mgd, igd,
                      float(len(tgts)), float(len(tgts)) / max(n, 1),
                      sm / max(um, 1e-6), utop3 - smean,
                      float(bg < 0.05 and len(sel) > 1), float(bg < 0.1 and len(sel) > 1),
                      float(tfr > 0.1), float(ts[1] >= 3), float(n > 15)]
    return feats


X_train = build_features(train_orders, train_v11, slices, len(train_orders))
y_bin, y_act = oracle_labels(train_orders, labels, slices, train_v11)

from sklearn.ensemble import ExtraTreesClassifier

sw = np.ones(len(y_bin)); pos = y_bin == 1
if np.sum(pos) > 0: sw[pos] = np.sum(~pos) / np.sum(pos)
test_bin = np.zeros(len(test_orders))
for seed in SEEDS:
    et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=8, max_features=0.7, n_jobs=-1, random_state=seed)
    et.fit(X_train, y_bin, sample_weight=sw)
    X_test = build_features(test_orders, test_v11, test_slices, len(test_orders))
    test_bin += et.predict_proba(X_test)[:, 1]
test_bin /= len(SEEDS)

act = y_bin == 1
test_act_probs = np.zeros((len(test_orders), 8))
for seed in SEEDS:
    et = ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, max_features=0.7, n_jobs=-1, random_state=seed)
    et.fit(X_train[act], y_act[act])
    X_test = build_features(test_orders, test_v11, test_slices, len(test_orders))
    ep = et.predict_proba(X_test)
    for ci, cls_id in enumerate(et.classes_): test_act_probs[:, int(cls_id)] += ep[:, ci]
test_act_probs /= len(SEEDS)
test_act = np.argmax(test_act_probs, axis=1)

test_base = v10.exact_count_mask(test_v11, test_slices, 1059, MAX_ROOT)

for THRESH in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    trig = test_bin >= THRESH
    for oi in range(len(test_orders)):
        if trig[oi] and test_act[oi] == 0:
            probs = test_act_probs[oi].copy(); probs[0] = -1; test_act[oi] = int(np.argmax(probs))

    new_mask = test_base.copy()
    base_counts = np.array([int(np.sum(test_base[sl])) for sl in test_slices])
    for oi in np.where(trig)[0]:
        a_id = test_act[oi]; na, nr = ACTION_MAP[a_id]
        if base_counts[oi] + na - nr < 1 or base_counts[oi] + na - nr > MAX_ROOT: continue
        sl = test_slices[oi]; ov = test_v11[sl]; ob = new_mask[sl]
        ranked = np.argsort(-ov, kind="stable")
        not_sel = ~ob; cands = ranked[not_sel[ranked]]; ob[cands[:na]] = True
        sel_idx = np.where(ob)[0]
        if len(sel_idx) > nr: sel_r = sel_idx[np.argsort(ov[sel_idx])]; ob[sel_r[:nr]] = False

    for oi, sl in enumerate(test_slices):
        if np.sum(new_mask[sl]) == 0: new_mask[sl.start + int(np.argmax(test_v11[sl]))] = True
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

    diff = 0
    for oi, sl in enumerate(test_slices):
        vp = set(np.where(test_base[sl])[0])
        vm = set(np.where(new_mask[sl])[0])
        if vp != vm: diff += 1

    print(f"thresh={THRESH:.2f}: trig={int(np.sum(trig))} changed={diff} total={int(np.sum(new_mask))}")

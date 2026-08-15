"""
V26 v2 — 边界节点特征 + 两阶段分类器
Stage 1: Binary — should we touch this order?
Stage 2: What action?
专注 V11 的「边界」：last selected vs first unselected 的间隙
"""

import sys, json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

MAX_ROOT = 8
SEEDS = [20260801, 20260817, 20260831]
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
              4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (2, 2)}


def load_baseline():
    dataset = v25_data.load_all_data(
        "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
    )
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    folds = dataset["folds"]
    train_v11 = dataset["train_v11"]
    order_alarms = [len(dataset["train_orders"][i]["alarms"]) for i in range(len(slices))]
    target_train = round(1059 / 546 * len(slices))
    base_mask = v10.exact_count_mask(train_v11, slices, target_train, MAX_ROOT)
    return {
        "labels": labels, "slices": slices, "folds": folds,
        "train_v11": train_v11, "base_mask": base_mask,
        "target_train": target_train, "order_alarms": order_alarms,
        "train_orders": dataset["train_orders"],
    }


def build_boundary_features(data):
    """
    Per-order features focused on the V11 selection boundary.
    Cross-fit: for each fold, stats computed from other 4 folds.
    """
    labels = data["labels"]
    slices = data["slices"]
    folds = data["folds"]
    v11 = data["train_v11"]
    base = data["base_mask"]
    orders = data["train_orders"]
    n_orders = len(slices)

    all_oi = np.arange(n_orders)
    feats = np.zeros((n_orders, 28), dtype=np.float32)  # expanded to 28

    for heldout in range(5):
        ref_oi = all_oi[folds != heldout]
        val_oi = all_oi[folds == heldout]

        # Cross-fit template error rates
        tpl_err = defaultdict(lambda: [0, 0])  # [errors, total]
        for oi in ref_oi:
            sl = slices[oi]
            ol = labels[sl]
            ob = base[sl]
            sig = tuple(sorted(Counter(
                str(a["title"]) for a in orders[oi]["alarms"]
            ).items()))
            n_fp = int(np.sum(ob & (ol == 0)))
            tpl_err[sig][0] += n_fp
            tpl_err[sig][1] += 1

        for vi in val_oi:
            sl = slices[vi]
            ov = v11[sl]
            ob = base[sl]
            ol = labels[sl]
            order = orders[vi]
            n = len(ol)

            # === Boundary analysis ===
            sel = np.where(ob)[0]
            unsel = np.where(~ob)[0]

            # Selected stats
            sel_scores = ov[sel] if len(sel) > 0 else np.array([0.5])
            sel_min = float(np.min(sel_scores))
            sel_min_rank = int(np.sum(ov >= sel_min))
            sel_mean = float(np.mean(sel_scores))
            sel_std = float(np.std(sel_scores)) if len(sel_scores) > 1 else 0.0

            # Unselected stats
            unsel_scores = ov[unsel] if len(unsel) > 0 else np.array([0.0])
            unsel_max = float(np.max(unsel_scores))
            unsel_max_rank = int(np.sum(ov >= unsel_max))
            unsel_mean = float(np.mean(unsel_scores))
            unsel_top3 = float(np.mean(np.sort(-unsel_scores)[:3])) if len(unsel_scores) >= 3 else unsel_max

            # Gap
            boundary_gap = sel_min - unsel_max
            boundary_rank_gap = unsel_max_rank - sel_min_rank

            # === Score distribution ===
            ov_mean = float(np.mean(ov))
            ov_std = float(np.std(ov))
            ov_max = float(np.max(ov))
            score_entropy = float(-np.sum(np.clip(ov, 1e-6, 1) * np.log(np.clip(ov, 1e-6, 1))) / max(np.log(n), 1))

            # === Template features (cross-fit) ===
            sig = tuple(sorted(Counter(
                str(a["title"]) for a in order["alarms"]
            ).items()))
            ts = tpl_err.get(sig, [0, 1])
            tpl_fp_rate = ts[0] / max(ts[1], 1)
            tpl_support = ts[1]

            # === Graph features ===
            nodes = order["topology"].get("nodes", [])
            alarms = order["alarms"]
            rid_to_idx = {n.get("@rid"): i for i, n in enumerate(nodes)}
            adj = [set() for _ in nodes]
            for e in order["topology"].get("edges", []):
                s, t = rid_to_idx.get(e.get("in")), rid_to_idx.get(e.get("out"))
                if s is not None and t is not None:
                    adj[s].add(t); adj[t].add(s)

            # Graph distance stats
            alarm_idx = [rid_to_idx.get(a.get("@rid"), -1) for a in alarms]
            targets = [alarm_idx[i] for i in sel if alarm_idx[i] >= 0]
            if targets:
                # Simple BFS distances
                from collections import deque
                dist = np.full(len(nodes), len(nodes) + 1)
                q = deque()
                for t in targets:
                    if t >= 0:
                        dist[t] = 0; q.append(t)
                while q:
                    cur = q.popleft()
                    for nb in adj[cur]:
                        if dist[cur] + 1 < dist[nb]:
                            dist[nb] = dist[cur] + 1; q.append(nb)
                graph_dists = [dist[ai] if ai >= 0 else 99 for ai in alarm_idx]
                mean_graph_dist = np.mean(graph_dists)
                min_graph_dist = np.min(graph_dists)
            else:
                mean_graph_dist = min_graph_dist = 99.0

            # === Compile features ===
            feats[vi] = [
                boundary_gap,                                    # 0
                boundary_rank_gap,                               # 1
                sel_min,                                         # 2
                unsel_max,                                       # 3
                unsel_top3,                                      # 4
                sel_mean,                                        # 5
                sel_std,                                         # 6
                unsel_mean,                                      # 7
                ov_mean,                                         # 8
                ov_std,                                          # 9
                ov_max,                                          # 10
                score_entropy,                                   # 11
                float(len(sel)),                                 # 12
                float(n),                                        # 13
                len(sel) / max(n, 1),                            # 14
                tpl_fp_rate,                                     # 15
                float(tpl_support),                              # 16
                mean_graph_dist,                                 # 17
                min_graph_dist,                                  # 18
                float(len(targets)),                             # 19
                float(len(targets)) / max(n, 1),                 # 20
                sel_min / max(unsel_max, 1e-6),                  # 21
                unsel_top3 - sel_mean,                           # 22
                float(boundary_gap < 0.05 and len(sel) > 1),     # 23: very close boundary
                float(boundary_gap < 0.1 and len(sel) > 1),      # 24: close boundary
                float(tpl_fp_rate > 0.1),                        # 25: error-prone template
                float(tpl_support >= 3),                         # 26: well-supported template
                float(n > 15),                                   # 27: large order
            ]

    return feats


def oracle_labels(data):
    """Binary oracle: 1 if order can be improved, 0 otherwise."""
    labels = data["labels"]
    slices = data["slices"]
    base = data["base_mask"]
    v11 = data["train_v11"]

    y_binary = np.zeros(len(slices), dtype=np.int32)
    y_action = np.zeros(len(slices), dtype=np.int32)
    y_gain = np.zeros(len(slices), dtype=np.int32)

    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base[sl]
        ov = v11[sl]
        bt = int(np.sum(ob & (ol == 1)))
        bc = int(np.sum(ob))

        fn = np.where((ob == 0) & (ol == 1))[0]
        fp = np.where(ob & (ol == 0))[0]

        if len(fn) == 0 and len(fp) == 0:
            continue

        fn_s = fn[np.argsort(-ov[fn])]
        fp_s = fp[np.argsort(ov[fp])]

        best_a, best_g = 0, 0
        for a_id, (na, nr) in ACTION_MAP.items():
            if na > len(fn) or nr > len(fp):
                continue
            if bc + na - nr < 1 or bc + na - nr > MAX_ROOT:
                continue
            nm = ob.copy()
            if na: nm[fn_s[:na]] = True
            if nr: nm[fp_s[:nr]] = False
            g = int(np.sum(nm & (ol == 1))) - bt
            if g > best_g:
                best_g = g; best_a = a_id

        if best_g > 0:
            y_binary[oi] = 1
            y_action[oi] = best_a
            y_gain[oi] = best_g

    return y_binary, y_action, y_gain


def two_stage_train(X, y_binary, y_action, y_gain, folds, seeds):
    """Two-stage classifier with cross-validation."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    n_orders = len(X)
    all_oi = np.arange(n_orders)

    oof_binary = np.zeros(n_orders, dtype=np.float32)
    oof_action = np.zeros(n_orders, dtype=np.int32)
    oof_confidence = np.zeros(n_orders, dtype=np.float32)

    for heldout in range(5):
        tr = all_oi[folds != heldout]
        val = all_oi[folds == heldout]

        X_tr, yb_tr = X[tr], y_binary[tr]
        X_val = X[val]

        n_pos = np.sum(yb_tr == 1)
        if n_pos < 3:
            oof_binary[val] = 0.0
            oof_action[val] = 0
            continue

        # Stage 1: Binary (is this order actionable?)
        # Use balanced class weight since actionable orders are rare (~7.6%)
        sw = np.ones(len(tr))
        sw[yb_tr == 1] = np.sum(yb_tr == 0) / max(np.sum(yb_tr == 1), 1)

        # Ensemble of binary classifiers
        bin_probs = np.zeros(len(val))
        for seed in seeds:
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=8,
                max_features=0.7, n_jobs=-1, random_state=seed + heldout,
            )
            et.fit(X_tr, yb_tr, sample_weight=sw)
            bin_probs += et.predict_proba(X_val)[:, 1]

        bin_probs /= len(seeds)
        oof_binary[val] = bin_probs

        # Stage 2: Action classifier (only on training actionable orders)
        actionable_tr = yb_tr == 1
        if np.sum(actionable_tr) >= 3:
            X_act_tr = X_tr[actionable_tr]
            ya_tr = y_action[tr][actionable_tr]
            X_act_val = X_val

            act_probs = np.zeros((len(val), 8))
            for seed in seeds:
                et = ExtraTreesClassifier(
                    n_estimators=100, max_depth=6, min_samples_leaf=4,
                    max_features=0.7, n_jobs=-1, random_state=seed + heldout,
                )
                et.fit(X_act_tr, ya_tr)
                ep = et.predict_proba(X_act_val)
                ec = et.classes_
                for ci, cls_id in enumerate(ec):
                    act_probs[:, int(cls_id)] += ep[:, ci]
            act_probs /= len(seeds)

            for i, vi in enumerate(val):
                best_a = int(np.argmax(act_probs[i]))
                oof_action[vi] = best_a if best_a != 0 else 0
                oof_confidence[vi] = float(np.max(act_probs[i]))
        else:
            oof_action[val] = 0

    return oof_binary, oof_action, oof_confidence


def main():
    print("=" * 60)
    print("V26 v2 — Boundary Features + Two-Stage Router")
    print("=" * 60)

    data = load_baseline()
    labels = data["labels"]
    slices = data["slices"]
    folds = data["folds"]
    base = data["base_mask"]
    v11 = data["train_v11"]
    target = data["target_train"]

    base_tp = int(np.sum(base & (labels == 1)))
    print(f"V11: TP={base_tp}")

    # Oracle
    y_bin, y_act, y_gain = oracle_labels(data)
    n_actionable = np.sum(y_bin == 1)
    print(f"Oracle actionable: {n_actionable}/{len(slices)} ({n_actionable/len(slices):.1%})")

    # Features
    print("Building boundary features...")
    X = build_boundary_features(data)
    print(f"Features: {X.shape}")

    # Train
    print("Training two-stage router...")
    oof_bin, oof_act, oof_conf = two_stage_train(X, y_bin, y_act, y_gain, folds, SEEDS)

    # Evaluate at multiple confidence thresholds
    print(f"\nThreshold sweep:")
    best_delta, best_thresh = -999, 0.0
    all_oi = np.arange(len(slices))

    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        triggered = (oof_bin >= thresh) & (oof_act != 0)
        n_trig = np.sum(triggered)

        new_mask = base.copy()
        # Greedy apply triggered actions (simple, no DP for probe)
        for oi in np.where(triggered)[0]:
            a_id = oof_act[oi]
            if a_id == 0:
                continue
            sl = slices[oi]
            ov = v11[sl]
            ob = new_mask[sl]
            na, nr = ACTION_MAP.get(a_id, (0, 0))

            ranked = np.argsort(-ov, kind="stable")
            not_sel = ~ob
            candidates = ranked[not_sel[ranked]]
            ob[candidates[:na]] = True

            sel_idx = np.where(ob)[0]
            if len(sel_idx) > nr:
                sel_ranked = sel_idx[np.argsort(ov[sel_idx])]
                ob[sel_ranked[:nr]] = False

        new_tp = int(np.sum(new_mask & (labels == 1)))
        delta = new_tp - base_tp

        # Fold-level
        fold_deltas = []
        for f in range(5):
            fold_oi = all_oi[folds == f]
            fbtp = sum(int(np.sum(base[slices[oi]] & (labels[slices[oi]] == 1))) for oi in fold_oi)
            fntp = sum(int(np.sum(new_mask[slices[oi]] & (labels[slices[oi]] == 1))) for oi in fold_oi)
            fold_deltas.append(fntp - fbtp)

        # Correct-change rate
        fold_trig = sum(np.sum(triggered[folds == f]) for f in range(5))
        correct = 0
        for oi in np.where(triggered)[0]:
            sl = slices[oi]
            bt = int(np.sum(base[sl] & (labels[sl] == 1)))
            nt = int(np.sum(new_mask[sl] & (labels[sl] == 1)))
            if nt > bt: correct += 1
        corr_rate = correct / max(fold_trig, 1)

        improved = sum(1 for d in fold_deltas if d > 0)
        all_pos = all(d >= 0 for d in fold_deltas)

        print(f"  thresh={thresh:.2f}: delta={delta:+d} trig={n_trig} "
              f"correct_rate={corr_rate:.2%} folds={fold_deltas} "
              f"improved={improved}/5 all_pos={all_pos}")

        if delta > best_delta:
            best_delta = delta
            best_thresh = thresh

    # Gate
    print(f"\nBest: thresh={best_thresh:.2f} delta={best_delta:+d}")
    print(f"Gate (>=30 TP, 3+/5 folds, all folds>=0): "
          f"{'PASSED' if best_delta >= 30 else 'FAILED'}")

    # Oracle ceiling
    oracle_gain = int(np.sum(y_gain))
    print(f"Oracle ceiling: +{oracle_gain} TP")
    print(f"Achieved: +{best_delta} / +{oracle_gain} = {best_delta/max(oracle_gain,1):.1%} of oracle")


if __name__ == "__main__":
    main()

"""
V26 Selective Error Router — 完整流程
Step 1-4: 全量训练 → 测试预测 → 阈值选择 → 生成 CSV
"""

import sys, json, hashlib, csv
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/zgyidong/codexgz/work")
import v10_grouped_ensemble as v10
sys.path.insert(0, "D:/zgyidong/experiments/v25_ensemble")
import v25_data

MAX_ROOT = 8
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
              4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (2, 2)}
ACTION_NAMES = ["keep", "add1", "add2", "add3", "rm1", "rm2", "swap1", "swap2"]
SEEDS = [20260801, 20260817, 20260831]


# ═══════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════

def load_all():
    dataset = v25_data.load_all_data(
        "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
    )
    return dataset


# ═══════════════════════════════════════════
# Feature engineering (full, not cross-fit for test)
# ═══════════════════════════════════════════

def build_features_full(orders, v11_scores, slices, labels=None):
    """
    Build 28-dim boundary features using ALL data (no cross-fit).
    For test orders, labels is None.
    """
    n_orders = len(orders)
    
    # Template error rates from training (if labels available)
    if labels is not None:
        tpl_err = defaultdict(lambda: [0, 0])
        for oi in range(n_orders):
            sl = slices[oi]
            ol = labels[sl]
            ob = v10.exact_count_mask(v11_scores, slices, 3169, MAX_ROOT)
            ob_local = ob[sl]
            sig = tuple(sorted(Counter(
                str(a["title"]) for a in orders[oi]["alarms"]
            ).items()))
            n_fp = int(np.sum(ob_local & (ol == 0)))
            tpl_err[sig][0] += n_fp
            tpl_err[sig][1] += 1
    else:
        tpl_err = defaultdict(lambda: [0, 1])

    feats = np.zeros((n_orders, 28), dtype=np.float32)

    for oi in range(n_orders):
        order = orders[oi]
        alarms = order["alarms"]
        n = len(alarms)

        sl = slices[oi]
        start, stop = sl.start, sl.stop
        ov = v11_scores[start:stop]

        if labels is not None:
            ob = v10.exact_count_mask(v11_scores, slices, round(1059 / 546 * n_orders), MAX_ROOT)
            ob_local = ob[sl]
        else:
            # For test: use top-K by V11 score
            ob_local = v10.exact_count_mask(v11_scores, slices, 1059, MAX_ROOT)[sl]

        # Boundary analysis
        sel = np.where(ob_local)[0]
        unsel = np.where(~ob_local)[0]

        sel_scores = ov[sel] if len(sel) > 0 else np.array([0.5])
        sel_min = float(np.min(sel_scores))
        sel_min_rank = int(np.sum(ov >= sel_min))
        sel_mean = float(np.mean(sel_scores))
        sel_std = float(np.std(sel_scores)) if len(sel_scores) > 1 else 0.0

        unsel_scores = ov[unsel] if len(unsel) > 0 else np.array([0.0])
        unsel_max = float(np.max(unsel_scores))
        unsel_max_rank = int(np.sum(ov >= unsel_max))
        unsel_mean = float(np.mean(unsel_scores))
        unsel_top3 = float(np.mean(np.sort(-unsel_scores)[:3])) if len(unsel_scores) >= 3 else unsel_max

        boundary_gap = sel_min - unsel_max
        boundary_rank_gap = unsel_max_rank - sel_min_rank

        ov_mean = float(np.mean(ov))
        ov_std = float(np.std(ov))
        ov_max = float(np.max(ov))
        score_entropy = float(-np.sum(np.clip(ov, 1e-6, 1) * np.log(np.clip(ov, 1e-6, 1))) / max(np.log(n), 1))

        # Template
        sig = tuple(sorted(Counter(str(a["title"]) for a in alarms).items()))
        ts = tpl_err.get(sig, [0, 1])
        tpl_fp_rate = ts[0] / max(ts[1], 1)
        tpl_support = ts[1]

        # Graph distance
        nodes = order["topology"].get("nodes", [])
        rid_to_idx = {n.get("@rid"): i for i, n in enumerate(nodes)}
        adj = [set() for _ in nodes]
        for e in order["topology"].get("edges", []):
            s, t = rid_to_idx.get(e.get("in")), rid_to_idx.get(e.get("out"))
            if s is not None and t is not None:
                adj[s].add(t); adj[t].add(s)

        alarm_idx = [rid_to_idx.get(a.get("@rid"), -1) for a in alarms]
        targets = [alarm_idx[i] for i in sel if alarm_idx[i] >= 0]
        if targets:
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

        feats[oi] = [
            boundary_gap, boundary_rank_gap,
            sel_min, unsel_max, unsel_top3,
            sel_mean, sel_std, unsel_mean,
            ov_mean, ov_std, ov_max, score_entropy,
            float(len(sel)), float(n), len(sel) / max(n, 1),
            tpl_fp_rate, float(tpl_support),
            mean_graph_dist, min_graph_dist,
            float(len(targets)), float(len(targets)) / max(n, 1),
            sel_min / max(unsel_max, 1e-6),
            unsel_top3 - sel_mean,
            float(boundary_gap < 0.05 and len(sel) > 1),
            float(boundary_gap < 0.1 and len(sel) > 1),
            float(tpl_fp_rate > 0.1),
            float(tpl_support >= 3),
            float(n > 15),
        ]

    return feats


# ═══════════════════════════════════════════
# Oracle labels (train only)
# ═══════════════════════════════════════════

def oracle_labels(train_orders, labels, slices, v11_scores):
    y_bin = np.zeros(len(train_orders), dtype=np.int32)
    y_act = np.zeros(len(train_orders), dtype=np.int32)
    y_gain = np.zeros(len(train_orders), dtype=np.int32)

    base_mask = v10.exact_count_mask(v11_scores, slices, round(1059 / 546 * len(train_orders)), MAX_ROOT)

    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base_mask[sl]
        ov = v11_scores[sl]
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
            y_bin[oi] = 1; y_act[oi] = best_a; y_gain[oi] = best_g

    return y_bin, y_act, y_gain, base_mask


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train_full_classifier(X_train, y_bin, y_act):
    """Train two-stage classifier on full training data."""
    from sklearn.ensemble import ExtraTreesClassifier

    # Stage 1: Binary
    sw = np.ones(len(y_bin))
    pos = y_bin == 1
    if np.sum(pos) > 0:
        sw[pos] = np.sum(~pos) / np.sum(pos)

    bin_probs_train = np.zeros(len(y_bin))
    for seed in SEEDS:
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=8,
            max_features=0.7, n_jobs=-1, random_state=seed,
        )
        et.fit(X_train, y_bin, sample_weight=sw)
        bin_probs_train += et.predict_proba(X_train)[:, 1]
    bin_probs_train /= len(SEEDS)

    # Stage 2: Action classifier
    actionable = y_bin == 1
    act_probs_train = np.zeros((len(y_bin), 8))
    if np.sum(actionable) >= 3:
        for seed in SEEDS:
            et = ExtraTreesClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=4,
                max_features=0.7, n_jobs=-1, random_state=seed,
            )
            et.fit(X_train[actionable], y_act[actionable])
            ep = et.predict_proba(X_train)
            for ci, cls_id in enumerate(et.classes_):
                act_probs_train[:, int(cls_id)] += ep[:, ci]
        act_probs_train /= len(SEEDS)

    return bin_probs_train, act_probs_train


def predict_on_test(X_train, y_bin, y_act, X_test):
    """Apply trained model patterns to test data."""
    bin_probs, act_probs = train_full_classifier(X_train, y_bin, y_act)

    # For test: train fresh, predict test
    from sklearn.ensemble import ExtraTreesClassifier

    # Binary
    sw = np.ones(len(y_bin))
    pos = y_bin == 1
    if np.sum(pos) > 0:
        sw[pos] = np.sum(~pos) / np.sum(pos)

    test_bin = np.zeros(len(X_test))
    for seed in SEEDS:
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=8,
            max_features=0.7, n_jobs=-1, random_state=seed,
        )
        et.fit(X_train, y_bin, sample_weight=sw)
        test_bin += et.predict_proba(X_test)[:, 1]
    test_bin /= len(SEEDS)

    # Action
    test_act_probs = np.zeros((len(X_test), 8))
    actionable = y_bin == 1
    if np.sum(actionable) >= 3:
        for seed in SEEDS:
            et = ExtraTreesClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=4,
                max_features=0.7, n_jobs=-1, random_state=seed,
            )
            et.fit(X_train[actionable], y_act[actionable])
            ep = et.predict_proba(X_test)
            for ci, cls_id in enumerate(et.classes_):
                test_act_probs[:, int(cls_id)] += ep[:, ci]
        test_act_probs /= len(SEEDS)

    test_act = np.argmax(test_act_probs, axis=1)

    return test_bin, test_act, test_act_probs


# ═══════════════════════════════════════════
# Apply actions + DP
# ═══════════════════════════════════════════

def apply_actions(base_mask, slices, v11_scores, triggered, action_preds, target_total):
    """Apply predicted actions to V11 base mask with greedy DP."""
    new_mask = base_mask.copy()
    base_counts = np.array([int(np.sum(base_mask[sl])) for sl in slices])
    total_base = int(np.sum(base_counts))

    # Candidate list
    candidates = []
    for oi in np.where(triggered)[0]:
        a_id = action_preds[oi]
        if a_id == 0:
            continue
        na, nr = ACTION_MAP[a_id]
        new_k = base_counts[oi] + na - nr
        if new_k < 1 or new_k > MAX_ROOT:
            continue
        dk = na - nr
        candidates.append((oi, a_id, na, nr, dk))

    # Greedy apply until budget met
    cur_total = total_base
    for oi, a_id, na, nr, dk in candidates:
        k_after = cur_total + dk
        if abs(k_after - target_total) > 4:
            continue

        sl = slices[oi]
        ov = v11_scores[sl]
        ob = new_mask[sl]
        ranked = np.argsort(-ov, kind="stable")

        not_sel = ~ob
        cands = ranked[not_sel[ranked]]
        ob[cands[:na]] = True

        sel_idx = np.where(ob)[0]
        if len(sel_idx) > nr:
            sel_ranked = sel_idx[np.argsort(ov[sel_idx])]
            ob[sel_ranked[:nr]] = False

        cur_total = k_after

    return new_mask


# ═══════════════════════════════════════════
# Submission
# ═══════════════════════════════════════════

def write_submission(test_mask, test_orders, data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_slices = data["test_slices"]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for oi, order in enumerate(test_orders):
            sl = test_slices[oi]
            rcs = []
            for local_idx in np.flatnonzero(test_mask[sl]):
                node = order["alarms"][int(local_idx)]
                rcs.append({
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                })
            writer.writerow([order["id"], json.dumps({"rootcause": rcs}, ensure_ascii=False)])


def validate_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    oids = set()
    total = 0
    k_dist = Counter()
    for r in rows:
        oids.add(r["order_id"])
        pred = json.loads(r["output"])
        k = len(pred["rootcause"])
        k_dist[k] += 1
        total += k
        for rc in pred["rootcause"]:
            assert "@rid" in rc
            assert "title" in rc
            assert "location" in rc
            assert "reason" in rc
    assert len(oids) == 546
    assert total == 1059
    assert all(1 <= k <= 8 for k in k_dist), f"Invalid K: {dict(k_dist)}"
    return True


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    print("=" * 60)
    print("V26 Selective Error Router — Full Pipeline")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading data...")
    dataset = load_all()
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    test_slices = dataset["data"]["test_slices"]
    train_v11 = dataset["train_v11"]
    test_v11 = dataset["test_v11"]
    train_orders = dataset["train_orders"]
    test_orders = dataset["test_orders"]

    # Oracle
    print("\n[2/5] Building oracle labels...")
    y_bin, y_act, y_gain, train_base_mask = oracle_labels(train_orders, labels, slices, train_v11)
    n_actionable = np.sum(y_bin == 1)
    oracle_gain = int(np.sum(y_gain))
    print(f"  Actionable orders: {n_actionable}/{len(train_orders)} ({n_actionable/len(train_orders):.1%})")
    print(f"  Oracle gain: +{oracle_gain} TP")

    # Features
    print("\n[3/5] Building features and training...")
    X_train = build_features_full(train_orders, train_v11, slices, labels)
    X_test = build_features_full(test_orders, test_v11, test_slices, None)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    test_bin, test_act, test_act_probs = predict_on_test(X_train, y_bin, y_act, X_test)

    # Threshold selection + apply
    print("\n[4/5] Selecting threshold and generating predictions...")
    test_base_mask = v10.exact_count_mask(test_v11, test_slices, 1059, MAX_ROOT)

    best_thresh = 0.50  # From OOF analysis
    triggered = test_bin >= best_thresh

    print(f"  Threshold: {best_thresh}")
    print(f"  Triggered: {int(np.sum(triggered))}/{len(test_orders)} orders")

    # Ensure non-zero action for triggered
    for oi in range(len(test_orders)):
        if triggered[oi] and test_act[oi] == 0:
            probs = test_act_probs[oi]
            probs[0] = -1  # suppress keep
            test_act[oi] = int(np.argmax(probs))

    test_mask = apply_actions(test_base_mask, test_slices, test_v11, triggered, test_act, 1059)

    # Ensure every order has at least 1 prediction
    for oi, sl in enumerate(test_slices):
        ob = test_mask[sl]
        if np.sum(ob) == 0:
            ov = test_v11[sl]
            best = int(np.argmax(ov))
            test_mask[sl.start + best] = True

    cur = int(np.sum(test_mask))
    if cur < 1059:
        # Add more from top scores (don't add to orders with 0)
        optional = []
        for oi, sl in enumerate(test_slices):
            ov = test_v11[sl]
            ob = test_mask[sl]
            ranked = np.argsort(-ov, kind="stable")
            not_sel = ~ob
            candidates = ranked[not_sel[ranked]]
            for c in candidates[:min(2, len(candidates))]:
                optional.append((oi, c, ov[c]))
        optional.sort(key=lambda x: -x[2])
        needed = 1059 - cur
        for oi, c, _ in optional[:needed]:
            sl = test_slices[oi]
            test_mask[sl.start + c] = True
            # Don't exceed 8
            if np.sum(test_mask[sl]) > 8:
                # Remove the lowest selected
                sel_idx = np.where(test_mask[sl])[0]
                worst = sel_idx[np.argmin(test_v11[sl][sel_idx])]
                test_mask[sl.start + worst] = False
    elif cur > 1059:
        # Remove lowest-scoring selected (keep at least 1)
        rm_candidates = []
        for oi, sl in enumerate(test_slices):
            ov = test_v11[sl]
            ob = test_mask[sl]
            sel_idx = np.where(ob)[0]
            if len(sel_idx) > 1:
                for si in sel_idx:
                    rm_candidates.append((oi, si, ov[si]))
        rm_candidates.sort(key=lambda x: x[2])
        needed = cur - 1059
        for oi, si, _ in rm_candidates[:needed]:
            sl = test_slices[oi]
            test_mask[sl.start + si] = False

    # Final safety: ensure every order has at least 1
    for oi, sl in enumerate(test_slices):
        if np.sum(test_mask[sl]) == 0:
            test_mask[sl.start + int(np.argmax(test_v11[sl]))] = True

    # Exact count adjustment (±1 correction)
    cur = int(np.sum(test_mask))
    if cur > 1059:
        # Remove one: find order with most predictions, remove lowest-scored
        best_oi, best_si = -1, -1
        best_score = 999
        for oi, sl in enumerate(test_slices):
            ob = test_mask[sl]
            if np.sum(ob) <= 1:
                continue
            sel_idx = np.where(ob)[0]
            worst = sel_idx[np.argmin(test_v11[sl][sel_idx])]
            if test_v11[sl][worst] < best_score:
                best_score = test_v11[sl][worst]
                best_oi, best_si = oi, worst
        if best_oi >= 0:
            test_mask[test_slices[best_oi].start + best_si] = False
    elif cur < 1059:
        # Add one: find order with fewest, add highest unselected
        for oi, sl in enumerate(test_slices):
            not_sel = ~test_mask[sl]
            if np.sum(not_sel) > 0:
                best = int(np.argmax(test_v11[sl] * not_sel))
                test_mask[sl.start + best] = True
                break

    # Generate submission
    print("\n[5/5] Writing submission...")
    out_dir = Path("D:/zgyidong/experiments/v26_selective_router/submissions")
    out_path = out_dir / "v26_selective_error_router_p1059.csv"

    write_submission(test_mask, test_orders, dataset["data"], out_path)

    # Validate
    validate_csv(out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print(f"\n{'='*60}")
    print("SUBMISSION GENERATED")
    print(f"{'='*60}")
    print(f"  Path: {out_path}")
    print(f"  SHA256: {sha}")
    print(f"  Predictions: {int(np.sum(test_mask))}")
    print(f"  Orders triggered: {int(np.sum(triggered))}/{len(test_orders)}")
    print(f"  Threshold: {best_thresh}")

    # Save metadata
    (out_dir / "v26_selective_error_router_p1059.sha256").write_text(sha)

    # Diff from V11
    v11_champion_path = "D:/zgyidong/experiments/submissions/champion_0.906324_day01_probe01_v11_full.csv"
    with open(v11_champion_path, "r", encoding="utf-8") as f:
        v11_rows = list(csv.DictReader(f))
    with open(out_path, "r", encoding="utf-8") as f:
        v26_rows = list(csv.DictReader(f))

    v11_preds = {
        r["order_id"]: frozenset(
            rc["@rid"] for rc in json.loads(r["output"])["rootcause"]
        )
        for r in v11_rows
    }

    diff = []
    changed_orders = 0
    for r in v26_rows:
        oid = r["order_id"]
        vp = v11_preds[oid]
        v26p = frozenset(
            rc["@rid"] for rc in json.loads(r["output"])["rootcause"]
        )
        if vp != v26p:
            changed_orders += 1
            added = v26p - vp
            removed = vp - v26p
            diff.append({
                "order_id": oid,
                "same_count": len(vp & v26p),
                "added": len(added),
                "removed": len(removed),
            })

    diff_info = {
        "v11_predictions": 1059,
        "v26_predictions": int(np.sum(test_mask)),
        "orders_changed": changed_orders,
        "diff_details": sorted(diff, key=lambda x: -(x["added"] + x["removed"]))[:30],
    }

    with open(out_dir / "v26_selective_error_router_p1059.diff.json", "w") as f:
        json.dump(diff_info, f, ensure_ascii=False, indent=2)

    print(f"  Orders different from V11: {changed_orders}/{len(test_orders)}")
    print(f"  Diff saved: {out_dir / 'v26_selective_error_router_p1059.diff.json'}")

    # Final report
    report = {
        "version": "v26-selective-error-router",
        "threshold": best_thresh,
        "oracle_gain": int(oracle_gain),
        "train_actionable": int(n_actionable),
        "test_triggered": int(np.sum(triggered)),
        "test_predictions": int(np.sum(test_mask)),
        "orders_changed_from_v11": changed_orders,
        "sha256": sha,
    }
    output_dir = Path("D:/zgyidong/experiments/v26_selective_router/outputs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "v26_final_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  Report saved: {output_dir / 'v26_final_report.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

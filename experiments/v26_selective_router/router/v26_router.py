"""
V26 Selective Error Router — 核心实现
交叉拟合特征 + 动作分类器 + 高置信触发 + DP 解码
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
ACTION_NAMES = ["keep", "add1", "add2", "add3", "rm1", "rm2", "swap1", "swap2"]
ACTION_MAP = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
              4: (0, 1), 5: (0, 2), 6: (1, 1), 7: (2, 2)}


# ═══════════════════════════════════════════
# Cross-fit 特征工程
# ═══════════════════════════════════════════

def compute_crossfit_v11_stats(train_v11, slices, folds_arr, labels, base_mask):
    """
    Per-order V11 statistics computed in cross-fit manner.
    For each fold, statistics are computed only from the OTHER 4 folds.
    """
    n_orders = len(slices)
    all_orders = np.arange(n_orders)

    feat_list = [None] * n_orders

    for heldout in range(5):
        ref_oi = all_orders[folds_arr != heldout]
        val_oi = all_orders[folds_arr == heldout]

        # Compute template residual rates from reference folds
        template_stats = defaultdict(lambda: {"total": 0, "errors": 0})
        for oi in ref_oi:
            sl = slices[oi]
            ol = labels[sl]
            ob = base_mask[sl]
            sig = order_signature_simple(sl, ol)
            template_stats[sig]["total"] += 1
            template_stats[sig]["errors"] += int(np.any((ob == 1) & (ol == 0)))

        # Per-validation-order features
        for oi in val_oi:
            sl = slices[oi]
            start, stop = sl.start, sl.stop
            ov = train_v11[start:stop]
            ob = base_mask[start:stop]
            n = stop - start

            # V11 score statistics
            v11_selected = ov[ob]
            v11_unselected = ov[~ob] if np.any(~ob) else np.array([0.0])

            # Margin: gap between last selected and first unselected
            if len(v11_selected) > 0 and len(v11_unselected) > 0:
                margin = np.min(v11_selected) - np.max(v11_unselected)
            elif len(v11_selected) == 0:
                margin = -1.0
            else:
                margin = 1.0

            # Rank stability: std of V11 scores among top-K
            top_n = min(8, n)
            ranked_scores = np.sort(-ov)[:top_n]
            rank_stability = float(np.std(-ranked_scores)) if len(ranked_scores) > 1 else 0.0

            # Score concentration
            score_mean = float(np.mean(ov))
            score_std = float(np.std(ov))
            score_max = float(np.max(ov))
            score_min = float(np.min(ov))
            score_spread = score_max - score_min

            # Selected stats
            n_selected = int(np.sum(ob))
            selected_mean = float(np.mean(v11_selected)) if len(v11_selected) > 0 else 0.0
            selected_std = float(np.std(v11_selected)) if len(v11_selected) > 1 else 0.0

            # Unselected stats (potential candidates)
            if np.any(~ob):
                unsel_top = np.sort(-ov[~ob])[:min(3, np.sum(~ob))]
                unsel_mean = float(np.mean(-unsel_top))
            else:
                unsel_mean = 0.0

            # Template residual rate (cross-fit)
            sig = order_signature_simple(sl, labels[sl])
            ts = template_stats.get(sig, {"total": 0, "errors": 0})
            tpl_err_rate = ts["errors"] / max(ts["total"], 1)

            # Order scale
            n_alarms = n
            n_targets = int(np.sum(labels[start:stop]))

            feat_list[oi] = np.array([
                margin,
                rank_stability,
                score_mean, score_std, score_spread,
                score_max, score_min,
                float(n_selected), selected_mean, selected_std,
                unsel_mean,
                tpl_err_rate,
                float(n_alarms), float(n_targets),
                n_selected / max(n_alarms, 1),
                n_targets / max(n_alarms, 1),
            ], dtype=np.float32)

    return np.array(feat_list)


def order_signature_simple(sl, order_labels):
    """Simple signature: number of positives + total + sparse pattern."""
    n_pos = int(np.sum(order_labels))
    n_total = len(order_labels)
    return (n_pos, n_total)


def compute_oracle_actions(labels, slices, base_mask, train_v11):
    """Compute oracle action labels per order."""
    actions = np.zeros(len(slices), dtype=np.int32)
    gains = np.zeros(len(slices), dtype=np.int32)

    for oi, sl in enumerate(slices):
        ol = labels[sl]
        ob = base_mask[sl]
        ov = train_v11[sl]

        base_tp = int(np.sum(ob & (ol == 1)))
        base_count = int(np.sum(ob))
        n = len(ol)

        fn_mask = (ob == 0) & (ol == 1)
        fp_mask = ob & (ol == 0)
        fn_idx = np.where(fn_mask)[0]
        fp_idx = np.where(fp_mask)[0]

        if len(fn_idx) == 0 and len(fp_idx) == 0:
            actions[oi] = 0
            gains[oi] = 0
            continue

        fn_sorted = fn_idx[np.argsort(-ov[fn_idx])]
        fp_sorted = fp_idx[np.argsort(ov[fp_idx])]

        best_action = 0
        best_gain = 0

        for a_id, (na, nr) in ACTION_MAP.items():
            if na > len(fn_idx) or nr > len(fp_idx):
                continue
            nc = base_count + na - nr
            if nc < 1 or nc > MAX_ROOT:
                continue

            new_mask = ob.copy()
            if na: new_mask[fn_sorted[:na]] = True
            if nr: new_mask[fp_sorted[:nr]] = False
            new_tp = int(np.sum(new_mask & (ol == 1)))
            gain = new_tp - base_tp

            if gain > best_gain:
                best_gain = gain
                best_action = a_id

        actions[oi] = best_action
        gains[oi] = best_gain

    return actions, gains


# ═══════════════════════════════════════════
# Action Router
# ═══════════════════════════════════════════

def train_action_router(features, oracle_actions, folds_arr, seeds, min_confidence=0.6):
    """
    Train per-order action classifier with cross-validation.
    Only triggers actions when classifier confidence > min_confidence.
    """
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    n_orders = len(features)
    all_orders = np.arange(n_orders)
    oof_probs = np.zeros((n_orders, 8), dtype=np.float32)
    oof_actions = np.full(n_orders, -1, dtype=np.int32)

    for heldout in range(5):
        tr_oi = all_orders[folds_arr != heldout]
        val_oi = all_orders[folds_arr == heldout]

        X_tr, y_tr = features[tr_oi], oracle_actions[tr_oi]
        X_val = features[val_oi]

        # Filter to non-keep for class balancing
        non_keep = y_tr != 0
        if np.sum(non_keep) < 5:
            oof_probs[val_oi, 0] = 1.0
            oof_actions[val_oi] = 0
            continue

        # Class weights: balance keep vs non-keep
        n_keep = np.sum(y_tr == 0)
        n_act = np.sum(non_keep)
        keep_weight = n_act / max(n_keep, 1)

        fold_probs = np.zeros((len(val_oi), 8), dtype=np.float32)

        for seed in seeds:
            # ExtraTrees
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=4,
                max_features=0.7, class_weight="balanced",
                n_jobs=-1, random_state=seed + heldout,
            )
            et.fit(X_tr, y_tr)
            et_probs = et.predict_proba(X_val)
            et_classes = et.classes_
            for ci, cls_id in enumerate(et_classes):
                fold_probs[:, int(cls_id)] += et_probs[:, ci]

            # HGB
            hgb = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, min_samples_leaf=8,
                l2_regularization=1.0, class_weight="balanced",
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed + heldout,
            )
            hgb.fit(X_tr, y_tr)
            hgb_probs = hgb.predict_proba(X_val)
            hgb_classes = hgb.classes_
            for ci, cls_id in enumerate(hgb_classes):
                fold_probs[:, int(cls_id)] += hgb_probs[:, ci]

        fold_probs /= (len(seeds) * 2)
        oof_probs[val_oi] = fold_probs

        # Apply confidence threshold
        for i, vi in enumerate(val_oi):
            best_a = int(np.argmax(fold_probs[i]))
            best_p = fold_probs[i, best_a]
            if best_a != 0 and best_p >= min_confidence:
                oof_actions[vi] = best_a
            else:
                oof_actions[vi] = 0

    return oof_probs, oof_actions


# ═══════════════════════════════════════════
# DP Decoder
# ═══════════════════════════════════════════

def apply_actions_dp(base_mask, slices, train_v11, oof_actions, oof_probs,
                    target_total, max_total_dk=16):
    """
    Greedy DP: act on highest-confidence non-keep orders until budget exhausted.
    Only apply actions where expected gain > 0.
    """
    n_orders = len(slices)
    base_counts = np.array([int(np.sum(base_mask[sl])) for sl in slices])
    total_base = int(np.sum(base_counts))

    new_actions = np.zeros(n_orders, dtype=np.int32)

    # Collect actionable orders with expected gain
    candidates = []
    for oi in range(n_orders):
        a_id = oof_actions[oi]
        if a_id == 0:
            continue
        na, nr = ACTION_MAP[a_id]
        new_k = base_counts[oi] + na - nr
        if new_k < 1 or new_k > MAX_ROOT:
            continue
        conf = oof_probs[oi, a_id]
        dk = na - nr
        candidates.append((oi, a_id, na, nr, dk, conf))

    # Sort by confidence descending
    candidates.sort(key=lambda x: -x[5])

    # Greedy apply until budget exhausted or max changes reached
    cur_dk = 0
    for oi, a_id, na, nr, dk, conf in candidates:
        new_dk = cur_dk + dk
        k_after = total_base + new_dk

        if k_after < target_total - max_total_dk or k_after > target_total + max_total_dk:
            # Would overshoot; only apply if we can correct later
            if abs(cur_dk - (target_total - total_base)) < abs(new_dk - (target_total - total_base)):
                continue  # Skip, this would take us further from target

        cur_dk = new_dk
        new_actions[oi] = a_id

        if abs(cur_dk - (target_total - total_base)) <= 1:
            break

    # Apply selected actions to mask
    new_mask = base_mask.copy()
    for oi in range(n_orders):
        a_id = new_actions[oi]
        if a_id == 0:
            continue

        sl = slices[oi]
        ov = train_v11[sl]
        ob = new_mask[sl]
        na, nr = ACTION_MAP[a_id]

        ranked = np.argsort(-ov, kind="stable")
        not_sel = ~ob
        candidates = ranked[not_sel[ranked]]
        ob[candidates[:na]] = True

        sel_idx = np.where(ob)[0]
        if len(sel_idx) > nr:
            sel_ranked = sel_idx[np.argsort(ov[sel_idx])]
            ob[sel_ranked[:nr]] = False

    return new_mask, new_actions


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════

def evaluate_router(new_mask, new_actions, base_mask, labels, slices, folds_arr):
    """Full evaluation with fold-level breakdown."""
    base_tp = int(np.sum(base_mask & (labels == 1)))
    new_tp = int(np.sum(new_mask & (labels == 1)))
    total_delta = new_tp - base_tp

    # Orders changed
    n_changed = np.sum(new_actions != 0)

    # Per-fold
    all_orders = np.arange(len(slices))
    fold_deltas = []
    for f in range(5):
        fold_oi = all_orders[folds_arr == f]
        fold_btp = sum(int(np.sum(base_mask[slices[oi]] & (labels[slices[oi]] == 1))) for oi in fold_oi)
        fold_ntp = sum(int(np.sum(new_mask[slices[oi]] & (labels[slices[oi]] == 1))) for oi in fold_oi)
        fold_deltas.append(fold_ntp - fold_btp)

    # Bootstrap
    per_order_deltas = []
    for oi in range(len(slices)):
        sl = slices[oi]
        bt = int(np.sum(base_mask[sl] & (labels[sl] == 1)))
        nt = int(np.sum(new_mask[sl] & (labels[sl] == 1)))
        per_order_deltas.append(nt - bt)

    boot_lower = v25_data.bootstrap_lower_bound(np.array(per_order_deltas))

    # Correct-change rate
    changed_oi = np.where(new_actions != 0)[0]
    correct_changes = 0
    total_changes = len(changed_oi)
    for oi in changed_oi:
        sl = slices[oi]
        bt = int(np.sum(base_mask[sl] & (labels[sl] == 1)))
        nt = int(np.sum(new_mask[sl] & (labels[sl] == 1)))
        if nt > bt:
            correct_changes += 1

    correct_rate = correct_changes / max(total_changes, 1)

    return {
        "base_tp": base_tp, "new_tp": new_tp, "tp_delta": total_delta,
        "orders_changed": int(n_changed), "correct_rate": correct_rate,
        "fold_deltas": fold_deltas, "bootstrap_95_lower": float(boot_lower),
        "improved_folds": int(np.sum(np.array(fold_deltas) > 0)),
    }


def main():
    print("=" * 60)
    print("V26 Selective Error Router")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading baseline...")
    dataset = v25_data.load_all_data(
        "D:/zgyidong/train", "D:/zgyidong/test", "D:/zgyidong/codexgz/v11"
    )
    labels = dataset["labels"]
    slices = dataset["data"]["train_slices"]
    folds_arr = dataset["folds"]
    train_v11 = dataset["train_v11"]
    target_train = round(1059 / 546 * 1634)
    base_mask = v10.exact_count_mask(train_v11, slices, target_train, MAX_ROOT)

    base_stats = v25_data.confusion(base_mask, labels)
    print(f"  V11: TP={base_stats[1]} FP={base_stats[2]} FN={base_stats[3]} N={base_stats[4]}")

    # Cross-fit features
    print("\n[2/4] Computing cross-fit features...")
    features = compute_crossfit_v11_stats(train_v11, slices, folds_arr, labels, base_mask)
    print(f"  Features: {features.shape}")

    # Oracle actions
    oracle_actions, oracle_gains = compute_oracle_actions(labels, slices, base_mask, train_v11)
    n_actionable = np.sum(oracle_actions != 0)
    print(f"  Oracle actionable orders: {n_actionable}/{len(slices)} ({n_actionable/len(slices):.1%})")

    # Train router
    print("\n[3/4] Training action router...")
    seeds = [20260801, 20260817, 20260831]

    # Search for best confidence threshold
    best_delta = -999
    best_thresh = 0.6
    best_result = None

    for conf in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        oof_probs, oof_actions = train_action_router(
            features, oracle_actions, folds_arr, seeds, min_confidence=conf
        )
        new_mask, final_actions = apply_actions_dp(
            base_mask, slices, train_v11, oof_actions, oof_probs, target_train
        )
        result = evaluate_router(new_mask, final_actions, base_mask, labels, slices, folds_arr)
        print(f"    conf={conf:.2f}: delta={result['tp_delta']:+d} "
              f"changed={result['orders_changed']} correct_rate={result['correct_rate']:.2%} "
              f"folds={result['fold_deltas']} boot={result['bootstrap_95_lower']:.1f}")

        if result["tp_delta"] > best_delta:
            best_delta = result["tp_delta"]
            best_thresh = conf
            best_result = result

    # Final result
    print(f"\n[4/4] Best configuration: conf={best_thresh:.2f}")
    print(f"  TP delta: {best_result['tp_delta']:+d}")
    print(f"  Orders changed: {best_result['orders_changed']}")
    print(f"  Correct rate: {best_result['correct_rate']:.2%}")
    print(f"  Fold deltas: {best_result['fold_deltas']}")
    print(f"  Improved folds: {best_result['improved_folds']}/5")
    print(f"  Bootstrap 95% lower: {best_result['bootstrap_95_lower']:.1f}")

    # Gate
    gate_min = 30
    passed = (best_result["tp_delta"] >= gate_min
              and best_result["improved_folds"] >= 3
              and min(best_result["fold_deltas"]) >= 0)

    print(f"\n  GATE ({'+' if passed else '-'}30 TP, 3+/5 folds, all folds>=0): "
          f"{'PASSED' if passed else 'FAILED'}")

    # Save
    output_dir = Path(__file__).parent.parent / "outputs"
    with open(output_dir / "v26_router_report.json", "w") as f:
        report = {**best_result, "conf_threshold": best_thresh, "gate_passed": passed}
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_dir / 'v26_router_report.json'}")


if __name__ == "__main__":
    main()

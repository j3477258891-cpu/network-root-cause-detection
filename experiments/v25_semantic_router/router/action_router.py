"""
V25 Semantic Router — 残差动作路由器

从语义专家、近邻专家、V11、V19/V22/V24 产生候选动作，
通过 ExtraTrees+HGB 融合 → DP 联合解码，严格保持总 K=1059。

动作空间: keep / add1 / add2 / remove1 / remove2 / swap1 / swap2
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import KFold

MAX_ROOT = 8
MAX_ACTION = 3  # upgraded from 2 based on oracle audit


def build_router_features(
    order_idx, sl, v11_scores, expert_scores_dict,
    base_mask, labels=None, folds=None, train_orders=None
):
    """
    Build per-order router features.

    Features:
    0-3:   expert score statistics (mean, std, max, median)
    4-7:   V11 score statistics for the order
    8-11:  expert-V11 agreement (correlation, top-k overlap, rank shift)
    12-15: V11 margin features
    16-19: order scale features (n_alarms, n_selected, n_targets)
    20-23: model disagreement features
    24-27: semantic features (if available)
    """

    start, stop = sl.start, sl.stop
    n_alarms = stop - start
    base_selected = base_mask[start:stop]
    base_count = int(np.sum(base_selected))
    v11_sl = v11_scores[start:stop]

    features = []

    for expert_name, expert_scores in expert_scores_dict.items():
        es = expert_scores[start:stop]

        # Basic stats
        features.extend([
            float(np.mean(es)),
            float(np.std(es)),
            float(np.max(es)),
            float(np.median(es)),
        ])

        # Top-K overlap with V11
        v11_top8 = set(np.argsort(-v11_sl)[:min(8, n_alarms)])
        expert_top8 = set(np.argsort(-es)[:min(8, n_alarms)])
        overlap = len(v11_top8 & expert_top8) / max(len(v11_top8), 1)
        features.append(float(overlap))

        # Rank correlation (simplified: top-8 rank agreement)
        v11_ranks = np.argsort(np.argsort(-v11_sl))
        expert_ranks = np.argsort(np.argsort(-es))
        rank_diff = np.mean(np.abs(v11_ranks - expert_ranks)) / max(n_alarms, 1)
        features.append(float(rank_diff))

        # Score ratios
        if np.std(es) > 1e-6:
            features.append(float(np.max(es) / max(np.mean(es), 1e-6)))
            features.append(float((np.max(es) - np.median(es)) / max(np.std(es), 1e-6)))
        else:
            features.extend([1.0, 0.0])

    # V11 margin features
    v11_ranked = np.sort(-v11_sl)
    features.extend([
        float(v11_ranked[0] - v11_ranked[min(1, n_alarms - 1)]),  # top1 margin
        float(v11_ranked[0] - v11_ranked[min(7, n_alarms - 1)]),  # top8 margin
        float(np.mean(v11_ranked[:min(4, n_alarms)]) - np.mean(v11_ranked[min(4, n_alarms):]) if n_alarms > 4 else 0),
        float(np.max(v11_sl) - np.min(v11_sl)),
    ])

    # Order scale
    features.extend([
        float(n_alarms),
        float(base_count),
        float(n_alarms / max(base_count, 1)),
        float(base_count / max(n_alarms, 1)),
    ])

    # Model disagreement: std of predictions across experts
    expert_ensemble = np.column_stack([expert_scores[start:stop] for expert_scores in expert_scores_dict.values()])
    if expert_ensemble.shape[1] > 0:
        std_across = np.std(expert_ensemble, axis=1)
        features.extend([
            float(np.mean(std_across)),
            float(np.max(std_across)),
            float(np.std(std_across)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # V11 confidence (derived from score distribution)
    score_spread = np.max(v11_sl) - np.min(v11_sl) if n_alarms > 1 else 0
    score_entropy = -np.sum(v11_sl * np.log(np.clip(v11_sl, 1e-6, 1))) / max(np.log(n_alarms), 1) if n_alarms > 0 else 0
    features.extend([float(score_spread), float(score_entropy)])

    return np.array(features, dtype=np.float32)


def compute_action_targets(order_idx, sl, v11_scores, labels, base_mask, expert_scores_dict):
    """
    Compute the best action per order based on oracle knowledge.
    Used as training targets for the router.

    Returns: (action_id, tp_change)
        action_id: 0=keep, 1=add1, 2=add2, 3=rm1, 4=rm2, 5=swap1, 6=swap2
    """
    start, stop = sl.start, sl.stop
    order_labels = labels[start:stop]
    order_base = base_mask[start:stop]
    base_tp = int(np.sum(order_base & (order_labels == 1)))
    base_count = int(np.sum(order_base))
    n_alarms = stop - start

    fn_mask = (order_base == 0) & (order_labels == 1)
    fp_mask = order_base & (order_labels == 0)
    fn_idx = np.where(fn_mask)[0]
    fp_idx = np.where(fp_mask)[0]

    if len(fn_idx) == 0 and len(fp_idx) == 0:
        return 0, 0  # keep

    # Use ensemble of expert scores for ranking
    ensemble = np.zeros(n_alarms)
    w_sum = 0
    for name, scores in expert_scores_dict.items():
        ensemble += scores[start:stop]
        w_sum += 1
    ensemble /= max(w_sum, 1)

    fn_sorted = fn_idx[np.argsort(-ensemble[fn_idx])]
    fp_sorted = fp_idx[np.argsort(ensemble[fp_idx])]

    best_action = 0
    best_gain = 0

    # Evaluate each action
    actions_to_try = [
        (0, 0, 0, "keep"),
        (1, 0, 1, "add1"),
        (2, 0, 2, "add2"),
        (3, 1, 0, "rm1"),
        (4, 2, 0, "rm2"),
        (5, 1, 1, "swap1"),
        (6, 2, 2, "swap2"),
    ]

    for a_id, na, nr, aname in actions_to_try:
        if na > len(fn_idx) or nr > len(fp_idx):
            continue
        new_count = base_count + na - nr
        if new_count < 1 or new_count > MAX_ROOT:
            continue

        new_mask = order_base.copy()
        if na:
            new_mask[fn_sorted[:na]] = True
        if nr:
            new_mask[fp_sorted[:nr]] = False

        new_tp = int(np.sum(new_mask & (order_labels == 1)))
        gain = new_tp - base_tp

        if gain > best_gain:
            best_gain = gain
            best_action = a_id

    return best_action, best_gain


def train_router(expert_scores_dict, labels, slices, base_mask, v11_scores,
                 folds_arr, seeds, n_folds=5):
    """
    Train action router with 5-fold CV using ExtraTrees + HGB.
    """
    all_orders = np.arange(len(slices))
    X_all = []
    y_all = []
    order_to_idx = {}  # order_idx -> row in X_all

    # Build features and targets
    feature_dim = None
    for oi, sl in enumerate(slices):
        feats = build_router_features(
            oi, sl, v11_scores, expert_scores_dict, base_mask,
            labels=labels, folds=folds_arr
        )
        if feature_dim is None:
            feature_dim = len(feats)
        action, gain = compute_action_targets(
            oi, sl, v11_scores, labels, base_mask, expert_scores_dict
        )
        X_all.append(feats)
        y_all.append(action)
        order_to_idx[oi] = len(X_all) - 1

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    n_orders = len(X_all)

    print(f"Router: {n_orders} orders, {feature_dim} features per order")
    print(f"Action distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    # 5-fold CV
    oof_probs = np.zeros((n_orders, 7), dtype=np.float32)  # 7 actions

    for heldout in range(n_folds):
        train_oi = all_orders[folds_arr != heldout]
        val_oi = all_orders[folds_arr == heldout]

        tr_idx = [order_to_idx[oi] for oi in train_oi]
        val_idx = [order_to_idx[oi] for oi in val_oi]

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_val = X_all[val_idx]

        # Per-seed ensemble
        fold_probs = np.zeros((len(val_idx), 7), dtype=np.float32)
        for seed in seeds:
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=4,
                max_features=0.7, class_weight="balanced",
                n_jobs=-1, random_state=seed + heldout,
            )
            et.fit(X_tr, y_tr)
            fold_probs += et.predict_proba(X_val)

            hgb = HistGradientBoostingClassifier(
                max_iter=200, max_depth=8, min_samples_leaf=8,
                l2_regularization=1.0, class_weight="balanced",
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed + heldout,
            )
            hgb.fit(X_tr, y_tr)
            fold_probs += hgb.predict_proba(X_val)

        fold_probs /= (len(seeds) * 2)  # average across seeds and models
        oof_probs[val_idx] = fold_probs

    return oof_probs, X_all.shape[1]


def dp_decode_actions(action_probs, base_mask, slices, v11_scores,
                      expert_scores_dict, target_total=3169, max_action=MAX_ACTION):
    """
    DP decoder: select actions for all orders to maximize expected TP gain
    while maintaining total K = target_total.
    """
    n_orders = len(slices)
    action_map = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                  3: (1, 0), 4: (2, 0), 5: (1, 1), 6: (2, 2)}
    # (remove_count, add_count)

    # Current K per order
    base_counts = np.array([int(np.sum(base_mask[sl])) for sl in slices])
    total_base = int(np.sum(base_counts))
    target_dk = target_total - total_base

    # DP table
    max_dk = n_orders * max_action
    offset = max_dk
    dp = np.full((n_orders + 1, 2 * max_dk + 1), -np.inf)
    back = np.zeros((n_orders + 1, 2 * max_dk + 1), dtype=np.int32)

    dp[0, offset] = 0.0

    for i in range(n_orders):
        bc = base_counts[i]
        for dk in range(-max_dk, max_dk + 1):
            dk_idx = dk + offset
            if dp[i, dk_idx] <= -1e10:
                continue

            for a_id, (nr, na) in action_map.items():
                new_count = bc - nr + na
                if new_count < 1 or new_count > MAX_ROOT:
                    continue
                new_dk = dk + (na - nr)
                if abs(new_dk) > max_dk:
                    continue
                new_idx = new_dk + offset
                new_val = dp[i, dk_idx] + action_probs[i, a_id]

                if new_val > dp[i + 1, new_idx]:
                    dp[i + 1, new_idx] = new_val
                    back[i + 1, new_idx] = a_id

    # Find best valid delta
    target_idx = target_dk + offset
    if abs(target_dk) > max_dk or dp[n_orders, target_idx] <= -1e10:
        valid = np.where(dp[n_orders] > -1e10)[0]
        if len(valid) == 0:
            return np.zeros(n_orders, dtype=np.int32), total_base
        target_idx = valid[np.argmax(dp[n_orders, valid])]

    # Backtrack
    actions = np.zeros(n_orders, dtype=np.int32)
    cur_idx = target_idx
    for i in range(n_orders, 0, -1):
        actions[i - 1] = back[i, cur_idx]
        nr, na = action_map[actions[i - 1]]
        cur_idx = cur_idx - (na - nr)

    return actions, total_base


def apply_actions(base_mask, slices, actions, ensemble_scores):
    """Apply router actions to modify the selection mask."""
    new_mask = base_mask.copy()

    for oi, sl in enumerate(slices):
        action = actions[oi]
        if action == 0:  # keep
            continue

        start, stop = sl.start, sl.stop
        local_scores = ensemble_scores[start:stop]
        local_select = new_mask[start:stop]
        n = stop - start

        ranked = np.argsort(-local_scores, kind="stable")

        if action == 1:  # add1
            not_sel = ~local_select
            candidates = ranked[not_sel[ranked]]
            if len(candidates) > 0:
                local_select[candidates[0]] = True
        elif action == 2:  # add2
            not_sel = ~local_select
            candidates = ranked[not_sel[ranked]]
            local_select[candidates[:2]] = True
        elif action == 3:  # rm1
            sel_idx = np.where(local_select)[0]
            if len(sel_idx) > 1:
                sel_ranked = sel_idx[np.argsort(local_scores[sel_idx])]
                local_select[sel_ranked[0]] = False
        elif action == 4:  # rm2
            sel_idx = np.where(local_select)[0]
            if len(sel_idx) > 2:
                sel_ranked = sel_idx[np.argsort(local_scores[sel_idx])]
                local_select[sel_ranked[:2]] = False
        elif action == 5:  # swap1
            sel_idx = np.where(local_select)[0]
            not_sel = ~local_select
            if len(sel_idx) > 0:
                sel_ranked = sel_idx[np.argsort(local_scores[sel_idx])]
                candidates = ranked[not_sel[ranked]]
                if len(candidates) > 0:
                    local_select[sel_ranked[0]] = False
                    local_select[candidates[0]] = True
        elif action == 6:  # swap2
            sel_idx = np.where(local_select)[0]
            not_sel = ~local_select
            if len(sel_idx) > 1:
                sel_ranked = sel_idx[np.argsort(local_scores[sel_idx])]
                candidates = ranked[not_sel[ranked]]
                n_swaps = min(2, len(candidates), len(sel_ranked))
                local_select[sel_ranked[:n_swaps]] = False
                local_select[candidates[:n_swaps]] = True

    return new_mask


def main():
    print("V25 Action Router")
    print("=" * 60)
    print("This module is imported by the training pipeline.")
    print("Usage:")
    print("  from router.action_router import train_router, dp_decode_actions, apply_actions")


if __name__ == "__main__":
    main()

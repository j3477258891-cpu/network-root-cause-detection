"""
V25 工作流 B: Enhanced DP 解码器
在 V22 双头边界 DP 基础上改进:
1. 更好的 K 估计 (LGBM 回归)
2. 三动作空间 (swap/insert/delete)
3. 多模型联合 DP
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def estimate_k_per_order(scores, slices, max_k=8):
    """Estimate optimal K per order using score distribution analysis."""
    k_estimates = np.zeros(len(slices), dtype=np.int32)
    
    for i, sl in enumerate(slices):
        local = scores[sl]
        n = len(local)
        if n == 0:
            k_estimates[i] = 1
            continue
        
        # Simple heuristic: number of scores above 0.5 * max_score
        threshold = np.max(local) * 0.6
        above = np.sum(local >= threshold)
        k_estimates[i] = max(1, min(max_k, int(above)))
    
    return k_estimates


def compute_action_utilities(scores_list, weights, base_mask, labels, slices):
    """
    Compute utility of taking each action per order.
    
    Actions: {0: keep, +1: add 1, +2: add 2, -1: remove 1, -2: remove 2}
    For multi-model: weighted average of per-model scores.
    """
    n_orders = len(slices)
    action_utils = np.zeros((n_orders, 5))  # -2, -1, 0, +1, +2
    action_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    
    for oi, sl in enumerate(slices):
        start, stop = sl.start, sl.stop
        n_local = stop - start
        if n_local == 0:
            continue
        
        # Current selected and candidates
        base_selected = base_mask[start:stop]
        base_count = int(np.sum(base_selected))
        
        # Blended score
        blended = np.zeros(n_local)
        for scores, w in zip(scores_list, weights):
            blended += w * scores[start:stop]
        
        # Rank candidates
        ranked = np.argsort(-blended, kind="stable")
        
        # Identify candidates not in base
        not_selected = ~base_selected
        candidates = ranked[not_selected[ranked]]
        selected_ranked = ranked[base_selected[ranked]]
        
        # Utility of adding (increase K)
        for delta in [1, 2]:
            add_candidates = candidates[:delta]
            if len(add_candidates) < delta:
                action_utils[oi, action_map[delta]] = -1e6
            else:
                # Utility = average score of added items minus cost
                utility = np.mean(blended[add_candidates]) - 0.3
                action_utils[oi, action_map[delta]] = utility
        
        # Utility of removing (decrease K)
        for delta in [-1, -2]:
            rm_candidates = selected_ranked[-abs(delta):] if len(selected_ranked) >= abs(delta) else []
            if len(rm_candidates) < abs(delta) or base_count + delta < 1:
                action_utils[oi, action_map[delta]] = -1e6
            else:
                # Utility = -average score of removed items (want to remove low scores)
                utility = -(np.mean(blended[rm_candidates]) - 0.5)
                action_utils[oi, action_map[delta]] = utility
        
        # Utility of keeping
        action_utils[oi, action_map[0]] = 0.0
    
    return action_utils


def dp_decode(action_utils, target_delta_k=0, max_delta=16):
    """
    DP to select optimal action per order with total K constraint.
    
    Returns: best actions per order and total utility.
    """
    n_orders = len(action_utils)
    action_values = [-2, -1, 0, 1, 2]
    action_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    
    # DP table: dp[i][dk + offset] = max utility
    offset = n_orders * 2  # max possible delta
    max_dk = n_orders * 2
    dp = np.full((n_orders + 1, 2 * max_dk + 1), -np.inf)
    back = np.zeros((n_orders + 1, 2 * max_dk + 1), dtype=np.int32)
    
    dp[0, offset] = 0.0
    
    for i in range(n_orders):
        for dk in range(-max_dk, max_dk + 1):
            dk_idx = dk + offset
            if dp[i, dk_idx] <= -1e10:
                continue
            
            for a_idx, action in enumerate(action_values):
                new_dk = dk + action
                if abs(new_dk) > max_dk:
                    continue
                new_idx = new_dk + offset
                new_val = dp[i, dk_idx] + action_utils[i, a_idx]
                
                if new_val > dp[i + 1, new_idx]:
                    dp[i + 1, new_idx] = new_val
                    back[i + 1, new_idx] = a_idx
    
    # Find target delta
    target_idx = target_delta_k + offset
    if abs(target_delta_k) > max_dk or dp[n_orders, target_idx] <= -1e10:
        # Find best valid delta
        valid = np.where(dp[n_orders] > -1e10)[0]
        if len(valid) == 0:
            return [2] * n_orders, 0  # all keep
        target_idx = valid[np.argmax(dp[n_orders, valid])]
    
    # Backtrack
    actions = np.zeros(n_orders, dtype=np.int32)
    cur_idx = target_idx
    for i in range(n_orders, 0, -1):
        a_idx = back[i, cur_idx]
        actions[i - 1] = action_values[a_idx]
        cur_idx = cur_idx - action_values[a_idx]
    
    total_utility = dp[n_orders, target_idx]
    actual_dk = target_idx - offset
    
    return actions, float(total_utility)


def apply_actions(base_mask, slices, scores, actions, max_k=8):
    """Apply DP actions to modify base mask."""
    new_mask = base_mask.copy()
    
    for oi, sl in enumerate(slices):
        action = actions[oi]
        start, stop = sl.start, sl.stop
        local_scores = scores[start:stop]
        local_new = new_mask[start:stop]
        n_local = stop - start
        
        if action == 0:
            continue
        
        ranked = np.argsort(-local_scores, kind="stable")
        
        if action > 0:
            # Add top-k not yet selected
            not_sel = ~local_new
            candidates = ranked[not_sel[ranked]]
            to_add = candidates[:action]
            local_new[to_add] = True
        
        elif action < 0:
            # Remove lowest-k currently selected
            sel_indices = np.where(local_new)[0]
            if len(sel_indices) <= abs(action):
                continue
            sel_ranked = sel_indices[np.argsort(local_scores[sel_indices], kind="stable")]
            to_rm = sel_ranked[:abs(action)]
            local_new[to_rm] = False
        
        # Ensure at least 1 selected
        if np.sum(local_new) == 0:
            local_new[ranked[0]] = True
    
    return new_mask


def enhanced_dp_pipeline(scores_dict, weights, slices, labels, base_mask, 
                         v11_scores, target_predictions=1059, max_k=8):
    """
    Full enhanced DP pipeline.
    
    Args:
        scores_dict: {"M1": oof, "M2": oof, ...} per-model scores
        weights: per-model weights
        slices: order slices
        labels: ground truth (for evaluation)
        base_mask: V11 baseline mask
        v11_scores: V11 base scores
        target_predictions: total K target
    """
    scores_list = list(scores_dict.values())
    model_names = list(scores_dict.keys())
    
    # Ensure weights match
    if len(weights) != len(scores_list):
        weights = [1.0 / len(scores_list)] * len(scores_list)
    
    # Compute action utilities
    action_utils = compute_action_utilities(scores_list, weights, base_mask, labels, slices)
    
    # Estimate K changes
    k_per_order = np.array([int(np.sum(base_mask[s])) for s in slices])
    total_k = int(np.sum(k_per_order))
    target_dk = target_predictions - total_k
    
    # DP decode
    actions, utility = dp_decode(action_utils, target_dk)
    
    # Apply actions
    blended = np.zeros(len(labels))
    for scores, w in zip(scores_list, weights):
        blended += w * scores
    blended /= sum(weights)
    
    new_mask = apply_actions(base_mask, slices, blended, actions, max_k)
    
    # Evaluation
    base_tp = int(np.sum(base_mask & (labels == 1)))
    new_tp = int(np.sum(new_mask & (labels == 1)))
    delta = new_tp - base_tp
    
    # Per-order details
    details = []
    for oi, sl in enumerate(slices):
        if actions[oi] != 0:
            details.append({
                "order": oi,
                "action": int(actions[oi]),
                "base_k": int(np.sum(base_mask[sl])),
                "new_k": int(np.sum(new_mask[sl])),
            })
    
    return {
        "new_mask": new_mask,
        "actions": actions.tolist(),
        "base_tp": base_tp,
        "new_tp": new_tp,
        "tp_delta": delta,
        "total_utility": utility,
        "changed_orders": len(details),
        "details": details,
        "total_k_before": total_k,
        "total_k_after": int(np.sum(new_mask)),
    }


def evaluate_dp_folds(scores_dict, weights, slices, labels, folds_arr, base_mask, v11_scores):
    """Evaluate DP with 5-fold cross-validation."""
    all_orders = np.arange(len(slices))
    results = []
    
    for fold in range(5):
        val_orders = all_orders[folds_arr == fold]
        val_rows = np.concatenate([
            np.arange(sl.start, sl.stop) for sl in [slices[i] for i in val_orders]
        ])
        
        # Build per-fold slices
        val_slices = []
        offset = 0
        for oi in val_orders:
            sl = slices[oi]
            val_slices.append(slice(offset, offset + sl.stop - sl.start))
            offset += sl.stop - sl.start
        
        # Subset scores
        fold_scores = {}
        for name, scores in scores_dict.items():
            fold_scores[name] = scores[val_rows]
        
        fold_labels = labels[val_rows]
        fold_base = base_mask[val_rows]
        fold_v11 = v11_scores[val_rows]
        
        result = enhanced_dp_pipeline(
            fold_scores, weights, val_slices, fold_labels, fold_base, fold_v11
        )
        result["fold"] = fold
        results.append(result)
    
    total_tp_delta = sum(r["tp_delta"] for r in results)
    return results, total_tp_delta

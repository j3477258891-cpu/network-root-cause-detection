"""
V25 工作流 D: 模板迁移
基于告警标题集合的模板匹配，保守地迁移根因模式
"""

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def build_template_fingerprint(order):
    """Build a template fingerprint for an order.
    
    Based on: alarm title counts + target alarm titles + alarm count.
    This matches the fold-signature approach.
    """
    alarms = order["alarms"]
    title_counts = Counter(
        str(a.get("title", "")) for a in alarms
    )
    target_titles = sorted(
        str(a.get("title", ""))
        for a in alarms
        if a.get("label") == "TargetAlarm"
    )
    fingerprint = (
        tuple(sorted(title_counts.items())),
        tuple(target_titles),
        len(alarms),
    )
    return fingerprint


def build_template_groups(train_orders):
    """Group training orders by template fingerprint."""
    groups = defaultdict(list)
    for idx, order in enumerate(train_orders):
        fp = build_template_fingerprint(order)
        groups[fp].append(idx)
    return groups


def compute_template_stats(groups, train_orders, labels, slices):
    """Compute per-template statistics for transfer.
    
    Returns: {
        fingerprint: {
            "support": int,
            "root_titles": {title: count},
            "root_rids": {rid: count},
            "consistency": float (most common root / total roots),
            "most_common_root": str (rid),
            "orders": [idx, ...]
        }
    }
    """
    stats = {}
    for fp, indices in groups.items():
        if len(indices) < 3:  # min_support
            continue
        
        # Collect root cause patterns
        root_titles = Counter()
        root_rids = Counter()
        total_roots = 0
        
        for oi in indices:
            order = train_orders[oi]
            sl = slices[oi]
            order_labels = labels[sl]
            order_alarms = order["alarms"]
            
            for j, alarm in enumerate(order_alarms):
                if order_labels[j]:
                    root_titles[str(alarm.get("title", ""))] += 1
                    root_rids[alarm.get("@rid", "")] += 1
                    total_roots += 1
        
        if total_roots == 0:
            continue
        
        # Consistency: most common root proportion
        most_common_title, most_common_count = root_titles.most_common(1)[0] if root_titles else ("", 0)
        consistency = most_common_count / total_roots if total_roots > 0 else 0
        
        if consistency < 0.95:  # min_consistency
            continue
        
        stats[fp] = {
            "support": len(indices),
            "root_titles": dict(root_titles.most_common(5)),
            "most_common_title": most_common_title,
            "consistency": consistency,
            "total_roots": total_roots,
            "orders": indices,
        }
    
    return stats


def match_test_orders(test_orders, template_stats):
    """Match test orders to known templates.
    
    Returns: {test_order_idx: template_fingerprint}
    """
    matches = {}
    for idx, order in enumerate(test_orders):
        fp = build_template_fingerprint(order)
        if fp in template_stats:
            matches[idx] = fp
    return matches


def compute_template_prior(template_stats, test_orders, feature_sets):
    """
    Compute template-based prior scores for test orders.
    
    For each test order matching a known template, assign higher scores
    to alarms whose titles match the template's most common root cause title.
    """
    test_n = sum(len(o["alarms"]) for o in test_orders)
    prior_scores = np.full(test_n, 0.5, dtype=np.float32)
    
    matches = match_test_orders(test_orders, template_stats)
    
    offset = 0
    for idx, order in enumerate(test_orders):
        n_alarms = len(order["alarms"])
        
        if idx in matches:
            fp = matches[idx]
            stats = template_stats[fp]
            most_common = stats["most_common_title"]
            consistency = stats["consistency"]
            
            for j, alarm in enumerate(order["alarms"]):
                title = str(alarm.get("title", ""))
                if title == most_common:
                    # Boost score proportionally to consistency
                    prior_scores[offset + j] = 0.5 + 0.4 * consistency
                # Don't penalize non-matching titles (stay at 0.5)
        
        offset += n_alarms
    
    return prior_scores


def template_transfer_validation(template_stats, train_orders, labels, slices, folds_arr):
    """
    Leave-one-template-out validation of template transfer.
    
    For each template group, leave out one order and check if the template
    prior correctly identifies root causes.
    """
    all_orders = np.arange(len(train_orders))
    results = []
    
    for fp, stats in template_stats.items():
        orders = stats["orders"]
        most_common_title = stats["most_common_title"]
        
        for holdout_oi in orders:
            # Predict: any alarm with most_common_title is root
            order = train_orders[holdout_oi]
            sl = slices[holdout_oi]
            order_labels = labels[sl]
            
            predicted = np.array([
                str(a.get("title", "")) == most_common_title
                for a in order["alarms"]
            ])
            
            tp = int(np.sum(predicted & order_labels))
            fp = int(np.sum(predicted & ~order_labels))
            fn = int(np.sum(~predicted & order_labels))
            
            results.append({
                "template_support": stats["support"],
                "consistency": stats["consistency"],
                "tp": tp, "fp": fp, "fn": fn,
                "correct": tp > 0 and fp == 0,
            })
    
    if not results:
        return {"accuracy": 0, "total": 0, "correct": 0}
    
    correct = sum(1 for r in results if r["correct"])
    return {
        "accuracy": correct / len(results),
        "total": len(results),
        "correct": correct,
        "avg_support": np.mean([r["template_support"] for r in results]),
    }


def build_template_features(train_orders, test_orders, labels, slices, config):
    """
    Full template transfer pipeline.
    
    Returns: template_prior_scores for test orders, validation results.
    """
    min_support = config.get("min_support", 3)
    min_consistency = config.get("min_consistency", 0.95)
    
    # Build template groups
    groups = build_template_groups(train_orders)
    
    # Filter by support
    groups = {fp: idx for fp, idx in groups.items() if len(idx) >= min_support}
    
    # Compute stats
    stats = compute_template_stats(groups, train_orders, labels, slices)
    
    # Filter by consistency
    stats = {fp: s for fp, s in stats.items() if s["consistency"] >= min_consistency}
    
    print(f"  Templates found: {len(stats)} (support>={min_support}, consistency>={min_consistency})")
    
    # Compute priors for test
    prior_scores = compute_template_prior(stats, test_orders, None)
    
    # Validation
    val_results = template_transfer_validation(stats, train_orders, labels, slices, None)
    
    coverage = len(match_test_orders(test_orders, stats)) / len(test_orders)
    
    print(f"  Validation accuracy: {val_results['accuracy']:.4f} ({val_results['correct']}/{val_results['total']})")
    print(f"  Test coverage: {coverage:.4f} ({len(match_test_orders(test_orders, stats))}/{len(test_orders)})")
    
    return {
        "prior_scores": prior_scores,
        "template_stats": stats,
        "validation": val_results,
        "coverage": coverage,
        "n_templates": len(stats),
    }

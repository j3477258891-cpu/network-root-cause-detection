"""V156 FN-classifier research track  (24h budget, zero submission quota).

Question: can a fold-honest classifier score UNSELECTED alarms well enough that
adding its top-K picks (per 546 orders) beats the strongest existing control on
the exact F1 objective?  This is the only remaining in-pool route to a larger
gain; it does NOT consume submissions unless it passes the gate below.

Reuse (read-only):
  * V25 bundle  train_v11/v13/v19, train_labels, train_alarm_ptr, train_folds,
                train_station_folds, train_alarm_x, train_path_*
  * V153 candidate pool (the <16 node new group must not touch the 17 candidates)

Fixed comparisons
  1 control       : existing CatBoost / V38 add-action scores
  2 +disagreement : control + pairwise |v11-v13|, |v11-v19|, |v13-v19|, std
  3 +graph        : control + alarm_x + path_node_type + path_edge_type + path_length
  4 combined      : 1 + 2 + 3  (semantic embeddings only with full outer-fold isolation)

Budget sweep: add 8 / 16 / 32 / 64 nodes per 546 orders.

Gate (all must hold):
  * 5 folds, >= 3 folds with positive F1 gain
  * pooled gain > best control
  * worst fold >= best control's worst fold
  * 2000 paired bootstrap: absolute and relative 95% lower bound > 0
  * parameter / calibration choice made only on the inner split

Run:  python research.py            (writes results.json + STATUS.json)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = Path("D:/zgyidong/experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz")
G = 1044
TARGET_TRAIN = 3169
MAX_ROOT = 8
BUDGETS = (8, 16, 32, 64)          # nodes per 546 orders
SEED = 20260910
START = time.time()
DEADLINE_S = 24 * 3600


# --------------------------------------------------------------------------- #
def load_arrays():
    with np.load(BUNDLE, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def v11_base_mask(scores, ptr, target, max_roots=MAX_ROOT):
    sel = np.zeros(len(scores), dtype=bool)
    optional = []
    for s, e in zip(ptr[:-1], ptr[1:]):
        s, e = int(s), int(e)
        rk = np.argsort(-scores[s:e], kind="stable")[:max_roots]
        sel[s + rk[0]] = True
        optional.extend((s + rk[1:]).tolist())
    rem = target - int(sel.sum())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    sel[optional[:rem]] = True
    return sel


def feature_groups(arrays):
    """Return the four fixed feature matrices over ALL train alarms."""
    v11 = arrays["train_v11"].astype(np.float32)
    v13 = arrays["train_v13"].astype(np.float32)
    v19 = arrays["train_v19"].astype(np.float32)
    control = np.column_stack([v11, v13, v19]).astype(np.float32)
    disagreement = np.column_stack([
        np.abs(v11 - v13), np.abs(v11 - v19), np.abs(v13 - v19),
        np.std(np.column_stack([v11, v13, v19]), axis=1),
    ]).astype(np.float32)
    graph = np.column_stack([
        arrays["train_alarm_x"].astype(np.float32),
        arrays["train_path_node_type"].astype(np.float32),
        arrays["train_path_edge_type"].astype(np.float32),
        arrays["train_path_length"].astype(np.float32).reshape(-1, 1),
    ]).astype(np.float32)
    return {
        "control": control,
        "control+disagreement": np.column_stack([control, disagreement]),
        "control+graph": np.column_stack([control, graph]),
        "combined": np.column_stack([control, disagreement, graph]),
    }


def fn_dataset(arrays, base):
    """Rows = nodes NOT selected by the baseline; label = is a true root cause."""
    labels = arrays["train_labels"].astype(np.int8)
    ptr = arrays["train_alarm_ptr"].astype(np.int64)
    folds = arrays["train_station_folds"].astype(np.int8)   # station-isolated
    unselected = ~base
    y = labels[unselected].astype(np.int8)
    node_fold = np.concatenate([np.full(int(e - s), folds[i], dtype=np.int8)
                                for i, (s, e) in enumerate(zip(ptr[:-1], ptr[1:]))])[unselected]
    order_of_node = np.concatenate([np.full(int(e - s), i, dtype=np.int64)
                                    for i, (s, e) in enumerate(zip(ptr[:-1], ptr[1:]))])[unselected]
    return unselected, y, node_fold, order_of_node


def cross_fit_scores(x, y, node_fold, n_folds=5):
    """Out-of-fold P(FN) for every unselected node (station-isolated folds)."""
    from lightgbm import LGBMClassifier
    scores = np.zeros(len(y), dtype=np.float32)
    for f in range(n_folds):
        tr, te = node_fold != f, node_fold == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(x[tr], y[tr])
        scores[te] = model.predict_proba(x[te])[:, 1]
    return scores


def add_topk_f1(arrays, base, unselected, scores, k_per_546):
    """F1 after adding the global top-K scored unselected nodes (per-order cap 8)."""
    labels = arrays["train_labels"].astype(np.int8)
    ptr = arrays["train_alarm_ptr"].astype(np.int64)
    n_orders = len(ptr) - 1
    k = int(round(k_per_546 * n_orders / 546))
    order_of_node = np.concatenate([np.full(int(e - s), i, dtype=np.int64)
                                    for i, (s, e) in enumerate(zip(ptr[:-1], ptr[1:]))])
    unsel_global = np.flatnonzero(unselected)
    order_ids = order_of_node[unsel_global]

    counts = np.zeros(n_orders, dtype=np.int64)
    np.add.at(counts, order_of_node[np.flatnonzero(base)], 1)

    sel = np.zeros(len(labels), dtype=bool)
    sel[np.flatnonzero(base)] = True
    added = 0
    for idx in np.argsort(-scores, kind="stable"):
        if added >= k:
            break
        o = int(order_ids[idx])
        if counts[o] >= MAX_ROOT:
            continue
        sel[unsel_global[idx]] = True
        counts[o] += 1
        added += 1
    tp = int((sel & (labels == 1)).sum())
    p = int(sel.sum())
    base_tp = int((base & (labels == 1)).sum())
    base_p = int(base.sum())
    return (2 * tp / (G + p)) - (2 * base_tp / (G + base_p)), added


def paired_bootstrap(deltas, iterations=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    samples = rng.choice(d, size=(iterations, len(d)), replace=True).sum(axis=1)
    return float(np.quantile(samples, 0.025))


def run_comparison(name, x, arrays, base, unselected, y, node_fold):
    scores = cross_fit_scores(x, y, node_fold)
    out = {"name": name, "budgets": {}}
    for k in BUDGETS:
        delta, added = add_topk_f1(arrays, base, unselected, scores, k)
        out["budgets"][str(k)] = {"delta_f1": round(delta, 8), "nodes_added": added}
    return out


def evaluate_gate(results):
    """Apply the fixed gate. Aggregate positivity is checked first; a negative
    aggregate is decisive and short-circuits the per-fold / bootstrap checks."""
    comps = {c["name"]: c for c in results["comparisons"]}
    control = comps["control"]
    ctrl_best = max(b["delta_f1"] for b in control["budgets"].values())
    best = max(((name, k, b["delta_f1"]) for name, c in comps.items() if name != "control"
                for k, b in c["budgets"].items()), key=lambda t: t[2])
    fails = []
    if best[2] <= 0:
        fails.append("pooled_gain_not_positive")
    if best[2] <= ctrl_best:
        fails.append("not_better_than_best_control")
    if fails:
        return {"passed": False, "fails": fails,
                "best_candidate": {"feature_set": best[0], "budget": best[1], "delta_f1": best[2]},
                "control_best_delta_f1": ctrl_best,
                "decision": "no_submit_quota_returns_to_pool"}
    return {"passed": True, "best_candidate": {"feature_set": best[0], "budget": best[1],
                                               "delta_f1": best[2]},
            "control_best_delta_f1": ctrl_best,
            "note": "per-fold + 2000-sample paired bootstrap still required",
            "decision": "run_full_gate"}


def main():
    arrays = load_arrays()
    base = v11_base_mask(arrays["train_v11"], arrays["train_alarm_ptr"], TARGET_TRAIN)
    unselected, y, node_fold, order_ids = fn_dataset(arrays, base)
    groups = feature_groups(arrays)
    results = {"fn_positive": int(y.sum()), "unselected": int(len(y)),
               "base_tp": int((base & (arrays["train_labels"] == 1)).sum()),
               "budgets_per_546": list(BUDGETS), "comparisons": [], "elapsed_s": None}
    for name, x in groups.items():
        xu = x[unselected]
        results["comparisons"].append(run_comparison(name, xu, arrays, base, unselected,
                                                      y, node_fold))
        results["elapsed_s"] = round(time.time() - START, 1)
    results["gate"] = evaluate_gate(results)
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (HERE / "STATUS.json").write_text(json.dumps({
        "track": "v156_fn_research", "deadline_hours": 24, "quota_cost": 0,
        "elapsed_s": results["elapsed_s"], "gate": results["gate"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": results["gate"]}, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()

"""Direct pair-delta classifier audit for the current fixed-count baseline.

Research only: no submission CSVs are written.  Candidate labels are built
from train root-cause annotations, while all splits are by whole order so a
template/station family cannot leak across train and validation pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT))
from experiments.research_pairwise_error_model import (  # noqa: E402
    DATA,
    TRAIN_DIR,
    TEST_DIR,
    CURRENT,
    OUT,
    v10,
    read_submission,
    baseline_mask_for_orders,
    exact_mask,
)

SEED = 20260823
MAX_ROOTS = 8
TRAIN_P = round(1035 / 546 * 1634)


def order_folds(orders, mode: str, nfold: int = 5):
    groups = defaultdict(list)
    for i, order in enumerate(orders):
        titles = Counter(str(a.get("title", "")) for a in order["alarms"])
        if mode == "site":
            # site is deliberately coarse; station split is a stress test.
            sites = []
            for a in order["alarms"]:
                text = str(a.get("location", ""))
                site = "unknown"
                for key in ("SubNetwork=", "STATION=", "NodeMe=", "gNodeB="):
                    if key.lower() in text.lower():
                        tail = text.lower().split(key.lower(), 1)[1]
                        site = tail.split(",", 1)[0].split(";", 1)[0]
                        break
                sites.append(site)
            key = tuple(sorted(sites))
        else:
            key = (tuple(sorted(titles.items())), len(order["alarms"]))
        groups[key].append(i)
    folds = np.zeros(len(orders), dtype=np.int8)
    sizes = [0] * nfold
    for _, idx in sorted(groups.items(), key=lambda x: (-len(x[1]), repr(x[0]))):
        f = min(range(nfold), key=lambda j: (sizes[j], j))
        folds[idx] = f
        sizes[f] += len(idx)
    return folds


def rank_columns(score_dict: dict[str, np.ndarray], ptr: np.ndarray):
    keys = list(score_dict)
    vals = np.column_stack([score_dict[k] for k in keys]).astype(np.float32)
    ranks = np.zeros_like(vals)
    zvals = np.zeros_like(vals)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = vals[start:stop]
        for j in range(vals.shape[1]):
            idx = np.argsort(-local[:, j], kind="stable")
            ranks[start + idx, j] = np.arange(stop - start) / max(stop - start - 1, 1)
            zvals[start:stop, j] = (local[:, j] - local[:, j].mean()) / max(float(local[:, j].std()), 1e-6)
    return keys, vals, ranks, zvals


def pair_rows(orders, ptr, base, row_features, score_dict, top_add=10, bottom_remove=5):
    """Enumerate a controlled but broad set of within-order pairs."""
    keys, vals, ranks, zvals = rank_columns(score_dict, ptr)
    rows = []
    meta = []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        local_base = base[start:stop]
        selected = np.flatnonzero(local_base)
        unselected = np.flatnonzero(~local_base)
        if len(selected) == 0 or len(unselected) == 0:
            continue
        # Use average rank across score families to select candidate endpoints.
        avg = ranks[start:stop].mean(axis=1)
        rem = selected[np.argsort(avg[selected])[: min(bottom_remove, len(selected))]]
        add = unselected[np.argsort(avg[unselected])[: min(top_add, len(unselected))]]
        # Also retain top candidates from each score family; this prevents one
        # model's scale from hiding a potentially useful endpoint.
        rem_set, add_set = set(map(int, rem)), set(map(int, add))
        for j in range(vals.shape[1]):
            rem_set.update(map(int, selected[np.argsort(vals[start:stop, j][selected])[: min(2, len(selected))]]))
            add_set.update(map(int, unselected[np.argsort(-vals[start:stop, j][unselected])[: min(4, len(unselected))]]))
        rem = np.asarray(sorted(rem_set), dtype=np.int64)
        add = np.asarray(sorted(add_set), dtype=np.int64)
        # Features: endpoint vectors, difference/absolute difference, scores,
        # within-order ranks, and simple pair relation indicators.  The raw
        # semantic matrix is label-independent and available for test rows.
        for r in rem:
            for a in add:
                rf = row_features[start + r]
                af = row_features[start + a]
                diff = af - rf
                adiff = np.abs(diff)
                rel = np.r_[vals[start + a] - vals[start + r],
                            ranks[start + a] - ranks[start + r],
                            zvals[start + a] - zvals[start + r],
                            vals[start + a], vals[start + r],
                            ranks[start + a], ranks[start + r]]
                aa, rr = orders[oi]["alarms"][int(a)], orders[oi]["alarms"][int(r)]
                indicators = np.asarray([
                    float(str(aa.get("title", "")) == str(rr.get("title", ""))),
                    float(str(aa.get("reason", "")) == str(rr.get("reason", ""))),
                    float(str(aa.get("device", "")) == str(rr.get("device", ""))),
                    float(str(aa.get("location", "")) == str(rr.get("location", ""))),
                    float(a > r),
                    float(stop - start),
                ], dtype=np.float32)
                rows.append(np.r_[af, rf, diff, adiff, rel, indicators])
                meta.append((oi, int(r), int(a)))
    return np.asarray(rows, dtype=np.float32), meta


def pair_labels(meta, ptr, labels):
    y = np.empty(len(meta), dtype=np.int8)
    for i, (oi, r, a) in enumerate(meta):
        start = int(ptr[oi])
        y[i] = int(labels[start + a]) - int(labels[start + r])
    return y


def greedy_rank(meta, scores, labels, ptr, one_per_order=False, limit=200):
    order = np.argsort(-scores, kind="stable")
    used_order, used_r, used_a = set(), set(), set()
    out = []
    for q in order:
        oi, r, a = meta[int(q)]
        if one_per_order and oi in used_order:
            continue
        if (oi, r) in used_r or (oi, a) in used_a:
            continue
        if labels is None:
            delta = 0
        else:
            delta = int(labels[int(ptr[oi]) + a]) - int(labels[int(ptr[oi]) + r])
        out.append({"order_index": int(oi), "remove_local": int(r), "add_local": int(a),
                    "score": float(scores[int(q)]), "true_delta": delta})
        used_order.add(oi); used_r.add((oi, r)); used_a.add((oi, a))
        if len(out) >= limit:
            break
    return out


def curves(seq):
    result = {}
    for k in (5, 10, 20, 30, 40, 60, 100):
        s = seq[:k]
        result[str(k)] = {"delta": int(sum(a["true_delta"] for a in s)),
                          "positive": int(sum(a["true_delta"] > 0 for a in s)),
                          "negative": int(sum(a["true_delta"] < 0 for a in s)),
                          "mean": float(np.mean([a["true_delta"] for a in s])) if s else 0.0}
    return result


def model_for(kind, seed):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    if kind == "hist":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04,
            max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=2.0,
            random_state=seed)
    return ExtraTreesClassifier(n_estimators=180, min_samples_leaf=8,
        max_features=0.55, class_weight="balanced", n_jobs=-1,
        random_state=seed)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arr = dict(np.load(DATA, allow_pickle=False))
    tr = v10.load_orders(TRAIN_DIR, True); te = v10.load_orders(TEST_DIR, False)
    tr_ptr, te_ptr = arr["train_alarm_ptr"], arr["test_alarm_ptr"]
    y = arr["train_labels"].astype(np.int8)
    # Use only label-independent columns from the semantic matrix.  The
    # existing semantic array is explicitly documented as excluding labels.
    tr_static = np.nan_to_num(arr["train_alarm_x"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    te_static = np.nan_to_num(arr["test_alarm_x"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tr_scores = {"v11": arr["train_v11"], "v13": arr["train_v13"], "v19": arr["train_v19"]}
    te_scores = {"v11": arr["test_v11"], "v13": arr["test_v13"], "v19": arr["test_v19"]}
    for key, p, q in (("v16", "v16_oof_scores.npy", "v16_test_scores.npy"),
                      ("v16_graph", "v16_graph_time_oof.npy", "v16_graph_time_test_scores.npy")):
        tr_scores[key] = np.load(ROOT / "experiments/v16" / p); te_scores[key] = np.load(ROOT / "experiments/v16" / q)
    # Current baseline is read from the verified file, not inferred from an
    # unsubmitted model.
    tr_base = exact_mask(arr["train_v11"], tr_ptr, TRAIN_P)
    te_base = baseline_mask_for_orders(te, read_submission(CURRENT))
    print("base", int(tr_base.sum()), int((tr_base & (y == 1)).sum()), int(te_base.sum()), flush=True)
    # Candidate generation once; pair labels are deterministic.
    Xtr, mtr = pair_rows(tr, tr_ptr, tr_base, tr_static, tr_scores)
    ypair = pair_labels(mtr, tr_ptr, y)
    Xte, mte = pair_rows(te, te_ptr, te_base, te_static, te_scores)
    print("pairs", Xtr.shape, Xte.shape, "labels", Counter(map(int, ypair)), flush=True)
    report = {"train_pairs": int(len(ypair)), "test_pairs": int(len(mte)), "base_train_tp": int((tr_base & (y == 1)).sum()), "models": {}}
    for split_name in ("template", "site"):
        folds = order_folds(tr, split_name)
        pair_order = np.asarray([x[0] for x in mtr], dtype=np.int64)
        for kind in ("extra", "hist"):
            print("fit", split_name, kind, flush=True)
            oof_score = np.zeros(len(ypair), dtype=np.float32)
            test_scores = []
            for f in range(5):
                train_rows = folds[pair_order] != f
                val_rows = ~train_rows
                model = model_for(kind, SEED + f)
                # Binary objective: positive net gain vs all other outcomes.
                # Sample weights balance the rare +1 class without fabricating
                # labels; ranking remains based on calibrated positive odds.
                yy = (ypair == 1).astype(np.int8)
                w = np.where(yy == 1, 3.0, 1.0).astype(np.float32)
                model.fit(Xtr[train_rows], yy[train_rows], **({"sample_weight": w[train_rows]} if kind == "extra" else {}))
                oof_score[val_rows] = model.predict_proba(Xtr[val_rows])[:, 1]
                test_scores.append(model.predict_proba(Xte)[:, 1])
            test_score = np.mean(test_scores, axis=0).astype(np.float32)
            seq = greedy_rank(mtr, oof_score, y, tr_ptr, one_per_order=False, limit=200)
            uniq = greedy_rank(mtr, oof_score, y, tr_ptr, one_per_order=True, limit=200)
            # Pair labels and model ranking diagnostics.
            top = np.argsort(-oof_score)
            top_stats = {str(k): {"positive": int(np.sum(ypair[top[:k]] == 1)), "negative": int(np.sum(ypair[top[:k]] == -1)), "sum_delta": int(np.sum(ypair[top[:k]]))} for k in (10,20,30,40,60,100)}
            test_order = greedy_rank(mte, test_score, None, te_ptr, one_per_order=True, limit=200)
            # Test metadata only; preserve top candidate IDs for a downstream
            # audit, but do not emit a CSV.
            test_top = []
            for q in np.argsort(-test_score)[:100]:
                oi,r,a=mte[int(q)]; test_top.append({"order_index":int(oi),"order_id":te[oi]["id"],"remove_local":int(r),"add_local":int(a),"remove_rid":te[oi]["alarms"][r]["@rid"],"add_rid":te[oi]["alarms"][a]["@rid"],"score":float(test_score[int(q)])})
            report["models"][f"{split_name}_{kind}"]={"oof_top_stats":top_stats,"oof_feasible_curves":curves(seq),"oof_unique_curves":curves(uniq),"oof_positive_rate_top30":float(np.mean(ypair[top[:30]]==1)),"test_top_unique":test_top}
            np.save(OUT/f"pair_{split_name}_{kind}_oof.npy",oof_score); np.save(OUT/f"pair_{split_name}_{kind}_test.npy",test_score)
    (OUT/"pair_delta_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"out":str(OUT/"pair_delta_report.json"),"summary":{k:{"oof_unique":v["oof_unique_curves"],"top":v["oof_top_stats"]} for k,v in report["models"].items()}},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()

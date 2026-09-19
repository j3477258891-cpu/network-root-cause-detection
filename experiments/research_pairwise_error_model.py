"""Research-only baseline error-correction / pairwise swap audit.

This module is deliberately not a submission generator.  It builds a train-side
proxy for the current fixed-count baseline, trains cross-fitted node models,
enumerates same-order swaps on test, and writes only diagnostics/reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
V10_PATH = ROOT / "codexgz/work"
if str(V10_PATH) not in sys.path:
    sys.path.insert(0, str(V10_PATH))
import v10_grouped_ensemble as v10  # type: ignore


TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
CURRENT = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/research_pairwise_error_model"
MAX_ROOTS = 8
TEST_P = 1035
TRAIN_P = round(TEST_P / 546 * 1634)


def read_submission(path: Path):
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            data = json.loads(row["output"])
            result[row["order_id"]] = {x["@rid"] for x in data["rootcause"]}
    return result


def slices_from_ptr(ptr: np.ndarray):
    return [slice(int(a), int(b)) for a, b in zip(ptr[:-1], ptr[1:])]


def exact_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    """The same one-mandatory + global optional fixed-count rule used by V10."""
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = np.argsort(-scores[start:stop], kind="stable")[:MAX_ROOTS]
        if len(local):
            selected[start + int(local[0])] = True
            optional.extend((start + local[1:]).tolist())
    optional_arr = np.asarray(optional, dtype=np.int64)
    if len(optional_arr):
        optional_arr = optional_arr[np.argsort(-scores[optional_arr], kind="stable")]
    remaining = max(0, target - int(selected.sum()))
    selected[optional_arr[:remaining]] = True
    return selected


def baseline_mask_for_orders(orders, submission: dict[str, set[str]]):
    bits = []
    for order in orders:
        chosen = submission.get(order["id"], set())
        bits.extend(a.get("@rid") in chosen for a in order["alarms"])
    return np.asarray(bits, dtype=bool)


def site_of(value: str) -> str:
    m = re.search(r"(?:SubNetwork|STATION|NodeMe|gNodeB)=([^,;]+)", str(value or ""), re.I)
    return m.group(1) if m else "unknown"


def scalar(v):
    return str(v or "")


def string_key(a: dict, order: dict, include_position: bool = False) -> str:
    """Stable non-RID alarm signature; values with numeric IDs are normalized."""
    def norm(x):
        x = scalar(x)
        x = re.sub(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", "<UUID>", x)
        return re.sub(r"\d+", "#", x).strip()
    vals = [
        norm(a.get("title")), norm(a.get("reason")), norm(a.get("device")),
        norm(a.get("location")), norm(a.get("vendor")), norm(a.get("device_type")),
        norm(a.get("board_type")), norm(a.get("cause")), norm(a.get("radio")),
        norm(a.get("deployment")), norm(a.get("label")), norm(a.get("timeline")),
    ]
    if include_position:
        vals.append(str(a.get("_local_index", -1)))
    return "|".join(vals)


def local_features(orders, arrays: np.ndarray, scores: dict[str, np.ndarray], split: str):
    """Construct compact, pair-friendly features with within-order ranks."""
    ptr = arrays[f"{split}_alarm_ptr"]
    # Existing 252 static semantic features plus all score families and ranks.
    static = np.nan_to_num(arrays[f"{split}_alarm_x"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    columns = [
        scores[k].reshape(-1, 1).astype(np.float32)
        for k in ("v11", "v13", "v19")
        if k in scores
    ]
    # Include V16 and V25 scores when available.
    for k in ("v16", "v16_graph", "m1", "m2", "m3", "m4", "stack"):
        if k in scores:
            columns.append(scores[k].reshape(-1, 1).astype(np.float32))
    score_mat = np.hstack(columns) if columns else np.zeros((len(static), 0), dtype=np.float32)
    ranks = np.zeros_like(score_mat)
    zscores = np.zeros_like(score_mat)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        if stop <= start:
            continue
        x = score_mat[start:stop]
        for j in range(x.shape[1]):
            order = np.argsort(-x[:, j], kind="stable")
            ranks[start + order, j] = np.arange(stop - start, dtype=np.float32) / max(stop - start - 1, 1)
            zscores[start:stop, j] = (x[:, j] - x[:, j].mean()) / max(float(x[:, j].std()), 1e-6)
    counts = np.diff(ptr).astype(np.float32)
    order_count = np.repeat(counts, counts.astype(np.int64)).reshape(-1, 1)
    # Local position, title frequency, and simple categorical hashes.
    extras = np.zeros((len(static), 12), dtype=np.float32)
    off = 0
    for order in orders:
        n = len(order["alarms"])
        titles = Counter(scalar(a.get("title")) for a in order["alarms"])
        devices = Counter(scalar(a.get("device")) for a in order["alarms"])
        for j, a in enumerate(order["alarms"]):
            a["_local_index"] = j
            extras[off + j] = [
                j / max(n - 1, 1),
                float(titles[scalar(a.get("title"))]),
                float(devices[scalar(a.get("device"))]),
                float(len(str(a.get("title", "")))),
                float(len(str(a.get("reason", "")))),
                float(len(str(a.get("location", "")))),
                # Do not use the train-only topology ``label`` field.  Test
                # orders do not expose it; including it would make OOF
                # correction estimates invalidly optimistic.
                0.0,
                float(bool(a.get("timeLists"))),
                float(len(a.get("timeLists", [])) if isinstance(a.get("timeLists", []), list) else 0),
                float(hash(scalar(a.get("title"))) % 1009) / 1009.0,
                float(hash(site_of(a.get("location", ""))) % 1009) / 1009.0,
                float(order["topology"].get("time", 0) or 0) % (86400 * 30) / (86400 * 30),
            ]
        off += n
    return np.hstack([static, score_mat, ranks, zscores, extras, order_count]).astype(np.float32)


def grouped_order_folds(orders, n=5, mode="template"):
    groups = defaultdict(list)
    for i, o in enumerate(orders):
        titles = Counter(scalar(a.get("title")) for a in o["alarms"])
        if mode == "site":
            key = tuple(sorted(site_of(a.get("location", "")) for a in o["alarms"]))
        else:
            key = (tuple(sorted(titles.items())), len(o["alarms"]))
        groups[key].append(i)
    folds = np.zeros(len(orders), dtype=np.int8)
    sizes = [0] * n
    for _, idx in sorted(groups.items(), key=lambda z: (-len(z[1]), repr(z[0]))):
        f = min(range(n), key=lambda x: (sizes[x], x))
        folds[idx] = f; sizes[f] += len(idx)
    return folds


def fit_oof(X, y, ptr, folds, model_kind="extra"):
    """Cross-fit node probabilities; imports are delayed so diagnostics stay light."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    oof = np.zeros(len(y), dtype=np.float32)
    tests = []
    for fold in range(int(folds.max()) + 1):
        tr = order_rows != fold if False else (folds[order_rows] != fold)
        va = ~tr
        if model_kind == "hist":
            model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04,
                    max_leaf_nodes=31, min_samples_leaf=18, l2_regularization=1.0,
                    random_state=20260823)
        else:
            model = ExtraTreesClassifier(n_estimators=220, min_samples_leaf=3,
                    max_features=0.65, class_weight="balanced", n_jobs=-1,
                    random_state=20260823 + fold)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        tests.append(model)
    return oof, tests


def swap_actions(orders, ptr, base, scores):
    actions = []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        b = base[start:stop]
        if not b.any() or b.all():
            continue
        local = scores[start:stop]
        selected = np.flatnonzero(b)
        unselected = np.flatnonzero(~b)
        # Enumerate top/bottom candidates, not just one pair per order.
        for ri in selected[np.argsort(local[selected])[: min(4, len(selected))]]:
            for ai in unselected[np.argsort(-local[unselected])[: min(4, len(unselected))]]:
                actions.append({
                    "order_index": oi,
                    "order_id": orders[oi]["id"],
                    "remove_local": int(ri), "add_local": int(ai),
                    "remove_rid": orders[oi]["alarms"][int(ri)]["@rid"],
                    "add_rid": orders[oi]["alarms"][int(ai)]["@rid"],
                    "score_margin": float(local[ai] - local[ri]),
                })
    return actions


def eval_actions(actions, labels, ptr, base):
    out = []
    for a in actions:
        sl = slice(int(ptr[a["order_index"]]), int(ptr[a["order_index"] + 1]))
        out.append({**a, "true_delta": int(labels[sl.start + a["add_local"]] - labels[sl.start + a["remove_local"]])})
    return out


def greedy_action_curve(actions, labels, ptr, one_per_order=False, limit=100):
    """Evaluate a feasible prefix (no duplicate remove/add; optionally one/order)."""
    used_orders: set[int] = set()
    used_remove: set[tuple[int, int]] = set()
    used_add: set[tuple[int, int]] = set()
    chosen = []
    for a in sorted(actions, key=lambda x: x["score_margin"], reverse=True):
        oi = int(a["order_index"])
        rem = (oi, int(a["remove_local"]))
        add = (oi, int(a["add_local"]))
        if one_per_order and oi in used_orders:
            continue
        if rem in used_remove or add in used_add:
            continue
        sl = slice(int(ptr[oi]), int(ptr[oi + 1]))
        delta = int(labels[sl.start + a["add_local"]] - labels[sl.start + a["remove_local"]])
        chosen.append({**a, "true_delta": delta})
        used_orders.add(oi); used_remove.add(rem); used_add.add(add)
        if len(chosen) >= limit:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(DATA, allow_pickle=False))
    train_orders = v10.load_orders(TRAIN_DIR, True)
    test_orders = v10.load_orders(TEST_DIR, False)
    tr_ptr, te_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(np.int8)
    scores_tr = {
        "v11": arrays["train_v11"], "v13": arrays["train_v13"], "v19": arrays["train_v19"],
        "v16": np.load(ROOT / "experiments/v16/v16_oof_scores.npy"),
        "v16_graph": np.load(ROOT / "experiments/v16/v16_graph_time_oof.npy"),
    }
    scores_te = {
        "v11": arrays["test_v11"], "v13": arrays["test_v13"], "v19": arrays["test_v19"],
        "v16": np.load(ROOT / "experiments/v16/v16_test_scores.npy"),
        "v16_graph": np.load(ROOT / "experiments/v16/v16_graph_time_test_scores.npy"),
    }
    # Existing V25 model scores if available.
    for key in ("m1", "m2", "m3", "m4", "stack"):
        p = ROOT / "experiments/v25_ensemble/outputs" / f"v25_{key}_oof.npy"
        q = ROOT / "experiments/v25_ensemble/outputs" / f"v25_{key}_test.npy"
        if p.exists() and q.exists():
            scores_tr[key] = np.load(p); scores_te[key] = np.load(q)
    Xtr = local_features(train_orders, arrays, scores_tr, "train")
    Xte = local_features(test_orders, arrays, scores_te, "test")
    # Proxy baseline: V11 exact-count at test-proportional train count.
    base_tr = exact_mask(arrays["train_v11"], tr_ptr, TRAIN_P)
    base_te = baseline_mask_for_orders(test_orders, read_submission(CURRENT))
    print("shapes", Xtr.shape, Xte.shape, "base", base_tr.sum(), base_te.sum(), "labels", labels.sum(), flush=True)
    report = {"train_p": int(base_tr.sum()), "test_p": int(base_te.sum()), "train_tp": int((base_tr & (labels == 1)).sum()), "models": {}}
    for mode in ("template", "site"):
        folds = grouped_order_folds(train_orders, mode=mode)
        for kind in (("extra", "hist") if args.quick else ("extra", "hist")):
            print("fit", mode, kind, flush=True)
            oof, models = fit_oof(Xtr, labels, tr_ptr, folds, kind)
            # Evaluate fixed-count replacement by replacing baseline with top scores.
            new = exact_mask(oof, tr_ptr, TRAIN_P)
            tp = int((new & (labels == 1)).sum())
            actions = eval_actions(swap_actions(train_orders, tr_ptr, base_tr, oof), labels, tr_ptr, base_tr)
            actions.sort(key=lambda x: x["score_margin"], reverse=True)
            curves = {}
            for k in (5, 10, 20, 30, 40, 60, 100):
                ss = actions[:k]
                curves[str(k)] = {"sum_delta": int(sum(a["true_delta"] for a in ss)),
                                  "positive": int(sum(a["true_delta"] > 0 for a in ss)),
                                  "negative": int(sum(a["true_delta"] < 0 for a in ss)),
                                  "mean": float(np.mean([a["true_delta"] for a in ss])) if ss else 0.0}
            # Feasible, order-unique curves are the relevant stress test for a
            # submission campaign; the earlier raw curve can contain repeated
            # swaps on one order and is retained only for comparison.
            feasible = greedy_action_curve(actions, labels, tr_ptr, one_per_order=False, limit=200)
            unique = greedy_action_curve(actions, labels, tr_ptr, one_per_order=True, limit=200)
            feasible_curves, unique_curves = {}, {}
            for k in (5, 10, 20, 30, 40, 60, 100):
                for dest, seq in ((feasible_curves, feasible), (unique_curves, unique)):
                    ss = seq[:k]
                    dest[str(k)] = {
                        "sum_delta": int(sum(a["true_delta"] for a in ss)),
                        "positive": int(sum(a["true_delta"] > 0 for a in ss)),
                        "negative": int(sum(a["true_delta"] < 0 for a in ss)),
                        "mean": float(np.mean([a["true_delta"] for a in ss])) if ss else 0.0,
                    }
            # Fit full models and rank test swaps.  No files emitted.
            fold_models = models
            test_pred = np.mean([m.predict_proba(Xte)[:, 1] for m in fold_models], axis=0).astype(np.float32)
            test_actions = swap_actions(test_orders, te_ptr, base_te, test_pred)
            test_actions.sort(key=lambda x: x["score_margin"], reverse=True)
            report["models"][f"{mode}_{kind}"] = {
                "oof_tp": tp, "oof_delta_tp": tp - int((base_tr & (labels == 1)).sum()),
                "action_curves": curves,
                "feasible_action_curves": feasible_curves,
                "unique_order_curves": unique_curves,
                "feasible_top60": feasible[:60],
                "unique_top60": unique[:60],
                "test_candidate_count": len(test_actions),
                "test_top30": test_actions[:30],
                "test_top60_expected_positive_count": int(sum(1 for a in test_actions[:60] if a["score_margin"] > 0)),
            }
            np.save(OUT / f"{mode}_{kind}_oof.npy", oof)
            np.save(OUT / f"{mode}_{kind}_test.npy", test_pred)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT / 'report.json'), "summary": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

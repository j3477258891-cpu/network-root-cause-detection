"""Research-only semantic/template transfer audit.

This script searches for same-order replacement actions using labels from the
training rootcause files and evaluates them with template-grouped cross-fold
validation.  It deliberately does not write a submission CSV.  The output is
an auditable JSON report containing OOF swap curves and test candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT / "codexgz" / "work"))
import v10_grouped_ensemble as v10  # noqa: E402

OUT = ROOT / "experiments" / "research_semantic_transfer.json"
BASELINE = ROOT / "experiments" / "v60_combined_checkpoint" / "highest_verified_combined.csv"
V11_DIR = ROOT / "codexgz" / "v11"
N_FOLDS = 5
TRUE_ROOTS = 1044
TEST_P = 1035
TEST_ORDERS = 546
TRAIN_TARGET = round(TEST_P / TEST_ORDERS * 1634)

UUID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def norm(value: object, numbers: bool = True) -> str:
    text = "" if value is None else str(value)
    text = UUID_RE.sub("<UUID>", text)
    if numbers:
        text = NUM_RE.sub("#", text)
    text = SPACE_RE.sub(" ", text).strip().lower()
    return text


def alarm_keys(a: dict, local_index: int, order_len: int) -> dict[str, str]:
    title = norm(a.get("title"), False)
    reason = norm(a.get("reason"), False)
    loc = norm(a.get("location"), True)
    # A compact semantic key intentionally excludes location.  The full key
    # includes the normalized location shape; both are evaluated separately.
    sem = "|".join(
        norm(a.get(k), False)
        for k in ("title", "reason", "device_type", "board_type", "cause", "radio", "deployment")
    )
    title_reason = f"{title}|{reason}"
    # Some files carry a station identifier in location.  Removing the first
    # station-like token yields a portable shape while retaining rack/slot
    # structure.
    loc_portable = re.sub(r"(managedelement|station|nodeme|sbn)\s*=\s*#?[^,;]+", r"\1=<station>", loc)
    return {
        "sem": sem,
        "full": sem + "|loc=" + loc,
        "portable": sem + "|loc=" + loc_portable,
        "title_reason": title_reason,
        "title": title,
        "position": str(local_index),
        "position_ratio": f"{round(local_index / max(order_len - 1, 1), 2):.2f}",
    }


def order_signature(order: dict) -> tuple:
    counts = Counter(norm(a.get("title"), False) for a in order["alarms"])
    return (tuple(sorted(counts.items())), len(order["alarms"]))


def grouped_folds(orders: list[dict]) -> np.ndarray:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, o in enumerate(orders):
        groups[order_signature(o)].append(i)
    folds = np.zeros(len(orders), dtype=np.int8)
    sizes = [0] * N_FOLDS
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), hashlib.sha256(repr(kv[0]).encode()).hexdigest()))
    for _, ids in ranked:
        f = min(range(N_FOLDS), key=lambda x: (sizes[x], x))
        for i in ids:
            folds[i] = f
        sizes[f] += len(ids)
    return folds


def labels_for_orders(orders: list[dict]) -> list[np.ndarray]:
    return [np.asarray([int(a.get("@rid") in o.get("roots", set())) for a in o["alarms"]], dtype=np.int8) for o in orders]


def make_baseline_mask(orders: list[dict], score: np.ndarray, target: int, slices: list[slice]) -> np.ndarray:
    selected = np.zeros(len(score), dtype=bool)
    optional: list[int] = []
    for sl in slices:
        s, e = sl.start, sl.stop
        ranked = np.argsort(-score[s:e], kind="stable")[:8]
        if len(ranked):
            selected[s + int(ranked[0])] = True
            optional.extend((s + ranked[1:]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-score[optional], kind="stable")]
    rem = int(target - selected.sum())
    if rem < 0 or rem > len(optional):
        raise ValueError((target, int(selected.sum()), len(optional)))
    selected[optional[:rem]] = True
    return selected


def collect_stats(orders: list[dict], labels: list[np.ndarray], train_ids: np.ndarray):
    """Collect hierarchical counts for a set of orders."""
    levels = ("full", "portable", "sem", "title_reason", "title")
    stats = {k: defaultdict(lambda: [0, 0]) for k in levels}
    # Signature/relative-position key is kept separately.  We sort equal
    # semantic alarms by normalized location to make positions transferable.
    for oi in train_ids:
        o = orders[int(oi)]
        n = len(o["alarms"])
        keys = [alarm_keys(a, j, n) for j, a in enumerate(o["alarms"])]
        for j, (k, y) in enumerate(zip(keys, labels[int(oi)])):
            for level in levels:
                z = stats[level][k[level]]
                z[0] += 1
                z[1] += int(y)
        # Exact template + canonical position among alarms of same semantic
        # type. This is less brittle than the original input index.
        groups = defaultdict(list)
        for j, k in enumerate(keys):
            groups[k["sem"]].append(j)
        sig = repr(order_signature(o))
        for sem, js in groups.items():
            js_sorted = sorted(js, key=lambda j: (keys[j]["portable"], keys[j]["full"], j))
            for rank, j in enumerate(js_sorted):
                z = stats.setdefault("sigpos", defaultdict(lambda: [0, 0]))[(sig, sem, rank)]
                z[0] += 1
                z[1] += int(labels[int(oi)][j])
    return stats


def p_lookup(stats, keys: dict[str, str], sig: str, sem_rank: tuple | None, prior: float = 0.30) -> tuple[float, str, int]:
    # Hierarchical backoff.  Require at least two observations for a specific
    # key; otherwise use a broader key.  Laplace smoothing is conservative.
    candidates = []
    if sem_rank is not None and (sig, keys["sem"], sem_rank[1]) in stats["sigpos"]:
        candidates.append(("sigpos", (sig, keys["sem"], sem_rank[1]), 2.0))
    for level, weight in (("full", 5.0), ("portable", 4.0), ("sem", 3.0), ("title_reason", 2.0), ("title", 1.0)):
        if keys[level] in stats[level]:
            candidates.append((level, keys[level], weight))
    # Weighted blend of all available levels, but cap influence of sparse keys.
    vals = []
    for level, key, weight in candidates:
        n, pos = stats[level][key]
        if n <= 0:
            continue
        # stronger smoothing for sparse exact templates
        alpha = 2.0 if n < 8 else 1.0
        p = (pos + alpha * prior) / (n + 2 * alpha)
        vals.append((p, weight * min(1.0, n / 8.0), level, n))
    if not vals:
        return prior, "prior", 0
    num = sum(p * w for p, w, _, _ in vals)
    den = sum(w for _, w, _, _ in vals)
    # Use the most specific level with maximal support as provenance.
    best = max(vals, key=lambda x: (x[1], x[3]))
    return float(num / max(den, 1e-9)), best[2], int(best[3])


def predict_fold(orders, labels, baseline_mask_flat, slices, heldout_orders, train_ids):
    stats = collect_stats(orders, labels, train_ids)
    pred = []
    for oi in heldout_orders:
        o = orders[int(oi)]
        n = len(o["alarms"])
        keys = [alarm_keys(a, j, n) for j, a in enumerate(o["alarms"])]
        groups = defaultdict(list)
        for j, k in enumerate(keys):
            groups[k["sem"]].append(j)
        ranks = {}
        for sem, js in groups.items():
            js_sorted = sorted(js, key=lambda j: (keys[j]["portable"], keys[j]["full"], j))
            for rank, j in enumerate(js_sorted):
                ranks[j] = rank
        sl = slices[int(oi)]
        base = baseline_mask_flat[sl]
        for j, k in enumerate(keys):
            p, source, support = p_lookup(stats, k, repr(order_signature(o)), (k["sem"], ranks.get(j, 0)))
            pred.append((int(oi), j, p, source, support, bool(base[j])))
    return pred


def candidate_curve(preds, label_arrays, max_n=200):
    # One swap per order, using predicted p(add)-p(remove), then globally rank.
    by_order: dict[int, list] = defaultdict(list)
    for row in preds:
        by_order[row[0]].append(row)
    actions = []
    for oi, rows in by_order.items():
        adds = [r for r in rows if not r[5]]
        rems = [r for r in rows if r[5]]
        if not adds or not rems:
            continue
        # Keep top few pair alternatives, then choose highest delta.  We also
        # report all pairs for precision diagnostics.
        best = max(((a[2] - r[2], a, r) for a in adds for r in rems), key=lambda x: x[0])
        d, a, r = best
        y = label_arrays[oi]
        actual = int(y[a[1]]) - int(y[r[1]])
        actions.append({"order_index": oi, "pred_delta": float(d), "actual_delta": actual, "add_index": a[1], "remove_index": r[1], "add_p": a[2], "remove_p": r[2], "add_source": a[3], "remove_source": r[3]})
    actions.sort(key=lambda x: (-x["pred_delta"], x["order_index"]))
    out = {}
    for n in (5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200):
        z = actions[:n]
        if not z:
            continue
        vals = [x["actual_delta"] for x in z]
        out[str(n)] = {"n": len(z), "sum_delta_tp": int(sum(vals)), "mean_delta": float(np.mean(vals)), "positive": int(sum(v > 0 for v in vals)), "zero": int(sum(v == 0 for v in vals)), "negative": int(sum(v < 0 for v in vals)), "precision_positive": float(np.mean(np.asarray(vals) > 0)), "top_pred_min": float(z[-1]["pred_delta"])}
    return actions, out


def main():
    train = v10.load_orders(ROOT / "codexgz" / "train", True)
    test = v10.load_orders(ROOT / "codexgz" / "test", False)
    all_orders = train + test
    train_labels = labels_for_orders(train)
    # Flatten test baseline from the verified CSV (used only for test ranking).
    baseline_rows = {}
    for row in csv.DictReader(BASELINE.open(encoding="utf-8-sig")):
        baseline_rows[row["order_id"]] = {x["@rid"] for x in json.loads(row["output"])["rootcause"]}
    test_base = []
    for o in test:
        test_base.append(np.asarray([a.get("@rid") in baseline_rows[o["id"]] for a in o["alarms"]], dtype=bool))
    folds = grouped_folds(train)
    # Train baseline proxy from V11 OOF at the same alarm-density ratio as test.
    v11_oof = np.load(V11_DIR / "v11_oof_meta.npy") * 0.75 + np.load(V11_DIR / "v11_oof_context.npy") * 0.25
    train_slices = []
    p = 0
    for o in train:
        train_slices.append(slice(p, p + len(o["alarms"])))
        p += len(o["alarms"])
    flat_labels = np.concatenate(train_labels)
    base_flat = make_baseline_mask(train, v11_oof, TRAIN_TARGET, train_slices)
    base_tp = int(np.sum(base_flat & (flat_labels == 1)))
    fold_actions = []
    fold_curves = {}
    for f in range(N_FOLDS):
        val = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        preds = predict_fold(train, train_labels, base_flat, train_slices, val, tr)
        actions, curve = candidate_curve(preds, train_labels)
        fold_actions.extend(actions)
        fold_curves[str(f)] = curve
    # Aggregate OOF actions across folds; each order appears once.
    fold_actions.sort(key=lambda x: (-x["pred_delta"], x["order_index"]))
    agg_curve = {}
    for n in (5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200):
        z = fold_actions[:n]
        if z:
            vals = [x["actual_delta"] for x in z]
            agg_curve[str(n)] = {"n": len(z), "sum_delta_tp": int(sum(vals)), "mean_delta": float(np.mean(vals)), "positive": int(sum(v > 0 for v in vals)), "zero": int(sum(v == 0 for v in vals)), "negative": int(sum(v < 0 for v in vals)), "precision_positive": float(np.mean(np.asarray(vals) > 0)), "top_pred_min": float(z[-1]["pred_delta"])}
    # Full-train probabilities for test candidates.
    stats = collect_stats(train, train_labels, np.arange(len(train)))
    test_preds = []
    for oi, o in enumerate(test):
        n = len(o["alarms"])
        keys = [alarm_keys(a, j, n) for j, a in enumerate(o["alarms"])]
        groups = defaultdict(list)
        for j, k in enumerate(keys):
            groups[k["sem"]].append(j)
        ranks = {}
        for sem, js in groups.items():
            js_sorted = sorted(js, key=lambda j: (keys[j]["portable"], keys[j]["full"], j))
            for rank, j in enumerate(js_sorted):
                ranks[j] = rank
        for j, k in enumerate(keys):
            pval, source, support = p_lookup(stats, k, repr(order_signature(o)), (k["sem"], ranks.get(j, 0)))
            test_preds.append((oi, j, pval, source, support, bool(test_base[oi][j])))
    actions = []
    by_order = defaultdict(list)
    for x in test_preds:
        by_order[x[0]].append(x)
    for oi, rs in by_order.items():
        adds = [x for x in rs if not x[5]]
        rems = [x for x in rs if x[5]]
        if not adds or not rems:
            continue
        # keep best predicted pair and a few alternatives for inspection
        pairs = sorted(((a[2] - r[2], a, r) for a in adds for r in rems), key=lambda x: -x[0])
        for d, a, r in pairs[:3]:
            actions.append({"order_id": test[oi]["id"], "order_index": oi, "pred_delta": float(d), "add_rid": test[oi]["alarms"][a[1]].get("@rid"), "remove_rid": test[oi]["alarms"][r[1]].get("@rid"), "add_p": float(a[2]), "remove_p": float(r[2]), "add_source": a[3], "remove_source": r[3], "add_support": a[4], "remove_support": r[4]})
    actions.sort(key=lambda x: -x["pred_delta"])
    report = {"version": "semantic-template-transfer-1", "baseline": {"path": str(BASELINE), "test_p": int(sum(map(np.sum, test_base))), "train_proxy_target": TRAIN_TARGET, "train_proxy_tp": base_tp}, "cv": {"folds": fold_curves, "aggregate": agg_curve, "n_actions": len(fold_actions)}, "test_candidates": actions[:300]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "baseline": report["baseline"], "cv_aggregate": agg_curve, "top_test": actions[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

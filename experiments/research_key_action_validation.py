"""Research-only action-level validation for descriptor-key candidates.

The earlier alarm-key report measured descriptor rates globally.  This script
tests whether those rates remain useful when they are used to *change a
fixed-count prediction set*: all key tables are fit without the validation
orders, and the current ranker mask is held fixed.  It emits no leaderboard
submission and never treats the test labels as known.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
NPZ = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
V30 = ROOT / "experiments/v30_meta_stack"
KEY_REPORT = ROOT / "experiments/research_alarm_key_candidates.json"
FIXED = ROOT / "experiments/research_equation_audit/fixed_labels.json"
OUT = ROOT / "experiments/research_key_action_validation"
TRUE_ROOTS = 3041
TEST_TRUE_ROOTS = 1044


def text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(text(x) for x in v)
    return str(v)


def norm(v: Any) -> str:
    # Keep semantic descriptors but remove identifiers that cannot transfer.
    return re.sub(r"\d+", "#", text(v)).strip().lower()


KINDS = [
    "full_no_location",
    "title_reason_cause_timeline",
    "title_cause_timeline",
    "title_reason",
    "title_label",
    "reason_label",
]


def key(a: dict[str, Any], kind: str) -> tuple[str, ...]:
    title, reason = norm(a.get("title")), norm(a.get("reason"))
    cause, timeline = norm(a.get("cause")), norm(a.get("timeline"))
    label, device, board = norm(a.get("label")), norm(a.get("device_type")), norm(a.get("board_type"))
    radio, deployment = norm(a.get("radio")), norm(a.get("deployment"))
    neigh = tuple(sorted(norm(x) for x in (a.get("neighbor_titles") or [])))
    if kind == "title_reason_cause_timeline":
        return title, reason, cause, timeline, label
    if kind == "full_no_location":
        return title, reason, cause, timeline, label, device, board, radio, deployment, "|".join(neigh)
    if kind == "title_cause_timeline":
        return title, cause, timeline, label
    if kind == "title_reason":
        return title, reason, label
    if kind == "title_label":
        return title, label
    if kind == "reason_label":
        return reason, label
    raise ValueError(kind)


def load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        d = json.load(f)
    z = np.load(NPZ)
    return d["train"], d["test"], {k: z[k] for k in z.files}


def flat_orders(orders: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    ptr = [0]
    oi = []
    for j, order in enumerate(orders):
        for alarm in order.get("alarms", []):
            rows.append(alarm)
            oi.append(j)
        ptr.append(len(rows))
    return rows, np.asarray(ptr, dtype=np.int64), np.asarray(oi, dtype=np.int64)


def load_submission(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["order_id"]] = {x["@rid"] for x in json.loads(row["output"])["rootcause"]}
    return out


def baseline_mask(test: list[dict[str, Any]], rows: list[dict[str, Any]]) -> np.ndarray:
    selected = load_submission(BASE)
    m = np.zeros(len(rows), dtype=bool)
    for i, (order, alarm) in enumerate(zip((o for o in test for _ in o.get("alarms", [])), rows)):
        if alarm["rid"] in selected.get(order["order_id"], set()):
            m[i] = True
    return m


def exact_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for s, t in zip(ptr[:-1], ptr[1:]):
        s, t = int(s), int(t)
        order = np.argsort(-scores[s:t], kind="stable")
        if len(order):
            selected[s + order[0]] = True
            optional.extend((s + order[1:8]).tolist())
    opts = np.asarray(optional, dtype=np.int64)
    opts = opts[np.argsort(-scores[opts], kind="stable")]
    remain = target - int(selected.sum())
    if remain < 0 or remain > len(opts):
        raise ValueError((target, int(selected.sum()), len(opts)))
    selected[opts[:remain]] = True
    return selected


def stats_for(orders: list[dict[str, Any]], fit_indices: np.ndarray, kind: str, alpha: float, prior: float) -> dict[tuple[str, ...], tuple[int, int]]:
    stats: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
    for oi in fit_indices:
        for alarm in orders[int(oi)].get("alarms", []):
            k = key(alarm, kind)
            stats[k][0] += int(alarm.get("is_root") or 0)
            stats[k][1] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


def predict(orders: list[dict[str, Any]], rows: list[dict[str, Any]], oi: np.ndarray, stats: dict[tuple[str, ...], tuple[int, int]], kind: str, alpha: float, prior: float) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float64)
    for i, alarm in enumerate(rows):
        pos, n = stats.get(key(alarm, kind), (0, 0))
        out[i] = (pos + alpha * prior) / (n + alpha)
    return out


def wilson_upper(k: int, n: int, z: float = 1.645) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return min(1.0, c + h)


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h)


def remove_ranked(mask: np.ndarray, p: np.ndarray, oi: np.ndarray, k: int) -> np.ndarray:
    """Remove k lowest-p selected nodes without emptying an order."""
    out = mask.copy()
    counts = np.bincount(oi, weights=out.astype(np.int8), minlength=int(oi.max()) + 1)
    candidates = np.flatnonzero(out)
    candidates = candidates[np.argsort(p[candidates], kind="stable")]
    removed = 0
    for i in candidates:
        o = int(oi[i])
        if counts[o] <= 1:
            continue
        out[i] = False
        counts[o] -= 1
        removed += 1
        if removed >= k:
            break
    return out


def add_ranked(mask: np.ndarray, p: np.ndarray, oi: np.ndarray, ptr: np.ndarray, k: int, max_roots: int = 8) -> np.ndarray:
    out = mask.copy()
    counts = np.bincount(oi, weights=out.astype(np.int8), minlength=int(oi.max()) + 1)
    candidates = np.flatnonzero(~out)
    candidates = candidates[np.argsort(-p[candidates], kind="stable")]
    added = 0
    for i in candidates:
        o = int(oi[i])
        if counts[o] >= max_roots:
            continue
        out[i] = True
        counts[o] += 1
        added += 1
        if added >= k:
            break
    return out


def metrics(mask: np.ndarray, y: np.ndarray, true_roots: int = TRUE_ROOTS) -> dict[str, float | int]:
    tp = int(np.sum(mask & y)); p = int(mask.sum())
    return {"tp": tp, "p": p, "f1": 2 * tp / (true_roots + p), "delta_tp": tp}


def eval_curve(mask: np.ndarray, p: np.ndarray, oi: np.ndarray, y: np.ndarray, ks: list[int], label: str) -> list[dict[str, Any]]:
    base = metrics(mask, y)
    out = []
    for k in ks:
        m = remove_ranked(mask, p, oi, k)
        q = metrics(m, y)
        q.update({"action": "delete", "k": k, "label": label, "base_tp": base["tp"], "net_gain": q["tp"] - base["tp"]})
        out.append(q)
        m = add_ranked(mask, p, oi, np.array([], dtype=np.int64), k)
        q = metrics(m, y)
        q.update({"action": "add", "k": k, "label": label, "base_tp": base["tp"], "net_gain": q["tp"] - base["tp"]})
        out.append(q)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, a = load_data()
    trrows, trptr, troi = flat_orders(train)
    terows, teptr, teoi = flat_orders(test)
    y = a["train_labels"].astype(bool)
    # Current V30 consensus is a reproducible cross-fitted baseline analogue.
    score_sets = {
        "v11": a["train_v11"].astype(float),
        "v30": np.load(V30 / "v30_consensus_oof.npy").astype(float),
        "v13": a["train_v13"].astype(float),
        "v19": a["train_v19"].astype(float),
    }
    test_scores = {
        "v11": a["test_v11"].astype(float),
        "v30": np.load(V30 / "v30_consensus_test.npy").astype(float),
        "v13": a["test_v13"].astype(float),
        "v19": a["test_v19"].astype(float),
    }
    folds = a["train_station_folds"].astype(int)
    prior = float(y.mean())
    # Evaluate both the current-count analogue (3097) and older 1059 analogue.
    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for base_name, scores in score_sets.items():
        for target in (3097, 3169):
            mask = exact_mask(scores, trptr, target)
            base = metrics(mask, y)
            for kind in KINDS:
                # out-of-fold key posterior for every row
                poof = np.zeros(len(trrows), dtype=float)
                for fold in range(5):
                    fit = np.flatnonzero(folds != fold)
                    val_orders = np.flatnonzero(folds == fold)
                    stats = stats_for(train, fit, kind, 10.0, prior)
                    val_rows = np.flatnonzero(np.isin(troi, val_orders))
                    poof[val_rows] = predict(train, [trrows[int(i)] for i in val_rows], troi[val_rows], stats, kind, 10.0, prior)
                label = f"{base_name}_{target}_{kind}"
                predictions[label] = poof
                for k in (5, 10, 20, 30, 40, 50, 60, 80, 100):
                    m = remove_ranked(mask, poof, troi, k)
                    q = metrics(m, y)
                    results.append({"label": label, "base": base, "k": k, "action": "delete", "net_gain": q["tp"] - base["tp"], **q})
                for k in (5, 10, 20, 30, 40, 50):
                    m = add_ranked(mask, poof, troi, trptr, k)
                    q = metrics(m, y)
                    results.append({"label": label, "base": base, "k": k, "action": "add", "net_gain": q["tp"] - base["tp"], **q})
    # A simple rank-ensemble of the most stable descriptor keys.
    for base_name, scores in score_sets.items():
        for target in (3097, 3169):
            mask = exact_mask(scores, trptr, target); base = metrics(mask, y)
            chosen = []
            for kind in ("full_no_location", "title_reason_cause_timeline", "title_label", "reason_label"):
                label = f"{base_name}_{target}_{kind}"
                chosen.append(predictions[label])
            p = np.mean(chosen, axis=0)
            label = f"{base_name}_{target}_ensemble4"
            for k in (5, 10, 20, 30, 40, 50, 60, 80, 100):
                m = remove_ranked(mask, p, troi, k); q = metrics(m, y)
                results.append({"label": label, "base": base, "k": k, "action": "delete", "net_gain": q["tp"] - base["tp"], **q})

    # Test-time ranked action catalog using all train labels.  This is a
    # candidate ranking only; no label is inferred for test nodes here.
    base_test_mask = np.zeros(len(terows), dtype=bool)
    submitted = load_submission(BASE)
    for i, (order, alarm) in enumerate(zip((o for o in test for _ in o.get("alarms", [])), terows)):
        base_test_mask[i] = alarm["rid"] in submitted.get(order["order_id"], set())
    test_catalog: list[dict[str, Any]] = []
    fixed = {}
    if FIXED.exists():
        fixed = {(x["order_id"], x["rid"]): x.get("label") for x in json.loads(FIXED.read_text(encoding="utf8")).get("labels", [])}
    for kind in KINDS:
        stats = stats_for(train, np.arange(len(train)), kind, 10.0, prior)
        p = predict(test, terows, teoi, stats, kind, 10.0, prior)
        for i, alarm in enumerate(terows):
            role = "delete" if base_test_mask[i] else "add"
            pos, n = stats.get(key(alarm, kind), (0, 0))
            if n < 20:
                continue
            upper = wilson_upper(n - pos, n) if role == "delete" else None
            lower = wilson_lower(pos, n) if role == "add" else None
            if role == "delete" and upper is not None and upper <= 0.15:
                test_catalog.append({"order_id": test[int(teoi[i])]["order_id"], "rid": alarm["rid"], "role": role, "kind": kind, "p_root": float(p[i]), "support": n, "upper90_fp": float(upper), "fixed_label": fixed.get((test[int(teoi[i])]["order_id"], alarm["rid"]))})
            if role == "add" and lower is not None and lower >= 0.80:
                test_catalog.append({"order_id": test[int(teoi[i])]["order_id"], "rid": alarm["rid"], "role": role, "kind": kind, "p_root": float(p[i]), "support": n, "lower90_root": float(lower), "fixed_label": fixed.get((test[int(teoi[i])]["order_id"], alarm["rid"]))})
    # strongest record per node, excluding equation-fixed wrong direction
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for c in test_catalog:
        if c["fixed_label"] is not None and ((c["role"] == "delete" and c["fixed_label"] == 1) or (c["role"] == "add" and c["fixed_label"] == 0)):
            continue
        k = (c["order_id"], c["rid"])
        strength = (-(c["upper90_fp"] if c["role"] == "delete" else 1 - c["lower90_root"]), c["support"])
        old = best.get(k)
        old_strength = (-(old["upper90_fp"] if old["role"] == "delete" else 1 - old["lower90_root"]), old["support"]) if old else None
        if old is None or strength > old_strength:
            best[k] = c
    dels = sorted([x for x in best.values() if x["role"] == "delete"], key=lambda x: (x["upper90_fp"], -x["support"]))
    adds = sorted([x for x in best.values() if x["role"] == "add"], key=lambda x: (-x["lower90_root"], -x["support"]))
    # Build disjoint same-order swap suggestions from calibrated p_root.
    by_order_del: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_order_add: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in dels: by_order_del[x["order_id"]].append(x)
    for x in adds: by_order_add[x["order_id"]].append(x)
    swaps = []
    for oid in sorted(set(by_order_del) & set(by_order_add)):
        for d in by_order_del[oid][:3]:
            for ad in by_order_add[oid][:3]:
                swaps.append({"order_id": oid, "remove_rid": d["rid"], "add_rid": ad["rid"], "p_remove": d["p_root"], "p_add": ad["p_root"], "expected_delta": ad["p_root"] - d["p_root"], "remove": d, "add": ad})
    swaps.sort(key=lambda x: -x["expected_delta"])
    report = {
        "version": 1,
        "data": {"train_orders": len(train), "test_orders": len(test), "train_labels": int(y.sum()), "station_folds": sorted(set(folds.tolist()))},
        "results_sorted": sorted(results, key=lambda x: (x["action"] != "delete", -x["net_gain"], x["k"]))[:250],
        "best_delete_by_k": {},
        "best_add_by_k": {},
        "test_catalog": {"count": len(best), "deletions": dels[:200], "additions": adds[:200], "swaps": swaps[:200]},
        "notes": ["All action curves are train OOF analogues; they are not leaderboard scores.", "Test catalog is a ranking only and is not emitted as a submission."],
    }
    for k in (5, 10, 20, 30, 40, 50, 60, 80, 100):
        rs = [x for x in results if x["action"] == "delete" and x["k"] == k]
        if rs:
            report["best_delete_by_k"][str(k)] = sorted(rs, key=lambda x: -x["net_gain"])[:12]
    for k in (5, 10, 20, 30, 40, 50):
        rs = [x for x in results if x["action"] == "add" and x["k"] == k]
        if rs:
            report["best_add_by_k"][str(k)] = sorted(rs, key=lambda x: -x["net_gain"])[:12]
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_delete_by_k": report["best_delete_by_k"], "test_counts": {k: len(v) if isinstance(v, list) else v for k,v in report["test_catalog"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

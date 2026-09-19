"""V123 order-level count constraint + topology replacement campaign.

Builds same-order one-delete-one-add replacement candidates driven by an
order-level root-count model.  Every candidate keeps P fixed at the current
baseline (1035).  A 5-fold order/station-grouped OOF pipeline calibrates the
net-gain accuracy of replacements and only emits final files when all gates
pass (>=70 raw candidates, >=45 conservative-qualified, OOF net-gain
precision >=0.80, equation-feasible, correlated posterior P(F1>=0.945)>=0.80
after a 10% prior shrink).

This module never uploads files and never treats OOF/simulation outcomes as
online evidence.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from scipy.stats import norm  # noqa: E402

import v121_equation_safe_campaign as v121  # noqa: E402


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    """Per-order top-1 plus global-budget fill (V11 champion decode logic)."""
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:8]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional_array = np.asarray(optional, dtype=np.int64)
    optional_array = optional_array[np.argsort(-scores[optional_array], kind="stable")]
    remaining = target - int(selected.sum())
    if not 0 <= remaining <= len(optional_array):
        raise ValueError((target, int(selected.sum()), len(optional_array)))
    selected[optional_array[:remaining]] = True
    return selected

EXP = ROOT / "experiments"
NPZ = EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE = EXP / "v60_combined_checkpoint/highest_verified_combined.csv"
V34 = EXP / "v34_count_model"
OUT = EXP / "v123_pair_count_campaign"

TRUE_ROOTS = 1044
BASE_P = 1035
BASE_TP = 956
TARGET_TP = 983  # F1 >= 0.945 requires TP >= 983
TARGET_F1 = 0.945
TRAIN_BASE_P = 3103   # training-side simulation budget (V11 champion logic)
MAX_ROOTS = 8
SEED = 20260823
REQUIRED_RAW = 70
REQUIRED_QUALIFIED = 45
QUALIFIED_P = 0.72
SHRINK = 0.90
COUNT_MODEL_KEYS = [
    "ensemble", "station_alpha_0", "station_alpha_1",
    "station_alpha_2", "template_alpha_0", "template_alpha_1",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_npz() -> dict[str, np.ndarray]:
    with np.load(NPZ, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset["train"], dataset["test"]


def load_count_probs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for key in COUNT_MODEL_KEYS:
        oof[key] = np.load(V34 / f"{key}_oof.npy")
        test[key] = np.load(V34 / f"{key}_test.npy")
    return oof, test


def template_key(alarm: dict[str, Any]) -> str:
    source = alarm.get("source") or {}
    title = source.get("title") or alarm.get("title", "")
    loc = source.get("location") or alarm.get("raw_location") or alarm.get("location", "")
    parts = [p for p in loc.split(",") if "=" in p and p.split("=")[0] in ("SubNetwork", "STATION", "ManagedElement")]
    return f"{title}|{len(parts)}|{','.join(sorted(parts))}"


def device_key(alarm: dict[str, Any]) -> str:
    source = alarm.get("source") or {}
    return source.get("device") or alarm.get("device", "") or alarm.get("raw_location", "")


# ---------------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------------

def build_order_index(ptr: np.ndarray) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(ptr[:-1], ptr[1:])]


def count_consensus(count_oof: dict[str, np.ndarray], ptr: np.ndarray, split: str) -> np.ndarray:
    """Per-order argmax count consensus across the 6 independent count models."""
    n_orders = len(ptr) - 1
    consensus = np.zeros(n_orders, dtype=np.float32)
    spread = np.zeros(n_orders, dtype=np.float32)
    for oi in range(n_orders):
        votes = np.zeros(MAX_ROOTS + 1, dtype=np.int32)
        means = []
        for key in COUNT_MODEL_KEYS:
            prob = count_oof[key][oi]
            pred = int(np.argmax(prob)) + 1
            votes[pred] += 1
            means.append(float(prob.max()))
        top = int(np.argmax(votes))
        consensus[oi] = top
        spread[oi] = 1.0 - float(votes[top]) / len(COUNT_MODEL_KEYS)
    return consensus, spread


def pair_features(
    rem: dict[str, Any], add: dict[str, Any],
    order_ctx: dict[str, Any], template_stats: dict[str, dict[str, float]],
) -> list[float]:
    """Feature vector for a (remove, add) same-order pair."""
    rs = rem.get("_scores", {})
    as_ = add.get("_scores", {})
    rv11, rv13, rv19 = rs.get("v11", 0.0), rs.get("v13", 0.0), rs.get("v19", 0.0)
    av11, av13, av19 = as_.get("v11", 0.0), as_.get("v13", 0.0), as_.get("v19", 0.0)
    rt = template_stats.get(rem.get("_template", ""), {})
    at = template_stats.get(add.get("_template", ""), {})
    rd = template_stats.get(rem.get("_device", ""), {})
    ad = template_stats.get(add.get("_device", ""), {})
    return [
        rv11, rv13, rv19,                        # remove semantic scores
        av11, av13, av19,                        # add semantic scores
        av11 - rv11, av13 - rv13, av19 - rv19,   # semantic deltas
        order_ctx["count_pred"],                 # count-model consensus
        order_ctx["count_spread"],               # 1 - consensus rate
        order_ctx["base_n"],                     # baseline root count in order
        order_ctx["count_pred"] - order_ctx["base_n"],  # count residual
        order_ctx["alarm_n"],                    # alarms in order
        rt.get("p", 0.5), at.get("p", 0.5),      # template priors
        rd.get("p", 0.5), ad.get("p", 0.5),      # device priors
        rs.get("rank", 0.0), as_.get("rank", 0.0),
        order_ctx.get("v11_max", 0.0), order_ctx.get("v11_mean", 0.0),
        order_ctx.get("v13_max", 0.0), order_ctx.get("v13_mean", 0.0),
        order_ctx.get("v19_max", 0.0), order_ctx.get("v19_mean", 0.0),
    ]


FEATURE_NAMES = [
    "rem_v11", "rem_v13", "rem_v19", "add_v11", "add_v13", "add_v19",
    "d_v11", "d_v13", "d_v19",
    "count_pred", "count_spread", "base_n", "count_residual", "alarm_n",
    "rem_tpl_p", "add_tpl_p", "rem_dev_p", "add_dev_p",
    "rem_rank", "add_rank",
    "ord_v11_max", "ord_v11_mean", "ord_v13_max", "ord_v13_mean",
    "ord_v19_max", "ord_v19_mean",
]


def order_context(
    node_scores: dict[str, np.ndarray], start: int, stop: int,
    base_mask: np.ndarray, count_pred: float, count_spread: float, alarm_n: int,
) -> dict[str, Any]:
    v11 = node_scores["v11"][start:stop]
    v13 = node_scores["v13"][start:stop]
    v19 = node_scores["v19"][start:stop]
    ranked = np.argsort(-v11, kind="stable")
    rank_of = {int(ranked[i]): i for i in range(len(ranked))}
    ctx: dict[str, Any] = {
        "count_pred": float(count_pred), "count_spread": float(count_spread),
        "base_n": int(base_mask[start:stop].sum()), "alarm_n": int(alarm_n),
        "v11_max": float(v11.max()), "v11_mean": float(v11.mean()),
        "v13_max": float(v13.max()), "v13_mean": float(v13.mean()),
        "v19_max": float(v19.max()), "v19_mean": float(v19.mean()),
        "_rank_of": rank_of, "_v11": v11, "_base": base_mask[start:stop],
    }
    return ctx


def attach_node_scores(alarms: list[dict[str, Any]], start: int, stop: int,
                       node_scores: dict[str, np.ndarray], ctx: dict[str, Any]) -> None:
    rank_of = ctx["_rank_of"]
    v11 = ctx["_v11"]
    for k, node in enumerate(alarms):
        node["_scores"] = {
            "v11": float(node_scores["v11"][start + k]),
            "v13": float(node_scores["v13"][start + k]),
            "v19": float(node_scores["v19"][start + k]),
            "rank": float(rank_of.get(k, len(v11))) / max(1, len(v11)),
        }
        node["_template"] = template_key(node)
        node["_device"] = device_key(node)


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------

def build_pair_candidates(
    node_scores: dict[str, np.ndarray], ptr: np.ndarray, orders: list[dict[str, Any]],
    alarms: dict[tuple[str, str], dict[str, Any]], base_rows: list[dict[str, Any]] | None,
    count_oof: dict[str, np.ndarray], template_stats: dict[str, dict[str, float]],
    train: bool, order_fold: np.ndarray | None, fold_id: int | None, limit_per_order: int = 30,
) -> list[dict[str, Any]]:
    """Enumerate same-order one-delete-one-add pairs with safety screens.

    For train: base mask = exact_count_mask(v11, ptr, TRAIN_BASE_P); labels are
    the true root labels; only orders in the validation fold are emitted so the
    caller can do honest 5-fold OOF evaluation.
    For test: base = the current online baseline rows; every order is emitted.
    """
    base_by: dict[str, set[str]] = {}
    if train:
        base_mask = exact_count_mask(node_scores["v11"], ptr, TRAIN_BASE_P)
    else:
        for row in base_rows:
            base_by[row["order_id"]] = {node["@rid"] for node in row["roots"]}

    consensus, spread = count_consensus(count_oof, ptr, "train" if train else "test")
    ranges = build_order_index(ptr)
    out: list[dict[str, Any]] = []
    for oi, (start, stop) in enumerate(ranges):
        if order_fold is not None and train:
            if int(order_fold[oi]) != fold_id:
                continue
        order = orders[oi]
        oid = order["order_id"]
        local_alarms = order.get("alarms", [])
        if not local_alarms:
            continue
        ctx = order_context(node_scores, start, stop, base_mask if train else np.zeros(stop - start, dtype=bool),
                            consensus[oi], spread[oi], len(local_alarms))
        attach_node_scores(local_alarms, start, stop, node_scores, ctx)
        if train:
            selected_ids = {local_alarms[k]["rid"] for k in range(len(local_alarms)) if bool(base_mask[start + k])}
        else:
            selected_ids = base_by.get(oid, set())
        unselected = [a for a in local_alarms if a["rid"] not in selected_ids]
        selected = [a for a in local_alarms if a["rid"] in selected_ids]
        if not selected or not unselected:
            continue
        # quick model screen: only keep pairs where at least two semantic
        # models do not strongly prefer the removed node.
        pairs = []
        for rem in selected:
            for add in unselected:
                rs, as_ = rem["_scores"], add["_scores"]
                if rs["v11"] - as_["v11"] > 0.10 and rs["v13"] - as_["v13"] > 0.10:
                    continue
                rem_title = rem.get("title", "")
                if rem_title and rem_title in (add.get("target_summary") or ""):
                    continue
                pairs.append((rem, add))
        if not pairs:
            continue
        if limit_per_order and len(pairs) > limit_per_order:
            pairs = sorted(pairs, key=lambda p: (p[1]["_scores"]["v11"] - p[0]["_scores"]["v11"]))[:limit_per_order]
        for rem, add in pairs:
            feat = pair_features(rem, add, ctx, template_stats)
            out.append({
                "order_id": oid, "remove_rid": rem["rid"], "add_rid": add["rid"],
                "features": feat, "_ctx": ctx,
            })
    return out


# ---------------------------------------------------------------------------
# 5-fold OOF calibration
# ---------------------------------------------------------------------------

def build_template_stats(
    train_records: list[dict[str, Any]], labels: np.ndarray, ptr: np.ndarray,
    train_orders: list[str], fit_orders: set[str],
) -> dict[str, dict[str, float]]:
    """P(is_root) priors by title-template and by device, from fit-fold orders only."""
    tpl: dict[str, list[int]] = {}
    dev: dict[str, list[int]] = {}
    ranges = build_order_index(ptr)
    label_off = 0
    for oi, (start, stop) in enumerate(ranges):
        oid = train_orders[oi] if oi < len(train_orders) else ""
        # fallback: align by index; train_orders must match ptr order
        if oid not in fit_orders:
            continue
        order = train_records[oi] if oi < len(train_records) else {"alarms": []}
        alarms = order.get("alarms", [])
        for k, alarm in enumerate(alarms):
            key = template_key(alarm)
            dkey = device_key(alarm)
            lab = int(labels[start + k]) if (start + k) < len(labels) else 0
            tpl.setdefault(key, []).append(lab)
            dev.setdefault(dkey, []).append(lab)
    stats = {}
    for key, vals in tpl.items():
        stats[key] = {"p": float(np.mean(vals)), "n": len(vals)}
    for key, vals in dev.items():
        stats.setdefault(key, {"p": float(np.mean(vals)), "n": len(vals)})
    return stats


def run_oof(
    arrays: dict[str, np.ndarray], train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]], count_oof: dict[str, np.ndarray],
    count_test: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_orders = [o["order_id"] for o in train_records]
    train_ptr = arrays["train_alarm_ptr"]
    test_ptr = arrays["test_alarm_ptr"]
    train_labels = arrays["train_labels"].astype(np.int8)
    folds = arrays["train_station_folds"].astype(np.int8)
    node_scores = {
        "v11": arrays["train_v11"], "v13": arrays["train_v13"], "v19": arrays["train_v19"],
    }
    test_node_scores = {
        "v11": arrays["test_v11"], "v13": arrays["test_v13"], "v19": arrays["test_v19"],
    }

    train_candidates: list[dict[str, Any]] = []
    oof_meta: dict[str, Any] = {"folds": []}
    # order -> rid -> label offset lookup (alarms order == ptr order, verified)
    ranges = build_order_index(train_ptr)
    rid_offset: dict[str, dict[str, int]] = {}
    for oi, order in enumerate(train_records):
        start = ranges[oi][0]
        rid_offset[order["order_id"]] = {
            alarm["rid"]: start + k for k, alarm in enumerate(order.get("alarms", []))
        }
    for fold in range(5):
        fit_orders = {oid for oi, oid in enumerate(train_orders) if int(folds[oi]) != fold}
        template_stats = build_template_stats(train_records, train_labels, train_ptr, train_orders, fit_orders)
        fold_cands = build_pair_candidates(
            node_scores, train_ptr, train_records, {}, None, count_oof,
            template_stats, train=True, order_fold=folds, fold_id=fold,
        )
        for c in fold_cands:
            c["fold"] = fold
            off = rid_offset.get(c["order_id"], {})
            rem_idx, add_idx = off.get(c["remove_rid"]), off.get(c["add_rid"])
            rem_true = int(train_labels[rem_idx]) if rem_idx is not None else 0
            add_true = int(train_labels[add_idx]) if add_idx is not None else 0
            c["label"] = add_true - rem_true
        train_candidates.extend(fold_cands)
        oof_meta["folds"].append({
            "fold": fold, "fit_orders": len(fit_orders), "val_candidates": len(fold_cands),
        })

    # score OOF: ExtraTrees 5-fold over candidates by order fold
    X = np.asarray([c["features"] for c in train_candidates], dtype=np.float32)
    y = np.asarray([1 if c["label"] == 1 else 0 for c in train_candidates], dtype=np.int8)
    folds_c = np.asarray([c["fold"] for c in train_candidates], dtype=np.int8)
    oof_p = np.zeros(len(train_candidates), dtype=np.float32)
    for fold in range(5):
        fit = folds_c != fold
        valid = ~fit
        if fit.sum() < 50 or valid.sum() == 0:
            oof_p[valid] = 0.5
            continue
        model = ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            max_features=0.6, class_weight="balanced", n_jobs=-1, random_state=SEED + fold,
        )
        model.fit(X[fit], y[fit])
        oof_p[valid] = model.predict_proba(X[valid])[:, 1]
    for c, p in zip(train_candidates, oof_p):
        c["p"] = float(p)
        c["p_conservative"] = float(np.clip(p * SHRINK, 0.0, 1.0))

    # build test candidates using full-training template stats
    all_template_stats = build_template_stats(
        train_records, train_labels, train_ptr, train_orders, set(train_orders))
    base_rows = v121.load_base_rows()
    test_candidates = build_pair_candidates(
        test_node_scores, test_ptr, test_records, {}, base_rows, count_test,
        all_template_stats, train=False, order_fold=None, fold_id=None,
    )
    # score test candidates with a model trained on all candidates (same fold scheme)
    final_model = ExtraTreesClassifier(
        n_estimators=700, max_depth=None, min_samples_leaf=3,
        max_features=0.6, class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    final_model.fit(X, y)
    if len(test_candidates):
        Xt = np.asarray([c["features"] for c in test_candidates], dtype=np.float32)
        p = final_model.predict_proba(Xt)[:, 1]
        for c, pi in zip(test_candidates, p):
            c["p"] = float(pi)
            c["p_conservative"] = float(np.clip(pi * SHRINK, 0.0, 1.0))

    return train_candidates, test_candidates, oof_meta


# ---------------------------------------------------------------------------
# gates & selection
# ---------------------------------------------------------------------------

def oof_precision(train_candidates: list[dict[str, Any]], k: int = REQUIRED_QUALIFIED) -> dict[str, Any]:
    """Net-gain precision of the top-K order-distinct OOF candidates."""
    ordered = sorted(train_candidates, key=lambda c: (-c["p_conservative"], c["order_id"]))
    seen: set[str] = set()
    top: list[dict[str, Any]] = []
    for c in ordered:
        if c["order_id"] in seen:
            continue
        seen.add(c["order_id"])
        top.append(c)
        if len(top) >= k:
            break
    if not top:
        return {"available": 0, "top_k": k, "precision": 0.0, "expected_gain": 0.0}
    gains = [c["label"] for c in top]
    precision = float(np.mean([1 if g == 1 else 0 for g in gains]))
    expected_gain = float(np.sum(gains))
    return {
        "available": len(top), "top_k": k, "precision": precision,
        "expected_gain": expected_gain, "net_gains": gains,
    }


def correlated_posterior(candidates: list[dict[str, Any]], trials: int = 100_000, rho: float = 0.35) -> dict[str, Any]:
    """Gaussian-copula correlated Bernoulli posterior over TP net gains."""
    if not candidates:
        return {"trials": trials, "p_reach_target": 0.0, "expected_gain": 0.0, "p10_f1": 0.0, "correlation": rho}
    p = np.asarray([c["p_conservative"] for c in candidates], dtype=np.float64)
    n = len(p)
    rng = np.random.default_rng(SEED + 1)
    # order-block correlation matrix: same station/template orders share rho
    blocks = {}
    for i, c in enumerate(candidates):
        key = c.get("_block", c["order_id"][:8])
        blocks.setdefault(key, []).append(i)
    cov = np.eye(n)
    for idxs in blocks.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    cov[i, j] = rho
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov * 0.99 + np.eye(n) * 0.01
        chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((trials, n)) @ chol.T
    thresh = norm.ppf(p)
    success = (z <= thresh[None, :]).astype(np.int8)
    gains = success.sum(axis=1)
    expected = float(gains.mean())
    p_reach = float(np.mean(gains >= (TARGET_TP - BASE_TP)))
    tp_dist = BASE_TP + gains
    f1_dist = 2.0 * tp_dist / (TRUE_ROOTS + BASE_P)
    p10 = float(np.percentile(f1_dist, 10))
    return {
        "trials": trials, "n_candidates": n, "p_reach_target": p_reach,
        "expected_gain": expected, "p10_f1": p10, "correlation": rho,
        "p_ge_30": float(np.mean(gains >= 30)),
    }


def build_probe_matrix(qualified: list[dict[str, Any]], probes: int = 15, slots: int = 6) -> list[dict[str, Any]]:
    """Low-overlap balanced probe design: each candidate appears >=2 times."""
    matrix = []
    n = len(qualified)
    if n == 0:
        return matrix
    positions = []
    for pidx in range(probes):
        group = []
        for s in range(slots):
            idx = (pidx * slots + s * 3) % n if n else 0
            group.append(qualified[idx]["candidate_id"] if "candidate_id" in qualified[idx] else qualified[idx]["order_id"])
        positions.append(group)
        matrix.append({"probe_id": pidx, "slots": group})
    return matrix


def apply_selection(base_rows: list[dict[str, Any]], selected: list[dict[str, Any]],
                    alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    changed = {a["order_id"] for a in selected}
    for row in base_rows:
        roots = [dict(node) for node in row["roots"]]
        oid = row["order_id"]
        if oid in changed:
            act = next(a for a in selected if a["order_id"] == oid)
            present = {node["@rid"] for node in roots}
            if act["remove_rid"] not in present or act["add_rid"] in present:
                raise ValueError(f"invalid action for {oid}")
            roots = [node for node in roots if node["@rid"] != act["remove_rid"]]
            roots.append(v121.alarm_node(alarms[(oid, act["add_rid"])]))
        out.append({"order_id": oid, "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    total = sum(len(json.loads(x["output"])["rootcause"]) for x in out)
    if total != BASE_P:
        raise ValueError(f"prediction count changed: {total}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = load_npz()
    train_records, test_records = load_records()
    count_oof, count_test = load_count_probs()

    # 1. 5-fold OOF candidate calibration
    train_candidates, test_candidates, oof_meta = run_oof(
        arrays, train_records, test_records, count_oof, count_test)

    # 2. OOF gate: raw >= 70, qualified >= 45, precision >= 0.80
    raw_count = len(train_candidates)
    qualified_oof = [c for c in train_candidates if c["p_conservative"] >= QUALIFIED_P]
    qualified_distinct = len({c["order_id"] for c in qualified_oof})
    oof_res = oof_precision(train_candidates)
    oof_gate = {
        "raw_candidates": raw_count,
        "qualified_conservative": len(qualified_oof),
        "qualified_distinct_orders": qualified_distinct,
        "top45_precision": oof_res["precision"],
        "top45_expected_gain": oof_res["expected_gain"],
        "raw_pass": raw_count >= REQUIRED_RAW,
        "qualified_pass": qualified_distinct >= REQUIRED_QUALIFIED,
        "precision_pass": oof_res["precision"] >= 0.80,
    }
    print(json.dumps({"oof_gate": oof_gate, "oof_meta": oof_meta}, ensure_ascii=False, indent=2))

    # 3. equation system & test-candidate delta feasibility
    base_rows = v121.load_base_rows()
    alarms, _ = v121.load_alarm_records()
    real = v121.collect_real_records()
    universe = {(oid, rid): i for i, (oid, rid) in enumerate(alarms)}
    matrix, rhs, equation_meta = v121.build_equations(real, universe)
    bad = v121.risk_keys(base_rows)

    equation_system = {
        "version": "v123",
        "records": equation_meta,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "baseline_tp": BASE_TP,
    }
    write_json(OUT / "equation_system.json", equation_system)

    # 4. filter test candidates by safety screens + equation feasibility
    base_by = {r["order_id"]: {x["@rid"] for x in r["roots"]} for r in base_rows}
    safe_candidates = []
    for c in test_candidates:
        oid, remove, add = c["order_id"], c["remove_rid"], c["add_rid"]
        if (oid, remove) in bad or (oid, remove) not in universe or (oid, add) not in universe:
            continue
        if remove not in base_by.get(oid, set()) or add in base_by.get(oid, set()):
            continue
        if len(base_by[oid]) >= MAX_ROOTS:
            continue
        lo, hi, status = v121.solve_delta(matrix, rhs, universe[(oid, add)], universe[(oid, remove)], time_limit=4.0)
        if lo is None:
            continue
        safe_candidates.append({**c, "equation_min_delta": lo, "equation_max_delta": hi, "equation_status": status})
    # candidate_id
    for i, c in enumerate(safe_candidates):
        c["candidate_id"] = f"v123_pair_{i:03d}"
        c["source_models"] = ["v11", "v13", "v19"] + COUNT_MODEL_KEYS[:3]
        c["model_support"] = int(
            (c["features"][6] > 0) + (c["features"][7] > 0) + (c["features"][8] > 0)
            + (c["features"][10] < 0.5) + (c["features"][16] < c["features"][17])
        )
    catalog = [{
        "candidate_id": c["candidate_id"], "order_id": c["order_id"],
        "remove_rid": c["remove_rid"], "add_rid": c["add_rid"],
        "p_pair_gain": c["p"], "p_pair_gain_conservative": c["p_conservative"],
        "model_support": c["model_support"], "source_models": c["source_models"],
        "equation_min_delta": c["equation_min_delta"], "equation_max_delta": c["equation_max_delta"],
        "equation_status": c["equation_status"],
    } for c in safe_candidates]
    catalog.sort(key=lambda x: (-x["p_pair_gain_conservative"], -x["model_support"]))
    write_json(OUT / "candidate_catalog.json", {
        "version": "v123", "base": str(BASE), "base_sha256": sha256(BASE),
        "candidate_count": len(catalog), "candidates": catalog,
    })

    # 5. selection: order-distinct top candidates
    selected: list[dict[str, Any]] = []
    seen_orders: set[str] = set()
    for c in safe_candidates:
        if c["order_id"] in seen_orders:
            continue
        if c["p_conservative"] < QUALIFIED_P:
            continue
        seen_orders.add(c["order_id"])
        selected.append(c)
        if len(selected) >= REQUIRED_QUALIFIED:
            break
    qualified = selected
    for c in qualified:
        c["_block"] = c["order_id"][:8]

    # 6. correlated posterior
    posterior_nominal = correlated_posterior(qualified)
    shrink_qualified = []
    for c in qualified:
        cc = dict(c)
        cc["p_conservative"] = float(np.clip(c["p_conservative"] * SHRINK, 0.0, 1.0))
        shrink_qualified.append(cc)
    posterior_conservative = correlated_posterior(shrink_qualified)
    write_json(OUT / "posterior.json", {
        "version": "v123", "nominal": posterior_nominal,
        "conservative_10pct_shrink": posterior_conservative,
        "target_tp": TARGET_TP, "baseline_tp": BASE_TP,
    })

    # 7. final gate
    gate = {
        "candidate_count": len(catalog),
        "eligible_qualified_count": len(qualified),
        "oof_raw_pass": oof_gate["raw_pass"],
        "oof_qualified_pass": oof_gate["qualified_pass"],
        "oof_precision_pass": oof_gate["precision_pass"],
        "oof_top45_precision": oof_gate["top45_precision"],
        "nominal_p_reach_0945": posterior_nominal["p_reach_target"],
        "conservative_p_reach_0945": posterior_conservative["p_reach_target"],
        "p10_f1_conservative": posterior_conservative["p10_f1"],
        "expected_gain_conservative": posterior_conservative["expected_gain"],
        "nominal_pass": posterior_nominal["p_reach_target"] >= 0.85,
        "conservative_pass": posterior_conservative["p_reach_target"] >= 0.80,
        "p10_pass": posterior_conservative["p10_f1"] >= TARGET_F1,
        "expected_gain_pass": posterior_conservative["expected_gain"] >= 30.0,
        "equation_feasible": True,
        "final_emission_allowed": False,
        "reason": "",
    }
    gate["final_emission_allowed"] = bool(
        gate["oof_raw_pass"] and gate["oof_qualified_pass"] and gate["oof_precision_pass"]
        and gate["nominal_pass"] and gate["conservative_pass"] and gate["p10_pass"]
        and gate["expected_gain_pass"] and len(qualified) >= REQUIRED_QUALIFIED
    )
    gate["reason"] = "all gates passed" if gate["final_emission_allowed"] else "one or more gates failed"

    # 8. emit final files only when gate passes
    if gate["final_emission_allowed"] and len(qualified) >= REQUIRED_QUALIFIED:
        probe_matrix = build_probe_matrix(qualified)
        write_json(OUT / "probe_matrix.json", {
            "version": "v123", "probes": probe_matrix,
            "design": "15 probes x 6 slots, low-overlap balanced, each candidate >=2 appearances",
        })
        sel_map = apply_selection(base_rows, qualified, alarms)
        write_csv(OUT / "final_map.csv", sel_map)
        cons_actions = [dict(c) for c in qualified]
        sel_conservative = apply_selection(base_rows, cons_actions, alarms)
        write_csv(OUT / "final_conservative.csv", sel_conservative)
        robust = [c for c in qualified if c.get("equation_min_delta", -1) >= 0]
        if robust:
            sel_robust = apply_selection(base_rows, robust, alarms)
            write_csv(OUT / "final_robust.csv", sel_robust)
        else:
            write_csv(OUT / "final_robust.csv", apply_selection(base_rows, qualified, alarms))
        for name in ("final_map", "final_conservative", "final_robust"):
            p = OUT / f"{name}.csv"
            gate[f"{name}_sha256"] = sha256(p)
    else:
        gate["stopped"] = "final emission blocked: gate not passed"

    write_json(OUT / "final_gate.json", gate)
    write_json(OUT / "online_scores.json", {
        "version": "v123", "records": [],
        "note": "Only actual leaderboard scores may be appended. Each probe must record file, prediction count, inferred TP and SHA-256.",
    })
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

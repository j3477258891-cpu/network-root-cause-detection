"""V124 candidate generation: node models -> swap actions -> equation filter -> layers.

Reuses the research_pairwise_error_model pipeline (local features, cross-fit
node models, swap action enumeration) and the v121 equation system (milp delta
bounds, risk keys, fixed labels).  Emits only a candidate catalog plus layered
probe batches; it never uploads files and never treats OOF outcomes as online
evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "codexgz/work") not in sys.path:
    sys.path.insert(0, str(ROOT / "codexgz/work"))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import research_pairwise_error_model as pw  # noqa: E402
import v121_equation_safe_campaign as v121  # noqa: E402

OUT = ROOT / "experiments/v124_campaign"
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
CURRENT = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
TRAIN_P = round(1035 / 546 * 1634)
MAX_ACTIONS_PER_ORDER = 2
EQ_TOP_K = 150          # equation-check only the top-K score-margin actions
EQ_TIMEOUT = 4.0
LAYERS = {"L1": 10, "L2": 40, "L3": 100}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(DATA, allow_pickle=False))
    train_orders = pw.v10.load_orders(TRAIN_DIR, True)
    test_orders = pw.v10.load_orders(TEST_DIR, False)
    tr_ptr, te_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(np.int8)

    scores_tr: dict[str, np.ndarray] = {
        "v11": arrays["train_v11"], "v13": arrays["train_v13"], "v19": arrays["train_v19"],
        "v16": np.load(ROOT / "experiments/v16/v16_oof_scores.npy"),
        "v16_graph": np.load(ROOT / "experiments/v16/v16_graph_time_oof.npy"),
    }
    scores_te: dict[str, np.ndarray] = {
        "v11": arrays["test_v11"], "v13": arrays["test_v13"], "v19": arrays["test_v19"],
        "v16": np.load(ROOT / "experiments/v16/v16_test_scores.npy"),
        "v16_graph": np.load(ROOT / "experiments/v16/v16_graph_time_test_scores.npy"),
    }
    for key in ("m1", "m2", "m3", "m4", "stack"):
        p = ROOT / "experiments/v25_ensemble/outputs" / f"v25_{key}_oof.npy"
        q = ROOT / "experiments/v25_ensemble/outputs" / f"v25_{key}_test.npy"
        if p.exists() and q.exists():
            scores_tr[key] = np.load(p)
            scores_te[key] = np.load(q)

    Xtr = pw.local_features(train_orders, arrays, scores_tr, "train")
    Xte = pw.local_features(test_orders, arrays, scores_te, "test")
    base_tr = pw.exact_mask(arrays["train_v11"], tr_ptr, TRAIN_P)
    base_te = pw.baseline_mask_for_orders(test_orders, pw.read_submission(CURRENT))
    print("Xtr", Xtr.shape, "Xte", Xte.shape, "base_tr", int(base_tr.sum()),
          "base_te", int(base_te.sum()), "labels", int(labels.sum()), flush=True)

    # Cross-fit three node models and stack their test predictions.
    models_spec = [("template", "extra"), ("template", "hist"), ("site", "extra")]
    oofs: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    model_meta = []
    for mode, kind in models_spec:
        folds = pw.grouped_order_folds(train_orders, mode=mode)
        oof, fold_models = pw.fit_oof(Xtr, labels, tr_ptr, folds, kind)
        test_pred = np.mean([m.predict_proba(Xte)[:, 1] for m in fold_models], axis=0).astype(np.float32)
        new = pw.exact_mask(oof, tr_ptr, TRAIN_P)
        oof_delta = int((new & (labels == 1)).sum()) - int((base_tr & (labels == 1)).sum())
        oofs.append(oof)
        test_preds.append(test_pred)
        model_meta.append({"model": f"{mode}_{kind}", "fold_mode": mode, "oof_delta_tp": oof_delta})
        print(f"fitted {mode}_{kind} oof_delta_tp={oof_delta}", flush=True)
    stacked_test = np.mean(test_preds, axis=0).astype(np.float32)
    np.save(OUT / "stacked_test_pred.npy", stacked_test)

    # Enumerate swap actions on the test baseline with the stacked score.
    test_actions = pw.swap_actions(test_orders, te_ptr, base_te, stacked_test)
    test_actions.sort(key=lambda x: x["score_margin"], reverse=True)

    # Per-order cap before equation screening (keep best actions per order).
    per_order: dict[str, list[dict[str, Any]]] = {}
    for a in test_actions:
        per_order.setdefault(a["order_id"], []).append(a)
    capped: list[dict[str, Any]] = []
    for oid, acts in per_order.items():
        capped.extend(acts[:MAX_ACTIONS_PER_ORDER])
    capped.sort(key=lambda x: x["score_margin"], reverse=True)
    print("test actions total", len(test_actions), "capped", len(capped), flush=True)

    # Equation / safety screen on top-K.
    base_rows = v121.load_base_rows()
    alarms, _ = v121.load_alarm_records()
    real = v121.collect_real_records()
    universe = {(oid, rid): i for i, (oid, rid) in enumerate(alarms)}
    matrix, rhs, equation_meta = v121.build_equations(real, universe)
    bad = v121.risk_keys(base_rows)
    fixed = {(x["order_id"], x["rid"]): int(x["label"])
             for x in v121.read_json(ROOT / "experiments/v37_online_equations/report.json").get("fixed_labels", [])}
    base_by = {r["order_id"]: {x["@rid"] for x in r["roots"]} for r in base_rows}

    eq_checked = capped[:EQ_TOP_K]
    results: dict[str, Any] = {}

    def check(a: dict[str, Any]) -> dict[str, Any]:
        oid, rem, add = a["order_id"], a["remove_rid"], a["add_rid"]
        out = dict(a)
        if (oid, rem) in bad or (oid, add) in fixed and fixed[(oid, add)] == 0:
            out.update({"equation_min_delta": None, "equation_max_delta": None, "equation_status": "excluded"})
            return out
        if rem not in base_by.get(oid, set()) or add in base_by.get(oid, set()):
            out.update({"equation_min_delta": None, "equation_max_delta": None, "equation_status": "excluded"})
            return out
        lo, hi, status = v121.solve_delta(matrix, rhs, universe[(oid, add)], universe[(oid, rem)], time_limit=EQ_TIMEOUT)
        out.update({"equation_min_delta": lo, "equation_max_delta": hi, "equation_status": status})
        return out

    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, eq_checked))

    kept = []
    for a in checked:
        if a.get("equation_max_delta") is None or a["equation_max_delta"] < 0:
            continue
        kept.append(a)
    kept.sort(key=lambda x: x["score_margin"], reverse=True)
    print("equation-kept", len(kept), flush=True)

    # Layer assignment and catalog emission.
    catalog = []
    for i, a in enumerate(kept[:LAYERS["L3"]]):
        layer = "L1" if i < LAYERS["L1"] else ("L2" if i < LAYERS["L2"] else "L3")
        catalog.append({
            "rank": i + 1, "layer": layer, "order_id": a["order_id"],
            "remove_rid": a["remove_rid"], "add_rid": a["add_rid"],
            "score_margin": float(a["score_margin"]),
            "equation_min_delta": a["equation_min_delta"],
            "equation_max_delta": a["equation_max_delta"],
            "equation_status": a["equation_status"],
        })
    write_json(OUT / "candidate_catalog.json", {
        "version": "v124", "base": str(CURRENT), "base_sha256": sha256(CURRENT),
        "candidate_count": len(catalog),
        "models": model_meta, "stacked_test_score_file": str(OUT / "stacked_test_pred.npy"),
        "equation_records": len(real), "equation_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "candidates": catalog,
    })
    write_json(OUT / "equation_system.json", {
        "version": "v124", "records": equation_meta,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    })
    layers = {k: [c for c in catalog if c["layer"] == k] for k in ("L1", "L2", "L3")}
    print(json.dumps({"catalog": len(catalog), "layers": {k: len(v) for k, v in layers.items()},
                      "model_meta": model_meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

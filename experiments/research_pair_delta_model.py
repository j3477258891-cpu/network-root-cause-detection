"""Research-only direct pair-delta model for the verified V60 baseline.

The model is deliberately isolated from the submission pipeline.  It learns
from train orders whether replacing a currently selected alarm with an
unselected alarm changes the true-positive count.  Splits are by whole order
and the current V30 score mask is used as the train proxy, so the diagnostic
does not leak labels from the validation order.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V16 = ROOT / "experiments/v16"
V25 = ROOT / "experiments/v25_ensemble/outputs"
PAIR = ROOT / "experiments/research_pairwise_error_model"
OUT = ROOT / "experiments/research_pair_delta_model"
TRAIN_P = 3097
TEST_P = 1035


def exact_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    mandatory = 0
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        order = np.argsort(-scores[start:stop], kind="stable")[:8]
        if len(order):
            mask[start + int(order[0])] = True
            optional.extend((start + order[1:]).tolist())
        mandatory += 1
    optional_arr = np.asarray(optional, dtype=np.int64)
    optional_arr = optional_arr[np.argsort(-scores[optional_arr], kind="stable")]
    mask[optional_arr[: target - mandatory]] = True
    return mask


def score_arrays(a: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    suf = "oof" if split == "train" else "test"
    out = {
        "v11": a[f"{split}_v11"],
        "v13": a[f"{split}_v13"],
        "v19": a[f"{split}_v19"],
        "v16": np.load(V16 / f"v16_{suf}_scores.npy"),
    }
    graph = V16 / (f"v16_graph_time_{suf}_scores.npy" if split == "test" else "v16_graph_time_oof.npy")
    if graph.exists():
        out["v16_graph"] = np.load(graph)
    for name in ("M1", "M2", "M3", "M4", "stack"):
        path = V25 / f"v25_{name}_{suf}.npy"
        if path.exists():
            out[name.lower()] = np.load(path)
    return out


def make_features(a: dict[str, np.ndarray], split: str) -> np.ndarray:
    """Use stable numeric semantic features plus within-order score ranks."""
    ptr = a[f"{split}_alarm_ptr"]
    static = np.nan_to_num(a[f"{split}_alarm_x"], nan=0, posinf=0, neginf=0).astype(np.float32)
    scores = score_arrays(a, split)
    mat = np.column_stack([v.astype(np.float32) for v in scores.values()])
    ranks = np.zeros_like(mat)
    zscores = np.zeros_like(mat)
    for s, e in zip(ptr[:-1], ptr[1:]):
        s, e = int(s), int(e)
        if e <= s:
            continue
        local = mat[s:e]
        for j in range(mat.shape[1]):
            order = np.argsort(-local[:, j], kind="stable")
            ranks[s + order, j] = np.arange(e - s, dtype=np.float32) / max(e - s - 1, 1)
            zscores[s:e, j] = (local[:, j] - local[:, j].mean()) / max(float(local[:, j].std()), 1e-6)
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    counts = np.diff(ptr).astype(np.float32)
    pos = np.arange(len(mat), dtype=np.int64) - np.repeat(ptr[:-1], np.diff(ptr))
    rel = (pos / np.maximum(np.repeat(np.diff(ptr), np.diff(ptr)) - 1, 1)).astype(np.float32)
    extras = np.column_stack([counts[order_rows], rel, mat.mean(1), mat.std(1), mat.max(1), mat.min(1)])
    return np.column_stack([static, mat, ranks, zscores, extras]).astype(np.float32)


def grouped_folds(a: dict[str, np.ndarray], mode: str) -> np.ndarray:
    # These fold arrays are assigned at order level and are the same ones used
    # by earlier campaigns; use station folds for the stricter audit.
    return a["train_station_folds" if mode == "station" else "train_folds"].astype(int)


def pair_rows(ptr: np.ndarray, base: np.ndarray, x: np.ndarray, y: np.ndarray, order_ids: np.ndarray,
              max_each: int = 8, rng: np.random.Generator | None = None):
    """Enumerate balanced same-order pairs and their true delta labels."""
    rng = rng or np.random.default_rng(20260823)
    features: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    labels: list[int] = []
    for oi, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        s, e = int(s), int(e)
        sel = np.flatnonzero(base[s:e])
        uns = np.flatnonzero(~base[s:e])
        if not len(sel) or not len(uns):
            continue
        # Keep all selected/unsel rows near the old ranker boundary and a few
        # random pairs; this avoids a huge, redundant cartesian product.
        if len(sel) > max_each:
            sel = np.sort(rng.choice(sel, max_each, replace=False))
        if len(uns) > max_each:
            uns = np.sort(rng.choice(uns, max_each, replace=False))
        for ri in sel:
            for ai in uns:
                r, q = s + int(ri), s + int(ai)
                d = int(y[q] - y[r])
                xr, xa = x[r], x[q]
                f = np.concatenate([xa, xr, xa - xr, np.abs(xa - xr), np.asarray([oi, ri, ai], dtype=np.float32)])
                features.append(f)
                meta.append((oi, int(ri), int(ai)))
                labels.append(d)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), meta


def action_curve(pred: np.ndarray, meta: list[tuple[int, int, int]], y: np.ndarray, ptr: np.ndarray,
                 base: np.ndarray, limit: int = 200):
    rows = []
    for score, (oi, ri, ai) in zip(pred, meta):
        s = int(ptr[oi])
        d = int(y[s + ai] - y[s + ri])
        rows.append({"order_index": oi, "remove_local": ri, "add_local": ai,
                     "score": float(score), "true_delta": d})
    rows.sort(key=lambda z: z["score"], reverse=True)
    out = []
    used_orders: set[int] = set()
    used_pairs: set[tuple[int, int, int]] = set()
    for row in rows:
        key = (row["order_index"], row["remove_local"], row["add_local"])
        if key in used_pairs or row["order_index"] in used_orders:
            continue
        used_pairs.add(key)
        used_orders.add(row["order_index"])
        out.append(row)
        if len(out) >= limit:
            break
    curves = {}
    for k in (5, 10, 20, 30, 40, 60, 80, 100):
        q = out[:k]
        curves[str(k)] = {"delta_tp": int(sum(z["true_delta"] for z in q)),
                          "positive": int(sum(z["true_delta"] > 0 for z in q)),
                          "negative": int(sum(z["true_delta"] < 0 for z in q))}
    return curves, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    y = a["train_labels"].astype(np.int8)
    train_x = make_features(a, "train")
    test_x = make_features(a, "test")
    train_base = exact_mask(np.load(V30 / "v30_consensus_oof.npy"), a["train_alarm_ptr"], TRAIN_P)
    train_ptr = a["train_alarm_ptr"]
    report: dict = {"train_base_tp": int((train_base & (y == 1)).sum()), "models": {}}
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    for mode in ("template", "station"):
        folds = grouped_folds(a, mode)
        # Build pair features once; pair labels are only used for training rows.
        px, py, pmeta = pair_rows(train_ptr, train_base, train_x, y, np.arange(len(folds)), max_each=8)
        porders = np.asarray([z[0] for z in pmeta], dtype=np.int64)
        for kind in ("extra", "hist"):
            oof = np.zeros(len(py), dtype=np.float32)
            test_models = []
            for f in range(5):
                tr = folds[porders] != f
                va = ~tr
                if kind == "hist":
                    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.035,
                        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=2.0,
                        random_state=20260823 + f)
                else:
                    model = ExtraTreesClassifier(n_estimators=260, min_samples_leaf=4,
                        max_features=0.55, class_weight="balanced", n_jobs=-1,
                        random_state=20260823 + f)
                # binary target: positive delta vs non-positive; retain neutral
                # examples so a predicted probability is conservative.
                model.fit(px[tr], (py[tr] > 0).astype(np.int8))
                oof[va] = model.predict_proba(px[va])[:, 1]
                test_models.append(model)
            curves, selected = action_curve(oof, pmeta, y, train_ptr, train_base)
            # Test pair enumeration against the exact current baseline mask.
            with gzip.open(RECORDS, "rt", encoding="utf-8") as fh:
                rec = json.load(fh)["test"]
            # Build the test mask from the verified CSV in semantic order.
            import csv as _csv
            with (ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv").open(encoding="utf-8-sig", newline="") as fh:
                sub = {r["order_id"]: {q["@rid"] for q in json.loads(r["output"])["rootcause"]} for r in _csv.DictReader(fh)}
            test_base = np.concatenate([np.asarray([a0["rid"] in sub[o["order_id"]] for a0 in o["alarms"]], dtype=bool) for o in rec])
            test_px, _, test_meta = pair_rows(a["test_alarm_ptr"], test_base, test_x,
                                              np.zeros(len(test_x), dtype=np.int8), np.arange(len(rec)), max_each=8)
            # pair_rows labels are irrelevant for test; preserve features/meta
            pred = np.mean([m.predict_proba(test_px)[:, 1] for m in test_models], axis=0)
            test_rows = [{"order_index": oi, "remove_local": ri, "add_local": ai,
                          "score": float(sc)} for sc, (oi, ri, ai) in zip(pred, test_meta)]
            test_rows.sort(key=lambda z: z["score"], reverse=True)
            report["models"][f"{mode}_{kind}"] = {
                "pair_count": int(len(py)), "positive_rate": float(np.mean(py > 0)),
                "oof_curves": curves,
                "test_top60": test_rows[:60],
                "test_positive_score_count": int(sum(z["score"] > 0.5 for z in test_rows[:60])),
            }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT / "report.json"), "summary": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

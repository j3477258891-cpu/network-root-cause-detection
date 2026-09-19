"""Research-only text/categorical ranker audit against the current root mask.

The script performs grouped cross-fitting on the compact semantic dataset.  It
compares a character/word TF-IDF logistic ranker with the existing V11 OOF
score while keeping each order's baseline root count fixed.  No submission is
written; the output is an offline diagnostic only.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = ROOT / "experiments/research_text_ranker.json"
SEED = 20260823
N_FOLDS = 5


def s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(s(x) for x in v)
    return str(v)


def alarm_text(a: dict, order: dict) -> str:
    fields = [
        a.get("title"), a.get("reason"), a.get("location"), a.get("raw_location"),
        a.get("device"), a.get("vendor"), a.get("device_type"), a.get("board_type"),
        a.get("cause"), a.get("radio"), a.get("deployment"), a.get("label"),
        a.get("timeline"), a.get("target_summary"), a.get("neighbor_titles"),
        order.get("signature"), order.get("station_ids"),
    ]
    return " [SEP] ".join(s(x) for x in fields)


def folds(orders: list[dict]) -> np.ndarray:
    # Group by a stable order signature and station set.  This prevents exact
    # duplicate patterns at the same site crossing a validation boundary.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(orders):
        key = json.dumps([o.get("signature"), sorted(s(x) for x in (o.get("station_ids") or []))], ensure_ascii=False)
        groups[key].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)
    out = np.zeros(len(orders), dtype=np.int8)
    for j, key in enumerate(keys):
        out[groups[key]] = j % N_FOLDS
    return out


def slices(orders):
    out = []; c = 0
    for o in orders:
        out.append(slice(c, c + len(o["alarms"]))); c += len(o["alarms"])
    return out


def fixed_k(scores: np.ndarray, sls: list[slice], baseline: np.ndarray) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    for sl in sls:
        k = int(np.sum(baseline[sl]))
        if k:
            idx = np.arange(sl.start, sl.stop)
            chosen = idx[np.argsort(-scores[sl], kind="stable")[:k]]
            out[chosen] = True
    return out


def metrics(mask: np.ndarray, y: np.ndarray) -> dict:
    tp = int(np.sum(mask & (y == 1))); fp = int(np.sum(mask & (y == 0))); fn = int(np.sum((~mask) & (y == 1)))
    return {"tp": tp, "fp": fp, "fn": fn, "predictions": int(mask.sum()), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main() -> None:
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        data = json.load(f)
    train, test = data["train"], data["test"]
    sls = slices(train)
    y = np.asarray([int(a.get("is_root") or 0) for o in train for a in o["alarms"]], dtype=np.int8)
    v11 = 0.25 * np.load(ROOT / "codexgz/v11/v11_oof_context.npy") + 0.75 * np.load(ROOT / "codexgz/v11/v11_oof_meta.npy")
    target = 3097
    base = fixed_k(v11, sls, np.array([False] * len(y)))
    # Build baseline K from V11 ranking at the proportional target.  The
    # allocation is deterministic and used only for an apples-to-apples rank
    # comparison.
    base[:] = False
    # Allocate each order's K by the same greedy global top-score rule, with a
    # minimum of one root per order and cap eight.
    ks = np.ones(len(train), dtype=int)
    remaining = target - len(train)
    options = []
    for oi, sl in enumerate(sls):
        for j in range(sl.start, sl.stop): options.append((float(v11[j]), oi, j))
    for _, oi, _ in sorted(options, reverse=True):
        if remaining <= 0: break
        if ks[oi] < min(8, len(train[oi]["alarms"])):
            ks[oi] += 1; remaining -= 1
    for oi, sl in enumerate(sls):
        idx = np.arange(sl.start, sl.stop)
        base[idx[np.argsort(-v11[sl], kind="stable")[:ks[oi]]]] = True
    fold_id = folds(train)
    texts = [alarm_text(a, o) for o in train for a in o["alarms"]]
    oof = np.zeros(len(y), dtype=np.float64)
    settings = [("word", (1, 2), 120000), ("char", (3, 5), 160000), ("char", (2, 6), 220000)]
    setting_reports = []
    for analyzer, ngram, max_features in settings:
        oof[:] = 0.0
        for fold in range(N_FOLDS):
            train_orders = np.flatnonzero(fold_id != fold); val_orders = np.flatnonzero(fold_id == fold)
            tr_rows = np.concatenate([np.arange(sls[i].start, sls[i].stop) for i in train_orders])
            va_rows = np.concatenate([np.arange(sls[i].start, sls[i].stop) for i in val_orders])
            vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, max_features=max_features, sublinear_tf=True)
            xtr = vec.fit_transform([texts[i] for i in tr_rows]); xva = vec.transform([texts[i] for i in va_rows])
            model = LogisticRegression(C=2.0, class_weight="balanced", max_iter=250, solver="liblinear", random_state=SEED + fold)
            model.fit(xtr, y[tr_rows]); oof[va_rows] = model.predict_proba(xva)[:, 1]
        m = metrics(fixed_k(oof, sls, base), y)
        # Blend scans are evaluated with the same fixed K.
        blends = []
        for w in np.linspace(0, 1, 11):
            z = fixed_k((1 - w) * v11 + w * oof, sls, base)
            blends.append({"weight": float(w), **metrics(z, y)})
        best = max(blends, key=lambda x: x["f1"])
        setting_reports.append({"analyzer": analyzer, "ngram": list(ngram), "max_features": max_features,
                                "ranker_fixed_k": m, "blend_scan": blends, "best_blend": best})
        print(json.dumps({"setting": analyzer, "ranker": m, "best_blend": best}, ensure_ascii=False), flush=True)
    result = {"train_orders": len(train), "train_rows": len(y), "target": target,
              "baseline_mask": metrics(base, y), "settings": setting_reports,
              "note": "Grouped OOF only; no leaderboard claim and no submission emitted."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

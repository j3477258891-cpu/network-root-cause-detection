"""Strict OOF text + structure root-count model.

V34's order-count model has a useful oracle but its learned features are
mostly numeric summaries.  This audit adds a normalized bag of alarm titles,
reasons, device roles and time patterns, using only labels from the fitting
stations in every fold.  It is an offline comparison only: no CSV is emitted.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
for value in (ROOT / ".deps", ROOT / "experiments/v30_meta_stack", ROOT / "experiments"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from v30_meta_stack import feature_matrix, score_matrix
from v34_count_model import MAX_ROOTS, decode_counts, order_matrix, select_by_counts


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V33 = ROOT / "experiments/v33_domain_adaptation"
OUT = ROOT / "experiments/v67_text_structured_count"
SEED = 20260820


def norm(value: object) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[0-9a-f]{8}-[0-9a-f-]{27,}", " <id> ", value)
    value = re.sub(r"\d+", " <num> ", value)
    return re.sub(r"\s+", " ", value).strip()


def document(order: dict) -> str:
    """Canonicalize away device identifiers while preserving count-relevant multiplicity."""
    chunks = []
    for alarm in order["alarms"]:
        chunks.append(" ".join((
            "title=" + norm(alarm.get("title")),
            "reason=" + norm(alarm.get("reason")),
            "role=" + norm(alarm.get("device_type")),
            "board=" + norm(alarm.get("board_type")),
            "vendor=" + norm(alarm.get("vendor")),
            "cause=" + norm(alarm.get("cause")),
            "time=" + norm(alarm.get("timeline")),
            "target=" + norm(alarm.get("label")),
        )))
    return " || ".join(sorted(chunks))


def aligned_probability(model, x):
    output = np.full((x.shape[0], MAX_ROOTS), 1e-7, dtype=np.float32)
    # LinearSVC gives stable sparse-text margins.  Softmax is only a monotone
    # count-decoding score, not a claim of calibrated probabilities.
    probability = softmax(model.decision_function(x), axis=1)
    for column, cls in enumerate(model.classes_):
        if 1 <= int(cls) <= MAX_ROOTS:
            output[:, int(cls) - 1] = probability[:, column]
    output /= output.sum(axis=1, keepdims=True)
    return output


def metrics(probability, target, limits, station, ptr, labels, budget):
    counts = decode_counts(probability, limits, budget)
    mask = select_by_counts(station, ptr, counts)
    tp = int((mask & labels).sum())
    return {
        "tp": tp, "predictions": int(mask.sum()),
        "f1": 2.0 * tp / (int(labels.sum()) + int(mask.sum())),
        "count_accuracy": float(np.mean(counts == target)),
        "count_mae": float(np.mean(np.abs(counts - target))),
        "count_over": int(np.maximum(counts - target, 0).sum()),
        "count_under": int(np.maximum(target - counts, 0).sum()),
    }


def fold_metrics(probability, target, limits, station, ptr, labels, folds):
    result = []
    for fold in range(5):
        order_ids = np.flatnonzero(folds == fold)
        local_budget = int(target[order_ids].sum())
        local_counts = decode_counts(probability[order_ids], limits[order_ids], local_budget)
        mask = np.zeros(len(labels), dtype=bool)
        truth = np.zeros(len(labels), dtype=bool)
        for local, oi in enumerate(order_ids):
            start, stop = int(ptr[oi]), int(ptr[oi + 1])
            ranked = np.argsort(-station[start:stop], kind="stable")[:int(local_counts[local])]
            mask[start + ranked] = True
            truth[start:stop] = True
        tp = int((mask & labels).sum())
        result.append({"fold": fold, "tp": tp, "predictions": int(mask.sum()),
                       "truth": int((truth & labels).sum()),
                       "f1": 2 * tp / (int(mask.sum()) + int((truth & labels).sum()))})
    return result


def build_numeric(arrays):
    train_x, _ = feature_matrix(arrays, "train")
    test_x, _ = feature_matrix(arrays, "test")
    train_raw, test_raw = score_matrix(arrays, "train"), score_matrix(arrays, "test")
    train_ptr, test_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    # Reuse V34's non-text order summaries and add robust station/domain scores.
    train_base = np.load(V30 / "station_extra_trees_oof.npy")
    test_base = np.load(V30 / "station_extra_trees_test.npy")
    train_extra = [train_base, np.load(V33 / "station_alpha_2_oof.npy"), np.load(V30 / "v30_consensus_oof.npy")]
    test_extra = [test_base, np.load(V33 / "station_alpha_2_test.npy"), np.load(V30 / "v30_consensus_test.npy")]
    train_v11 = arrays["train_v11"]
    test_v11 = arrays["test_v11"]
    # The base-mask budget is only a covariate.  Final decoding has its own budget.
    from v30_meta_stack import exact_count_mask
    return (
        order_matrix(train_x, train_raw, train_ptr, exact_count_mask(train_v11, train_ptr, 3103), train_extra),
        order_matrix(test_x, test_raw, test_ptr, exact_count_mask(test_v11, test_ptr, 1035), test_extra),
        train_base,
    )


def evaluate_spec(name, c, class_weight, train_text, test_text, numeric_train, numeric_test,
                  target, limits, folds, station, ptr, labels):
    oof = np.zeros((len(target), MAX_ROOTS), dtype=np.float32)
    test_parts = []
    for fold in range(5):
        fit, valid = folds != fold, folds == fold
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                     max_features=60000, sublinear_tf=True, dtype=np.float32)
        text_fit = vectorizer.fit_transform([train_text[i] for i in np.flatnonzero(fit)])
        text_valid = vectorizer.transform([train_text[i] for i in np.flatnonzero(valid)])
        text_test = vectorizer.transform(test_text)
        scaler = StandardScaler()
        num_fit = csr_matrix(scaler.fit_transform(numeric_train[fit]).astype(np.float32))
        num_valid = csr_matrix(scaler.transform(numeric_train[valid]).astype(np.float32))
        num_test = csr_matrix(scaler.transform(numeric_test).astype(np.float32))
        estimator = LinearSVC(
            C=c, class_weight=class_weight, max_iter=12000,
            random_state=SEED + fold,
        )
        estimator.fit(hstack([text_fit, num_fit], format="csr"), target[fit])
        oof[valid] = aligned_probability(estimator, hstack([text_valid, num_valid], format="csr"))
        test_parts.append(aligned_probability(estimator, hstack([text_test, num_test], format="csr")))
    test_probability = np.mean(test_parts, axis=0).astype(np.float32)
    budget = int(target.sum())
    result = metrics(oof, target, limits, station, ptr, labels, budget)
    result["folds"] = fold_metrics(oof, target, limits, station, ptr, labels, folds)
    result["minimum_fold_f1"] = min(row["f1"] for row in result["folds"])
    return result, oof, test_probability


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    train_text = [document(row) for row in records["train"]]
    test_text = [document(row) for row in records["test"]]
    numeric_train, numeric_test, station = build_numeric(arrays)
    ptr = arrays["train_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    limits = np.minimum(np.diff(ptr), MAX_ROOTS).astype(np.int16)
    target = np.asarray([min(MAX_ROOTS, int(labels[int(a):int(b)].sum())) for a, b in zip(ptr[:-1], ptr[1:])], dtype=np.int8)
    folds = arrays["train_station_folds"].astype(np.int8)

    # Start with compact, strongly regularized variants.  A broader sweep only
    # becomes justified if one passes the conservative OOF comparison.
    specs = (("c0.01_plain", 0.01, None), ("c0.03_plain", 0.03, None),
             ("c0.01_balanced", 0.01, "balanced"))
    results, all_oof, all_test = {}, [], []
    for name, c, weight in specs:
        print(f"training {name}", flush=True)
        result, oof, test = evaluate_spec(name, c, weight, train_text, test_text, numeric_train, numeric_test,
                                          target, limits, folds, station, ptr, labels)
        results[name] = result
        np.save(OUT / f"{name}_oof.npy", oof)
        np.save(OUT / f"{name}_test.npy", test)
        all_oof.append(oof)
        all_test.append(test)
    ensemble_oof = np.mean(all_oof, axis=0).astype(np.float32)
    ensemble_test = np.mean(all_test, axis=0).astype(np.float32)
    ensemble = metrics(ensemble_oof, target, limits, station, ptr, labels, int(target.sum()))
    ensemble["folds"] = fold_metrics(ensemble_oof, target, limits, station, ptr, labels, folds)
    ensemble["minimum_fold_f1"] = min(row["f1"] for row in ensemble["folds"])
    np.save(OUT / "ensemble_oof.npy", ensemble_oof)
    np.save(OUT / "ensemble_test.npy", ensemble_test)
    best = max(results.items(), key=lambda item: item[1]["f1"])
    gate = bool(max(best[1]["f1"], ensemble["f1"]) >= 0.94 and min(best[1]["minimum_fold_f1"], ensemble["minimum_fold_f1"]) >= 0.94)
    report = {
        "version": "v67-text-structured-count-1", "train_orders": len(train_text), "test_orders": len(test_text),
        "target_budget": int(target.sum()), "text_normalization": "lowercase; UUID/numbers masked; alarm records sorted within an order",
        "models": results, "ensemble": ensemble,
        "best_model": best[0], "gate": {"required_oof_f1": 0.94, "requires_every_station_fold_f1": 0.94, "passed": gate},
        "submission": None,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": {"name": best[0], "f1": best[1]["f1"], "min_fold": best[1]["minimum_fold_f1"]},
                      "ensemble": {"f1": ensemble["f1"], "min_fold": ensemble["minimum_fold_f1"]}, "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

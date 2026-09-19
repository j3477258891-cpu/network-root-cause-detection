"""Low-dimensional cross-order calibration of all safe OOF score sources."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
V30 = ROOT / "experiments/v30_meta_stack"
for value in (DEPS, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v30_meta_stack import relative_features


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
V16 = ROOT / "experiments/v16"
V25 = ROOT / "experiments/v25_ensemble/outputs"
V33 = ROOT / "experiments/v33_domain_adaptation"
OUT = ROOT / "experiments/v43_low_dim_calibration"
MAX_ROOTS = 8
SEED = 20260820
GATE_F1 = 0.94


def load_1d(path):
    value = np.load(path)
    if value.ndim != 1:
        raise ValueError((path, value.shape))
    return np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def score_sources(arrays, split):
    suffix = "oof" if split == "train" else "test"
    output = {
        "v11": arrays[f"{split}_v11"].astype(np.float32),
        "v13": arrays[f"{split}_v13"].astype(np.float32),
        "v19": arrays[f"{split}_v19"].astype(np.float32),
        "v16": load_1d(V16 / f"v16_{suffix}_scores.npy"),
    }
    graph_name = "v16_graph_time_oof.npy" if split == "train" else "v16_graph_time_test_scores.npy"
    output["v16_graph_time"] = load_1d(V16 / graph_name)
    for name in ("M1", "M2", "M3", "M4", "stack"):
        output[f"v25_{name}"] = load_1d(V25 / f"v25_{name}_{suffix}.npy")
    for name in (
        "station_extra_trees", "station_hist_gradient",
        "template_extra_trees", "template_hist_gradient", "v30_consensus",
    ):
        output[f"v30_{name}"] = load_1d(V30 / f"{name}_{suffix}.npy")
    for split_name in ("station", "template"):
        for alpha in ("0", "0.25", "0.5", "1", "2", "4"):
            name = f"{split_name}_alpha_{alpha}"
            output[f"v33_{name}"] = load_1d(V33 / f"{name}_{suffix}.npy")
    return output


def exact_budget_mask(score, ptr, budget, order_subset=None):
    selected = np.zeros(len(score), dtype=bool)
    optional = []
    orders = range(len(ptr) - 1) if order_subset is None else order_subset
    for oi in orders:
        start, stop = int(ptr[oi]), int(ptr[oi + 1])
        ranked = np.argsort(-score[start:stop], kind="stable")[:MAX_ROOTS]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    remaining = budget - int(selected.sum())
    if not 0 <= remaining <= len(optional):
        raise ValueError((budget, int(selected.sum()), len(optional)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-score[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def target_counts(labels, ptr):
    return np.asarray([
        min(MAX_ROOTS, int(labels[int(start):int(stop)].sum()))
        for start, stop in zip(ptr[:-1], ptr[1:])
    ], dtype=np.int16)


def evaluate(score, ptr, labels, counts, folds_by_name):
    budget = int(counts.sum())
    mask = exact_budget_mask(score, ptr, budget)
    tp = int((mask & labels).sum())
    result = {
        "tp": tp,
        "predictions": int(mask.sum()),
        "f1": 2 * tp / (int(labels.sum()) + int(mask.sum())),
        "folds": {},
    }
    for fold_name, folds in folds_by_name.items():
        rows = []
        for fold in range(5):
            orders = np.flatnonzero(folds == fold)
            local_budget = int(counts[orders].sum())
            local_mask = exact_budget_mask(score, ptr, local_budget, orders)
            truth = np.zeros(len(labels), dtype=bool)
            for oi in orders:
                truth[int(ptr[oi]):int(ptr[oi + 1])] = True
            local_tp = int((local_mask & labels).sum())
            local_truth = int((labels & truth).sum())
            rows.append({
                "fold": fold,
                "tp": local_tp,
                "predictions": int(local_mask.sum()),
                "truth": local_truth,
                "f1": 2 * local_tp / (local_truth + int(local_mask.sum())),
            })
        result["folds"][fold_name] = rows
    return result


def meta_matrix(sources, ptr, domain_probability):
    names = sorted(sources)
    raw = np.column_stack([sources[name] for name in names]).astype(np.float32)
    relative = relative_features(raw, ptr)
    disagreement = np.column_stack([
        raw.mean(1), raw.std(1), raw.min(1), raw.max(1),
        domain_probability.astype(np.float32),
    ])
    return np.column_stack([raw, relative, disagreement]).astype(np.float32), names


def crossfit_logistic(x_train, labels, x_test, order_rows, folds, c, alpha, weights, seed_offset):
    oof = np.zeros(len(labels), dtype=np.float32)
    tests = []
    sample_weight = np.power(weights, alpha).astype(np.float32)
    sample_weight /= sample_weight.mean()
    for fold in range(5):
        fit = folds[order_rows] != fold
        valid = ~fit
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, max_iter=1200, class_weight=None,
                solver="lbfgs", random_state=SEED + seed_offset + fold,
            ),
        )
        estimator.fit(x_train[fit], labels[fit],
                      logisticregression__sample_weight=sample_weight[fit])
        oof[valid] = estimator.predict_proba(x_train[valid])[:, 1]
        tests.append(estimator.predict_proba(x_test)[:, 1])
    return oof, np.mean(tests, axis=0).astype(np.float32)


def minimum_fold_delta(candidate, baseline):
    return min(
        row["f1"] - baseline[fold_name][index]["f1"]
        for fold_name, rows in candidate["folds"].items()
        for index, row in enumerate(rows)
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    labels = arrays["train_labels"].astype(bool)
    ptr = arrays["train_alarm_ptr"]
    order_rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    counts = target_counts(labels, ptr)
    folds_by_name = {
        "station": arrays["train_station_folds"].astype(np.int8),
        "template": arrays["train_folds"].astype(np.int8),
    }
    train_sources = score_sources(arrays, "train")
    test_sources = score_sources(arrays, "test")
    if list(sorted(train_sources)) != list(sorted(test_sources)):
        raise RuntimeError("train/test score sources do not align")
    source_results = {
        name: evaluate(score, ptr, labels, counts, folds_by_name)
        for name, score in train_sources.items()
    }
    source_ranking = sorted(
        ({"name": name, "f1": row["f1"], "tp": row["tp"]}
         for name, row in source_results.items()),
        key=lambda row: (-row["f1"], row["name"]),
    )

    base_name = "v30_station_extra_trees"
    base = source_results[base_name]
    baseline_folds = base["folds"]
    blends = []
    station = train_sources[base_name]
    station_test = test_sources[base_name]
    for source_name, source_score in train_sources.items():
        if source_name == base_name:
            continue
        for station_weight in np.linspace(0.0, 1.0, 21):
            score = station_weight * station + (1.0 - station_weight) * source_score
            result = evaluate(score, ptr, labels, counts, folds_by_name)
            row = {
                "source": source_name,
                "station_weight": float(station_weight),
                "f1": result["f1"],
                "tp": result["tp"],
                "minimum_fold_delta_f1": minimum_fold_delta(result, baseline_folds),
            }
            blends.append(row)
    blends.sort(key=lambda row: (
        -int(row["minimum_fold_delta_f1"] >= 0), -row["f1"],
        -row["minimum_fold_delta_f1"],
    ))

    x_train, feature_names = meta_matrix(
        train_sources, ptr, load_1d(V33 / "domain_train_probability.npy")
    )
    x_test, test_feature_names = meta_matrix(
        test_sources, arrays["test_alarm_ptr"], load_1d(V33 / "domain_test_probability.npy")
    )
    if feature_names != test_feature_names:
        raise RuntimeError("meta feature names do not align")
    importance = np.load(V33 / "importance_weights.npy").astype(np.float32)
    importance /= importance.mean()
    specs = []
    for fold_name, folds in folds_by_name.items():
        for c in (0.003, 0.01, 0.03, 0.1, 0.3):
            for alpha in (0.0, 1.0):
                specs.append((fold_name, folds, c, alpha))
    meta_results, meta_predictions = {}, {}
    for index, (fold_name, folds, c, alpha) in enumerate(specs):
        name = f"{fold_name}_c{c:g}_a{alpha:g}"
        print(f"training {name}", flush=True)
        oof, test_probability = crossfit_logistic(
            x_train, labels.astype(np.int8), x_test, order_rows, folds,
            c, alpha, importance, index * 10,
        )
        result = evaluate(oof, ptr, labels, counts, folds_by_name)
        result["minimum_fold_delta_f1"] = minimum_fold_delta(result, baseline_folds)
        meta_results[name] = result
        meta_predictions[name] = (oof, test_probability)

    meta_ranking = sorted(
        ({
            "name": name, "f1": row["f1"], "tp": row["tp"],
            "minimum_fold_delta_f1": row["minimum_fold_delta_f1"],
        } for name, row in meta_results.items()),
        key=lambda row: (
            -int(row["minimum_fold_delta_f1"] >= 0), -row["f1"],
            -row["minimum_fold_delta_f1"],
        ),
    )
    best_meta = meta_ranking[0]
    best_blend = blends[0]
    gate_passed = bool(
        max(best_meta["f1"], best_blend["f1"]) >= GATE_F1
        and (
            (best_meta["f1"] >= best_blend["f1"] and best_meta["minimum_fold_delta_f1"] >= 0)
            or (best_blend["f1"] > best_meta["f1"] and best_blend["minimum_fold_delta_f1"] >= 0)
        )
    )
    report = {
        "version": "v43-low-dim-calibration-1",
        "score_source_count": len(train_sources),
        "meta_feature_count": int(x_train.shape[1]),
        "budget": int(counts.sum()),
        "baseline": {"name": base_name, **base},
        "source_ranking": source_ranking,
        "best_blends": blends[:30],
        "best_meta": meta_ranking[:20],
        "meta_results": meta_results,
        "gate": {
            "required_f1": GATE_F1,
            "requires_minimum_fold_delta_nonnegative": True,
            "passed": gate_passed,
        },
        "submission": None,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "source_ranking": source_ranking[:10],
        "baseline_f1": base["f1"],
        "best_blend": best_blend,
        "best_meta": best_meta,
        "gate": report["gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

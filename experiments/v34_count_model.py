"""Cross-fitted order root-count model with exact global-budget decoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))
if str(ROOT / "experiments/v30_meta_stack") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments/v30_meta_stack"))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from v30_meta_stack import exact_count_mask, feature_matrix, score_matrix


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
DOMAIN = ROOT / "experiments/v33_domain_adaptation"
V30 = ROOT / "experiments/v30_meta_stack"
OUT = ROOT / "experiments/v34_count_model"
SEED = 20260820
MAX_ROOTS = 8
TRAIN_BASE_P = 3103


def order_matrix(alarm_x, raw_scores, ptr, base_mask, extra_scores=None):
    rows = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = alarm_x[start:stop]
        scores = raw_scores[start:stop]
        sorted_scores = np.sort(scores, axis=0)[::-1]
        top = np.full((8, scores.shape[1]), -1.0, dtype=np.float32)
        top[:min(8, len(scores))] = sorted_scores[:8]
        summary = np.asarray([
            stop - start,
            int(base_mask[start:stop].sum()),
            float(scores.mean()),
            float(scores.std()),
            float(scores.max()),
            float(scores.min()),
        ], dtype=np.float32)
        extra = []
        if extra_scores is not None:
            for extra_score in extra_scores:
                local_extra = extra_score[start:stop]
                sorted_extra = np.sort(local_extra)[::-1]
                extra_top = np.full(8, -1.0, dtype=np.float32)
                extra_top[:min(8, len(local_extra))] = sorted_extra[:8]
                extra.extend(extra_top.tolist())
                gaps = np.zeros(8, dtype=np.float32)
                count = min(7, len(local_extra) - 1)
                if count > 0:
                    gaps[:count] = sorted_extra[:count] - sorted_extra[1:count + 1]
                gaps[7] = sorted_extra[min(7, len(local_extra) - 1)]
                extra.extend(gaps.tolist())
                extra.extend([
                    float(local_extra.mean()), float(local_extra.std()),
                    float(local_extra.max()), float(local_extra.min()),
                ])
        rows.append(np.concatenate([
            local.mean(0), local.std(0), local.min(0), local.max(0),
            top.ravel(), summary,
            np.asarray(extra, dtype=np.float32),
        ]))
    return np.asarray(rows, dtype=np.float32)


def aligned_probability(model, x):
    output = np.full((len(x), MAX_ROOTS), 1e-6, dtype=np.float32)
    probability = model.predict_proba(x)
    for source, cls in enumerate(model.classes_):
        cls = int(cls)
        if 1 <= cls <= MAX_ROOTS:
            output[:, cls - 1] = probability[:, source]
    output /= output.sum(axis=1, keepdims=True)
    return output


def crossfit(x_train, target, x_test, folds, weights, alpha):
    oof = np.zeros((len(target), MAX_ROOTS), dtype=np.float32)
    tests = []
    sample_weight = np.power(weights, alpha).astype(np.float32)
    sample_weight /= sample_weight.mean()
    for fold in range(5):
        fit = folds != fold
        valid = ~fit
        model = ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.55,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + fold + round(alpha * 100),
        )
        model.fit(x_train[fit], target[fit], sample_weight=sample_weight[fit])
        oof[valid] = aligned_probability(model, x_train[valid])
        tests.append(aligned_probability(model, x_test))
    return oof, np.mean(tests, axis=0).astype(np.float32)


def decode_counts(probability, alarm_counts, budget):
    """Maximize summed log probability subject to an exact count budget."""
    n = len(probability)
    if not n <= budget <= int(np.minimum(alarm_counts, MAX_ROOTS).sum()):
        raise ValueError((n, budget, int(np.minimum(alarm_counts, MAX_ROOTS).sum())))
    logp = np.log(np.clip(probability, 1e-9, 1.0))
    neg = -1e30
    dp = np.full(budget + 1, neg, dtype=np.float64)
    dp[0] = 0.0
    back = np.zeros((n, budget + 1), dtype=np.int8)
    for oi in range(n):
        new = np.full(budget + 1, neg, dtype=np.float64)
        choice = np.zeros(budget + 1, dtype=np.int8)
        limit = min(MAX_ROOTS, int(alarm_counts[oi]))
        for count in range(1, limit + 1):
            candidate = dp[:-count] + float(logp[oi, count - 1])
            better = candidate > new[count:]
            if np.any(better):
                target_rows = np.flatnonzero(better) + count
                new[target_rows] = candidate[better]
                choice[target_rows] = count
        dp = new
        back[oi] = choice
    if dp[budget] <= neg / 2:
        raise RuntimeError("count budget is unreachable")
    counts = np.zeros(n, dtype=np.int8)
    remaining = budget
    for oi in range(n - 1, -1, -1):
        count = int(back[oi, remaining])
        if count <= 0:
            raise RuntimeError((oi, remaining))
        counts[oi] = count
        remaining -= count
    if remaining != 0 or int(counts.sum()) != budget:
        raise RuntimeError((remaining, int(counts.sum()), budget))
    return counts


def select_by_counts(scores, ptr, counts):
    mask = np.zeros(len(scores), dtype=bool)
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:int(counts[oi])]
        mask[start + ranked] = True
    return mask


def evaluate(probability, target_count, alarm_counts, node_scores, ptr, labels, budgets):
    argmax = np.argmax(probability, axis=1) + 1
    argmax = np.minimum(argmax, np.minimum(alarm_counts, MAX_ROOTS))
    report = {
        "argmax_accuracy": float(np.mean(argmax == target_count)),
        "argmax_mae": float(np.mean(np.abs(argmax - target_count))),
        "argmax_sum": int(argmax.sum()),
        "budgets": {},
    }
    for budget in budgets:
        counts = decode_counts(probability, alarm_counts, budget)
        mask = select_by_counts(node_scores, ptr, counts)
        tp = int((mask & labels).sum())
        report["budgets"][str(budget)] = {
            "tp": tp,
            "predictions": int(mask.sum()),
            "f1": 2.0 * tp / (int(labels.sum()) + int(mask.sum())),
            "count_accuracy": float(np.mean(counts == target_count)),
            "count_mae": float(np.mean(np.abs(counts - target_count))),
            "count_over": int(np.maximum(counts - target_count, 0).sum()),
            "count_under": int(np.maximum(target_count - counts, 0).sum()),
        }
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    train_x, _ = feature_matrix(arrays, "train")
    test_x, _ = feature_matrix(arrays, "test")
    train_scores = score_matrix(arrays, "train")
    test_scores = score_matrix(arrays, "test")
    train_ptr = arrays["train_alarm_ptr"]
    test_ptr = arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    train_alarm_counts = np.diff(train_ptr).astype(np.int16)
    test_alarm_counts = np.diff(test_ptr).astype(np.int16)
    target_count = np.asarray([
        min(MAX_ROOTS, int(labels[int(start):int(stop)].sum()))
        for start, stop in zip(train_ptr[:-1], train_ptr[1:])
    ], dtype=np.int8)
    capped_true_total = int(target_count.sum())
    base = exact_count_mask(arrays["train_v11"], train_ptr, TRAIN_BASE_P)
    train_extra_scores = [
        np.load(V30 / "station_extra_trees_oof.npy"),
        np.load(DOMAIN / "station_alpha_2_oof.npy"),
        np.load(V30 / "v30_consensus_oof.npy"),
    ]
    test_extra_scores = [
        np.load(V30 / "station_extra_trees_test.npy"),
        np.load(DOMAIN / "station_alpha_2_test.npy"),
        np.load(V30 / "v30_consensus_test.npy"),
    ]
    train_order_x = order_matrix(
        train_x, train_scores, train_ptr, base, train_extra_scores
    )
    # The test base-count feature uses the current prediction budget only as a
    # covariate; final counts are decoded independently below.
    test_base = exact_count_mask(arrays["test_v11"], test_ptr, 1035)
    test_order_x = order_matrix(
        test_x, test_scores, test_ptr, test_base, test_extra_scores
    )
    alarm_weights = np.load(DOMAIN / "importance_weights.npy")
    order_weights = np.asarray([
        float(alarm_weights[int(start):int(stop)].mean())
        for start, stop in zip(train_ptr[:-1], train_ptr[1:])
    ], dtype=np.float32)
    order_weights /= order_weights.mean()

    node_models = {
        "v30_station": np.load(V30 / "station_extra_trees_oof.npy"),
        "v33_alpha2": np.load(DOMAIN / "station_alpha_2_oof.npy"),
    }
    results = {}
    probabilities = []
    for split, fold_key in (("station", "train_station_folds"),
                            ("template", "train_folds")):
        for alpha in (0.0, 1.0, 2.0):
            key = f"{split}_alpha_{alpha:g}"
            print(f"training {key}", flush=True)
            oof, test = crossfit(
                train_order_x, target_count, test_order_x,
                arrays[fold_key].astype(np.int8), order_weights, alpha
            )
            np.save(OUT / f"{key}_oof.npy", oof)
            np.save(OUT / f"{key}_test.npy", test)
            probabilities.append((key, oof, test))
            results[key] = {
                node_name: evaluate(
                    oof, target_count, train_alarm_counts, node_score,
                    train_ptr, labels, (capped_true_total, 3041, 3103)
                )
                for node_name, node_score in node_models.items()
            }

    ensemble_oof = np.mean([item[1] for item in probabilities], axis=0)
    ensemble_test = np.mean([item[2] for item in probabilities], axis=0)
    np.save(OUT / "ensemble_oof.npy", ensemble_oof)
    np.save(OUT / "ensemble_test.npy", ensemble_test)
    results["ensemble"] = {
        node_name: evaluate(
            ensemble_oof, target_count, train_alarm_counts, node_score,
            train_ptr, labels, (capped_true_total, 3041, 3103)
        )
        for node_name, node_score in node_models.items()
    }

    oracle = {}
    for node_name, node_score in node_models.items():
        mask = select_by_counts(node_score, train_ptr, target_count)
        tp = int((mask & labels).sum())
        oracle[node_name] = {
            "tp": tp,
            "predictions": int(mask.sum()),
            "f1": 2.0 * tp / (int(labels.sum()) + int(mask.sum())),
        }
    best = max(
        (
            value["v30_station"]["budgets"][str(capped_true_total)]["f1"], key
        )
        for key, value in results.items()
    )
    report = {
        "version": "v34-count-model-1",
        "train_orders": int(len(target_count)),
        "test_orders": int(len(test_alarm_counts)),
        "target_count_distribution": {
            str(count): int((target_count == count).sum())
            for count in range(1, MAX_ROOTS + 1)
        },
        "capped_true_total": capped_true_total,
        "oracle": oracle,
        "results": results,
        "best_capped_budget_model": best[1],
        "best_capped_budget_f1": best[0],
        "gate": {
            "requires_f1": 0.94,
            "passed": bool(best[0] >= 0.94),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

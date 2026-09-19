"""Strict station-held-out structural-neighbor audit.

This is an offline feasibility experiment.  It uses only normalized, local
alarm structure to retrieve training orders, transfers root-count evidence and
alarm-key label rates, then evaluates a joint count/ranking decoder.  Every
OOF query is restricted to references outside its station fold.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
for value in (ROOT / ".deps", ROOT / "experiments/v30_meta_stack"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from v30_meta_stack import feature_matrix, score_matrix


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V34 = ROOT / "experiments/v34_count_model"
OUT = ROOT / "experiments/v68_structural_neighbor_joint"
SEED = 20260820
MAX_ROOTS = 8
TARGET_BUDGET = 3041
NEIGHBORS = 24


def normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())[:240]


def alarm_key(alarm: dict) -> str:
    # Deliberately excludes station and device IDs: those do not transfer to a
    # held-out site.  Reason/cause/type preserve the failure mechanism.
    return "|".join(
        normalize(alarm.get(name))
        for name in ("title", "reason", "vendor", "device_type", "board_type", "cause", "timeline")
    )


def order_text(order: dict) -> str:
    tokens = []
    for alarm in order["alarms"]:
        for name in ("title", "reason", "vendor", "device_type", "board_type", "cause", "radio", "deployment", "timeline"):
            value = normalize(alarm.get(name))
            if value:
                tokens.append(f"{name}={value}")
    # The target-title summary is available at prediction time and encodes the
    # order's topology role without using the root labels.
    target = normalize(order["alarms"][0].get("target_summary", "")) if order["alarms"] else ""
    if target:
        tokens.append(f"context={target}")
    return " ".join(tokens)


def root_count(order: dict) -> int:
    return min(MAX_ROOTS, sum(int(alarm.get("is_root", 0)) for alarm in order["alarms"]))


def reference_stats(query, references, similarities, labels_by_order=None):
    """Return count-transfer and alarm-transfer features from labeled refs."""
    top = np.argsort(-similarities, kind="stable")[: min(NEIGHBORS, len(references))]
    top = [int(value) for value in top if similarities[value] > 0]
    result = np.zeros(8 + 7, dtype=np.float32)
    alarm_probability = np.full(len(query["alarms"]), np.nan, dtype=np.float32)
    if not top:
        return result, alarm_probability
    weights = np.asarray([max(float(similarities[index]), 1e-5) for index in top], dtype=np.float32)
    weights /= weights.sum()
    counts = np.asarray([root_count(references[index]) for index in top], dtype=np.int8)
    for count, weight in zip(counts, weights):
        result[int(count) - 1] += weight
    ordered = np.asarray([float(similarities[index]) for index in top], dtype=np.float32)
    result[8:] = [
        float(ordered[0]),
        float(ordered.mean()),
        float(ordered.std()),
        float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0]),
        float((ordered >= 0.80).sum()),
        float((ordered >= 0.65).sum()),
        float((ordered >= 0.50).sum()),
    ]
    votes = defaultdict(lambda: [0.0, 0.0])
    for neighbor, weight in zip(top, weights):
        for alarm in references[neighbor]["alarms"]:
            key = alarm_key(alarm)
            votes[key][0] += float(weight) * int(alarm.get("is_root", 0))
            votes[key][1] += float(weight)
    for index, alarm in enumerate(query["alarms"]):
        positive, total = votes[alarm_key(alarm)]
        if total > 0:
            alarm_probability[index] = positive / total
    return result, alarm_probability


def transfer_features(train_records, test_records, folds):
    texts = [order_text(order) for order in train_records]
    test_texts = [order_text(order) for order in test_records]
    vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True, norm="l2")
    train_matrix = vectorizer.fit_transform(texts)
    test_matrix = vectorizer.transform(test_texts)
    train_count = len(train_records)
    test_count = len(test_records)
    order_features = np.zeros((train_count, 15), dtype=np.float32)
    test_order_features = np.zeros((test_count, 15), dtype=np.float32)
    train_alarm = [None] * train_count
    test_alarm = [None] * test_count
    for fold in range(5):
        query = np.flatnonzero(folds == fold)
        reference = np.flatnonzero(folds != fold)
        similarity = (train_matrix[query] @ train_matrix[reference].T).toarray()
        for local, order_index in enumerate(query):
            values, alarms = reference_stats(train_records[int(order_index)], [train_records[int(i)] for i in reference], similarity[local])
            order_features[int(order_index)] = values
            train_alarm[int(order_index)] = alarms
    # All labeled train data may be used for test transfer.  This does not
    # change the fold-honest OOF evaluation above.
    similarity = (test_matrix @ train_matrix.T).toarray()
    for order_index in range(test_count):
        values, alarms = reference_stats(test_records[order_index], train_records, similarity[order_index])
        test_order_features[order_index] = values
        test_alarm[order_index] = alarms
    return order_features, test_order_features, train_alarm, test_alarm


def count_probability(x_train, y, x_test, folds):
    def aligned(model, x):
        out = np.full((len(x), MAX_ROOTS), 1e-7, dtype=np.float32)
        values = model.predict_proba(x)
        for column, cls in enumerate(model.classes_):
            if 1 <= int(cls) <= MAX_ROOTS:
                out[:, int(cls) - 1] = values[:, column]
        out /= out.sum(axis=1, keepdims=True)
        return out
    oof = np.zeros((len(y), MAX_ROOTS), dtype=np.float32)
    test = []
    for fold in range(5):
        fit = folds != fold
        model = ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=3, max_features=0.60,
            class_weight="balanced", n_jobs=-1, random_state=SEED + fold,
        )
        model.fit(x_train[fit], y[fit])
        oof[~fit] = aligned(model, x_train[~fit])
        test.append(aligned(model, x_test))
    return oof, np.mean(test, axis=0).astype(np.float32)


def aggregate_alarm_features(node_x, ptr):
    """Convert the existing alarm-level stack features to order-level input."""
    rows = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        local = node_x[int(start):int(stop)]
        # Mean/spread/extrema preserve the distribution around each order's
        # ranking boundary, while avoiding row IDs or any label-derived input.
        rows.append(np.concatenate([
            local.mean(axis=0), local.std(axis=0),
            local.max(axis=0), local.min(axis=0),
            np.asarray([len(local)], dtype=np.float32),
        ]))
    return np.asarray(rows, dtype=np.float32)


def decode_counts(probability, alarm_counts, budget):
    logp = np.log(np.clip(probability, 1e-9, 1.0))
    dp = np.full(budget + 1, -1e30, dtype=np.float64)
    dp[0] = 0.0
    back = np.zeros((len(probability), budget + 1), dtype=np.int8)
    for order, limit_raw in enumerate(alarm_counts):
        limit = min(int(limit_raw), MAX_ROOTS)
        new = np.full(budget + 1, -1e30, dtype=np.float64)
        choice = np.zeros(budget + 1, dtype=np.int8)
        for count in range(1, limit + 1):
            trial = dp[:-count] + float(logp[order, count - 1])
            better = trial > new[count:]
            new[count:][better] = trial[better]
            choice[count:][better] = count
        dp, back[order] = new, choice
    result = np.zeros(len(probability), dtype=np.int8)
    remaining = budget
    for order in range(len(probability) - 1, -1, -1):
        count = int(back[order, remaining])
        if count < 1:
            raise RuntimeError((order, remaining))
        result[order] = count
        remaining -= count
    return result


def select(scores, ptr, counts):
    mask = np.zeros(len(scores), dtype=bool)
    for order, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        rank = np.argsort(-scores[start:stop], kind="stable")[: int(counts[order])]
        mask[start + rank] = True
    return mask


def rank_blend(base, transfer, ptr, alpha):
    result = base.astype(np.float32).copy()
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local = transfer[start:stop]
        valid = np.isfinite(local)
        if not valid.any():
            continue
        z_base = (base[start:stop] - base[start:stop].mean()) / (base[start:stop].std() + 1e-6)
        centered = np.where(valid, local, np.nanmean(local[valid]))
        z_transfer = (centered - centered.mean()) / (centered.std() + 1e-6)
        result[start:stop] = z_base + alpha * z_transfer
    return result


def metrics(mask, labels, truth):
    tp = int((mask & labels).sum())
    pred = int(mask.sum())
    return {"tp": tp, "predictions": pred, "f1": 2 * tp / (truth + pred)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    train_records, test_records = records["train"], records["test"]
    ptr, test_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    folds = arrays["train_station_folds"].astype(np.int8)
    truth_counts = np.asarray([root_count(order) for order in train_records], dtype=np.int8)
    train_node_base = feature_matrix(arrays, "train")[0]
    test_node_base = feature_matrix(arrays, "test")[0]
    train_order_base = aggregate_alarm_features(train_node_base, ptr)
    test_order_base = aggregate_alarm_features(test_node_base, test_ptr)
    transfer, test_transfer, alarm_transfer, test_alarm_transfer = transfer_features(train_records, test_records, folds)
    x_train = np.column_stack([train_order_base, transfer])
    x_test = np.column_stack([test_order_base, test_transfer])
    oof_count, test_count = count_probability(x_train, truth_counts, x_test, folds)
    # V34 is the strongest existing station-held-out count posterior.  Blend
    # only OOF-to-OOF and test-to-test so retrieval is assessed as an additive
    # signal, never as a replacement for the established model.
    v34_oof = np.load(V34 / "station_alpha_1_oof.npy")
    v34_test = np.load(V34 / "station_alpha_1_test.npy")
    station = np.load(V30 / "station_extra_trees_oof.npy")
    test_station = np.load(V30 / "station_extra_trees_test.npy")
    train_alarm_transfer = np.concatenate([np.nan_to_num(value, nan=0.5) for value in alarm_transfer])
    test_alarm_transfer_vector = np.concatenate([np.nan_to_num(value, nan=0.5) for value in test_alarm_transfer])
    budgets = (int(truth_counts.sum()), 3041, TARGET_BUDGET)
    results = {}
    count_models = {"retrieval": (oof_count, test_count), "v34": (v34_oof, v34_test)}
    for weight in (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        if weight not in (0.0, 1.0):
            count_models[f"v34_plus_retrieval_{weight:g}"] = (
                (1.0 - weight) * v34_oof + weight * oof_count,
                (1.0 - weight) * v34_test + weight * test_count,
            )
    for count_name, (count_oof, _) in count_models.items():
        for alpha in (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
            score = rank_blend(station, train_alarm_transfer, ptr, alpha)
            for budget in budgets:
                counts = decode_counts(count_oof, np.diff(ptr), budget)
                result = metrics(select(score, ptr, counts), labels, int(labels.sum()))
                result["count_accuracy"] = float(np.mean(counts == truth_counts))
                result["count_mae"] = float(np.mean(np.abs(counts - truth_counts)))
                per_fold = []
                for fold in range(5):
                    rows = np.repeat(folds, np.diff(ptr)) == fold
                    local = select(score, ptr, counts) & rows
                    per_fold.append(metrics(local[rows], labels[rows], int(labels[rows].sum()))["f1"])
                result["minimum_fold_f1"] = min(per_fold)
                result["fold_f1"] = per_fold
                results[f"count_{count_name}_alpha_{alpha:g}_budget_{budget}"] = result
    best_key, best_result = max(results.items(), key=lambda item: item[1]["f1"])
    parts = best_key.split("_")
    alpha = float(parts[parts.index("alpha") + 1])
    count_name = best_key.split("_alpha_")[0].removeprefix("count_")
    selected_count_test = count_models[count_name][1]
    test_score = rank_blend(test_station, test_alarm_transfer_vector, test_ptr, alpha)
    # Match the selected OOF budget, but do not emit a submission: this audit
    # is accepted only if it clears the deliberately high feasibility gate.
    selected_budget = int(best_key.rsplit("_", 1)[1])
    test_counts = decode_counts(selected_count_test, np.diff(test_ptr), selected_budget - (3041 - 1035))
    report = {
        "version": "v68-structural-neighbor-joint-audit-1",
        "method": "station-fold-isolated TF-IDF structural retrieval + count transfer + alarm-key rank blend",
        "train_orders": len(train_records), "test_orders": len(test_records),
        "oof_results": results, "best": {"key": best_key, **best_result},
        "test_projection": {
            "selected_count_model": count_name, "selected_alpha": alpha, "selected_budget": int(test_counts.sum()),
            "count_distribution": {str(k): int((test_counts == k).sum()) for k in range(1, MAX_ROOTS + 1)},
            "transfer_alarm_coverage": float(np.isfinite(np.concatenate(test_alarm_transfer)).mean()),
        },
        "gate": {"required_oof_f1": 0.94, "required_minimum_fold_f1": 0.92,
                 "passed": bool(best_result["f1"] >= 0.94 and best_result["minimum_fold_f1"] >= 0.92),
                 "submission_emitted": False},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(OUT / "count_oof.npy", oof_count)
    np.save(OUT / "count_test.npy", test_count)
    np.save(OUT / "test_rank_score.npy", test_score)
    print(json.dumps({"best": report["best"], "test_projection": report["test_projection"], "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

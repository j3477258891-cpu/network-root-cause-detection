"""Cross-fitted marginal root allocator with an exact global budget.

Instead of predicting one count class per order, this model estimates the TP
utility of selecting the k-th ranked alarm.  A dynamic program then chooses a
prefix length for every order under the known total-root budget.
"""

from __future__ import annotations

import gzip
import hashlib
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

from sklearn.ensemble import ExtraTreesClassifier

from build_cross_order_probes import load_submission, write_submission
from v30_meta_stack import feature_matrix, score_matrix


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
DOMAIN = ROOT / "experiments/v33_domain_adaptation"
OUT = ROOT / "experiments/v42_marginal_count"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
MAX_ROOTS = 8
TRUE_ROOTS_TEST = 1044
SEED = 20260820
GATE_F1 = 0.94


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked_context(score_sets, ptr):
    rows = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        values = [float(stop - start) / 32.0]
        for scores in score_sets:
            local = scores[start:stop]
            ranked = np.sort(local)[::-1]
            top = np.full(MAX_ROOTS, -1.0, dtype=np.float32)
            top[:min(MAX_ROOTS, len(ranked))] = ranked[:MAX_ROOTS]
            gaps = np.zeros(MAX_ROOTS, dtype=np.float32)
            gap_count = min(MAX_ROOTS - 1, len(ranked) - 1)
            if gap_count > 0:
                gaps[:gap_count] = ranked[:gap_count] - ranked[1:gap_count + 1]
            values.extend(top.tolist())
            values.extend(gaps.tolist())
            values.extend([
                float(local.mean()), float(local.std()),
                float(local.max()), float(local.min()),
            ])
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def marginal_rows(node_x, rank_score, ptr, context, labels=None, weights=None):
    features, targets, order_rows, node_rows, ranks, row_weights = [], [], [], [], [], []
    limits = np.minimum(np.diff(ptr), MAX_ROOTS).astype(np.int8)
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-rank_score[start:stop], kind="stable")[:int(limits[oi])]
        ranked = start + ranked
        local_scores = rank_score[ranked]
        for rank, node in enumerate(ranked, 1):
            previous = float(local_scores[rank - 2]) if rank > 1 else 1.0
            current = float(local_scores[rank - 1])
            following = float(local_scores[rank]) if rank < len(local_scores) else 0.0
            positional = np.asarray([
                rank / MAX_ROOTS,
                limits[oi] / MAX_ROOTS,
                current,
                previous - current,
                current - following,
                current / max(previous, 1e-6),
            ], dtype=np.float32)
            features.append(np.concatenate([node_x[node], context[oi], positional]))
            order_rows.append(oi)
            node_rows.append(int(node))
            ranks.append(rank)
            if labels is not None:
                targets.append(int(labels[node]))
            if weights is not None:
                row_weights.append(float(weights[node]))
    return {
        "x": np.asarray(features, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.int8) if labels is not None else None,
        "orders": np.asarray(order_rows, dtype=np.int32),
        "nodes": np.asarray(node_rows, dtype=np.int32),
        "ranks": np.asarray(ranks, dtype=np.int8),
        "weights": np.asarray(row_weights, dtype=np.float32) if weights is not None else None,
        "limits": limits,
    }


def model(seed):
    return ExtraTreesClassifier(
        n_estimators=450,
        max_depth=None,
        min_samples_leaf=5,
        max_features=0.5,
        class_weight=None,
        n_jobs=-1,
        random_state=seed,
    )


def crossfit(train, test, folds, alpha, seed_offset):
    oof = np.zeros(len(train["y"]), dtype=np.float32)
    tests = []
    order_folds = folds[train["orders"]]
    sample_weight = np.power(train["weights"], alpha).astype(np.float32)
    sample_weight /= sample_weight.mean()
    for fold in range(5):
        fit = order_folds != fold
        valid = ~fit
        estimator = model(SEED + seed_offset + fold)
        estimator.fit(train["x"][fit], train["y"][fit], sample_weight=sample_weight[fit])
        oof[valid] = estimator.predict_proba(train["x"][valid])[:, 1]
        tests.append(estimator.predict_proba(test["x"])[:, 1])
    return oof, np.mean(tests, axis=0).astype(np.float32)


def utility_matrix(probability, marginal, order_count):
    output = np.full((order_count, MAX_ROOTS), -1e12, dtype=np.float64)
    cumulative = np.zeros(order_count, dtype=np.float64)
    for value, oi, rank in zip(probability, marginal["orders"], marginal["ranks"]):
        cumulative[oi] += float(value)
        output[oi, int(rank) - 1] = cumulative[oi]
    return output


def decode_utility(utility, limits, budget):
    n = len(utility)
    if not n <= budget <= int(limits.sum()):
        raise ValueError((n, budget, int(limits.sum())))
    neg = -1e30
    dp = np.full(budget + 1, neg, dtype=np.float64)
    dp[0] = 0.0
    back = np.zeros((n, budget + 1), dtype=np.int8)
    for oi in range(n):
        new = np.full(budget + 1, neg, dtype=np.float64)
        choice = np.zeros(budget + 1, dtype=np.int8)
        for count in range(1, int(limits[oi]) + 1):
            candidate = dp[:-count] + utility[oi, count - 1]
            better = candidate > new[count:]
            if np.any(better):
                rows = np.flatnonzero(better) + count
                new[rows] = candidate[better]
                choice[rows] = count
        dp = new
        back[oi] = choice
    if dp[budget] <= neg / 2:
        raise RuntimeError("budget is unreachable")
    counts = np.zeros(n, dtype=np.int8)
    remaining = budget
    for oi in range(n - 1, -1, -1):
        count = int(back[oi, remaining])
        if count <= 0:
            raise RuntimeError((oi, remaining))
        counts[oi] = count
        remaining -= count
    if remaining != 0:
        raise RuntimeError(remaining)
    return counts


def selected_mask(rank_score, ptr, counts, order_subset=None):
    output = np.zeros(len(rank_score), dtype=bool)
    use = range(len(counts)) if order_subset is None else order_subset
    for oi in use:
        start, stop = int(ptr[oi]), int(ptr[oi + 1])
        ranked = np.argsort(-rank_score[start:stop], kind="stable")[:int(counts[oi])]
        output[start + ranked] = True
    return output


def metrics(counts, rank_score, ptr, labels, target_counts, order_subset=None):
    mask = selected_mask(rank_score, ptr, counts, order_subset)
    if order_subset is None:
        truth_mask = np.ones(len(labels), dtype=bool)
        local_counts = counts
        local_target = target_counts
    else:
        truth_mask = np.zeros(len(labels), dtype=bool)
        for oi in order_subset:
            truth_mask[int(ptr[oi]):int(ptr[oi + 1])] = True
        local_counts = counts[order_subset]
        local_target = target_counts[order_subset]
    tp = int((mask & labels).sum())
    predictions = int(mask.sum())
    truth = int((labels & truth_mask).sum())
    return {
        "tp": tp,
        "predictions": predictions,
        "truth": truth,
        "f1": 2 * tp / (truth + predictions),
        "count_accuracy": float(np.mean(local_counts == local_target)),
        "count_mae": float(np.mean(np.abs(local_counts - local_target))),
    }


def fold_audit(utility, station_utility, limits, folds, target_counts,
               rank_score, ptr, labels):
    output = []
    for fold in range(5):
        orders = np.flatnonzero(folds == fold)
        budget = int(target_counts[orders].sum())
        local_counts = decode_utility(utility[orders], limits[orders], budget)
        local_station = decode_utility(station_utility[orders], limits[orders], budget)
        counts = np.ones(len(limits), dtype=np.int8)
        baseline = np.ones(len(limits), dtype=np.int8)
        counts[orders] = local_counts
        baseline[orders] = local_station
        result = metrics(counts, rank_score, ptr, labels, target_counts, orders)
        base_result = metrics(baseline, rank_score, ptr, labels, target_counts, orders)
        result.update({
            "fold": fold,
            "baseline_f1": base_result["f1"],
            "delta_f1": result["f1"] - base_result["f1"],
        })
        output.append(result)
    return output


def roots_from_counts(records, score, ptr, counts):
    roots = {}
    for oi, order in enumerate(records):
        start, stop = int(ptr[oi]), int(ptr[oi + 1])
        ranked = np.argsort(-score[start:stop], kind="stable")[:int(counts[oi])]
        values = []
        for local in ranked:
            alarm = order["alarms"][int(local)]
            source = alarm["source"]
            values.append({
                "@rid": alarm["rid"],
                "title": source.get("title", ""),
                "location": source.get("location", ""),
                "reason": source.get("reason", ""),
            })
        roots[order["order_id"]] = values
    return roots


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    train_x, _ = feature_matrix(arrays, "train")
    test_x, _ = feature_matrix(arrays, "test")
    train_raw = score_matrix(arrays, "train")
    test_raw = score_matrix(arrays, "test")
    train_ptr = arrays["train_alarm_ptr"]
    test_ptr = arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(bool)
    target_counts = np.asarray([
        min(MAX_ROOTS, int(labels[int(start):int(stop)].sum()))
        for start, stop in zip(train_ptr[:-1], train_ptr[1:])
    ], dtype=np.int8)
    train_budget = int(target_counts.sum())

    train_station = np.load(V30 / "station_extra_trees_oof.npy")
    test_station = np.load(V30 / "station_extra_trees_test.npy")
    train_domain = np.load(DOMAIN / "station_alpha_2_oof.npy")
    test_domain = np.load(DOMAIN / "station_alpha_2_test.npy")
    train_consensus = np.load(V30 / "v30_consensus_oof.npy")
    test_consensus = np.load(V30 / "v30_consensus_test.npy")
    train_sets = [train_station, train_domain, train_consensus] + [train_raw[:, i] for i in range(train_raw.shape[1])]
    test_sets = [test_station, test_domain, test_consensus] + [test_raw[:, i] for i in range(test_raw.shape[1])]
    train_context = ranked_context(train_sets, train_ptr)
    test_context = ranked_context(test_sets, test_ptr)
    importance = np.load(DOMAIN / "importance_weights.npy").astype(np.float32)
    importance /= importance.mean()
    train = marginal_rows(
        train_x, train_station, train_ptr, train_context, labels, importance
    )
    test = marginal_rows(test_x, test_station, test_ptr, test_context)

    specs = (
        ("station_alpha0", "train_station_folds", 0.0, 0),
        ("station_alpha1", "train_station_folds", 1.0, 100),
        ("template_alpha0", "train_folds", 0.0, 200),
        ("template_alpha1", "train_folds", 1.0, 300),
    )
    model_results, oofs, tests = {}, [], []
    limits = train["limits"]
    station_probability = train_station[train["nodes"]]
    station_utility = utility_matrix(station_probability, train, len(limits))
    station_counts = decode_utility(station_utility, limits, train_budget)
    baseline = metrics(
        station_counts, train_station, train_ptr, labels, target_counts
    )
    for name, fold_key, alpha, seed_offset in specs:
        print(f"training {name}", flush=True)
        oof, test_probability = crossfit(
            train, test, arrays[fold_key].astype(np.int8), alpha, seed_offset
        )
        np.save(OUT / f"{name}_oof.npy", oof)
        np.save(OUT / f"{name}_test.npy", test_probability)
        oofs.append(oof)
        tests.append(test_probability)
        utility = utility_matrix(oof, train, len(limits))
        counts = decode_utility(utility, limits, train_budget)
        result = metrics(counts, train_station, train_ptr, labels, target_counts)
        result["folds"] = fold_audit(
            utility, station_utility, limits, arrays[fold_key].astype(np.int8),
            target_counts, train_station, train_ptr, labels,
        )
        result["minimum_fold_delta_f1"] = min(row["delta_f1"] for row in result["folds"])
        model_results[name] = result

    ensemble_oof = np.mean(oofs, axis=0).astype(np.float32)
    ensemble_test = np.mean(tests, axis=0).astype(np.float32)
    np.save(OUT / "ensemble_oof.npy", ensemble_oof)
    np.save(OUT / "ensemble_test.npy", ensemble_test)
    ensemble_utility = utility_matrix(ensemble_oof, train, len(limits))
    ensemble_counts = decode_utility(ensemble_utility, limits, train_budget)
    ensemble_result = metrics(
        ensemble_counts, train_station, train_ptr, labels, target_counts
    )
    fold_sets = {
        "station": arrays["train_station_folds"].astype(np.int8),
        "template": arrays["train_folds"].astype(np.int8),
    }
    ensemble_result["fold_audits"] = {}
    for name, folds in fold_sets.items():
        rows = fold_audit(
            ensemble_utility, station_utility, limits, folds, target_counts,
            train_station, train_ptr, labels,
        )
        ensemble_result["fold_audits"][name] = rows
    ensemble_result["minimum_fold_delta_f1"] = min(
        row["delta_f1"]
        for rows in ensemble_result["fold_audits"].values()
        for row in rows
    )

    oracle_probability = labels[train["nodes"]].astype(np.float64)
    oracle_utility = utility_matrix(oracle_probability, train, len(limits))
    prefix_oracle_counts = decode_utility(oracle_utility, limits, train_budget)
    prefix_oracle = metrics(
        prefix_oracle_counts, train_station, train_ptr, labels, target_counts
    )
    count_oracle = metrics(
        target_counts, train_station, train_ptr, labels, target_counts
    )

    test_utility = utility_matrix(ensemble_test, test, len(test["limits"]))
    test_counts = decode_utility(test_utility, test["limits"], TRUE_ROOTS_TEST)
    gate_passed = bool(
        ensemble_result["f1"] >= GATE_F1
        and ensemble_result["minimum_fold_delta_f1"] >= 0.0
    )
    submission = None
    if gate_passed:
        with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
            records = json.load(handle)["test"]
        order_ids, _ = load_submission(CHAMPION)
        roots = roots_from_counts(records, test_station, test_ptr, test_counts)
        path = OUT / "v42_marginal_budget1044.csv"
        write_submission(path, order_ids, roots)
        submission = {
            "path": str(path), "predictions": int(test_counts.sum()),
            "sha256": sha256(path),
        }

    report = {
        "version": "v42-marginal-count-1",
        "feature_count": int(train["x"].shape[1]),
        "marginal_train_rows": int(len(train["x"])),
        "train_budget": train_budget,
        "test_budget": TRUE_ROOTS_TEST,
        "baseline_global_station": baseline,
        "models": model_results,
        "ensemble": ensemble_result,
        "oracles": {
            "true_count": count_oracle,
            "best_prefix_at_exact_budget": prefix_oracle,
        },
        "test_count_distribution": {
            str(count): int((test_counts == count).sum())
            for count in range(1, MAX_ROOTS + 1)
        },
        "gate": {
            "required_oof_f1": GATE_F1,
            "requires_every_fold_not_below_baseline": True,
            "passed": gate_passed,
        },
        "submission": submission,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": baseline,
        "models": {name: {
            "f1": row["f1"],
            "minimum_fold_delta_f1": row["minimum_fold_delta_f1"],
        } for name, row in model_results.items()},
        "ensemble": {
            "f1": ensemble_result["f1"],
            "minimum_fold_delta_f1": ensemble_result["minimum_fold_delta_f1"],
        },
        "oracles": report["oracles"],
        "gate": report["gate"],
        "submission": submission,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

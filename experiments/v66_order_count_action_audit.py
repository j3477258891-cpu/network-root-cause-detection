"""Audit whether one-step per-order count changes can plausibly reach F1=.95.

This is deliberately an *offline gate*, not a submission generator.  It uses
the V11 exact-budget selection as the train analogue of the public champion,
then learns only from station-held-out order actions:

* add: include the highest station-ranked currently unselected alarm;
* delete: remove the lowest station-ranked currently selected alarm.

For each action the relevant target-margin change at F1=.95 is known exactly
on train.  This avoids the invalid shortcut of treating a correct deletion as
an increase in TP: a deletion always lowers P, and loses one TP when the
removed alarm is a true root.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
for value in (ROOT / ".deps", ROOT / "experiments/v30_meta_stack"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sklearn.ensemble import ExtraTreesClassifier

from build_cross_order_probes import apply_actions, load_submission
from v30_meta_stack import exact_count_mask, feature_matrix, score_matrix


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = ROOT / "experiments/v30_meta_stack"
V33 = ROOT / "experiments/v33_domain_adaptation"
V37 = ROOT / "experiments/v37_online_equations"
OUT = ROOT / "experiments/v66_order_count_action_audit"
CHAMPION = V37 / "submissions/v37_exact_corrections.csv"

MAX_ROOTS = 8
TRAIN_BUDGET = 3103
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
TARGET = 0.95
SEED = 20260820


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_from_submission(records, roots) -> np.ndarray:
    selected = np.zeros(sum(len(row["alarms"]) for row in records), dtype=bool)
    offset = 0
    for row in records:
        chosen = {item["@rid"] for item in roots[row["order_id"]]}
        selected[offset:offset + len(row["alarms"])] = [
            alarm["rid"] in chosen for alarm in row["alarms"]
        ]
        offset += len(row["alarms"])
    return selected


def rank_positions(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.int16)
    result[np.argsort(-values, kind="stable")] = np.arange(1, len(values) + 1)
    return result


def action_rows(node_x, raw_scores, station, domain, consensus, ptr, base, labels=None):
    """Return exactly one add and/or delete action per order."""
    rows, targets, orders, nodes, kinds, meta = [], [], [], [], [], []
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        count = int(base[start:stop].sum())
        local_station = station[start:stop]
        local_domain = domain[start:stop]
        local_consensus = consensus[start:stop]
        local_v11 = raw_scores[start:stop, 0]
        station_rank = rank_positions(local_station)
        domain_rank = rank_positions(local_domain)
        consensus_rank = rank_positions(local_consensus)
        v11_rank = rank_positions(local_v11)
        boundary_selected = float(local_station[base[start:stop]].min())
        boundary_unselected = float(local_station[~base[start:stop]].max()) if (~base[start:stop]).any() else -1.0
        order_summary = np.asarray([
            (stop - start) / 32.0, count / MAX_ROOTS,
            float(local_station.mean()), float(local_station.std()),
            float(local_station.max()), float(local_station.min()),
            float(local_domain.mean()), float(local_consensus.mean()),
            boundary_selected, boundary_unselected,
            boundary_unselected - boundary_selected,
        ], dtype=np.float32)

        candidates = []
        if count < min(MAX_ROOTS, stop - start):
            # Add precisely the next station candidate.  Selecting a lower one
            # without this node would violate the intended prefix count policy.
            local = int(np.argmax(np.where(~base[start:stop], local_station, -np.inf)))
            candidates.append(("add", local))
        if count > 1:
            local = int(np.argmin(np.where(base[start:stop], local_station, np.inf)))
            candidates.append(("delete", local))

        for kind, local in candidates:
            row = start + local
            chosen = bool(base[row])
            if (kind == "add") == chosen:
                raise RuntimeError((oi, kind, local, chosen))
            positional = np.asarray([
                1.0 if kind == "add" else 0.0,
                station_rank[local] / max(stop - start, 1),
                domain_rank[local] / max(stop - start, 1),
                consensus_rank[local] / max(stop - start, 1),
                v11_rank[local] / max(stop - start, 1),
                float(local_station[local]), float(local_domain[local]), float(local_consensus[local]),
                float(local_v11[local]), float(local_station[local] - boundary_selected),
                float(local_station[local] - boundary_unselected),
            ], dtype=np.float32)
            rows.append(np.concatenate([node_x[row], raw_scores[row], order_summary, positional]))
            orders.append(oi)
            nodes.append(row)
            kinds.append(kind)
            meta.append({
                "order_index": oi, "node_row": row, "kind": kind,
                "count": count, "station_rank": int(station_rank[local]),
                "station_score": float(local_station[local]),
            })
            if labels is not None:
                targets.append(int(labels[row]))
    return {
        "x": np.asarray(rows, dtype=np.float32),
        "y": None if labels is None else np.asarray(targets, dtype=np.int8),
        "orders": np.asarray(orders, dtype=np.int32),
        "nodes": np.asarray(nodes, dtype=np.int32),
        "kinds": np.asarray(kinds),
        "meta": meta,
    }


def fit_predict(train, test, folds):
    """Strict station-held-out action probabilities.

    Calibration is reported separately.  Calibrating an ExtraTrees model on
    its own fit predictions would look better than it is, so this gate keeps
    the outer-fold probabilities raw rather than introducing that leakage.
    """
    oof = np.zeros(len(train["y"]), dtype=np.float32)
    test_predictions = []
    train_folds = folds[train["orders"]]
    for fold in range(5):
        fit = train_folds != fold
        valid = ~fit
        model = ExtraTreesClassifier(
            n_estimators=900, min_samples_leaf=8, max_features=0.60,
            class_weight="balanced", n_jobs=-1, random_state=SEED + fold,
        )
        model.fit(train["x"][fit], train["y"][fit])
        oof[valid] = model.predict_proba(train["x"][valid])[:, 1]
        test_predictions.append(model.predict_proba(test["x"])[:, 1])
    return oof, np.mean(test_predictions, axis=0).astype(np.float32)


def margin(kind: str, label: int) -> float:
    """Delta of (2*TP - TARGET*P) induced by the action."""
    return (2.0 * label - TARGET) if kind == "add" else (TARGET - 2.0 * label)


def expected_margin(kind: str, probability: float) -> float:
    return (2.0 * probability - TARGET) if kind == "add" else (TARGET - 2.0 * probability)


def choose_nonconflicting(rows, probabilities, labels=None, require_positive=True):
    """At most one action per order; select by expected, then actual margin."""
    best = {}
    for i, (oi, kind) in enumerate(zip(rows["orders"], rows["kinds"])):
        score = expected_margin(str(kind), float(probabilities[i]))
        if require_positive and score <= 0:
            continue
        if int(oi) not in best or score > best[int(oi)][1]:
            best[int(oi)] = (i, score)
    indices = np.asarray([value[0] for _, value in sorted(best.items())], dtype=np.int32)
    estimated = float(sum(value[1] for value in best.values()))
    actual = None
    if labels is not None:
        actual = float(sum(margin(str(rows["kinds"][i]), int(labels[i])) for i in indices))
    return indices, estimated, actual


def action_summary(rows, probabilities, labels=None, top_k=(5, 10, 20, 40, 80, 160, 320)):
    candidates = []
    for i, (oi, kind) in enumerate(zip(rows["orders"], rows["kinds"])):
        expected = expected_margin(str(kind), float(probabilities[i]))
        actual = None if labels is None else margin(str(kind), int(labels[i]))
        candidates.append((expected, i, int(oi), str(kind), actual))
    # Deduplicate by order before producing a trajectory.
    best = {}
    for row in sorted(candidates, reverse=True):
        if row[0] <= 0 or row[2] in best:
            continue
        best[row[2]] = row
    ranked = sorted(best.values(), reverse=True)
    output = []
    for limit in top_k:
        subset = ranked[:limit]
        if not subset:
            continue
        output.append({
            "actions": len(subset),
            "expected_margin": float(sum(item[0] for item in subset)),
            "actual_margin": None if labels is None else float(sum(item[4] for item in subset)),
            "adds": int(sum(item[3] == "add" for item in subset)),
            "deletes": int(sum(item[3] == "delete" for item in subset)),
            "expected_tp_delta": float(sum((probabilities[item[1]] if item[3] == "add" else probabilities[item[1]] - 1) for item in subset)),
            "actual_tp_delta": None if labels is None else int(sum((int(labels[item[1]]) if item[3] == "add" else int(labels[item[1]]) - 1) for item in subset)),
        })
    return output, ranked


def stratified_calibration(rows, probabilities, labels):
    result = {}
    for kind in ("add", "delete"):
        mask = rows["kinds"] == kind
        if not mask.any():
            continue
        bins = np.quantile(probabilities[mask], np.linspace(0, 1, 6))
        bins[0] -= 1e-6
        for index in range(1, len(bins)):
            if bins[index] <= bins[index - 1]:
                bins[index] = bins[index - 1] + 1e-6
        groups = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            use = mask & (probabilities > lo) & (probabilities <= hi)
            if not use.any():
                continue
            groups.append({
                "count": int(use.sum()), "mean_probability": float(probabilities[use].mean()),
                "positive_rate": float(labels[use].mean()),
                "actual_margin": float(sum(margin(kind, int(y)) for y in labels[use])),
            })
        result[kind] = groups
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        test_records = json.load(handle)["test"]
    train_x, _ = feature_matrix(arrays, "train")
    test_x, _ = feature_matrix(arrays, "test")
    train_raw = score_matrix(arrays, "train")
    test_raw = score_matrix(arrays, "test")
    train_ptr, test_ptr = arrays["train_alarm_ptr"], arrays["test_alarm_ptr"]
    labels = arrays["train_labels"].astype(np.int8)
    train_base = exact_count_mask(arrays["train_v11"], train_ptr, TRAIN_BUDGET)

    order_ids, champion_roots = load_submission(CHAMPION)
    if len(order_ids) != len(test_records):
        raise RuntimeError((len(order_ids), len(test_records)))
    test_base = selected_from_submission(test_records, champion_roots)
    if int(test_base.sum()) != BASE_P:
        raise RuntimeError((int(test_base.sum()), BASE_P))

    train_station = np.load(V30 / "station_extra_trees_oof.npy")
    test_station = np.load(V30 / "station_extra_trees_test.npy")
    train_domain = np.load(V33 / "station_alpha_2_oof.npy")
    test_domain = np.load(V33 / "station_alpha_2_test.npy")
    train_consensus = np.load(V30 / "v30_consensus_oof.npy")
    test_consensus = np.load(V30 / "v30_consensus_test.npy")
    train = action_rows(train_x, train_raw, train_station, train_domain, train_consensus, train_ptr, train_base, labels)
    test = action_rows(test_x, test_raw, test_station, test_domain, test_consensus, test_ptr, test_base)
    oof_probability, test_probability = fit_predict(train, test, arrays["train_station_folds"].astype(np.int8))

    # Oracle proves the maximum recoverable signal in this restricted action
    # universe.  It is not a forecast and must never be submitted as a model.
    oracle_prob = train["y"].astype(np.float32)
    _, _, oracle_actual = choose_nonconflicting(train, oracle_prob, train["y"])
    chosen, expected, actual = choose_nonconflicting(train, oof_probability, train["y"])
    oof_curve, _ = action_summary(train, oof_probability, train["y"])
    test_curve, test_ranked = action_summary(test, test_probability)
    needed = TARGET * (TRUE_ROOTS + BASE_P) - 2 * BASE_TP
    max_expected_test = float(sum(row[0] for row in test_ranked))
    test_actions = []
    for rank, (score, index, oi, kind, _) in enumerate(test_ranked, 1):
        node = int(test["nodes"][index])
        local = node - int(test_ptr[oi])
        alarm = test_records[oi]["alarms"][local]
        test_actions.append({
            "rank": rank, "order_id": test_records[oi]["order_id"], "rid": alarm["rid"],
            "kind": kind, "expected_margin": float(score),
            "probability_true_root": float(test_probability[index]),
            "station_rank": test["meta"][index]["station_rank"],
            "station_score": test["meta"][index]["station_score"],
            "title": alarm.get("title", ""),
        })

    train_base_tp = int((train_base & labels.astype(bool)).sum())
    report = {
        "version": "v66-order-count-action-audit-1",
        "target": TARGET,
        "test_baseline": {"tp": BASE_TP, "predictions": BASE_P, "f1": 2 * BASE_TP / (TRUE_ROOTS + BASE_P), "target_margin_needed": needed, "sha256": sha256(CHAMPION)},
        "train_analogue": {"predictions": int(train_base.sum()), "tp": train_base_tp, "truth": int(labels.sum()), "f1": 2 * train_base_tp / (int(labels.sum()) + int(train_base.sum()))},
        "candidate_counts": {"train": int(len(train["y"])), "test": int(len(test["x"])), "test_orders": int(len(set(test["orders"].tolist())))},
        "strict_station_oof": {
            "chosen_positive_actions": int(len(chosen)), "estimated_margin": expected, "actual_margin": actual,
            "oracle_action_universe_margin": oracle_actual,
            "calibration": stratified_calibration(train, oof_probability, train["y"]),
            "curve": oof_curve,
        },
        "test_projection": {
            "positive_nonconflicting_actions": int(len(test_ranked)), "expected_margin": max_expected_test,
            "needed_margin": needed, "coverage_ratio": max_expected_test / needed if needed > 0 else None,
            "curve": test_curve,
            "feasibility_gate": bool(max_expected_test >= needed and actual is not None and actual >= 0),
            "warning": "Expected margins are OOF-calibrated estimates, not leaderboard evidence. No submission is emitted until the OOF trajectory and coverage gate are both strong.",
        },
        "candidate_actions": test_actions,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(OUT / "train_oof_probability.npy", oof_probability)
    np.save(OUT / "test_probability.npy", test_probability)
    print(json.dumps({
        "train_f1": report["train_analogue"]["f1"],
        "candidate_counts": report["candidate_counts"],
        "oof": {"estimated_margin": expected, "actual_margin": actual, "oracle_margin": oracle_actual},
        "test": {"expected_margin": max_expected_test, "needed_margin": needed, "gate": report["test_projection"]["feasibility_gate"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

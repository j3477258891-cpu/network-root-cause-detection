"""Probabilistic MAP inference under exact historical leaderboard equations."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
V30 = ROOT / "experiments/v30_meta_stack"
EXPERIMENTS = ROOT / "experiments"
for value in (DEPS, V30, EXPERIMENTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LogisticRegression

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import (
    CHAMPION, CHAMPION_TP, TRUE_ROOTS, build_system, collect_scored_submissions,
)


DATA = EXPERIMENTS / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V16 = EXPERIMENTS / "v16"
V25 = EXPERIMENTS / "v25_ensemble/outputs"
V33 = EXPERIMENTS / "v33_domain_adaptation"
V37 = EXPERIMENTS / "v37_online_equations"
V11_REPORT = EXPERIMENTS / "submissions/template_ranked/v11_constrained_report.json"
OUT = EXPERIMENTS / "v45_constrained_map"
BASE_TP_AFTER_EXACT = CHAMPION_TP + 1
BASE_P = 1035
MAX_ROOTS = 8


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


def platt(train_score, labels, test_score):
    train_logit = np.log(np.clip(train_score, 1e-5, 1 - 1e-5) / np.clip(1 - train_score, 1e-5, 1))
    test_logit = np.log(np.clip(test_score, 1e-5, 1 - 1e-5) / np.clip(1 - test_score, 1e-5, 1))
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight=None)
    model.fit(train_logit.reshape(-1, 1), labels)
    return model.predict_proba(test_logit.reshape(-1, 1))[:, 1].astype(np.float64)


def priors(arrays):
    labels = arrays["train_labels"].astype(np.int8)
    source = {
        "v11": (arrays["train_v11"], arrays["test_v11"]),
        "v13": (arrays["train_v13"], arrays["test_v13"]),
        "v19": (arrays["train_v19"], arrays["test_v19"]),
        "station": (
            np.load(V30 / "station_extra_trees_oof.npy"),
            np.load(V30 / "station_extra_trees_test.npy"),
        ),
        "domain": (
            np.load(V33 / "station_alpha_2_oof.npy"),
            np.load(V33 / "station_alpha_2_test.npy"),
        ),
        "consensus": (
            np.load(V30 / "v30_consensus_oof.npy"),
            np.load(V30 / "v30_consensus_test.npy"),
        ),
    }
    output = {}
    calibrated = {}
    for name, (train_score, test_score) in source.items():
        calibrated[name] = platt(train_score, labels, test_score)
        output[f"platt_{name}"] = calibrated[name]
    output["platt_station_domain_mean"] = 0.5 * calibrated["station"] + 0.5 * calibrated["domain"]
    output["platt_station_domain_blend"] = 0.4 * calibrated["station"] + 0.6 * calibrated["domain"]
    output["raw_station_domain_blend"] = (
        0.4 * source["station"][1] + 0.6 * source["domain"][1]
    ).astype(np.float64)
    return output


def solve_map(matrix, rhs, objective, forced=None):
    n = len(objective)
    lower = np.zeros(n)
    upper = np.ones(n)
    if forced:
        for index, value in forced.items():
            lower[index] = value
            upper[index] = value
    constraints = () if not len(matrix) else LinearConstraint(matrix, rhs, rhs)
    result = milp(
        objective, integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(lower, upper), constraints=constraints,
        options={"time_limit": 30},
    )
    return result


def objective_from_probability(probability):
    probability = np.clip(probability, 1e-4, 1 - 1e-4)
    return -np.log(probability / (1 - probability))


def equation_backtest(matrix, rhs, objective):
    full_rank = int(np.linalg.matrix_rank(matrix))
    rows = []
    for index in range(len(matrix)):
        keep = np.arange(len(matrix)) != index
        reduced_rank = int(np.linalg.matrix_rank(matrix[keep]))
        result = solve_map(matrix[keep], rhs[keep], objective)
        if not result.success:
            raise RuntimeError((index, result.message))
        assignment = np.rint(result.x).astype(np.int8)
        predicted = int(round(matrix[index] @ assignment))
        actual = int(rhs[index])
        rows.append({
            "equation": index,
            "independent_when_removed": reduced_rank < full_rank,
            "predicted_rhs": predicted,
            "actual_rhs": actual,
            "absolute_error": abs(predicted - actual),
            "exact": predicted == actual,
        })
    informative = [row for row in rows if row["independent_when_removed"]]
    return {
        "full_rank": full_rank,
        "informative_equations": len(informative),
        "exact_rate": float(np.mean([row["exact"] for row in informative])) if informative else 1.0,
        "mean_absolute_error": float(np.mean([row["absolute_error"] for row in informative])) if informative else 0.0,
        "maximum_absolute_error": max((row["absolute_error"] for row in informative), default=0),
        "rows": rows,
    }


def fixed_accuracy(keys, probability, fixed_rows):
    index = {key: i for i, key in enumerate(keys)}
    values = []
    for row in fixed_rows:
        key = (row["order_id"], row["rid"])
        if key in index:
            p = float(probability[index[key]])
            label = int(row["label"])
            values.append((label, p))
    accuracy = np.mean([(p >= 0.5) == bool(label) for label, p in values])
    logloss = -np.mean([
        label * math.log(max(p, 1e-9)) + (1 - label) * math.log(max(1 - p, 1e-9))
        for label, p in values
    ])
    return {"count": len(values), "accuracy": float(accuracy), "logloss": float(logloss)}


def protected_nodes():
    report = read_json(V11_REPORT)
    output = set()
    for name in ("protected_in", "protected_out"):
        output |= {(row["order_id"], row["rid"]) for row in report.get(name, [])}
    return output


def candidate_actions(keys, assignment, gaps, champion_selected, fixed, exact_orders, protected):
    candidates = []
    for index, key in enumerate(keys):
        if key in fixed or key in protected or key[0] in exact_orders:
            continue
        selected = key in champion_selected
        label = bool(assignment[index])
        if selected == label:
            continue
        candidates.append({
            "key": key, "index": index, "map_label": int(label),
            "selected_by_champion": selected, "force_flip_gap": float(gaps[index]),
        })
    by_order = defaultdict(list)
    for row in candidates:
        by_order[row["key"][0]].append(row)
    actions = []
    counts = defaultdict(int)
    for oid, rid in champion_selected:
        counts[oid] += 1
    for oid, rows in by_order.items():
        removals = sorted(
            [row for row in rows if row["selected_by_champion"]],
            key=lambda row: -row["force_flip_gap"],
        )
        additions = sorted(
            [row for row in rows if not row["selected_by_champion"]],
            key=lambda row: -row["force_flip_gap"],
        )
        remove = removals[:1] if removals and (additions or counts[oid] > 1) else []
        add = additions[:1] if additions and (removals or counts[oid] < MAX_ROOTS) else []
        if not remove and not add:
            continue
        evidence_rows = remove + add
        actions.append({
            "action_id": "",
            "order_id": oid,
            "remove_rids": [row["key"][1] for row in remove],
            "add_rids": [row["key"][1] for row in add],
            "source": "v45_constrained_map",
            "expected_gain": None,
            "evidence": {
                "minimum_force_flip_gap": min(row["force_flip_gap"] for row in evidence_rows),
                "nodes": evidence_rows,
            },
        })
    actions.sort(key=lambda row: -row["evidence"]["minimum_force_flip_gap"])
    for index, action in enumerate(actions, 1):
        action["action_id"] = f"v45_map_{index:03d}"
    return actions


def emit(name, candidate_actions, exact, order_ids, champion_roots, records_by_order):
    actions = list(exact["actions"]) + candidate_actions
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    add_count = sum(len(action["add_rids"]) for action in candidate_actions)
    remove_count = sum(len(action["remove_rids"]) for action in candidate_actions)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "path": str(path), "actions": actions,
        "candidate_action_count": len(candidate_actions),
        "predictions": predictions, "sha256": sha256(path),
        "score_possibilities": [
            {
                "candidate_delta_tp": delta,
                "tp": BASE_TP_AFTER_EXACT + delta,
                "score": round(2 * (BASE_TP_AFTER_EXACT + delta) / (TRUE_ROOTS + predictions), 9),
            }
            for delta in range(-remove_count, add_count + 1)
        ],
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    row_for_key = {}
    row = 0
    for order in records:
        for alarm in order["alarms"]:
            row_for_key[(order["order_id"], alarm["rid"])] = row
            row += 1

    order_ids, champion_roots = load_submission(CHAMPION)
    champion_selected = root_nodes(champion_roots)
    scored = collect_scored_submissions()
    keys, matrix, rhs, equations = build_system(scored, champion_selected)
    fixed_report = read_json(V37 / "report.json")
    fixed_rows = fixed_report["fixed_labels"]
    fixed = {(row["order_id"], row["rid"]) for row in fixed_rows}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    exact_orders = {action["order_id"] for action in exact["actions"]}
    records_by_order = {order["order_id"]: order for order in records}

    all_priors = priors(arrays)
    audits = {}
    mapped_priors = {}
    for name, full_probability in all_priors.items():
        probability = np.asarray([full_probability[row_for_key[key]] for key in keys])
        mapped_priors[name] = probability
        objective = objective_from_probability(probability)
        backtest = equation_backtest(matrix, rhs, objective)
        fixed_result = fixed_accuracy(keys, probability, fixed_rows)
        result = solve_map(matrix, rhs, objective)
        if not result.success:
            raise RuntimeError((name, result.message))
        audits[name] = {
            "backtest": backtest,
            "fixed_labels": fixed_result,
            "map_objective": float(result.fun),
        }
    ranking = sorted(audits, key=lambda name: (
        -audits[name]["backtest"]["exact_rate"],
        audits[name]["backtest"]["mean_absolute_error"],
        -audits[name]["fixed_labels"]["accuracy"],
        audits[name]["fixed_labels"]["logloss"],
    ))
    best_name = ranking[0]
    probability = mapped_priors[best_name]
    objective = objective_from_probability(probability)
    base = solve_map(matrix, rhs, objective)
    assignment = np.rint(base.x).astype(np.int8)
    gaps = np.full(len(keys), np.inf, dtype=np.float64)
    for index, value in enumerate(assignment):
        opposite = solve_map(matrix, rhs, objective, {index: 1 - int(value)})
        if opposite.success:
            gaps[index] = max(0.0, float(opposite.fun - base.fun))

    actions = candidate_actions(
        keys, assignment, gaps, champion_selected, fixed, exact_orders, protected_nodes()
    )
    best_audit = audits[best_name]
    gate_passed = bool(
        best_audit["backtest"]["exact_rate"] >= 0.60
        and best_audit["backtest"]["mean_absolute_error"] <= 0.75
        and best_audit["fixed_labels"]["accuracy"] >= 0.75
        and len(actions) >= 4
        and actions[3]["evidence"]["minimum_force_flip_gap"] >= 2.0
    )
    probes = {}
    if gate_passed:
        for count in (4, 8):
            if len(actions) >= count:
                name = f"v45_constrained_map_top{count}_hold"
                probes[name] = emit(
                    name, actions[:count], exact, order_ids, champion_roots, records_by_order
                )
    report = {
        "version": "v45-constrained-map-1",
        "system": {
            "equations": len(matrix), "variables": len(keys),
            "rank": int(np.linalg.matrix_rank(matrix)), "fixed_labels": len(fixed),
        },
        "prior_ranking": ranking,
        "prior_audits": audits,
        "selected_prior": best_name,
        "map_candidate_action_count": len(actions),
        "map_candidate_actions": actions,
        "gate": {
            "requires_informative_loo_exact_rate": 0.60,
            "requires_informative_loo_mae_at_most": 0.75,
            "requires_fixed_accuracy": 0.75,
            "requires_top4_flip_gap": 2.0,
            "passed": gate_passed,
        },
        "probes": probes,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "system": report["system"],
        "prior_ranking": ranking,
        "selected_prior": best_name,
        "best_backtest": best_audit["backtest"],
        "best_fixed_labels": best_audit["fixed_labels"],
        "candidate_actions": len(actions),
        "top_action_gaps": [
            row["evidence"]["minimum_force_flip_gap"] for row in actions[:10]
        ],
        "gate": report["gate"],
        "probes": {name: {
            "path": row["path"], "predictions": row["predictions"],
            "sha256": row["sha256"],
        } for name, row in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

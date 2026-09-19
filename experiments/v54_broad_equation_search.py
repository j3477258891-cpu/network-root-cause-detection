"""Search all legal historical-node queries by expected label resolution."""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import load_submission
from v37_online_equation_solver import build_system, collect_scored_submissions, solve_fixed_labels
from v45_constrained_map_inference import priors


RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
DATA = EXP / "v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
OUT = EXP / "v54_broad_equation_search"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prior_distribution(vector, probability, possible):
    distribution = {0: 1.0}
    for coefficient, p in zip(vector, probability):
        if not coefficient:
            continue
        updated = defaultdict(float)
        for value, mass in distribution.items():
            updated[value] += mass * (1 - p)
            updated[value + int(coefficient)] += mass * p
        distribution = dict(updated)
    selected = {value: distribution.get(value, 0.0) for value in possible}
    total = sum(selected.values())
    return {value: mass / total for value, mass in selected.items()} if total else {
        value: 1 / len(possible) for value in possible
    }


def query_stats(vector, matrix, rhs, probability):
    feasible = []
    negative_count = int(np.count_nonzero(vector < 0))
    positive_count = int(np.count_nonzero(vector > 0))
    for value in range(-negative_count, positive_count + 1):
        augmented = np.vstack([matrix, vector])
        try:
            solve_fixed_labels(augmented, np.r_[rhs, value], len(vector))
        except RuntimeError:
            continue
        feasible.append(value)
    # The solve above already checked feasibility but repeated label-bound
    # solving is needed to know which variables become exact.
    if not feasible:
        return None
    dist = prior_distribution(vector, probability, feasible)
    outcomes = []
    for value in feasible:
        lower, upper, _ = solve_fixed_labels(
            np.vstack([matrix, vector]), np.r_[rhs, value], len(vector)
        )
        fixed = int(np.sum(lower == upper))
        outcomes.append({"delta": int(value), "probability": dist[value], "fixed": fixed})
    return {
        "possible": feasible,
        "outcomes": outcomes,
        "expected_fixed": float(sum(row["probability"] * row["fixed"] for row in outcomes)),
        "minimum_fixed": min(row["fixed"] for row in outcomes),
        "expected_new_fixed": float(sum(row["probability"] * max(0, row["fixed"] - 29) for row in outcomes)),
        "bits": math.log2(len(feasible)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, champion = load_submission(V30 / "submissions/v30_cross_order_top5.csv")
    champion_nodes = {(oid, node["@rid"]) for oid, values in champion.items() for node in values}
    keys, matrix, rhs, _ = build_system(collect_scored_submissions(), champion_nodes)
    key_index = {key: i for i, key in enumerate(keys)}
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    probability_full = priors(arrays)["platt_station_domain_mean"]
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    row_for_key = {}
    row = 0
    order_counts = {}
    for order in records:
        order_counts[order["order_id"]] = len(champion[order["order_id"]])
        for alarm in order["alarms"]:
            row_for_key[(order["order_id"], alarm["rid"])] = row
            row += 1
    probability = np.asarray([probability_full[row_for_key[key]] for key in keys])
    fixed_report = read_json(EXP / "v37_online_equations/report.json")
    fixed_keys = {(x["order_id"], x["rid"]) for x in fixed_report["fixed_labels"]}
    previous = set()
    manifest = EXP / "v53_active_equation/manifests/v53_active_equation_probe.json"
    if manifest.exists():
        previous = {(a["order_id"], rid) for a in read_json(manifest).get("candidate_actions", [])
                    for rid in a.get("remove_rids", []) + a.get("add_rids", [])}

    candidates = []
    for i, key in enumerate(keys):
        if key in fixed_keys or key in previous:
            continue
        oid, rid = key
        selected = key in champion_nodes
        if selected and order_counts[oid] <= 1:
            continue
        if not selected and order_counts[oid] >= 8:
            continue
        vector = np.zeros(len(keys), dtype=np.float64)
        vector[i] = -1 if selected else 1
        stats = query_stats(vector, matrix, rhs, probability)
        if stats is None:
            continue
        candidates.append({
            "key": [oid, rid], "selected": selected, "probability": float(probability[i]),
            "action_coefficient": int(vector[i]), **stats,
        })
    candidates.sort(key=lambda row: (-row["expected_new_fixed"], -row["expected_fixed"], row["key"]))
    report = {
        "version": "v54-broad-equation-search-1",
        "system": {"equations": len(matrix), "variables": len(keys), "fixed": len(fixed_keys)},
        "candidate_count": len(candidates), "candidates": candidates,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "system": report["system"],
        "candidate_count": len(candidates),
        "top": [{k: row[k] for k in ("key", "selected", "probability", "possible", "expected_fixed", "expected_new_fixed", "minimum_fixed")} for row in candidates[:30]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

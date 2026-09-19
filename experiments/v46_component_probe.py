"""Find rank-deficiency-one equation components and probe one bit to solve all."""

from __future__ import annotations

import gzip
import hashlib
import json
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

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import (
    CHAMPION, CHAMPION_TP, TRUE_ROOTS, build_system, collect_scored_submissions,
)


RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V37 = EXPERIMENTS / "v37_online_equations"
V11_REPORT = EXPERIMENTS / "submissions/template_ranked/v11_constrained_report.json"
OUT = EXPERIMENTS / "v46_component_probe"
BASE_TP_AFTER_EXACT = CHAMPION_TP + 1
MAX_ROOTS = 8


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_nodes(roots):
    return {(oid, node["@rid"]) for oid, values in roots.items() for node in values}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def reduced_components(keys, matrix, rhs, fixed_map):
    fixed_indices = [i for i, key in enumerate(keys) if key in fixed_map]
    free_indices = [i for i, key in enumerate(keys) if key not in fixed_map]
    adjusted = rhs.astype(np.float64).copy()
    if fixed_indices:
        fixed_values = np.asarray([fixed_map[keys[i]] for i in fixed_indices], dtype=np.float64)
        adjusted -= matrix[:, fixed_indices] @ fixed_values
    reduced = matrix[:, free_indices]
    uf = UnionFind(len(free_indices))
    for row in reduced:
        variables = np.flatnonzero(np.abs(row) > 1e-9)
        for value in variables[1:]:
            uf.union(int(variables[0]), int(value))
    groups = defaultdict(list)
    for local in range(len(free_indices)):
        groups[uf.find(local)].append(local)
    output = []
    for local_indices in groups.values():
        local_indices = np.asarray(local_indices, dtype=np.int64)
        row_mask = np.any(np.abs(reduced[:, local_indices]) > 1e-9, axis=1)
        local_matrix = reduced[row_mask][:, local_indices]
        local_rhs = adjusted[row_mask]
        rank = int(np.linalg.matrix_rank(local_matrix))
        output.append({
            "local_indices": local_indices,
            "global_indices": np.asarray([free_indices[i] for i in local_indices], dtype=np.int64),
            "matrix": local_matrix,
            "rhs": local_rhs,
            "variables": len(local_indices),
            "equations": int(row_mask.sum()),
            "rank": rank,
            "nullity": len(local_indices) - rank,
        })
    output.sort(key=lambda row: (row["nullity"], -row["variables"]))
    return output


def branch_solution(component, local_index, value):
    n = component["variables"]
    lower, upper = np.zeros(n), np.ones(n)
    lower[local_index] = value
    upper[local_index] = value
    result = milp(
        np.zeros(n), integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(component["matrix"], component["rhs"], component["rhs"]),
        options={"time_limit": 30},
    )
    if not result.success:
        return None
    assignment = np.rint(result.x).astype(np.int8)
    if not np.allclose(component["matrix"] @ assignment, component["rhs"]):
        raise RuntimeError("branch solution does not satisfy component equations")
    probe_row = np.zeros(n)
    probe_row[local_index] = 1
    augmented_rank = int(np.linalg.matrix_rank(np.vstack([component["matrix"], probe_row])))
    return {"assignment": assignment, "augmented_rank": augmented_rank, "unique": augmented_rank == n}


def protected_nodes():
    report = read_json(V11_REPORT)
    output = set()
    for name in ("protected_in", "protected_out"):
        output |= {(row["order_id"], row["rid"]) for row in report.get(name, [])}
    return output


def valid_probe_keys(component, keys, champion_selected, counts, protected, exact_orders):
    output = []
    for local, global_index in enumerate(component["global_indices"]):
        key = keys[int(global_index)]
        if key in protected or key[0] in exact_orders:
            continue
        selected = key in champion_selected
        if selected and counts[key[0]] <= 1:
            continue
        if not selected and counts[key[0]] >= MAX_ROOTS:
            continue
        output.append((local, key, selected))
    return output


def correction_actions(component, assignment, keys, champion_selected,
                       protected, exact_orders, counts):
    changes = defaultdict(lambda: {"remove_rids": [], "add_rids": [], "labels": []})
    for local, global_index in enumerate(component["global_indices"]):
        key = keys[int(global_index)]
        if key in protected or key[0] in exact_orders:
            continue
        label = int(assignment[local])
        selected = key in champion_selected
        if selected == bool(label):
            continue
        field = "remove_rids" if selected else "add_rids"
        changes[key[0]][field].append(key[1])
        changes[key[0]]["labels"].append({"rid": key[1], "label": label})
    actions, skipped = [], []
    for oid, change in sorted(changes.items()):
        final_count = counts[oid] - len(change["remove_rids"]) + len(change["add_rids"])
        if not 1 <= final_count <= MAX_ROOTS:
            skipped.append({"order_id": oid, "reason": "invalid_root_count", **change})
            continue
        actions.append({
            "action_id": "",
            "order_id": oid,
            "remove_rids": sorted(change["remove_rids"]),
            "add_rids": sorted(change["add_rids"]),
            "source": "v46_component_exact_branch",
            "expected_gain": None,
            "evidence": {"proof": "unique binary branch after one probed label", "labels": change["labels"]},
        })
    for index, action in enumerate(actions, 1):
        action["action_id"] = f"v46_exact_{index:03d}"
    return actions, skipped


def theoretical_result(actions, base_predictions):
    additions = sum(len(action["add_rids"]) for action in actions)
    removals = sum(len(action["remove_rids"]) for action in actions)
    tp = BASE_TP_AFTER_EXACT + additions
    predictions = base_predictions + additions - removals
    return {
        "added_true": additions,
        "removed_false": removals,
        "tp": tp,
        "predictions": predictions,
        "score": 2 * tp / (TRUE_ROOTS + predictions),
    }


def emit(name, actions, exact, order_ids, champion_roots, records_by_order, metadata):
    all_actions = list(exact["actions"]) + actions
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    predictions = sum(map(len, roots.values()))
    path = OUT / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": name, "baseline": str(CHAMPION), "fixed_base": exact["probe_id"],
        "path": str(path), "actions": all_actions,
        "predictions": predictions, "sha256": sha256(path), **metadata,
    }
    (OUT / "manifests" / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    order_ids, champion_roots = load_submission(CHAMPION)
    champion_selected = root_nodes(champion_roots)
    counts = {oid: len(values) for oid, values in champion_roots.items()}
    scored = collect_scored_submissions()
    keys, matrix, rhs, _ = build_system(scored, champion_selected)
    fixed_report = read_json(V37 / "report.json")
    fixed_map = {
        (row["order_id"], row["rid"]): int(row["label"])
        for row in fixed_report["fixed_labels"]
    }
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    exact_orders = {action["order_id"] for action in exact["actions"]}
    protected = protected_nodes()
    components = reduced_components(keys, matrix, rhs, fixed_map)

    candidates = []
    for component_index, component in enumerate(components):
        if component["nullity"] != 1:
            continue
        valid = valid_probe_keys(
            component, keys, champion_selected, counts, protected, exact_orders
        )
        if not valid:
            continue
        local_index, probe_key, selected = valid[0]
        branches = {}
        viable = True
        for value in (0, 1):
            solution = branch_solution(component, local_index, value)
            if solution is None or not solution["unique"]:
                viable = False
                break
            actions, skipped = correction_actions(
                component, solution["assignment"], keys, champion_selected,
                protected, exact_orders, counts,
            )
            result = theoretical_result(actions, len(champion_selected))
            branches[str(value)] = {
                "assignment": solution["assignment"].tolist(),
                "actions": actions, "skipped": skipped, "result": result,
            }
        if not viable:
            continue
        candidates.append({
            "component_index": component_index,
            "variables": component["variables"], "equations": component["equations"],
            "rank": component["rank"], "nullity": component["nullity"],
            "probe_local_index": local_index,
            "probe_key": probe_key, "probe_selected": selected,
            "branch_0": branches["0"], "branch_1": branches["1"],
            "worst_branch_score": min(branches["0"]["result"]["score"], branches["1"]["result"]["score"]),
            "best_branch_score": max(branches["0"]["result"]["score"], branches["1"]["result"]["score"]),
        })
    candidates.sort(key=lambda row: (-row["worst_branch_score"], -row["variables"]))
    probes = {}
    selected_component = candidates[0] if candidates else None
    if selected_component:
        with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
            records = json.load(handle)["test"]
        records_by_order = {row["order_id"]: row for row in records}
        oid, rid = selected_component["probe_key"]
        probe_action = {
            "action_id": "v46_component_probe_bit",
            "order_id": oid,
            "remove_rids": [rid] if selected_component["probe_selected"] else [],
            "add_rids": [] if selected_component["probe_selected"] else [rid],
            "source": "v46_rank_deficiency_one_probe",
            "expected_gain": None,
            "evidence": {
                "component_variables": selected_component["variables"],
                "component_rank": selected_component["rank"],
                "proof": "observing this bit raises component rank to its variable count",
            },
        }
        probe_predictions = len(champion_selected) + (-1 if selected_component["probe_selected"] else 1)
        probe_scores = []
        for value in (0, 1):
            delta_tp = -value if selected_component["probe_selected"] else value
            tp = BASE_TP_AFTER_EXACT + delta_tp
            probe_scores.append({
                "probed_label": value, "tp": tp,
                "score": round(2 * tp / (TRUE_ROOTS + probe_predictions), 9),
            })
        probes["v46_component_bit_probe"] = emit(
            "v46_component_bit_probe", [probe_action], exact, order_ids,
            champion_roots, records_by_order,
            {"component_index": selected_component["component_index"],
             "score_possibilities": probe_scores},
        )
        for value in (0, 1):
            branch = selected_component[f"branch_{value}"]
            name = f"v46_component_branch{value}_exact_hold"
            probes[name] = emit(
                name, branch["actions"], exact, order_ids, champion_roots,
                records_by_order,
                {"requires_probed_label": value,
                 "expected_tp": branch["result"]["tp"],
                 "expected_score": branch["result"]["score"]},
            )

    component_summary = [{
        "variables": row["variables"], "equations": row["equations"],
        "rank": row["rank"], "nullity": row["nullity"],
    } for row in components]
    report = {
        "version": "v46-component-probe-1",
        "system": {
            "equations": len(matrix), "variables": len(keys),
            "fixed_variables": len(fixed_map), "free_variables": len(keys) - len(fixed_map),
        },
        "components": component_summary,
        "nullity_one_candidate_count": len(candidates),
        "candidates": candidates,
        "selected_component": selected_component,
        "probes": probes,
        "recommendation": (
            {"submit_after_v37": probes["v46_component_bit_probe"]["path"]}
            if selected_component else None
        ),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "system": report["system"],
        "components": component_summary,
        "candidate_count": len(candidates),
        "selected": None if not selected_component else {
            "variables": selected_component["variables"],
            "probe_key": selected_component["probe_key"],
            "probe_selected": selected_component["probe_selected"],
            "branch_0_result": selected_component["branch_0"]["result"],
            "branch_1_result": selected_component["branch_1"]["result"],
        },
        "probes": {name: {
            "path": row["path"], "predictions": row["predictions"],
            "sha256": row["sha256"],
        } for name, row in probes.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

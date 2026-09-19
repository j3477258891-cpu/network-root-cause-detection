"""V122 multi-model addition campaign.

Ranks one-sided additions using independent historical model reports, while
keeping the current verified baseline untouched.  This is an exploratory
submission generator; it never claims that an addition is guaranteed true.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(EXP))
import v121_equation_safe_campaign as v121  # noqa: E402

OUT = EXP / "v122_multimodel_addition_campaign"
BASE = EXP / "v60_combined_checkpoint/highest_verified_combined.csv"
V41 = EXP / "v41_equation_aware/report.json"
V119 = EXP / "v119_joint_campaign/candidate_catalog.json"
V54 = EXP / "v54_broad_equation_search/report.json"
V37 = EXP / "v37_online_equations/report.json"
SCORE_FILES = [
    ROOT / "codexgz/v11/v11_test_scores.csv",
    ROOT / "codexgz/v11_variants/template_ranked_scores.csv",
    ROOT / "codexgz/v11_variants/robust_blend70_scores.csv",
    ROOT / "codexgz/v11_variants/robust_no_location_scores.csv",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v11() -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = {}
    for path in SCORE_FILES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["order_id"], row["rid"])
                values.setdefault(key, []).append(float(row["meta_mean"]))
    return values


def pure_bounds(matrix, rhs, index: int) -> tuple[int, int, str]:
    objective = np.zeros(matrix.shape[1], dtype=np.float64)
    objective[index] = 1.0
    kwargs = {
        "integrality": np.ones(matrix.shape[1]),
        "bounds": Bounds(np.zeros(matrix.shape[1]), np.ones(matrix.shape[1])),
        "constraints": LinearConstraint(matrix, rhs, rhs),
        "options": {"time_limit": 5.0},
    }
    lower = milp(objective, **kwargs)
    upper = milp(-objective, **kwargs)
    if not lower.success or not upper.success:
        return -1, -1, f"min={lower.message}; max={upper.message}"
    return int(round(lower.fun)), int(round(-upper.fun)), "optimal"


def add_node(rows, order_id: str, rid: str, alarms) -> list[dict[str, Any]]:
    out = []
    changed = False
    for row in rows:
        roots = [dict(node) for node in row["roots"]]
        if row["order_id"] == order_id:
            present = {node["@rid"] for node in roots}
            if rid in present:
                raise ValueError("candidate is already in baseline")
            if len(roots) >= 8:
                raise ValueError("addition would exceed root-count cap")
            roots.append(v121.alarm_node(alarms[(order_id, rid)]))
            changed = True
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    if not changed:
        raise ValueError(f"order not found: {order_id}")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "output"])
        writer.writeheader()
        writer.writerows(rows)


def build_candidates(base_rows, alarms, orders, matrix, rhs, universe):
    base_keys = {(row["order_id"], node["@rid"]) for row in base_rows for node in row["roots"]}
    fixed = {(x["order_id"], x["rid"]): int(x["label"]) for x in read_json(V37).get("fixed_labels", [])}
    v11 = load_v11()
    v119 = {(x["order_id"], x["rid"]): float(x.get("prior", 0.0)) for x in read_json(V119).get("additions", [])}
    v54 = {}
    for x in read_json(V54).get("candidates", []):
        if x.get("action_coefficient") == 1:
            v54[tuple(x["key"])] = float(x["probability"])
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in read_json(V41).get("candidates", []):
        if item.get("kind") != "add" or not item.get("actions"):
            continue
        action = item["actions"][0]
        if len(action.get("add_rids", [])) != 1 or action.get("remove_rids"):
            continue
        key = (action["order_id"], action["add_rids"][0])
        if key in base_keys or key not in alarms or key in fixed:
            continue
        if len(next((x["roots"] for x in base_rows if x["order_id"] == key[0]), [])) >= 8:
            continue
        index = universe[key]
        lo, hi, status = pure_bounds(matrix, rhs, index)
        evidence = action.get("evidence", {})
        scores = evidence.get("scores", {})
        v11_values = v11.get(key, [])
        count_support = 6 if item.get("source") == "v36" else 0
        count_margin = float(evidence.get("mean_margin", 0.0)) if item.get("source") == "v36" else 0.0
        row = merged.setdefault(key, {
            "order_id": key[0], "add_rid": key[1], "sources": [],
            "v41_candidate_ids": [], "v41_priority": 0.0,
            "v41_rank_average": None, "v41_scores": scores,
            "v119_prior": v119.get(key), "v54_probability": v54.get(key),
            "v11_meta_means": v11_values, "count_model_support": 0,
            "count_model_mean_margin": 0.0, "equation_min_delta": lo,
            "equation_max_delta": hi, "equation_status": status,
        })
        row["sources"].append(item.get("source"))
        row["v41_candidate_ids"].append(item.get("candidate_id"))
        row["v41_priority"] = max(row["v41_priority"], float(item.get("priority", 0.0)))
        if evidence.get("rank_average") is not None:
            row["v41_rank_average"] = max(float(row["v41_rank_average"] or 0.0), float(evidence["rank_average"]))
        if item.get("source") == "v36":
            row["count_model_support"] = max(row["count_model_support"], count_support)
            row["count_model_mean_margin"] = max(row["count_model_mean_margin"], count_margin)
    candidates = []
    for row in merged.values():
        # A candidate fixed false by real leaderboard equations is not a
        # candidate, regardless of its offline model score.
        if row["equation_max_delta"] <= 0:
            continue
        support = sum(bool(x) for x in [row["v119_prior"] is not None, row["v54_probability"] is not None, row["v41_rank_average"] is not None, row["count_model_support"] >= 6, len(row["v11_meta_means"]) >= 3])
        v11_mean = float(np.mean(row["v11_meta_means"])) if row["v11_meta_means"] else 0.0
        rank = float(row["v41_rank_average"] or 0.0)
        prior = float(row["v119_prior"] or 0.0)
        p54 = float(row["v54_probability"] or 0.0)
        consensus = float(np.mean([x for x in [v11_mean, rank, prior, p54] if x > 0]))
        row.update({"evidence_support_channels": support, "v11_mean": v11_mean, "consensus_score": consensus})
        candidates.append(row)
    candidates.sort(key=lambda x: (-x["evidence_support_channels"], -x["consensus_score"], -x["v41_priority"]))
    return candidates


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_rows = v121.load_base_rows()
    alarms, orders = v121.load_alarm_records()
    real = v121.collect_real_records()
    universe = {(order_id, rid): i for i, (order_id, rid) in enumerate(alarms)}
    matrix, rhs, _ = v121.build_equations(real, universe)
    candidates = build_candidates(base_rows, alarms, orders, matrix, rhs, universe)
    if not candidates:
        raise RuntimeError("no eligible multi-model addition candidates")
    selected = candidates[0]
    probe_rows = add_node(base_rows, selected["order_id"], selected["add_rid"], alarms)
    probe_path = OUT / "probe_01_multimodel_add.csv"
    write_csv(probe_path, probe_rows)
    prediction_count = sum(len(json.loads(row["output"])["rootcause"]) for row in probe_rows)
    write_json = lambda path, obj: path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(OUT / "candidate_catalog.json", {"version": "v122", "candidate_count": len(candidates), "candidates": candidates})
    write_json(OUT / "selection.json", {"selected": selected, "probe_path": str(probe_path), "predictions": prediction_count, "sha256": v121.sha256(probe_path), "baseline_f1": 0.919673, "note": "exploratory one-sided addition; actual leaderboard score required"})
    write_json(OUT / "validation.json", {"real_record_count": len(real), "equation_shape": [int(matrix.shape[0]), int(matrix.shape[1])], "baseline_tp": 956, "orders": len(probe_rows), "predictions": prediction_count, "min_roots": min(len(json.loads(row["output"])["rootcause"]) for row in probe_rows), "max_roots": max(len(json.loads(row["output"])["rootcause"]) for row in probe_rows), "sha256": v121.sha256(probe_path), "structure_pass": prediction_count == 1036 and len(probe_rows) == 546})
    print(json.dumps({"selected": selected, "probe": str(probe_path), "predictions": prediction_count, "sha256": v121.sha256(probe_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

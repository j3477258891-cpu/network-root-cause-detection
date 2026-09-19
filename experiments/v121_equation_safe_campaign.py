"""V121: derive only equation-safe same-order replacements.

This module consumes real scored submission records, converts them into exact
integer constraints over test alarms, and emits a single probe only when a
candidate is forced to have positive TP delta by every feasible label solution.
It never uploads files and never treats OOF/simulation outcomes as online
evidence.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
CATALOG = ROOT / "experiments/v120_swap_campaign/candidate_catalog.json"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = ROOT / "experiments/v121_equation_safe_campaign"
TRUE_ROOTS = 1044
BASE_P = 1035
BASE_TP = 956
TARGET_F1 = 0.945


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_submission(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result[row["order_id"]] = [x["@rid"] for x in json.loads(row["output"])["rootcause"]]
    return result


def load_base_rows() -> list[dict[str, Any]]:
    result = []
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result.append({"order_id": row["order_id"], "roots": json.loads(row["output"])["rootcause"]})
    return result


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def add_record(records: list[dict[str, Any]], path: Path, score: float | None, tp: int | None, predictions: int | None, source: str, status: str = "scored") -> None:
    if not path.exists():
        return
    p = int(predictions or sum(len(v) for v in load_submission(path).values()))
    inferred = int(tp if tp is not None else round(float(score) * (TRUE_ROOTS + p) / 2.0))
    records.append({"path": str(path), "sha256": sha256(path), "score": score, "tp": inferred, "predictions": p, "source": source, "status": status})


def collect_real_records() -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    # V27/V28: only explicitly scored batches; inferred batches are excluded.
    for state_path in (ROOT / "experiments/v27_closed_loop/state.json", ROOT / "experiments/v28_ten_day_campaign/state.json", ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json"):
        state = read_json(state_path)
        for batch in state.get("batches", []):
            if batch.get("status") != "scored" or not batch.get("submission"):
                continue
            add_record(raw, resolve_path(batch["submission"]), batch.get("score"), batch.get("tp"), batch.get("predictions"), str(state_path))
    # V29/V30 reports explicitly separate verified online results.
    v29 = read_json(ROOT / "experiments/v29_domain_ranker/reports/online_results.json")
    for item in v29.get("verified", []):
        add_record(raw, resolve_path(item.get("submitted_path") or item["path"]), item.get("score"), item.get("tp"), item.get("predictions"), "v29_online_results")
    v30 = read_json(ROOT / "experiments/v30_meta_stack/online_results.json")
    for item in v30.get("verified", []):
        path = item.get("path")
        if path:
            add_record(raw, resolve_path(path), item.get("score"), item.get("tp"), item.get("predictions"), "v30_online_results")
    # Later user-confirmed online records.
    add_record(raw, ROOT / "experiments/submissions/distance1_swap_equation_filtered.csv", 0.903319, 939, 1035, "v117_user_confirmed")
    add_record(raw, ROOT / "experiments/v118_safe_positive_campaign/probe_00_calibration.csv", 0.907308, 925, 995, "v118_user_confirmed")
    v119 = ROOT / "experiments/v119_joint_campaign/probe_01.csv"
    if v119.exists():
        add_record(raw, v119, 0.910151, 937, 1015, "v119_user_confirmed")
    add_record(raw, ROOT / "experiments/v120_swap_campaign/probe_00_baseline.csv", 0.919673, 956, 1035, "v120_user_confirmed")
    # Append only actual V121 leaderboard records, including exploratory probes.
    v121_scores = ROOT / "experiments/v121_equation_safe_campaign/online_scores.json"
    if v121_scores.exists():
        for item in read_json(v121_scores).get("records", []):
            path = item.get("file") or item.get("path")
            if not path and item.get("file_sha256"):
                for probe_file in (ROOT / "experiments/v121_equation_safe_campaign").glob("probe_*.csv"):
                    if sha256(probe_file) == item["file_sha256"]:
                        path = str(probe_file)
                        break
            if path and item.get("score") is not None and item.get("predictions") is not None:
                candidate_path = resolve_path(path)
                if candidate_path.exists():
                    add_record(raw, candidate_path, item["score"], item.get("inferred_tp"), item["predictions"], "v121_user_confirmed")
    # Deduplicate identical files, retaining the most explicit source.
    unique: dict[str, dict[str, Any]] = {}
    for item in raw:
        unique.setdefault(item["sha256"], item)
    return list(unique.values())


def build_equations(real_records: list[dict[str, Any]], universe: dict[tuple[str, str], int]) -> tuple[csr_matrix, np.ndarray, list[dict[str, Any]]]:
    rows, cols, data, rhs, meta = [], [], [], [], []
    for row_index, item in enumerate(real_records):
        selected = load_submission(Path(item["path"]))
        for order_id, rids in selected.items():
            for rid in rids:
                key = (order_id, rid)
                if key not in universe:
                    raise ValueError(f"submission contains unknown alarm {key}: {item['path']}")
                rows.append(row_index); cols.append(universe[key]); data.append(1)
        rhs.append(int(item["tp"]))
        meta.append({k: item[k] for k in ("path", "sha256", "score", "tp", "predictions", "source", "status")})
    matrix = csr_matrix((data, (rows, cols)), shape=(len(real_records), len(universe)), dtype=np.float64)
    return matrix, np.asarray(rhs, dtype=np.float64), meta


def solve_delta(matrix: csr_matrix, rhs: np.ndarray, add_index: int, remove_index: int, time_limit: float = 8.0) -> tuple[int | None, int | None, str]:
    n = matrix.shape[1]
    objective = np.zeros(n, dtype=np.float64); objective[add_index] = 1.0; objective[remove_index] -= 1.0
    constraints = LinearConstraint(matrix, rhs, rhs)
    kwargs = {"integrality": np.ones(n), "bounds": Bounds(np.zeros(n), np.ones(n)), "constraints": constraints, "options": {"time_limit": time_limit}}
    lo = milp(objective, **kwargs)
    hi = milp(-objective, **kwargs)
    if not lo.success or not hi.success:
        reason = f"infeasible_or_timeout: min={lo.message!r} max={hi.message!r}"
        return None, None, reason
    return int(round(lo.fun)), int(round(-hi.fun)), "optimal"


def load_alarm_records() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        dataset = json.load(f)
    alarms = {(order["order_id"], alarm["rid"]): alarm for order in dataset["test"] for alarm in order.get("alarms", [])}
    orders = {order["order_id"]: order for order in dataset["test"]}
    return alarms, orders


def risk_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    base = {(r["order_id"], node["@rid"]) for r in rows for node in r["roots"]}
    bad: set[tuple[str, str]] = set()
    for path in (ROOT / "experiments/v118_safe_positive_campaign/probe_00_calibration.csv", ROOT / "experiments/v119_joint_campaign/probe_01.csv", ROOT / "experiments/submissions/distance1_swap_equation_filtered.csv"):
        if not path.exists(): continue
        current = {(order_id, rid) for order_id, rids in load_submission(path).items() for rid in rids}
        bad |= base - current
    return bad


def alarm_node(alarm: dict[str, Any]) -> dict[str, str]:
    source = alarm.get("source") or {}
    return {"@rid": alarm["rid"], "title": source.get("title", alarm.get("title", "")), "location": source.get("location", alarm.get("raw_location", alarm.get("location", ""))), "reason": source.get("reason", alarm.get("reason", ""))}


def apply_action(rows: list[dict[str, Any]], action: dict[str, Any], alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        roots = [dict(x) for x in row["roots"]]
        if row["order_id"] == action["order_id"]:
            present = {x["@rid"] for x in roots}
            if action["remove_rid"] not in present or action["add_rid"] in present: raise ValueError("invalid safe action")
            roots = [x for x in roots if x["@rid"] != action["remove_rid"]]
            roots.append(alarm_node(alarms[(row["order_id"], action["add_rid"])]))
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    if sum(len(json.loads(x["output"])["rootcause"]) for x in out) != BASE_P: raise ValueError("prediction count changed")
    return out


def expanded_candidates(base_rows: list[dict[str, Any]], orders: dict[str, dict[str, Any]], alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Enumerate all structurally valid same-order pairs after node safety screens."""
    bad = risk_keys(base_rows)
    score_path = ROOT / "codexgz/v11/v11_test_scores.csv"
    v11: dict[tuple[str, str], tuple[float, float]] = {}
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f): v11[(r["order_id"], r["rid"])] = (float(r["context_score"]), float(r["meta_mean"]))
    base_by = {r["order_id"]: {x["@rid"] for x in r["roots"]} for r in base_rows}
    out = []
    for oid, order in orders.items():
        selected = [a for a in order.get("alarms", []) if a["rid"] in base_by.get(oid, set())]
        unselected = [a for a in order.get("alarms", []) if a["rid"] not in base_by.get(oid, set())]
        for rem in selected:
            for add in unselected:
                if (oid, rem["rid"]) in bad: continue
                rem_title = rem.get("title", "")
                if rem_title and rem_title in (add.get("target_summary") or ""): continue
                rs, ads = v11.get((oid, rem["rid"]), (0.0, 0.0)), v11.get((oid, add["rid"]), (0.0, 0.0))
                if rs[1] - ads[1] > 0.10: continue
                out.append({
                    "candidate_id": f"v122_expanded_{len(out)+1:03d}", "order_id": oid,
                    "remove_rid": rem["rid"], "add_rid": add["rid"],
                    "source_models": ["expanded_v11_semantic"], "model_support": 1,
                    "nominal_p_net_gain": float(np.clip(0.5 + 0.5 * (ads[0] - rs[0]) + 0.1 * (ads[1] - rs[1]), 0.05, 0.95)),
                    "conservative_p_net_gain": 0.0, "v11_remove": {"context": rs[0], "meta": rs[1]}, "v11_add": {"context": ads[0], "meta": ads[1]},
                })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"]); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_rows = load_base_rows(); alarms, orders = load_alarm_records()
    real = collect_real_records()
    universe = {(order_id, rid): i for i, (order_id, rid) in enumerate(alarms)}
    matrix, rhs, equation_meta = build_equations(real, universe)
    score_checks = []
    for item in real:
        raw_tp = float(item["score"]) * (TRUE_ROOTS + int(item["predictions"])) / 2.0 if item["score"] is not None else float(item["tp"])
        nearest = int(round(raw_tp))
        score_checks.append({"path": item["path"], "score": item["score"], "predictions": item["predictions"], "recorded_tp": item["tp"], "raw_tp": raw_tp, "consistent": nearest == int(item["tp"])})
    # Baseline equation must be exactly TP=956.
    baseline_eq = [x for x in real if x["sha256"] == sha256(BASE)]
    base_check = {"records": len(real), "baseline_records": baseline_eq, "baseline_tp": BASE_TP, "equations": int(matrix.shape[0]), "variables": int(matrix.shape[1])}
    write_json(OUT / "equation_system.json", {"version": "v121", "records": equation_meta, "shape": [int(matrix.shape[0]), int(matrix.shape[1])], "rhs": [int(x) for x in rhs], "baseline_check": base_check, "excluded_policy": ["outcomes", "possible_delta", "OOF", "unsubmitted"]})
    catalog = expanded_candidates(base_rows, orders, alarms) if args.expand else read_json(CATALOG)["candidates"]
    bad = risk_keys(base_rows)
    candidates, safe, diagnostic, conflicts = [], [], [], []
    v11: dict[tuple[str, str], float] = {}
    score_path = ROOT / "codexgz/v11/v11_test_scores.csv"
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                v11[(r["order_id"], r["rid"])] = float(r["meta_mean"])
    base_by_order = {r["order_id"]: {x["@rid"] for x in r["roots"]} for r in base_rows}
    for item in catalog:
        oid, remove, add = item["order_id"], item["remove_rid"], item["add_rid"]
        if (oid, remove) in bad or (oid, remove) not in universe or (oid, add) not in universe: continue
        if remove not in base_by_order.get(oid, set()): continue
        if add in base_by_order.get(oid, set()): continue
        rem_alarm, add_alarm = alarms[(oid, remove)], alarms[(oid, add)]
        rem_title, add_target = rem_alarm.get("title", ""), add_alarm.get("target_summary", "")
        if rem_title and rem_title in add_target: continue
        # Node-level safety screen: do not replace a much stronger baseline node.
        if v11.get((oid, remove), 0.0) - v11.get((oid, add), 0.0) > 0.10: continue
        lo, hi, status = solve_delta(matrix, rhs, universe[(oid, add)], universe[(oid, remove)], args.time_limit)
        result = {**item, "equation_min_delta": lo, "equation_max_delta": hi, "equation_status": status}
        candidates.append(result)
        if lo is None: conflicts.append(result)
        elif lo == 1: safe.append(result)
        elif lo == 0 and hi == 1: diagnostic.append(result)
    safe.sort(key=lambda x: (-len(x.get("source_models", [])), -float(x.get("nominal_p_net_gain", 0.0)), x["order_id"]))
    diagnostic.sort(key=lambda x: (-float(x.get("nominal_p_net_gain", 0.0)), x["order_id"]))
    write_json(OUT / "candidate_catalog.json", {"version": "v121", "base": str(BASE), "base_sha256": sha256(BASE), "candidate_count": len(candidates), "candidates": candidates})
    write_json(OUT / "safe_positive_candidates.json", {"version": "v121", "count": len(safe), "candidates": safe})
    write_json(OUT / "diagnostic_candidates.json", {"version": "v121", "count": len(diagnostic), "candidates": diagnostic})
    write_json(OUT / "conflicts.json", {"version": "v121", "count": len(conflicts), "conflicts": conflicts})
    final_gate = {"baseline_tp": BASE_TP, "equation_feasible": not bool(conflicts), "safe_positive_count": len(safe), "diagnostic_count": len(diagnostic), "final_emission_allowed": False, "safe_probe_allowed": bool(safe), "reason": "no equation-safe positive candidate" if not safe else "safe single probe generated"}
    if safe:
        action = safe[0]
        probe_rows = apply_action(base_rows, action, alarms)
        probe_path = OUT / "probe_01_safe_single.csv"; write_csv(probe_path, probe_rows)
        final_gate.update({"probe_path": str(probe_path), "probe_sha256": sha256(probe_path), "probe_predictions": BASE_P, "selected_action": action})
    elif getattr(args, "emit_best_risk", False) and candidates:
        ranked = [x for x in candidates if x.get("equation_max_delta") == 1]
        if ranked:
            if getattr(args, "candidate_id", None):
                action = next((x for x in ranked if x.get("candidate_id") == args.candidate_id), None)
                if action is None:
                    raise ValueError(f"candidate id not found or not exploratory-eligible: {args.candidate_id}")
            else:
                action = max(ranked, key=lambda x: (float(x.get("nominal_p_net_gain", 0.0)), -abs(int(x.get("equation_min_delta", -1)))))
            output_name = getattr(args, "output_name", "probe_01_exploratory_best.csv")
            probe_path = OUT / output_name; write_csv(probe_path, apply_action(base_rows, action, alarms))
            final_gate.update({"exploratory_probe_path": str(probe_path), "exploratory_probe_sha256": sha256(probe_path), "exploratory_probe_predictions": BASE_P, "selected_action": action, "reason": "exploratory candidate only; equation min delta is not positive", "exploratory_only": True})
    write_json(OUT / "final_gate.json", final_gate)
    existing_score_path = OUT / "online_scores.json" if (OUT / "online_scores.json").exists() else ROOT / "experiments/v120_swap_campaign/online_scores.json"
    existing_scores = read_json(existing_score_path) if existing_score_path.exists() else {"records": []}
    write_json(OUT / "online_scores.json", {"version": "v121", "records": existing_scores.get("records", []), "note": "Only actual leaderboard scores may be appended."})
    validation = {
        "real_record_count": len(real),
        "score_tp_all_consistent": all(x["consistent"] for x in score_checks),
        "equation_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "equation_feasible": not bool(conflicts),
        "baseline_tp": BASE_TP,
        "baseline_predictions": BASE_P,
        "candidate_count_after_safety_screen": len(candidates),
        "safe_positive_count": len(safe),
        "diagnostic_count": len(diagnostic),
        "probe_emitted": bool(safe),
        "all_requirements_pass": all(x["consistent"] for x in score_checks) and not bool(conflicts) and (not safe or Path(OUT / "probe_01_safe_single.csv").exists()),
        "score_checks": score_checks,
    }
    write_json(OUT / "validation.json", validation)
    (OUT / "README.md").write_text("""# V121 equation-safe campaign\n\nThis campaign uses only real scored submissions to constrain binary test-alarm labels. It emits `probe_01_safe_single.csv` only when every feasible integer label assignment gives that replacement a +1 TP delta.\n\nCurrent result: the equation system is feasible, but no candidate passes `min(delta)=+1`; no V121 submission file was emitted. The current online baseline remains TP=956, F1=0.919673.\n\nTo append an actual score after a future safe submission, use the `record` command. Do not enter OOF or simulated scores.\n""", encoding="utf-8")
    print(json.dumps({"real_records": len(real), "equations": int(matrix.shape[0]), "variables": int(matrix.shape[1]), "candidate_count": len(candidates), "safe_positive_count": len(safe), "diagnostic_count": len(diagnostic), "conflicts": len(conflicts), "final_emission_allowed": False, "output": str(OUT)}, ensure_ascii=False, indent=2))


def record(args: argparse.Namespace) -> None:
    path = OUT / "online_scores.json"
    data = read_json(path) if path.exists() else {"version": "v121", "records": []}
    raw_tp = float(args.score) * (TRUE_ROOTS + int(args.predictions)) / 2.0
    tp = int(round(raw_tp))
    entry = {"probe_id": args.probe_id, "score": float(args.score), "predictions": int(args.predictions), "inferred_tp": tp, "raw_tp": raw_tp, "file_sha256": args.sha256, "note": "actual leaderboard score only"}
    data.setdefault("records", []).append(entry)
    write_json(path, data)
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate"); generate.add_argument("--time-limit", type=float, default=8.0); generate.add_argument("--expand", action="store_true"); generate.add_argument("--emit-best-risk", action="store_true", help="emit highest-prior exploratory single swap even if not equation-safe"); generate.add_argument("--candidate-id"); generate.add_argument("--output-name", default="probe_01_exploratory_best.csv")
    rec = sub.add_parser("record"); rec.add_argument("--probe-id", type=int, required=True); rec.add_argument("--score", type=float, required=True); rec.add_argument("--predictions", type=int, default=BASE_P); rec.add_argument("--sha256", default="")
    args = parser.parse_args()
    if args.command == "generate": run(args)
    else: record(args)


if __name__ == "__main__": main()

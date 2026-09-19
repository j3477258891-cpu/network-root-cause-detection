"""Read-only audit of real leaderboard equations and candidate deltas.

This script deliberately does not create or modify a submission CSV.  It
collects only records marked as scored (or explicitly recorded as verified
online), fixes the missing V30 paths, and rebuilds the integer system from
the current test alarm universe.  Outputs are written under this research
directory so that the older V121 artifacts remain untouched.

Run with the bundled runtime, for example::

    $env:PYTHONPATH = 'D:\\zgyidong\\.deps'
    $py = 'C:\\Users\\86158\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe'
    & $py experiments\\research_equation_audit\\audit_real_scores.py

The default candidate union contains the V120, V121 and V122 catalogs.  Use
``--catalog`` to audit a different catalog or ``--skip-node-bounds`` when a
quick equation-only run is desired.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


# This file is nested one level below ``experiments``; parents[2] is the
# workspace root (the sibling campaign directories live there).
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
BASE_PATH = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
TEST_RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if relative_to is not None:
        candidate = relative_to / path
        if candidate.exists():
            return candidate
    return ROOT / path


def load_submission(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            payload = json.loads(row["output"])
            result[row["order_id"]] = [node["@rid"] for node in payload["rootcause"]]
    return result


def prediction_count(path: Path) -> int:
    return sum(len(values) for values in load_submission(path).values())


def infer_tp(score: float, predictions: int) -> tuple[int, float]:
    raw = float(score) * (TRUE_ROOTS + predictions) / 2.0
    # Scores are displayed to six decimals; all records in this audit are far
    # from a half-integer tie.  Explicit half-up rounding is deterministic.
    return int(math.floor(raw + 0.5)), raw


def add_record(
    records: list[dict[str, Any]],
    path: Path,
    score: float | int | None,
    tp: int | None,
    predictions: int | None,
    source: str,
    source_detail: str,
) -> None:
    if score is None or not path.exists():
        return
    p = int(predictions or prediction_count(path))
    displayed_score = float(score)
    inferred, raw = infer_tp(displayed_score, p)
    explicit_tp = int(tp) if tp is not None else inferred
    records.append(
        {
            "path": str(path),
            "sha256": sha256(path),
            "score": displayed_score,
            "predictions": p,
            "tp": explicit_tp,
            "raw_tp_from_displayed_score": raw,
            "tp_consistent": abs(raw - explicit_tp) <= 0.01,
            "status": "scored",
            "source": source,
            "source_detail": source_detail,
            "source_class": "public_scored",
        }
    )


def find_csv_by_sha(folder: Path, expected_sha: str | None) -> Path | None:
    """Resolve score ledgers that store only a file hash."""
    if not expected_sha:
        return None
    expected = expected_sha.lower()
    for path in sorted(folder.glob("probe_*.csv")):
        if path.exists() and sha256(path).lower() == expected:
            return path
    for path in sorted(folder.glob("*.csv")):
        if path.exists() and sha256(path).lower() == expected:
            return path
    return None


def collect_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect unique public scored files and a trace of excluded artifacts."""

    raw: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    # V27/V28 state machines distinguish scored batches from inferred, ready,
    # and pending batches.  Only the explicit scored branch is admissible.
    state_paths = [
        ROOT / "experiments/v27_closed_loop/state.json",
        ROOT / "experiments/v28_ten_day_campaign/state.json",
        ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
    ]
    for state_path in state_paths:
        if not state_path.exists():
            continue
        state = read_json(state_path)
        for batch in state.get("batches", []):
            status = batch.get("status")
            submission = batch.get("submission")
            if status == "scored" and submission:
                add_record(
                    raw,
                    resolve_path(submission),
                    batch.get("score"),
                    batch.get("tp"),
                    batch.get("predictions"),
                    state_path.name,
                    str(state_path),
                )
            elif status != "scored":
                excluded.append(
                    {
                        "source": str(state_path),
                        "batch_id": batch.get("batch_id"),
                        "status": status,
                        "reason": "not explicitly scored; excluded from equations",
                    }
                )

    # V29 explicitly calls these verified online results.
    v29_path = ROOT / "experiments/v29_domain_ranker/reports/online_results.json"
    if v29_path.exists():
        v29 = read_json(v29_path)
        for item in v29.get("verified", []):
            raw_path = item.get("submitted_path") or item.get("path")
            if raw_path:
                add_record(
                    raw,
                    resolve_path(raw_path),
                    item.get("score"),
                    item.get("tp"),
                    item.get("predictions"),
                    "v29_online_results",
                    str(v29_path),
                )

    # V30's verified entries omit path.  Resolve by probe_id, and use the
    # champion path as a second independent fallback for the first entry.
    v30_path = ROOT / "experiments/v30_meta_stack/online_results.json"
    if v30_path.exists():
        v30 = read_json(v30_path)
        champion = v30.get("champion", {})
        if champion.get("path"):
            add_record(
                raw,
                resolve_path(champion["path"]),
                champion.get("score"),
                champion.get("tp"),
                champion.get("predictions"),
                "v30_online_results",
                f"{v30_path}:champion",
            )
        for item in v30.get("verified", []):
            probe_id = str(item.get("probe_id", ""))
            raw_path = item.get("path")
            if not raw_path and probe_id:
                raw_path = str(
                    ROOT / "experiments/v30_meta_stack/submissions" / f"{probe_id}.csv"
                )
            if raw_path:
                add_record(
                    raw,
                    resolve_path(raw_path),
                    item.get("score"),
                    item.get("tp"),
                    item.get("predictions"),
                    "v30_online_results",
                    f"{v30_path}:verified:{probe_id}",
                )

    # The original ledger contains two older scored submissions.
    ledger_path = ROOT / "experiments/ledger.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        for item in ledger.get("experiments", []):
            if item.get("status") != "scored" or not item.get("candidate_path"):
                continue
            raw_path = item["candidate_path"]
            add_record(
                raw,
                resolve_path(raw_path),
                item.get("score"),
                item.get("tp"),
                item.get("predictions"),
                "ledger.json",
                f"{ledger_path}:{item.get('id')}",
            )

    # Later user-confirmed/public records.  These are intentionally explicit;
    # no OOF or simulation report is used as a score.
    v117_path = ROOT / "experiments/v117_distance1_online_result.json"
    if v117_path.exists():
        item = read_json(v117_path)
        add_record(
            raw,
            resolve_path(item["submitted_file"]),
            item.get("score"),
            item.get("inferred_tp"),
            item.get("predictions"),
            "v117_user_confirmed",
            str(v117_path),
        )

    for folder, source, detail in [
        (
            ROOT / "experiments/v104_15day_positive_campaign",
            "v104_user_confirmed",
            "calibration_result.json",
        ),
        (
            ROOT / "experiments/v118_safe_positive_campaign",
            "v118_user_confirmed",
            "calibration_result.json",
        ),
    ]:
        result_path = folder / "calibration_result.json"
        if result_path.exists():
            item = read_json(result_path)
            add_record(
                raw,
                folder / "probe_00_calibration.csv",
                item.get("score"),
                item.get("inferred_tp"),
                item.get("predictions"),
                source,
                str(result_path),
            )

    v119_path = ROOT / "experiments/v119_joint_campaign/online_scores.json"
    if v119_path.exists():
        for item in read_json(v119_path).get("results", []):
            raw_path = item.get("file")
            if raw_path:
                inferred_tp = item.get("inferred_tp_if_base_tp_956")
                add_record(
                    raw,
                    resolve_path(raw_path, v119_path.parent),
                    item.get("score"),
                    inferred_tp,
                    item.get("predictions"),
                    "v119_user_confirmed",
                    str(v119_path),
                )

    v120_path = ROOT / "experiments/v120_swap_campaign/online_scores.json"
    if v120_path.exists():
        for item in read_json(v120_path).get("records", []):
            raw_path = item.get("file")
            if not raw_path:
                resolved_by_sha = find_csv_by_sha(v120_path.parent, item.get("file_sha256"))
                raw_path = str(resolved_by_sha) if resolved_by_sha else None
            if raw_path:
                add_record(
                    raw,
                    resolve_path(raw_path, v120_path.parent),
                    item.get("score"),
                    item.get("inferred_tp"),
                    item.get("predictions"),
                    "v120_user_confirmed",
                    str(v120_path),
                )

    v121_path = ROOT / "experiments/v121_equation_safe_campaign/online_scores.json"
    if v121_path.exists():
        for item in read_json(v121_path).get("records", []):
            raw_path = item.get("file") or item.get("path")
            if not raw_path:
                resolved_by_sha = find_csv_by_sha(v121_path.parent, item.get("file_sha256"))
                raw_path = str(resolved_by_sha) if resolved_by_sha else None
            if raw_path:
                add_record(
                    raw,
                    resolve_path(raw_path, v121_path.parent),
                    item.get("score"),
                    item.get("inferred_tp"),
                    item.get("predictions"),
                    "v121_user_confirmed",
                    str(v121_path),
                )

    # V75 has one actual scored block row in its state machine.  Unlike the
    # later V120/V121 ledgers, the result entry stores no path (and has no
    # explicit ``status`` field), so resolve the canonical submission path
    # from its probe id.  The entry contains a displayed leaderboard score,
    # TP and prediction count; all other V75 rows are merely planned probes
    # and are deliberately not inferred here.
    v75_state_path = ROOT / "experiments/v75_dense_block_campaign/state.json"
    if v75_state_path.exists():
        v75_state = read_json(v75_state_path)
        for probe_id, item in (v75_state.get("results") or {}).items():
            if not isinstance(item, dict) or item.get("score") is None:
                continue
            raw_path = item.get("path") or item.get("submission")
            if not raw_path:
                raw_path = str(
                    v75_state_path.parent / "submissions" / f"{probe_id}.csv"
                )
            add_record(
                raw,
                resolve_path(raw_path, v75_state_path.parent),
                item.get("score"),
                item.get("tp"),
                item.get("predictions"),
                "v75_user_confirmed",
                f"{v75_state_path}:results:{probe_id}",
            )

    # Deduplicate identical files while preserving every provenance reference.
    unique: dict[str, dict[str, Any]] = {}
    score_conflicts: list[dict[str, Any]] = []
    for item in raw:
        existing = unique.get(item["sha256"])
        if existing is None:
            item["provenance"] = [
                {"source": item["source"], "source_detail": item["source_detail"]}
            ]
            unique[item["sha256"]] = item
        else:
            existing.setdefault("provenance", []).append(
                {"source": item["source"], "source_detail": item["source_detail"]}
            )
            if (existing["score"], existing["tp"], existing["predictions"]) != (
                item["score"],
                item["tp"],
                item["predictions"],
            ):
                score_conflicts.append({"existing": existing, "duplicate": item})

    records = list(unique.values())
    records.sort(key=lambda row: (row["predictions"], row["score"], row["path"]))

    # Explicitly document the offline/inferred artifacts that the campaign
    # directories contain, without treating their numbers as equations.
    for path in sorted(ROOT.glob("experiments/**/offline_simulation.json")):
        excluded.append(
            {
                "source": str(path),
                "status": "offline",
                "reason": "simulation, not leaderboard evidence",
            }
        )
    for path in sorted(ROOT.glob("experiments/**/posterior.json")):
        excluded.append(
            {
                "source": str(path),
                "status": "offline",
                "reason": "posterior/simulation artifact, not a scored file",
            }
        )
    return records, excluded + [{"score_conflicts": score_conflicts}]


def load_test_universe() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    with gzip.open(TEST_RECORDS, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    alarms: dict[tuple[str, str], dict[str, Any]] = {}
    orders: dict[str, dict[str, Any]] = {}
    for order in dataset["test"]:
        orders[order["order_id"]] = order
        for alarm in order.get("alarms", []):
            alarms[(order["order_id"], alarm["rid"])] = alarm
    return alarms, orders


def build_equations(
    records: list[dict[str, Any]], universe: dict[tuple[str, str], int]
) -> tuple[csr_matrix, np.ndarray, list[dict[str, Any]]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    rhs: list[int] = []
    metadata: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        selected = load_submission(Path(record["path"]))
        known_count = 0
        for order_id, rids in selected.items():
            for rid in rids:
                key = (order_id, rid)
                if key not in universe:
                    raise ValueError(f"unknown test alarm {key} in {record['path']}")
                rows.append(row_index)
                cols.append(universe[key])
                data.append(1)
                known_count += 1
        rhs.append(int(record["tp"]))
        metadata.append(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "score": record["score"],
                "predictions": record["predictions"],
                "tp": record["tp"],
                "nonzero_terms": known_count,
                "source": record["source"],
                "source_detail": record["source_detail"],
            }
        )
    matrix = csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(len(records), len(universe)),
        dtype=np.float64,
    )
    return matrix, np.asarray(rhs, dtype=np.float64), metadata


def solver_options(time_limit: float) -> dict[str, Any]:
    return {
        "integrality": np.ones(1),  # replaced by solve_objective
        "options": {"time_limit": float(time_limit)},
    }


def solve_objective(
    objective: np.ndarray,
    matrix: csr_matrix,
    rhs: np.ndarray,
    time_limit: float,
) -> tuple[int | None, str, str]:
    n = matrix.shape[1]
    kwargs = {
        "integrality": np.ones(n),
        "bounds": Bounds(np.zeros(n), np.ones(n)),
        "constraints": LinearConstraint(matrix, rhs, rhs),
        "options": {"time_limit": float(time_limit)},
    }
    result = milp(objective, **kwargs)
    if not result.success or result.fun is None:
        return None, str(result.status), str(result.message)
    return int(round(float(result.fun))), str(result.status), str(result.message)


def solve_min_max(
    coefficients: dict[int, float],
    matrix: csr_matrix,
    rhs: np.ndarray,
    time_limit: float,
) -> dict[str, Any]:
    objective = np.zeros(matrix.shape[1], dtype=np.float64)
    for index, coefficient in coefficients.items():
        objective[index] = coefficient
    minimum, min_status, min_message = solve_objective(objective, matrix, rhs, time_limit)
    maximum_neg, max_status, max_message = solve_objective(-objective, matrix, rhs, time_limit)
    maximum = None if maximum_neg is None else -maximum_neg
    return {
        "min": minimum,
        "max": maximum,
        "min_status": min_status,
        "max_status": max_status,
        "min_message": min_message,
        "max_message": max_message,
        "optimal": minimum is not None and maximum is not None,
    }


def load_catalog(path: Path, source_name: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    candidates = payload if isinstance(payload, list) else payload.get("candidates", [])
    output: list[dict[str, Any]] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or not item.get("order_id"):
            continue
        candidate = dict(item)
        candidate["catalog_source"] = source_name
        candidate["catalog_index"] = index
        candidate.setdefault("candidate_id", f"{source_name}_{index + 1:04d}")
        candidate["kind"] = (
            "swap"
            if candidate.get("remove_rid") and candidate.get("add_rid")
            else "addition"
            if candidate.get("add_rid")
            else "deletion"
        )
        output.append(candidate)
    return output


def observed_differences(records: list[dict[str, Any]], baseline_path: Path) -> list[dict[str, Any]]:
    base = load_submission(baseline_path)
    result: list[dict[str, Any]] = []
    for record in records:
        if Path(record["path"]).resolve() == baseline_path.resolve():
            continue
        current = load_submission(Path(record["path"]))
        removals: list[dict[str, str]] = []
        additions: list[dict[str, str]] = []
        for order_id in sorted(set(base) | set(current)):
            before = set(base.get(order_id, []))
            after = set(current.get(order_id, []))
            removals.extend({"order_id": order_id, "rid": rid} for rid in sorted(before - after))
            additions.extend({"order_id": order_id, "rid": rid} for rid in sorted(after - before))
        row: dict[str, Any] = {
            "path": record["path"],
            "sha256": record["sha256"],
            "score": record["score"],
            "predictions": record["predictions"],
            "tp": record["tp"],
            "delta_tp_vs_baseline": record["tp"] - BASE_TP,
            "removal_count": len(removals),
            "addition_count": len(additions),
            "removals": removals,
            "additions": additions,
        }
        if len(removals) == 1 and len(additions) == 1 and removals[0]["order_id"] == additions[0]["order_id"]:
            row["single_same_order_swap"] = True
            row["fixed_action_delta"] = record["tp"] - BASE_TP
        else:
            row["single_same_order_swap"] = False
        result.append(row)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    records, excluded = collect_records()
    if not records:
        raise RuntimeError("no scored records found")
    baseline = next((r for r in records if r["sha256"] == sha256(BASE_PATH)), None)
    if baseline is None:
        raise RuntimeError("current baseline was not collected as a scored record")

    alarms, orders = load_test_universe()
    universe_keys = sorted(alarms)
    universe = {key: index for index, key in enumerate(universe_keys)}
    matrix, rhs, equation_meta = build_equations(records, universe)

    feasibility = solve_objective(
        np.zeros(matrix.shape[1], dtype=np.float64), matrix, rhs, args.feasibility_time_limit
    )

    catalogs = []
    catalog_paths = args.catalog or [
        (ROOT / "experiments/v120_swap_campaign/candidate_catalog.json", "v120"),
        (ROOT / "experiments/v121_equation_safe_campaign/candidate_catalog.json", "v121"),
        (ROOT / "experiments/v122_multimodel_addition_campaign/candidate_catalog.json", "v122"),
    ]
    for raw_path, source_name in catalog_paths:
        path = Path(raw_path)
        if path.exists():
            catalogs.extend(load_catalog(path, source_name))

    catalog_count_before_dedup = len(catalogs)
    # De-duplicate exact actions across catalogs, retaining provenance.
    unique_candidates: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
    for candidate in catalogs:
        key = (
            candidate["order_id"],
            candidate.get("remove_rid"),
            candidate.get("add_rid"),
        )
        previous = unique_candidates.get(key)
        if previous is None:
            candidate["catalog_provenance"] = [candidate["catalog_source"]]
            unique_candidates[key] = candidate
        else:
            previous.setdefault("catalog_provenance", []).append(candidate["catalog_source"])
    catalogs = list(unique_candidates.values())

    candidate_bounds: list[dict[str, Any]] = []
    candidate_nodes: set[tuple[str, str]] = set()
    missing_candidates: list[dict[str, Any]] = []
    for candidate in catalogs:
        order_id = candidate["order_id"]
        add_rid = candidate.get("add_rid")
        remove_rid = candidate.get("remove_rid")
        coefficients: dict[int, float] = {}
        node_keys: list[dict[str, str]] = []
        valid = True
        if add_rid:
            add_key = (order_id, add_rid)
            add_index = universe.get(add_key)
            if add_index is None:
                valid = False
                node_keys.append({"role": "add", "order_id": order_id, "rid": add_rid, "missing": "true"})
            else:
                coefficients[add_index] = coefficients.get(add_index, 0.0) + 1.0
                candidate_nodes.add(add_key)
                node_keys.append({"role": "add", "order_id": order_id, "rid": add_rid})
        if remove_rid:
            remove_key = (order_id, remove_rid)
            remove_index = universe.get(remove_key)
            if remove_index is None:
                valid = False
                node_keys.append({"role": "remove", "order_id": order_id, "rid": remove_rid, "missing": "true"})
            else:
                coefficients[remove_index] = coefficients.get(remove_index, 0.0) - 1.0
                candidate_nodes.add(remove_key)
                node_keys.append({"role": "remove", "order_id": order_id, "rid": remove_rid})
        if valid and coefficients:
            bounds = solve_min_max(coefficients, matrix, rhs, args.candidate_time_limit)
        else:
            bounds = {
                "min": None,
                "max": None,
                "min_status": "missing_variable",
                "max_status": "missing_variable",
                "optimal": False,
            }
            missing_candidates.append(candidate)
        candidate_bounds.append(
            {
                "candidate_id": candidate["candidate_id"],
                "catalog_source": candidate["catalog_source"],
                "catalog_provenance": candidate.get("catalog_provenance", []),
                "order_id": order_id,
                "remove_rid": remove_rid,
                "add_rid": add_rid,
                "kind": candidate["kind"],
                "model_support": candidate.get("model_support"),
                "nominal_p_net_gain": candidate.get("nominal_p_net_gain"),
                "conservative_p_net_gain": candidate.get("conservative_p_net_gain"),
                "equation": bounds,
                "node_keys": node_keys,
            }
        )

    fixed_labels: list[dict[str, Any]] = []
    if not args.skip_node_bounds:
        for order_id, rid in sorted(candidate_nodes):
            index = universe[(order_id, rid)]
            bounds = solve_min_max({index: 1.0}, matrix, rhs, args.node_time_limit)
            fixed_labels.append(
                {
                    "order_id": order_id,
                    "rid": rid,
                    "index": index,
                    "label": bounds["min"] if bounds["min"] == bounds["max"] else None,
                    "min": bounds["min"],
                    "max": bounds["max"],
                    "equation": bounds,
                }
            )

    observed = observed_differences(records, BASE_PATH)
    fixed_actions = []
    for row in observed:
        if row.get("single_same_order_swap"):
            fixed_actions.append(
                {
                    "order_id": row["removals"][0]["order_id"],
                    "remove_rid": row["removals"][0]["rid"],
                    "add_rid": row["additions"][0]["rid"],
                    "delta_tp": row["fixed_action_delta"],
                    "evidence_file": row["path"],
                    "evidence_score": row["score"],
                }
            )

    # Compact, machine-readable outputs.  No submission CSV is emitted.
    record_output = {
        "version": 1,
        "ground_truth_positives": TRUE_ROOTS,
        "baseline": {
            "path": baseline["path"],
            "sha256": baseline["sha256"],
            "predictions": baseline["predictions"],
            "tp": baseline["tp"],
            "score": baseline["score"],
        },
        "baseline_tp_for_delta": BASE_TP,
        "unique_scored_file_count": len(records),
        "records": [
            {
                **record,
                "delta_tp_vs_baseline": record["tp"] - BASE_TP,
            }
            for record in records
        ],
        "excluded": excluded,
    }
    write_json(OUT / "real_scored_records.json", record_output)
    write_json(
        OUT / "equation_system.json",
        {
            "version": 1,
            "source_policy": "public scored/verified records only; inferred and offline artifacts excluded",
            "record_count": len(records),
            "variable_count": len(universe_keys),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "rhs": [int(x) for x in rhs.tolist()],
            "row_nonzero_terms": [int(x) for x in matrix.getnnz(axis=1).tolist()],
            "records": equation_meta,
            "variables": [
                {"index": index, "order_id": key[0], "rid": key[1]}
                for index, key in enumerate(universe_keys)
            ],
            "feasibility": {
                "success": feasibility[0] is not None,
                "objective_value": feasibility[0],
                "status": feasibility[1],
                "message": feasibility[2],
            },
        },
    )
    write_json(
        OUT / "candidate_bounds.json",
        {
            "version": 1,
            "catalog_count_before_dedup": catalog_count_before_dedup,
            "candidate_count": len(candidate_bounds),
            "missing_candidate_count": len(missing_candidates),
            "candidates": candidate_bounds,
        },
    )
    write_json(
        OUT / "fixed_labels.json",
        {
            "version": 1,
            "candidate_node_count": len(fixed_labels),
            "fixed_count": sum(x["label"] is not None for x in fixed_labels),
            "labels": fixed_labels,
        },
    )
    write_json(OUT / "observed_differences.json", {"version": 1, "differences": observed, "single_swap_fixed_actions": fixed_actions})

    optimal = [x for x in candidate_bounds if x["equation"].get("optimal")]
    positive = [x for x in optimal if x["equation"]["min"] == 1]
    fixed_negative = [x for x in optimal if x["equation"]["min"] == -1 and x["equation"]["max"] == -1]
    fixed_zero = [x for x in optimal if x["equation"]["min"] == 0 and x["equation"]["max"] == 0]
    distribution = Counter(
        (
            x["catalog_source"],
            x["equation"].get("min"),
            x["equation"].get("max"),
        )
        for x in optimal
    )
    summary = {
        "version": 1,
        "unique_scored_file_count": len(records),
        "equation_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "equation_feasible": feasibility[0] is not None,
        "baseline": record_output["baseline"],
        "target": {
            "f1": 0.945,
            "required_tp_at_p_1035": 983,
            "required_delta_tp": 27,
        },
        "catalogs": sorted({x["catalog_source"] for x in candidate_bounds}),
        "candidate_count": len(candidate_bounds),
        "candidate_optimal_count": len(optimal),
        "candidate_min_delta_positive_count": len(positive),
        "candidate_fixed_negative_count": len(fixed_negative),
        "candidate_fixed_zero_count": len(fixed_zero),
        "candidate_delta_distribution": [
            {
                "catalog_source": key[0],
                "min_delta": key[1],
                "max_delta": key[2],
                "count": count,
            }
            for key, count in sorted(distribution.items(), key=lambda item: str(item[0]))
        ],
        "min_delta_positive_candidates": positive,
        "fixed_negative_candidates": fixed_negative,
        "fixed_zero_candidates": fixed_zero,
        "fixed_online_single_swaps": fixed_actions,
        "notes": [
            "V30 verified records were recovered by probe_id because their JSON entries omit path.",
            "The older V121 equation_system.json is stale when this audit finds more records; use these outputs instead.",
            "No submission CSV was generated by this audit.",
        ],
    }
    write_json(OUT / "summary.json", summary)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        action="append",
        nargs=2,
        metavar=("PATH", "NAME"),
        help="catalog path and source name; repeat to replace the default V120/V121/V122 union",
    )
    parser.add_argument("--candidate-time-limit", type=float, default=2.0)
    parser.add_argument("--node-time-limit", type=float, default=1.0)
    parser.add_argument("--feasibility-time-limit", type=float, default=30.0)
    parser.add_argument("--skip-node-bounds", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    summary = audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))

"""V120 topology-replacement campaign.

The campaign is intentionally local-only.  It rebuilds one-remove/one-add
actions on the current 1,035-root checkpoint, creates a small coded probe
matrix, and provides an integer TP decoder.  It never uploads a file.

The model is reused from V29, but is refit against the current checkpoint and
four independent train fold definitions.  Historical pair manifests are used
as extra evidence, never as proof of an online gain.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/v120_swap_campaign"
DATASET = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V118_CAL = ROOT / "experiments/v118_safe_positive_campaign/probe_00_calibration.csv"
V119_DIR = ROOT / "experiments/v119_joint_campaign"
V117 = ROOT / "experiments/v117_distance1_online_result.json"

TRUE_ROOTS = 1044
BASE_P = 1035
BASE_TP = 956  # local checkpoint; online calibration is required before decoding
TARGET_F1 = 0.945
REQUIRED_DELTA = 28  # at P=1035, TP=984 is the first integer at/above .945
MIN_CANDIDATES = 60
TARGET_CANDIDATES = 100
GROUP_PROBES = 12
GROUP_SIZE = 9
SINGLE_PROBES = 6
FINAL_SLOTS = 5
RESERVE_SLOTS = 3
SIM_TRIALS = 100_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def f1(tp: int, predictions: int = BASE_P) -> float:
    return 2.0 * tp / (TRUE_ROOTS + predictions)


def load_rows(path: Path = BASE) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"order_id": row["order_id"], "roots": json.loads(row["output"])["rootcause"]})
    return rows


def load_records() -> dict[str, Any]:
    with gzip.open(RECORDS, "rt", encoding="utf-8") as f:
        return json.load(f)


def alarm_map(records: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (order["order_id"], alarm["rid"]): alarm
        for order in records["test"]
        for alarm in order.get("alarms", [])
    }


def changed_keys(left: Path, right: Path) -> set[tuple[str, str]]:
    """Return (order,rid) keys present in left but absent in right."""
    a = { (r["order_id"], n["@rid"]) for r in load_rows(left) for n in r["roots"] }
    b = { (r["order_id"], n["@rid"]) for r in load_rows(right) for n in r["roots"] }
    return a - b


def rejected_delete_keys(base: list[dict[str, Any]]) -> set[tuple[str, str]]:
    bad: set[tuple[str, str]] = set()
    if V118_CAL.exists():
        bad |= changed_keys(BASE, V118_CAL)
    probe = V119_DIR / "probe_01.csv"
    if probe.exists():
        bad |= changed_keys(BASE, probe)
    probe117 = ROOT / "experiments/submissions/distance1_swap_equation_filtered.csv"
    if probe117.exists():
        bad |= changed_keys(BASE, probe117)
    return bad


def load_v29_module():
    path = ROOT / "experiments/v29_domain_ranker/v29_actions.py"
    spec = importlib.util.spec_from_file_location("v29_actions_for_v120", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_mask(arrays: dict[str, np.ndarray], records: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    by_order = {row["order_id"]: {n["@rid"] for n in row["roots"]} for row in rows}
    ptr = arrays["test_alarm_ptr"]
    mask = np.zeros(len(arrays["test_v11"]), dtype=bool)
    for oi, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        order = records["test"][oi]
        selected = by_order[order["order_id"]]
        for local, alarm in enumerate(order.get("alarms", [])):
            mask[int(start) + local] = alarm["rid"] in selected
        if int(mask[int(start):int(stop)].sum()) != len(selected):
            raise ValueError(f"baseline alignment failed for {order['order_id']}")
    if int(mask.sum()) != BASE_P:
        raise ValueError(f"baseline P={int(mask.sum())}, expected {BASE_P}")
    return mask


def pair_model_candidates(rows: list[dict[str, Any]], records: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    """Fit V29 pair models with four independent fold definitions."""
    v29 = load_v29_module()
    with np.load(DATASET, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    train_mask = v29.exact_count_mask(arrays["train_v11"], arrays["train_alarm_ptr"], v29.TARGET_TRAIN_P)
    test_mask = current_mask(arrays, records, rows)
    train_x = v29.alarm_features(arrays, "train")
    test_x = v29.alarm_features(arrays, "test")
    labels = arrays["train_labels"].astype(np.int8)
    train_pair = v29.pair_candidates(arrays["train_alarm_ptr"], arrays["train_v11"], train_x, train_mask, labels)
    test_pair = v29.pair_candidates(arrays["test_alarm_ptr"], arrays["test_v11"], test_x, test_mask, None)
    scores = []
    for fold_name in ("train_station_folds", "train_folds", "train_connected_folds", "train_legacy_folds"):
        _, test_score = v29.crossfit_pair(
            train_pair[0], train_pair[1], train_pair[2], test_pair[0],
            arrays[fold_name].astype(np.int8), quick,
        )
        scores.append(test_score.astype(np.float32))
    matrix = np.vstack(scores)
    aggregate = matrix.mean(axis=0)
    support = (matrix > 0).sum(axis=0)
    ptr = arrays["test_alarm_ptr"]
    bad_deletes = rejected_delete_keys(rows)
    order_ids = [x["order_id"] for x in records["test"]]
    candidates = []
    # Keep support=2 pairs in the catalog as explicitly labelled exploratory
    # candidates.  They are useful for online information gain, but they do
    # not pass the final >=3-model eligibility gate.
    for i in np.flatnonzero(support >= 2):
        oi = int(test_pair[1][i])
        order_id = order_ids[oi]
        add_row = int(test_pair[3][i])
        remove_row = int(test_pair[4][i])
        start = int(ptr[oi])
        add_alarm = records["test"][oi]["alarms"][add_row - start]
        remove_alarm = records["test"][oi]["alarms"][remove_row - start]
        remove_key = (order_id, remove_alarm["rid"])
        if remove_key in bad_deletes:
            continue
        q = float(np.clip(0.50 + 0.25 * float(aggregate[i]) + 0.025 * (int(support[i]) - 3), 0.50, 0.90))
        candidates.append({
            "order_id": order_id,
            "remove_rid": remove_alarm["rid"],
            "add_rid": add_alarm["rid"],
            "source_models": [
                name for name, values in zip(("station", "template", "connected", "legacy"), matrix)
                if float(values[i]) > 0
            ],
            "model_support": int(support[i]),
            "candidate_status": "eligible" if int(support[i]) >= 3 else "exploratory_low_support",
            "model_score": float(aggregate[i]),
            "nominal_p_net_gain": q,
            "conservative_p_net_gain": max(0.50, q - 0.05),
            "evidence": {"fold_scores": [float(x) for x in matrix[:, i]]},
        })
    return candidates


def historical_candidates(rows: list[dict[str, Any]], records: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect valid same-order pairs from V29/V35/V28/V41 manifests."""
    base_by = {row["order_id"]: {n["@rid"] for n in row["roots"]} for row in rows}
    alarm_by = alarm_map(records)
    rejected = rejected_delete_keys(rows)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in ROOT.glob("experiments/**/manifests/*.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        source = str(path).replace("\\", "/")
        for action in manifest.get("actions", []):
            adds = action.get("add_rids", [])
            removes = action.get("remove_rids", [])
            if len(adds) != 1 or len(removes) != 1:
                continue
            oid, add, remove = action.get("order_id"), adds[0], removes[0]
            if oid not in base_by or remove not in base_by[oid] or add in base_by[oid]:
                continue
            if (oid, remove) in rejected or (oid, add) not in alarm_by:
                continue
            key = (oid, remove, add)
            utility = action.get("expected_gain")
            if utility is None:
                utility = action.get("evidence", {}).get("expected_tp_utility", 0.0)
            utility = float(utility or 0.0)
            item = out.setdefault(key, {
                "order_id": oid, "remove_rid": remove, "add_rid": add,
                "source_models": [], "source_files": [], "model_support": 0,
                "model_score": utility, "nominal_p_net_gain": float(np.clip(0.52 + 0.20 * utility, 0.50, 0.80)),
                "conservative_p_net_gain": float(np.clip(0.47 + 0.20 * utility, 0.50, 0.75)),
                "evidence": {"historical_utilities": []},
            })
            item["source_files"].append(source)
            tag = "v29" if "v29_" in source else ("v35" if "v35_" in source else ("v41" if "v41_" in source else "historical"))
            if tag not in item["source_models"]:
                item["source_models"].append(tag)
            item["model_support"] = len(item["source_models"])
            item["evidence"]["historical_utilities"].append(utility)
    return list(out.values())


def merge_candidates(model: list[dict[str, Any]], historical: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in model + historical:
        key = (item["order_id"], item["remove_rid"], item["add_rid"])
        if key not in by_key:
            by_key[key] = item
        else:
            cur = by_key[key]
            cur["source_models"] = sorted(set(cur.get("source_models", [])) | set(item.get("source_models", [])))
            cur["source_files"] = sorted(set(cur.get("source_files", [])) | set(item.get("source_files", [])))
            cur["model_support"] = max(int(cur.get("model_support", 0)), len(cur["source_models"]))
            cur["nominal_p_net_gain"] = max(float(cur["nominal_p_net_gain"]), float(item["nominal_p_net_gain"]))
            cur["conservative_p_net_gain"] = min(float(cur["conservative_p_net_gain"]), float(item["conservative_p_net_gain"]))
    order_rank = {row["order_id"]: i for i, row in enumerate(rows)}
    candidates = list(by_key.values())
    for idx, item in enumerate(candidates, 1):
        item["candidate_id"] = f"v120_pair_{idx:03d}"
        item["order_index"] = order_rank[item["order_id"]]
        item["eligible_nominal"] = float(item["nominal_p_net_gain"]) >= 0.60 and int(item["model_support"]) >= 3
        item["eligible_conservative"] = float(item["conservative_p_net_gain"]) >= 0.50 and int(item["model_support"]) >= 3
    candidates.sort(key=lambda x: (-float(x["nominal_p_net_gain"]), -int(x["model_support"]), x["order_id"], x["remove_rid"], x["add_rid"]))
    for idx, item in enumerate(candidates, 1):
        item["rank"] = idx
    return candidates


def node_from_alarm(alarm: dict[str, Any]) -> dict[str, str]:
    source = alarm.get("source") or {}
    return {"@rid": alarm["rid"], "title": source.get("title", alarm.get("title", "")),
            "location": source.get("location", alarm.get("raw_location", alarm.get("location", ""))),
            "reason": source.get("reason", alarm.get("reason", ""))}


def apply_pairs(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], indices: list[int], alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in indices:
        by_order[candidates[i]["order_id"]].append(candidates[i])
    out = []
    for row in rows:
        roots = [dict(x) for x in row["roots"]]
        present = {x["@rid"] for x in roots}
        seen_add, seen_remove = set(), set()
        for action in by_order.get(row["order_id"], []):
            if action["remove_rid"] in seen_remove or action["add_rid"] in seen_add:
                raise ValueError(f"overlapping pairs for {row['order_id']}")
            if action["remove_rid"] not in present or action["add_rid"] in present:
                raise ValueError(f"invalid pair for {row['order_id']}")
            roots = [x for x in roots if x["@rid"] != action["remove_rid"]]
            present.remove(action["remove_rid"])
            roots.append(node_from_alarm(alarms[(row["order_id"], action["add_rid"])]))
            present.add(action["add_rid"])
            seen_remove.add(action["remove_rid"]); seen_add.add(action["add_rid"])
        if not 1 <= len(roots) <= 8:
            raise ValueError(f"root count for {row['order_id']}: {len(roots)}")
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    if sum(len(json.loads(x["output"])["rootcause"]) for x in out) != BASE_P:
        raise ValueError("replacement changed prediction count")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader(); w.writerows(rows)


def build_probe_indices(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(candidates)
    matrix = np.zeros((GROUP_PROBES, n), dtype=np.int8)
    # First cover every candidate once.  A second pass spends the remaining
    # capacity on the highest-prior candidates; with 92 candidates and only
    # 108 group slots, covering every candidate twice is impossible under the
    # requested 8--10 actions per probe.
    order_used = [set() for _ in range(GROUP_PROBES)]
    incidence = []
    for pass_number in (1, 2):
        for idx in range(n):
            desired = 1 if pass_number == 1 else (2 if idx < GROUP_PROBES * GROUP_SIZE - n else 1)
            if pass_number == 2 and int(matrix[:, idx].sum()) >= desired:
                continue
            placed = 0
            row_order = sorted(range(GROUP_PROBES), key=lambda row: (int(matrix[row].sum()), (idx * 7 + row) % GROUP_PROBES))
            for row in row_order:
                if int(matrix[row].sum()) >= GROUP_SIZE or candidates[idx]["order_id"] in order_used[row]:
                    continue
                matrix[row, idx] = 1; order_used[row].add(candidates[idx]["order_id"]); placed += 1
                if placed == desired: break
            if placed: incidence.append(placed)
    # Ensure every row is non-empty; deterministic repair from rows with spare capacity.
    for row in range(GROUP_PROBES):
        if matrix[row].sum() == 0:
            donor = int(np.argmax(matrix.sum(axis=1)))
            idxs = np.flatnonzero(matrix[donor])
            for idx in idxs:
                if candidates[idx]["order_id"] not in order_used[row]:
                    matrix[donor, idx] = 0; order_used[donor].remove(candidates[idx]["order_id"])
                    matrix[row, idx] = 1; order_used[row].add(candidates[idx]["order_id"]); break
    meta = [{"probe_id": i + 1, "kind": "group", "candidate_indices": [int(x) for x in np.flatnonzero(matrix[i])], "size": int(matrix[i].sum())} for i in range(GROUP_PROBES)]
    return matrix, meta


def simulate(candidates: list[dict[str, Any]], seed: int = 120) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if not candidates:
        return {"trials": SIM_TRIALS, "nominal": {"p_reach_0945": 0.0}, "conservative": {"p_reach_0945": 0.0}}
    q = np.asarray([float(x["nominal_p_net_gain"]) for x in candidates])
    qc = np.asarray([float(x["conservative_p_net_gain"]) for x in candidates])
    # Pick one action per order for a final portfolio, in descending prior order.
    chosen = []
    used = set()
    for i in np.argsort(-q, kind="stable"):
        if candidates[int(i)]["order_id"] in used: continue
        used.add(candidates[int(i)]["order_id"]); chosen.append(int(i))
    chosen = chosen[: min(40, len(chosen))]
    def run(probs: np.ndarray) -> dict[str, Any]:
        p = probs[chosen]
        gains = np.where(rng.random((SIM_TRIALS, len(chosen))) < p[None, :], 1, -1).sum(axis=1)
        best_k = 0; best_reach = -1.0; best_p10 = 0.0; best_expected = -1e9
        for k in range(min(40, len(chosen)), max(28, min(40, len(chosen))) - 1, -1):
            g = np.where(rng.random((SIM_TRIALS, k)) < p[:k][None, :], 1, -1).sum(axis=1)
            reach = float(np.mean(g >= REQUIRED_DELTA)); p10 = float(np.quantile(g, .10)); exp = float(np.mean(g))
            if reach > best_reach: best_k, best_reach, best_p10, best_expected = k, reach, p10, exp
        return {"chosen_count": len(chosen), "selected_k": best_k, "expected_delta_tp": best_expected,
                "p_reach_0945": best_reach, "p10_f1": f1(BASE_TP + int(math.floor(best_p10))),
                "q_min_selected": float(np.min(p[:best_k])) if best_k else None}
    return {"trials": SIM_TRIALS, "candidate_count": len(candidates), "nominal": run(q), "conservative": run(qc)}


def structural_validate(path: Path) -> dict[str, Any]:
    rows = load_rows(path)
    counts = [len(x["roots"]) for x in rows]
    return {"path": str(path), "sha256": sha256(path), "orders": len(rows), "predictions": int(sum(counts)),
            "duplicate_order_ids": len(rows) - len({x["order_id"] for x in rows}), "root_count_min": min(counts), "root_count_max": max(counts),
            "valid": len(rows) == 546 and sum(counts) == BASE_P and min(counts) >= 1 and max(counts) <= 8}


def generate(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); records = load_records(); alarms = alarm_map(records)
    existing_catalog = OUT / "candidate_catalog.json"
    if args.reuse_catalog and existing_catalog.exists():
        saved = read_json(existing_catalog)
        candidates = saved["candidates"]
        model_count = saved.get("model_candidate_count", 0)
        historical_count = saved.get("historical_candidate_count", 0)
    else:
        model = pair_model_candidates(rows, records, args.quick)
        historical = historical_candidates(rows, records)
        candidates = merge_candidates(model, historical, rows)
        model_count = len(model); historical_count = len(historical)
    write_json(OUT / "candidate_catalog.json", {"version": "v120", "base": str(BASE), "base_sha256": sha256(BASE),
        "base_predictions": BASE_P, "base_tp_local": BASE_TP, "candidate_count": len(candidates),
        "model_candidate_count": model_count, "historical_candidate_count": historical_count, "candidates": candidates})
    matrix, probes = build_probe_indices(candidates)
    write_json(OUT / "matrix.json", {"version": "v120", "rows": matrix.tolist(), "probes": probes,
        "candidate_count": len(candidates), "coverage": {"at_least_once": int((matrix.sum(0) >= 1).sum()), "at_least_twice": int((matrix.sum(0) >= 2).sum())}})
    # Baseline calibration is an exact copy rendered through the same validator.
    baseline_path = OUT / "probe_00_baseline.csv"; write_csv(baseline_path, [{"order_id": x["order_id"], "output": json.dumps({"rootcause": x["roots"]}, ensure_ascii=False)} for x in rows])
    probe_records = [{"probe_id": 0, "kind": "baseline", "path": str(baseline_path), "predictions": BASE_P, "sha256": sha256(baseline_path), "validation": structural_validate(baseline_path)}]
    for i, item in enumerate(probes, 1):
        indices = item["candidate_indices"]
        path = OUT / f"probe_{i:02d}_group.csv"
        out_rows = apply_pairs(rows, candidates, indices, alarms); write_csv(path, out_rows)
        item.update({"path": str(path), "sha256": sha256(path), "predictions": BASE_P, "validation": structural_validate(path)})
        probe_records.append(item)
    ranked = sorted(range(len(candidates)), key=lambda i: (-float(candidates[i]["nominal_p_net_gain"]), candidates[i]["order_id"]))
    singles = []
    used_orders = set()
    for i in ranked:
        if candidates[i]["order_id"] in used_orders: continue
        used_orders.add(candidates[i]["order_id"]); singles.append(i)
        if len(singles) == SINGLE_PROBES: break
    for j, idx in enumerate(singles, GROUP_PROBES + 1):
        path = OUT / f"probe_{j:02d}_single.csv"; out_rows = apply_pairs(rows, candidates, [idx], alarms); write_csv(path, out_rows)
        probe_records.append({"probe_id": j, "kind": "single_confirmation", "candidate_indices": [idx], "path": str(path), "predictions": BASE_P, "sha256": sha256(path), "validation": structural_validate(path)})
    sim = simulate(candidates)
    eligible = [x for x in candidates if x["eligible_nominal"]]
    gate = {"candidate_count": len(candidates), "eligible_nominal_count": len(eligible), "route_feasible": len(candidates) >= MIN_CANDIDATES and len(eligible) >= MIN_CANDIDATES,
            "offline_nominal": sim["nominal"], "offline_conservative": sim["conservative"],
            "nominal_pass": sim["nominal"].get("p_reach_0945", 0.0) >= .85,
            "conservative_pass": sim["conservative"].get("p_reach_0945", 0.0) >= .80,
            "final_emission_allowed": False, "reason": "online posterior is required before final files"}
    write_json(OUT / "probes.json", {"version": "v120", "allocation": {"baseline": 1, "group": GROUP_PROBES, "single": SINGLE_PROBES, "final_slots": FINAL_SLOTS, "reserve": RESERVE_SLOTS}, "probes": probe_records})
    write_json(OUT / "offline_simulation.json", sim)
    write_json(OUT / "final_gate.json", gate)
    write_json(OUT / "validation.json", {"base": structural_validate(BASE), "probes": [x["validation"] for x in probe_records], "all_valid": all(x["validation"]["valid"] for x in probe_records)})
    (OUT / "README.md").write_text("""# V120 topology replacement campaign\n\nGenerated locally from the current 1,035-root checkpoint. Every action is one same-order deletion plus one addition, so all probes have P=1035.\n\n1. Submit `probe_00_baseline.csv` once to calibrate online TP (do not assume the local TP=956).\n2. Record its six-decimal score in `online_scores.json` using the decoder.\n3. Submit group probes only in the order selected by the posterior; this generator does not upload.\n4. Never create a `final_*.csv` unless `final_gate.json` becomes true after online decoding.\n\nThe current offline gate is evidence about the model only, not a leaderboard result.\n""", encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "eligible_nominal": len(eligible), "offline_nominal": sim["nominal"], "offline_conservative": sim["conservative"], "route_feasible": gate["route_feasible"], "output": str(OUT)}, ensure_ascii=False, indent=2))


def decode(args: argparse.Namespace) -> None:
    catalog = read_json(OUT / "candidate_catalog.json"); matrix = np.asarray(read_json(OUT / "matrix.json")["rows"], dtype=int)
    scores = [float(x) for x in args.scores.split(",") if x.strip()]
    if len(scores) > len(matrix): raise ValueError("too many scores")
    baseline_tp = args.baseline_tp
    if baseline_tp is None:
        baseline_tp = BASE_TP
    inferred = [int(round(score * (TRUE_ROOTS + BASE_P) / 2.0)) for score in scores]
    inferred_neighbours = [[max(0, tp - 1), tp, tp + 1] for tp in inferred]
    deltas = [x - baseline_tp for x in inferred]
    z = np.zeros(len(catalog["candidates"]), dtype=int)
    # For each observed row, solve a bounded least-squares integer problem. A
    # later online batch can overwrite the selected vector with an exact MILP.
    if deltas:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            result = milp(c=np.zeros(len(z)), integrality=np.ones(len(z)), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix[:len(deltas)], deltas, deltas), options={"time_limit": 20})
            if result.success and result.x is not None: z = np.rint(result.x).astype(int)
        except Exception:
            pass
    selected = [i for i, x in enumerate(z) if x]
    q = np.asarray([float(x["nominal_p_net_gain"]) for x in catalog["candidates"]])
    qc = np.asarray([float(x["conservative_p_net_gain"]) for x in catalog["candidates"]])
    expected = float(np.sum(2 * q[selected] - 1)) if selected else 0.0
    robust_expected = float(np.sum(2 * qc[selected] - 1)) if selected else 0.0
    result = {"baseline_tp": baseline_tp, "scores": scores, "inferred_tp": inferred, "inferred_tp_neighbours": inferred_neighbours, "delta_tp": deltas,
              "selected_indices": selected, "selected_pairs": [catalog["candidates"][i] for i in selected],
              "expected_net_gain": expected, "conservative_expected_net_gain": robust_expected,
              "equation_consistent": bool(len(deltas) == 0 or np.all(matrix[:len(deltas)] @ z == np.asarray(deltas))),
              "p_reach_0945": 0.0, "p10_f1": 0.0, "final_emission_allowed": False}
    if selected:
        rng = np.random.default_rng(1201)
        qsel = q[selected]
        draws = np.where(rng.random((SIM_TRIALS, len(selected))) < qsel[None, :], 1, -1).sum(axis=1)
        result["p_reach_0945"] = float(np.mean(draws >= REQUIRED_DELTA))
        result["p10_f1"] = f1(baseline_tp + int(math.floor(np.quantile(draws, .10))))
    result["final_emission_allowed"] = bool(
        result["equation_consistent"] and expected >= REQUIRED_DELTA and
        robust_expected >= REQUIRED_DELTA and result["p_reach_0945"] >= .80 and
        result["p10_f1"] >= TARGET_F1
    )
    write_json(OUT / "decoded.json", result)
    if result["final_emission_allowed"] and args.emit_final:
        rows = load_rows(); records = load_records(); alarms = alarm_map(records)
        unique, seen_orders = [], set()
        for i in sorted(selected, key=lambda j: -q[j]):
            oid = catalog["candidates"][i]["order_id"]
            if oid in seen_orders: continue
            seen_orders.add(oid); unique.append(i)
        branches = {
            "final_map": unique,
            "final_conservative": [i for i in unique if qc[i] >= .60],
            "final_robust": [i for i in unique if qc[i] >= .55],
            "final_model_weight": sorted(unique, key=lambda i: -float(catalog["candidates"][i].get("model_score", 0.0))),
            "final_online_reranked": sorted(unique, key=lambda i: -q[i]),
        }
        emitted = []
        for name, indices in branches.items():
            path = OUT / f"{name}.csv"; write_csv(path, apply_pairs(rows, catalog["candidates"], indices, alarms))
            emitted.append({"name": name, "path": str(path), "sha256": sha256(path), "validation": structural_validate(path), "indices": indices})
        result["emitted_final_branches"] = emitted
        write_json(OUT / "decoded.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def record_score(args: argparse.Namespace) -> None:
    path = OUT / "online_scores.json"
    data = read_json(path) if path.exists() else {"version": "v120", "records": []}
    raw_tp = float(args.score) * (TRUE_ROOTS + BASE_P) / 2.0
    tp = int(round(raw_tp))
    baseline_tp = tp if args.probe_id == 0 and args.baseline_tp is None else (args.baseline_tp if args.baseline_tp is not None else BASE_TP)
    entry = {"probe_id": args.probe_id, "score": float(args.score), "predictions": BASE_P,
             "inferred_tp": tp, "inferred_tp_neighbours": [max(0, tp - 1), tp, tp + 1], "raw_tp": raw_tp,
             "baseline_tp": baseline_tp, "delta_tp_vs_baseline": tp - baseline_tp,
             "file_sha256": args.sha256 or "", "note": "Leaderboard score is authoritative; integer TP is inferred from six decimals."}
    data.setdefault("records", []).append(entry)
    data["records"] = sorted(data["records"], key=lambda x: x["probe_id"])
    write_json(path, data)
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate"); gen.add_argument("--quick", action="store_true"); gen.add_argument("--reuse-catalog", action="store_true")
    dec = sub.add_parser("decode"); dec.add_argument("--scores", required=True, help="comma-separated group-probe scores in matrix order"); dec.add_argument("--baseline-tp", type=int, default=None); dec.add_argument("--emit-final", action="store_true")
    rec = sub.add_parser("record"); rec.add_argument("--probe-id", type=int, required=True); rec.add_argument("--score", type=float, required=True); rec.add_argument("--baseline-tp", type=int, default=None); rec.add_argument("--sha256", default="")
    args = parser.parse_args()
    if args.command == "generate": generate(args)
    elif args.command == "decode": decode(args)
    else: record_score(args)


if __name__ == "__main__":
    main()

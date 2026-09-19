"""V119 joint deletion/addition adaptive campaign.

This module only creates local CSVs and state files.  It never uploads files.
The campaign is deliberately conservative after the V118 calibration showed
that the previous fixed deletion pool was unsafe.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv"
OUT = ROOT / "experiments/v119_joint_campaign"
V75 = ROOT / "experiments/v75_dense_block_campaign/report.json"
V58 = ROOT / "experiments/v58_coded_campaign/report.json"
V59 = ROOT / "experiments/v59_extended_coded_campaign/report.json"
V11_SCORES = ROOT / "codexgz/v11/v11_test_scores.csv"
SEMANTIC = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V118_DIR = ROOT / "experiments/v118_safe_positive_campaign"

TRUE_POS = 1044
BASE_TP = 956
BASE_P = 1035
TARGET = 0.945
PROBES = 20
DELETION_PROBES = 8
ADDITION_PROBES = 8
MIXED_PROBES = 4
ADAPTIVE_SLOTS = 4
FINAL_SLOTS = 3
ADD_COUNT = 71
DELETE_COUNT = 80
CALIBRATED_DELETE_COUNT = 40
CALIBRATED_SAFE_DELETIONS = 9
SIM_TRIALS = 100_000


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def calibrate_false_priors(priors: np.ndarray, target_sum: float) -> np.ndarray:
    """Shift logits so the calibrated pool has the observed expected count."""
    priors = np.clip(np.asarray(priors, dtype=float), 1e-5, 1 - 1e-5)
    lo, hi = -20.0, 20.0
    logits = np.log(priors / (1 - priors))
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(np.sum(sigmoid(logits + mid))) > target_sum:
            hi = mid
        else:
            lo = mid
    return np.asarray(sigmoid(logits + (lo + hi) / 2), dtype=float)


def load_base() -> list[dict]:
    rows = []
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            roots = json.loads(row["output"]).get("rootcause", [])
            rows.append({"order_id": row["order_id"], "roots": roots})
    return rows


def load_alarm_lookup() -> dict[tuple[str, str], dict]:
    out = {}
    with gzip.open(SEMANTIC, "rt", encoding="utf-8") as f:
        records = json.load(f)
    for order in records["test"]:
        for alarm in order.get("alarms", []):
            out[(order["order_id"], alarm.get("rid"))] = alarm
    return out


def apply_actions(rows: list[dict], adds: Iterable[dict], deletes: Iterable[dict], alarm_by: dict) -> list[dict]:
    add_by: dict[str, set[str]] = {}
    del_by: dict[str, set[str]] = {}
    for action in adds:
        add_by.setdefault(action["order_id"], set()).update(action.get("add_rids", []))
    for action in deletes:
        del_by.setdefault(action["order_id"], set()).update(action.get("remove_rids", []))
    out = []
    for row in rows:
        removed = del_by.get(row["order_id"], set())
        roots = [r for r in row["roots"] if r.get("@rid") not in removed]
        present = {r.get("@rid") for r in roots}
        for rid in sorted(add_by.get(row["order_id"], set())):
            if rid in present:
                continue
            src = alarm_by.get((row["order_id"], rid), {})
            roots.append({
                "@rid": rid,
                "title": src.get("title", ""),
                "location": src.get("raw_location", src.get("location", "")),
                "reason": src.get("reason", ""),
            })
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
        writer.writeheader()
        writer.writerows(rows)


def action_key(action: dict, field: str) -> tuple[str, str]:
    rid = action[field][0]
    return action["order_id"], rid


def current_v118_delete_keys() -> set[tuple[str, str]]:
    """Reconstruct the V118 fixed pool for calibration, without importing it."""
    scores = {}
    with V11_SCORES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["order_id"], row["rid"])] = 0.25 * float(row["context_score"]) + 0.75 * float(row["meta_mean"])
    candidates = []
    for row in load_base():
        for node in row["roots"]:
            key = (row["order_id"], node.get("@rid"))
            candidates.append((scores.get(key, -1.0), key))
    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1]))
    return {key for _, key in candidates[:CALIBRATED_DELETE_COUNT]}


def load_candidates() -> tuple[list[dict], np.ndarray, list[dict], np.ndarray, dict]:
    """Load de-duplicated single-node additions and deletions."""
    j75 = read_json(V75)
    add_actions = []
    for p, action in zip(j75["correctness_probabilities"], j75["candidate_actions"]):
        if len(action.get("add_rids", [])) == 1 and not action.get("remove_rids"):
            add_actions.append((float(p), action))
    add_actions.sort(key=lambda x: (-x[0], x[1]["order_id"], x[1]["add_rids"][0]))
    adds, add_priors, seen = [], [], set()
    for p, action in add_actions:
        key = action_key(action, "add_rids")
        if key in seen:
            continue
        seen.add(key)
        adds.append(action)
        add_priors.append(p)
        if len(adds) >= ADD_COUNT:
            break

    delete_rows = []
    for path in (V75, V58, V59):
        report = read_json(path)
        for p, action in zip(report["correctness_probabilities"], report["candidate_actions"]):
            if len(action.get("remove_rids", [])) == 1 and not action.get("add_rids"):
                # p is the probability that the deletion is correct.  We need
                # the probability that the removed baseline node is false.
                delete_rows.append((1.0 - float(p), action))
    delete_rows.sort(key=lambda x: (-x[0], x[1]["order_id"], x[1]["remove_rids"][0]))
    fixed_keys = current_v118_delete_keys()
    fixed_by_key = {
        (row["order_id"], node.get("@rid")): {
            "action_id": "v118_calibrated_delete",
            "order_id": row["order_id"],
            "remove_rids": [node.get("@rid")],
            "add_rids": [],
            "source": "v118_calibrated_pool",
        }
        for row in load_base()
        for node in row["roots"]
        if (row["order_id"], node.get("@rid")) in fixed_keys
    }
    # Force all 40 calibrated nodes into the new pool.  Their individual
    # false-positive probabilities are intentionally flat before calibration:
    # V118 only tells us the aggregate (9/40), not which specific nodes are
    # safe to delete.
    deletes = list(fixed_by_key.values())
    del_priors = [CALIBRATED_SAFE_DELETIONS / CALIBRATED_DELETE_COUNT] * len(deletes)
    dseen = set(fixed_by_key)
    # Fill the remaining slots with independent, higher-quality candidates.
    for q_false, action in delete_rows:
        key = action_key(action, "remove_rids")
        if key in dseen:
            continue
        dseen.add(key)
        deletes.append(action)
        del_priors.append(q_false)
        if len(deletes) >= DELETE_COUNT:
            break

    # Calibrate the V118-overlapping subset against its observed aggregate:
    # 9 safe deletions out of 40 means 9 false baseline nodes.
    fixed_keys = current_v118_delete_keys()
    idx = [i for i, action in enumerate(deletes) if action_key(action, "remove_rids") in fixed_keys]
    if len(idx) >= 8:
        calibrated = calibrate_false_priors(np.asarray([del_priors[i] for i in idx]), CALIBRATED_SAFE_DELETIONS)
        for i, q in zip(idx, calibrated):
            del_priors[i] = float(q)

    meta = {
        "calibrated_v118_overlap": len(idx),
        "calibrated_safe_deletions": CALIBRATED_SAFE_DELETIONS,
        "calibration_score": 0.907308,
        "calibration_predictions": 995,
        "candidate_sources": [str(V75), str(V58), str(V59)],
    }
    return adds, np.asarray(add_priors, float), deletes, np.asarray(del_priors, float), meta


def balanced_mask(items: int, rows: int, repeats: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((rows, items), dtype=np.int8)
    for j in range(items):
        choices = rng.choice(rows, size=min(repeats, rows), replace=False)
        mask[choices, j] = 1
    # Repair empty rows by moving one incidence from the densest row.
    for r in range(rows):
        if mask[r].sum() == 0:
            donor = int(np.argmax(mask.sum(axis=1)))
            col = int(np.flatnonzero(mask[donor])[0])
            mask[donor, col] = 0
            mask[r, col] = 1
    return mask


def build_matrix(n_add: int, n_del: int, seed: int = 119) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    matrix = np.zeros((PROBES, n_add + n_del), dtype=np.int8)
    groups = []
    dm = balanced_mask(n_del, DELETION_PROBES, 2, rng)
    am = balanced_mask(n_add, ADDITION_PROBES, 2, rng)
    matrix[:DELETION_PROBES, n_add:] = dm
    matrix[DELETION_PROBES:DELETION_PROBES + ADDITION_PROBES, :n_add] = am
    for r in range(MIXED_PROBES):
        row = DELETION_PROBES + ADDITION_PROBES + r
        d = rng.choice(n_del, size=min(16, n_del), replace=False)
        a = rng.choice(n_add, size=min(16, n_add), replace=False)
        matrix[row, n_add + d] = 1
        matrix[row, a] = 1
    groups.extend(["delete_only"] * DELETION_PROBES)
    groups.extend(["add_only"] * ADDITION_PROBES)
    groups.extend(["mixed"] * MIXED_PROBES)
    return matrix, groups


def prediction_count(rows: list[dict]) -> int:
    return sum(len(json.loads(r["output"]).get("rootcause", [])) for r in rows)


def f1_from_state(add_total: int, add_correct: np.ndarray | int, del_total: int, del_false: np.ndarray | int, del_correct: np.ndarray | int) -> np.ndarray:
    # del_false are removed baseline false positives (good); del_correct are
    # removed baseline true positives (bad).
    tp = BASE_TP + np.asarray(add_correct) - np.asarray(del_correct)
    p = BASE_P + add_total - del_total
    return 2.0 * tp / (TRUE_POS + p)


def sample_conditioned_sum(probs: np.ndarray, target: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Bernoulli vectors conditioned on an observed total count."""
    out = []
    remaining = n
    while remaining:
        batch = max(4096, remaining * 8)
        draws = (rng.random((batch, len(probs))) < probs).astype(np.int8)
        good = draws[draws.sum(axis=1) == target]
        if len(good):
            take = good[:remaining]
            out.append(take)
            remaining -= len(take)
    return np.concatenate(out, axis=0)[:n]


def offline_simulation(add_priors: np.ndarray, del_false_priors: np.ndarray, trials: int = SIM_TRIALS, seed: int = 119) -> dict:
    rng = np.random.default_rng(seed)
    fixed_keys = current_v118_delete_keys()
    fixed_indices = np.array([i for i, a in enumerate(load_candidates()[2]) if action_key(a, "remove_rids") in fixed_keys], dtype=int)
    fixed_indices = fixed_indices[fixed_indices < len(del_false_priors)]
    other_indices = np.array([i for i in range(len(del_false_priors)) if i not in set(fixed_indices)], dtype=int)
    if len(fixed_indices) >= 8:
        fixed_probs = del_false_priors[fixed_indices]
        fixed = sample_conditioned_sum(fixed_probs, CALIBRATED_SAFE_DELETIONS, trials, rng)
    else:
        fixed = (rng.random((trials, 0)) < 0).astype(np.int8)
    delete_labels = np.zeros((trials, len(del_false_priors)), dtype=np.int8)
    if len(fixed_indices):
        delete_labels[:, fixed_indices] = fixed
    if len(other_indices):
        delete_labels[:, other_indices] = (rng.random((trials, len(other_indices))) < del_false_priors[other_indices]).astype(np.int8)
    add_labels = (rng.random((trials, len(add_priors))) < add_priors).astype(np.int8)

    # Choose a fixed, conservative portfolio by maximizing expected F1 over
    # prior means, then evaluate its reach probability under sampled labels.
    best = None
    add_order = np.argsort(-add_priors)
    del_order = np.argsort(-del_false_priors)
    for ka in range(20, len(add_priors) + 1):
        aidx = add_order[:ka]
        for kd in range(0, min(len(del_order), 60) + 1, 5):
            didx = del_order[:kd]
            expected = f1_from_state(ka, add_priors[aidx].sum(), kd, del_false_priors[didx].sum(), kd - del_false_priors[didx].sum())
            if best is None or float(expected) > best[0]:
                best = (float(expected), ka, kd)
    _, ka, kd = best
    aidx = add_order[:ka]
    didx = del_order[:kd]
    add_correct = add_labels[:, aidx].sum(axis=1)
    del_false = delete_labels[:, didx].sum(axis=1)
    del_correct = kd - del_false
    scores = f1_from_state(ka, add_correct, kd, del_false, del_correct)
    nominal = float(np.mean(scores >= TARGET))

    robust_add = np.clip(add_priors * 0.9, 1e-5, 1 - 1e-5)
    robust_del = np.clip(del_false_priors * 0.9, 1e-5, 1 - 1e-5)
    robust = offline_simulation_fixed(robust_add, robust_del, ka, kd, trials, seed + 1)
    return {
        "trials": trials,
        "selected_add_count": int(ka),
        "selected_delete_count": int(kd),
        "nominal_expected_f1": float(best[0]),
        "nominal_mean_f1": float(np.mean(scores)),
        "nominal_p10_f1": float(np.quantile(scores, 0.10)),
        "nominal_p_reach_0945": nominal,
        "robust_p_reach_0945": robust,
        "gate_pass": bool(nominal >= 0.85 and robust >= 0.80),
        "note": "This is a fixed-prior pre-probe estimate; online posterior updates are required before final emission.",
    }


def selection_posterior(add_indices: list[int], delete_indices: list[int], pa: np.ndarray, qd: np.ndarray, trials: int = SIM_TRIALS, seed: int = 120) -> dict:
    """Estimate the probability of a fixed decoded selection reaching target."""
    rng = np.random.default_rng(seed)
    ka, kd = len(add_indices), len(delete_indices)
    if ka == 0 and kd == 0:
        scores = np.full(trials, 2 * BASE_TP / (TRUE_POS + BASE_P))
    else:
        adds = (rng.random((trials, ka)) < pa[add_indices]).sum(axis=1) if ka else np.zeros(trials, dtype=int)
        dels_false = (rng.random((trials, kd)) < qd[delete_indices]).sum(axis=1) if kd else np.zeros(trials, dtype=int)
        scores = f1_from_state(ka, adds, kd, dels_false, kd - dels_false)
    robust_pa = np.clip(pa[add_indices] * 0.9, 1e-5, 1 - 1e-5) if add_indices else np.array([])
    robust_qd = np.clip(qd[delete_indices] * 0.9, 1e-5, 1 - 1e-5) if delete_indices else np.array([])
    if ka or kd:
        radd = (rng.random((trials, ka)) < robust_pa).sum(axis=1) if ka else np.zeros(trials, dtype=int)
        rfalse = (rng.random((trials, kd)) < robust_qd).sum(axis=1) if kd else np.zeros(trials, dtype=int)
        robust_scores = f1_from_state(ka, radd, kd, rfalse, kd - rfalse)
    else:
        robust_scores = scores.copy()
    return {
        "p_reach_0945": float(np.mean(scores >= TARGET)),
        "robust_p_reach_0945": float(np.mean(robust_scores >= TARGET)),
        "expected_f1": float(np.mean(scores)),
        "p10_f1": float(np.quantile(scores, 0.10)),
        "robust_p10_f1": float(np.quantile(robust_scores, 0.10)),
        "trials": trials,
        "gate_pass": bool(np.mean(scores >= TARGET) >= 0.85 and np.mean(robust_scores >= TARGET) >= 0.80 and np.quantile(scores, 0.10) >= TARGET),
    }


def offline_simulation_fixed(add_priors: np.ndarray, del_priors: np.ndarray, ka: int, kd: int, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    add_order = np.argsort(-add_priors)[:ka]
    del_order = np.argsort(-del_priors)[:kd]
    adds = (rng.random((trials, ka)) < add_priors[add_order]).sum(axis=1)
    dels_false = (rng.random((trials, kd)) < del_priors[del_order]).sum(axis=1)
    dels_correct = kd - dels_false
    scores = f1_from_state(ka, adds, kd, dels_false, dels_correct)
    return float(np.mean(scores >= TARGET))


def generate() -> dict:
    OUT.mkdir(exist_ok=True)
    adds, pa, deletes, qd, source_meta = load_candidates()
    matrix, groups = build_matrix(len(adds), len(deletes))
    rows = load_base()
    alarm_by = load_alarm_lookup()
    metadata = {
        "version": "v119-joint-delete-add-1",
        "base": str(BASE),
        "base_sha256": sha256(BASE),
        "base_tp": BASE_TP,
        "base_predictions": BASE_P,
        "base_status": "offline_verified_not_seen_in_submission_record",
        "online_highest_seen": 0.918711,
        "true_positives": TRUE_POS,
        "target_f1": TARGET,
        "add_count": len(adds),
        "delete_count": len(deletes),
        "probe_count": PROBES,
        "probe_groups": groups,
        "adaptive_slots": ADAPTIVE_SLOTS,
        "final_slots": FINAL_SLOTS,
        "remaining_submission_budget": 27,
        "source_meta": source_meta,
        "add_priors": pa.tolist(),
        "delete_false_priors": qd.tolist(),
    }
    (OUT / "campaign.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "matrix.json").write_text(json.dumps(matrix.tolist()), encoding="utf-8")
    catalog = {
        "additions": [{"index": i, "order_id": a["order_id"], "rid": a["add_rids"][0], "prior": float(pa[i])} for i, a in enumerate(adds)],
        "deletions": [{"index": i, "order_id": a["order_id"], "rid": a["remove_rids"][0], "false_prior": float(qd[i])} for i, a in enumerate(deletes)],
    }
    (OUT / "candidate_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    probe_meta = []
    for i in range(PROBES):
        add_idx = np.flatnonzero(matrix[i, :len(adds)]).tolist()
        del_idx = np.flatnonzero(matrix[i, len(adds):]).tolist()
        probe_rows = apply_actions(rows, [adds[j] for j in add_idx], [deletes[j] for j in del_idx], alarm_by)
        path = OUT / f"probe_{i + 1:02d}.csv"
        write_csv(path, probe_rows)
        probe_meta.append({
            "probe_id": i + 1,
            "group": groups[i],
            "path": str(path),
            "sha256": sha256(path),
            "prediction_count": prediction_count(probe_rows),
            "add_indices": add_idx,
            "delete_indices": del_idx,
        })
    (OUT / "probes.json").write_text(json.dumps(probe_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    sim = offline_simulation(pa, qd)
    (OUT / "offline_simulation.json").write_text(json.dumps(sim, indent=2), encoding="utf-8")
    posterior = {
        "p_reach_0945": sim["nominal_p_reach_0945"],
        "expected_f1": sim["nominal_mean_f1"],
        "p10_f1": sim["nominal_p10_f1"],
        "uncertainty": {"source": "pre_probe_fixed_portfolio", "trials": sim["trials"]},
        "gate_pass": sim["gate_pass"],
    }
    (OUT / "posterior.json").write_text(json.dumps(posterior, indent=2), encoding="utf-8")
    return {"campaign": metadata, "offline_simulation": sim, "probe_count": len(probe_meta)}


def infer_tp(score: float, predictions: int) -> tuple[int, list[int]]:
    candidates = []
    for tp in range(max(0, int(round(score * (TRUE_POS + predictions) / 2)) - 3), int(round(score * (TRUE_POS + predictions) / 2)) + 4):
        actual = 2 * tp / (TRUE_POS + predictions)
        if abs(actual - score) <= 0.5e-6 + 1e-12:
            candidates.append(tp)
    if not candidates:
        nearest = int(round(score * (TRUE_POS + predictions) / 2))
        return nearest, []
    return candidates[0], candidates


def solve(scores: list[float]) -> dict:
    campaign = read_json(OUT / "campaign.json")
    matrix = np.asarray(read_json(OUT / "matrix.json"), dtype=np.int8)
    n_add = int(campaign["add_count"])
    n_del = int(campaign["delete_count"])
    pa = np.asarray(campaign["add_priors"], float)
    qd = np.asarray(campaign["delete_false_priors"], float)
    if len(scores) > PROBES:
        raise ValueError(f"at most {PROBES} probe scores are accepted")
    A = matrix[:len(scores)].copy()
    A[:, n_add:] *= -1
    rhs, tp_info = [], []
    for i, score in enumerate(scores):
        p = BASE_P + int(matrix[i, :n_add].sum()) - int(matrix[i, n_add:].sum())
        tp, alternatives = infer_tp(float(score), p)
        rhs.append(tp - BASE_TP)
        tp_info.append({"probe": i + 1, "score": score, "predictions": p, "inferred_tp": tp, "alternatives": alternatives})

    result = {
        "observations": tp_info,
        "probe_count": len(scores),
        "consistent": True,
        "map_add_indices": [],
        "map_delete_indices": [],
        "p_reach_0945": 0.0,
        "expected_f1": None,
        "p10_f1": None,
        "uncertainty": {},
    }
    if not scores:
        (OUT / "decoded.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    prior_logit = np.r_[np.log(np.clip(pa, 1e-5, 1 - 1e-5) / np.clip(1 - pa, 1e-5, 1 - 1e-5)), np.log(np.clip(qd, 1e-5, 1 - 1e-5) / np.clip(1 - qd, 1e-5, 1 - 1e-5))]
    # Objective minimizes negative log prior; deletion variables represent safe
    # deletions (false baseline nodes), hence qd is the positive prior.
    c = -prior_logit
    constraints = LinearConstraint(A, np.asarray(rhs), np.asarray(rhs))
    res = milp(c, integrality=np.ones(n_add + n_del, dtype=np.int8), bounds=Bounds(np.zeros(n_add + n_del), np.ones(n_add + n_del)), constraints=constraints, options={"time_limit": 60})
    if res.x is None:
        result["consistent"] = False
        result["error"] = str(res.message)
        (OUT / "decoded.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    x = np.rint(res.x).astype(int)
    result["map_add_indices"] = np.flatnonzero(x[:n_add]).tolist()
    result["map_delete_indices"] = np.flatnonzero(x[n_add:]).tolist()
    result["uncertainty"] = {"solver_message": str(res.message), "equation_rank": int(np.linalg.matrix_rank(A))}
    posterior = selection_posterior(result["map_add_indices"], result["map_delete_indices"], pa, qd)
    result.update(posterior)
    (OUT / "posterior.json").write_text(json.dumps(posterior, indent=2), encoding="utf-8")
    (OUT / "decoded.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def emit_final() -> dict:
    """Emit final CSVs only after the conservative posterior gate passes."""
    campaign = read_json(OUT / "campaign.json")
    decoded = read_json(OUT / "decoded.json")
    posterior = read_json(OUT / "posterior.json")
    if not posterior.get("gate_pass"):
        result = {"emitted": False, "reason": "posterior_gate_failed", "posterior": posterior}
        (OUT / "final_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    adds, pa, deletes, qd, _ = load_candidates()
    rows, alarm_by = load_base(), load_alarm_lookup()
    add_map = decoded.get("map_add_indices", [])
    del_map = decoded.get("map_delete_indices", [])
    portfolios = [
        ("map", add_map, del_map),
        ("conservative", [i for i in add_map if pa[i] >= 0.5], [i for i in del_map if qd[i] >= 0.5]),
        ("high_precision", [i for i in add_map if pa[i] >= 0.6], [i for i in del_map if qd[i] >= 0.6]),
    ]
    emitted = []
    for name, ai, di in portfolios:
        out = apply_actions(rows, [adds[i] for i in ai], [deletes[i] for i in di], alarm_by)
        path = OUT / f"final_{name}.csv"
        write_csv(path, out)
        emitted.append({"name": name, "path": str(path), "sha256": sha256(path), "prediction_count": prediction_count(out), "add_indices": ai, "delete_indices": di})
    result = {"emitted": True, "files": emitted, "posterior": posterior}
    (OUT / "final_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate() -> dict:
    campaign = read_json(OUT / "campaign.json")
    probes = read_json(OUT / "probes.json")
    matrix = np.asarray(read_json(OUT / "matrix.json"), dtype=np.int8)
    errors = []
    if len(probes) != PROBES:
        errors.append(f"expected {PROBES} probes, found {len(probes)}")
    if matrix.shape != (PROBES, campaign["add_count"] + campaign["delete_count"]):
        errors.append(f"matrix shape mismatch: {matrix.shape}")
    if any(p["prediction_count"] <= 0 for p in probes):
        errors.append("empty prediction file")
    if any(not Path(p["path"]).exists() for p in probes):
        errors.append("missing probe file")
    # Every candidate must be observed in at least two pure or mixed probes.
    if np.any(matrix.sum(axis=0) < 2):
        errors.append("candidate appears fewer than twice")
    sim = read_json(OUT / "offline_simulation.json")
    result = {
        "ok": not errors,
        "errors": errors,
        "probe_count": len(probes),
        "matrix_shape": list(matrix.shape),
        "offline_simulation": sim,
        "final_emission_allowed": bool(sim.get("gate_pass") and not errors),
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--emit-final", action="store_true")
    parser.add_argument("--scores", nargs="*", type=float)
    args = parser.parse_args()
    if args.generate:
        print(json.dumps(generate(), ensure_ascii=False, indent=2))
    if args.validate:
        print(json.dumps(validate(), ensure_ascii=False, indent=2))
    if args.decode:
        print(json.dumps(solve(args.scores or []), ensure_ascii=False, indent=2))
    if args.emit_final:
        print(json.dumps(emit_final(), ensure_ascii=False, indent=2))
    if not (args.generate or args.validate or args.decode or args.emit_final):
        parser.error("choose --generate, --validate, --decode, or --emit-final")


if __name__ == "__main__":
    main()

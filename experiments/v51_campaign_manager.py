"""Record V49/V47/V50 scores and emit the best verified combined checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V30 = ROOT / "experiments/v30_meta_stack"
if str(V30) not in sys.path:
    sys.path.insert(0, str(V30))

from build_cross_order_probes import apply_actions, load_submission, write_submission


EXPERIMENTS = ROOT / "experiments"
CHAMPION = V30 / "submissions/v30_cross_order_top5.csv"
RECORDS = EXPERIMENTS / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V37 = EXPERIMENTS / "v37_online_equations"
V47 = EXPERIMENTS / "v47_v36_count_batch"
V49 = EXPERIMENTS / "v49_extended_count_batch"
V50 = EXPERIMENTS / "v50_disjoint_batches"
OUT = EXPERIMENTS / "v51_campaign"
STATE = OUT / "state.json"
TRUE_ROOTS = 1044
BASE_TP = 956
BASE_P = 1035
TARGET = 0.95


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    output = {}
    for report_path in (V47 / "report.json", V49 / "report.json"):
        manifest = read_json(report_path)["manifest"]
        output[manifest["probe_id"]] = manifest
    for name, manifest in read_json(V50 / "report.json")["probes"].items():
        output[name] = manifest
    return output


def load_state():
    if STATE.exists():
        return read_json(STATE)
    return {"version": "v51-campaign-state-1", "updated": date.today().isoformat(), "results": {}}


def save_state(state):
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_result(state, probe_id, score, registry_map):
    if probe_id not in registry_map:
        raise SystemExit(f"unknown probe_id: {probe_id}")
    manifest = registry_map[probe_id]
    predictions = int(manifest["predictions"])
    tp = round(score * (TRUE_ROOTS + predictions) / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(score - reconstructed) > 0.5e-6 + 1e-12:
        raise SystemExit(f"score does not map to a six-decimal TP: TP={tp}, exact={reconstructed:.12f}")
    possible_tps = {int(row["tp"]) for row in manifest["score_possibilities"]}
    if tp not in possible_tps:
        raise SystemExit(f"TP={tp} is outside the manifest possibilities")
    state["updated"] = date.today().isoformat()
    state["results"][probe_id] = {
        "score": score, "reconstructed_score": reconstructed,
        "tp": tp, "predictions": predictions,
        "delta_tp_vs_exact": tp - BASE_TP,
        "prediction_delta_vs_exact": predictions - BASE_P,
        "sha256": manifest["sha256"],
    }


def candidate_group(probe_id, manifest, result):
    candidate_actions = [
        action for action in manifest["actions"]
        if action["source"] != "historical_leaderboard_equations"
    ]
    return {
        "probe_id": probe_id,
        "actions": candidate_actions,
        "delta_tp": int(result["delta_tp_vs_exact"]),
        "delta_p": int(result["prediction_delta_vs_exact"]),
    }


def best_combination(state, registry_map):
    alternatives = [None]
    for probe_id in ("v47_exact_plus_v36_all3", "v49_exact_plus_strict4"):
        if probe_id in state["results"]:
            alternatives.append(candidate_group(
                probe_id, registry_map[probe_id], state["results"][probe_id]
            ))
    independent = []
    for probe_id, result in state["results"].items():
        if not probe_id.startswith("v50_"):
            continue
        independent.append(candidate_group(probe_id, registry_map[probe_id], result))
    best = None
    for alternative in alternatives:
        for flags in itertools.product((False, True), repeat=len(independent)):
            groups = ([alternative] if alternative else []) + [
                group for group, include in zip(independent, flags) if include
            ]
            tp = BASE_TP + sum(group["delta_tp"] for group in groups)
            predictions = BASE_P + sum(group["delta_p"] for group in groups)
            score = 2 * tp / (TRUE_ROOTS + predictions)
            row = {"groups": groups, "tp": tp, "predictions": predictions, "score": score}
            if best is None or (row["score"], row["tp"], -row["predictions"]) > (
                best["score"], best["tp"], -best["predictions"]
            ):
                best = row
    return best


def emit_checkpoint(best, registry_map):
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    actions = list(exact["actions"])
    for group in best["groups"]:
        actions.extend(group["actions"])
    orders = [action["order_id"] for action in actions]
    if len(orders) != len(set(orders)):
        raise RuntimeError("selected groups overlap by order")
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    if predictions != best["predictions"]:
        raise RuntimeError((predictions, best["predictions"]))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "highest_verified_checkpoint.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": "v51_highest_verified_checkpoint",
        "path": str(path), "baseline": str(CHAMPION),
        "selected_probe_ids": [group["probe_id"] for group in best["groups"]],
        "actions": actions, "tp": best["tp"], "predictions": predictions,
        "expected_score": best["score"], "sha256": sha256(path),
        "target_reached": best["score"] >= TARGET,
    }
    (OUT / "highest_verified_checkpoint.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def optimistic_remaining(best, state, registry_map):
    tp, predictions = best["tp"], best["predictions"]
    for probe_id, manifest in registry_map.items():
        if not probe_id.startswith("v50_") or probe_id in state["results"]:
            continue
        count = int(manifest["candidate_count"])
        if manifest["kind"] == "add":
            tp += count
            predictions += count
        else:
            predictions -= count
    score = 2 * tp / (TRUE_ROOTS + predictions)
    return {"tp": tp, "predictions": predictions, "score": score, "can_still_reach_target": score >= TARGET}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-id")
    parser.add_argument("--score", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.probe_id is None) != (args.score is None):
        raise SystemExit("--probe-id and --score must be provided together")
    registry_map = registry()
    state = load_state()
    if args.probe_id:
        record_result(state, args.probe_id, args.score, registry_map)
    best = best_combination(state, registry_map)
    checkpoint = emit_checkpoint(best, registry_map)
    remaining = optimistic_remaining(best, state, registry_map)
    state["best_checkpoint"] = {
        "selected_probe_ids": checkpoint["selected_probe_ids"],
        "tp": checkpoint["tp"], "predictions": checkpoint["predictions"],
        "score": checkpoint["expected_score"], "path": checkpoint["path"],
        "sha256": checkpoint["sha256"],
    }
    state["optimistic_remaining"] = remaining
    state["next_pending_v50_probe"] = next((
        row["probe_id"] for row in read_json(V50 / "report.json")["batch_plan"]
        if row["probe_id"] not in state["results"]
    ), None)
    if not args.dry_run:
        save_state(state)
    print(json.dumps({
        "recorded": None if not args.probe_id else state["results"][args.probe_id],
        "checkpoint": checkpoint,
        "optimistic_remaining": remaining,
        "next_pending_v50_probe": state["next_pending_v50_probe"],
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

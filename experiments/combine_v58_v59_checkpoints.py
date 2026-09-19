"""Combine independently verified V58 and V59 outcomes into the best F1."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS


V37 = EXP / "v37_online_equations"
CAMPAIGNS = (EXP / "v58_coded_campaign", EXP / "v59_extended_coded_campaign")
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v60_combined_checkpoint"
BASE_TP = 956
BASE_P = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def campaign_options(directory):
    report = read_json(directory / "report.json")
    state = read_json(directory / "state.json")
    options = [{
        "option_id": f"{directory.name}_none", "source": "none",
        "tp_delta": 0, "prediction_delta": 0, "candidate_actions": [],
    }]
    decode = state.get("decode", {})
    fixed_indices = decode.get("fixed_correct_indices", [])
    if fixed_indices:
        actions = [report["candidate_actions"][index] for index in fixed_indices]
        additions = sum(bool(action["add_rids"]) for action in actions)
        deletions = sum(bool(action["remove_rids"]) for action in actions)
        options.append({
            "option_id": f"{directory.name}_fixed", "source": "fixed_labels",
            "tp_delta": additions, "prediction_delta": additions - deletions,
            "candidate_actions": actions,
        })
    for probe_id, result in state.get("results", {}).items():
        manifest = report["probes"][probe_id]
        options.append({
            "option_id": f"{directory.name}_{probe_id}", "source": "verified_probe",
            "tp_delta": int(result["batch_delta_tp"]),
            "prediction_delta": int(manifest["prediction_delta"]),
            "candidate_actions": manifest["candidate_actions"],
        })
    for probe_id, result in state.get("adaptive_results", {}).items():
        manifest = state["adaptive_checkpoints"][probe_id]
        options.append({
            "option_id": f"{directory.name}_{probe_id}", "source": "verified_adaptive_checkpoint",
            "tp_delta": int(result["batch_delta_tp"]),
            "prediction_delta": int(manifest["prediction_delta"]),
            "candidate_actions": manifest["candidate_actions"],
        })
    return options


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    option_sets = [campaign_options(directory) for directory in CAMPAIGNS]
    best = None
    for left in option_sets[0]:
        for right in option_sets[1]:
            tp = BASE_TP + left["tp_delta"] + right["tp_delta"]
            predictions = BASE_P + left["prediction_delta"] + right["prediction_delta"]
            score = 2 * tp / (TRUE_ROOTS + predictions)
            row = {"tp": tp, "predictions": predictions, "score": score,
                   "selected_options": [left, right]}
            if best is None or score > best["score"]:
                best = row
    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    actions = list(exact["actions"])
    for option in best["selected_options"]:
        actions.extend(option["candidate_actions"])
    roots = apply_actions(champion_roots, records_by_order, actions)
    if sum(len(values) for values in roots.values()) != best["predictions"]:
        raise RuntimeError("combined checkpoint prediction mismatch")
    path = OUT / "highest_verified_combined.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "version": "v60-combined-checkpoint-1", "path": str(path),
        "tp": best["tp"], "predictions": best["predictions"],
        "score": best["score"], "target_reached": best["score"] >= 0.95,
        "selected_options": [{k: row[k] for k in (
            "option_id", "source", "tp_delta", "prediction_delta"
        )} for row in best["selected_options"]],
        "actions": actions, "sha256": sha256(path),
    }
    (OUT / "highest_verified_combined.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

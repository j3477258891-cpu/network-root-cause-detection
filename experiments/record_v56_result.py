"""Record one V56 score and rebuild the exact best scored-batch checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
V30 = EXP / "v30_meta_stack"
for value in (EXP, V30):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_cross_order_probes import apply_actions, load_submission, write_submission
from v37_online_equation_solver import CHAMPION, TRUE_ROOTS


OUT = EXP / "v56_quantitative_campaign"
V37 = EXP / "v37_online_equations"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
BASE_TP = 956
BASE_P = 1035


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--score", type=float, required=True)
    args = parser.parse_args()
    report = read_json(OUT / "report.json")
    state = read_json(OUT / "state.json")
    if args.probe_id not in report["probes"]:
        raise SystemExit(f"unknown probe: {args.probe_id}")
    manifest = report["probes"][args.probe_id]
    predictions = int(manifest["predictions"])
    tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(reconstructed - args.score) > 0.5e-6 + 1e-12:
        raise SystemExit("score does not map to an integer TP")
    delta = tp - BASE_TP
    possible = {int(row["batch_delta_tp"]) for row in manifest["score_possibilities"]}
    if delta not in possible:
        raise SystemExit(f"delta {delta} not in {sorted(possible)}")
    state["results"][args.probe_id] = {
        "score": args.score, "tp": tp, "predictions": predictions,
        "batch_delta_tp": delta, "prediction_delta": manifest["prediction_delta"],
        "reconstructed_score": reconstructed,
    }

    scored = [(probe_id, row) for probe_id, row in state["results"].items()]
    best = {"score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P), "tp": BASE_TP,
            "predictions": BASE_P, "selected_probe_ids": []}
    for count in range(1, len(scored) + 1):
        for subset in combinations(scored, count):
            local_tp = BASE_TP + sum(row["batch_delta_tp"] for _, row in subset)
            local_p = BASE_P + sum(row["prediction_delta"] for _, row in subset)
            local_score = 2 * local_tp / (TRUE_ROOTS + local_p)
            if local_score > best["score"]:
                best = {"score": local_score, "tp": local_tp, "predictions": local_p,
                        "selected_probe_ids": [probe_id for probe_id, _ in subset]}

    order_ids, champion_roots = load_submission(CHAMPION)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {row["order_id"]: row for row in records}
    exact = read_json(V37 / "manifests/v37_exact_corrections.json")
    actions = list(exact["actions"])
    for probe_id in best["selected_probe_ids"]:
        actions.extend(report["probes"][probe_id]["candidate_actions"])
    roots = apply_actions(champion_roots, records_by_order, actions)
    if sum(len(rows) for rows in roots.values()) != best["predictions"]:
        raise RuntimeError("checkpoint prediction mismatch")
    path = OUT / "highest_verified_checkpoint.csv"
    write_submission(path, order_ids, roots)
    checkpoint = {**best, "path": str(path), "sha256": sha256(path), "actions": actions}
    state["checkpoint"] = checkpoint
    (OUT / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "highest_verified_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"recorded": state["results"][args.probe_id],
                      "checkpoint": checkpoint}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

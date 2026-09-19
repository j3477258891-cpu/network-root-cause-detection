"""Validate a displayed V47 score and print its exact decision-tree branch."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/v47_v36_count_batch/report.json"
STATE = ROOT / "experiments/v47_v36_count_batch/online_state.json"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = read_json(REPORT)
    manifest = report["manifest"]
    predictions = int(manifest["predictions"])
    tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    exact_score = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(args.score - exact_score) > 0.5e-6 + 1e-12:
        raise SystemExit(f"score does not map to a six-decimal TP value: TP={tp}, exact={exact_score:.12f}")
    candidates = [
        row for row in manifest["score_possibilities"] if int(row["tp"]) == tp
    ]
    if not candidates:
        raise SystemExit(f"score implies unexpected TP={tp}")
    delta = tp - 956
    displayed = f"{args.score:.6f}"
    branch = report["decision_tree"]["outcomes"].get(displayed)
    if branch is None:
        # The report stores rounded exact outcomes; tolerate equivalent display values.
        expected_display = f"{exact_score:.6f}"
        branch = report["decision_tree"]["outcomes"].get(expected_display)
    if branch is None:
        raise SystemExit(f"no decision branch for displayed score {displayed}")
    state = {
        "updated": date.today().isoformat(),
        "probe_id": manifest["probe_id"],
        "score": args.score,
        "reconstructed_score": exact_score,
        "tp": tp,
        "predictions": predictions,
        "candidate_delta_tp": delta,
        "compatible_label_counts": [
            {"deleted_true": row["deleted_true"], "true_adds": row["true_adds"]}
            for row in candidates
        ],
        "decision": branch,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

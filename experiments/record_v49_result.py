"""Validate a V49 score and return the exact next campaign branch."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/v49_extended_count_batch/report.json"
STATE = ROOT / "experiments/v49_extended_count_batch/online_state.json"
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
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(args.score - reconstructed) > 0.5e-6 + 1e-12:
        raise SystemExit(f"invalid displayed score: TP={tp}, exact={reconstructed:.12f}")
    compatible = [row for row in manifest["score_possibilities"] if row["tp"] == tp]
    if not compatible:
        raise SystemExit(f"unexpected TP={tp}")
    delta = tp - 956
    outcome = next(
        (row for row in report["decision_tree"]["outcomes"].values()
         if row["candidate_delta_tp"] == delta),
        None,
    )
    if outcome is None:
        raise SystemExit(f"missing decision for delta={delta}")
    state = {
        "updated": date.today().isoformat(), "probe_id": manifest["probe_id"],
        "score": args.score, "reconstructed_score": reconstructed,
        "tp": tp, "predictions": predictions, "candidate_delta_tp": delta,
        "compatible_label_counts": [
            {"deleted_true": row["deleted_true"], "true_adds": row["true_adds"]}
            for row in compatible
        ],
        "decision": outcome, "dry_run": args.dry_run,
    }
    if not args.dry_run:
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

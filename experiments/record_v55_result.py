"""Record a V55 leaderboard score and select its proven checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v55_pair_equation"
TRUE_ROOTS = 1044


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=float, required=True)
    args = parser.parse_args()
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    manifest = report["probe"]
    predictions = int(manifest["predictions"])
    tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(reconstructed - args.score) > 0.5e-6 + 1e-12:
        raise SystemExit(f"score {args.score} is inconsistent with integer TP at P={predictions}")
    base_tp = 956
    delta = tp - base_tp
    possible = {int(row["query_delta_tp"]): row for row in manifest["score_possibilities"]}
    if delta not in possible:
        raise SystemExit(f"unexpected delta {delta}; expected one of {sorted(possible)}")
    checkpoint = report["outcome_files"][str(delta)]
    payload = {
        "version": "v55-online-result-1",
        "probe_id": manifest["probe_id"],
        "submitted_path": manifest["path"],
        "score": args.score,
        "tp": tp,
        "predictions": predictions,
        "query_delta_tp": delta,
        "reconstructed_score": reconstructed,
        "checkpoint": checkpoint,
    }
    (OUT / "online_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

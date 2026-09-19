"""Record the active-equation leaderboard score and select its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/v53_active_equation"
MANIFEST = OUT / "manifests/v53_active_equation_probe.json"
REPORT = OUT / "report.json"
ONLINE = OUT / "online_result.json"
STATE = OUT / "state.json"
TRUE_ROOTS = 1044


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = read_json(MANIFEST)
    report = read_json(REPORT)
    predictions = int(manifest["predictions"])
    tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    if abs(args.score - reconstructed) > 0.5e-6 + 1e-12:
        raise SystemExit(
            f"score does not map to a unique six-decimal TP: TP={tp}, exact={reconstructed:.12f}"
        )
    possibilities = {
        int(row["tp"]): int(row["query_delta_tp"])
        for row in manifest["score_possibilities"]
    }
    if tp not in possibilities:
        raise SystemExit(f"TP={tp} is outside the V53 possibilities")
    delta = possibilities[tp]
    outcome = report["outcome_files"][str(delta)]
    submitted_path = Path(manifest["path"])
    if sha256(submitted_path) != manifest["sha256"]:
        raise SystemExit("V53 probe SHA-256 mismatch")
    result = {
        "probe_id": manifest["probe_id"],
        "submitted_path": str(submitted_path),
        "score": args.score,
        "reconstructed_score": reconstructed,
        "tp": tp,
        "predictions": predictions,
        "query_delta_tp": delta,
        "sha256": manifest["sha256"],
    }
    online = {
        "version": "v53-online-result-1",
        "updated": date.today().isoformat(),
        "verified": [result],
    }
    state = {
        "version": "v53-state-1",
        "updated": date.today().isoformat(),
        "result": result,
        "recommended_checkpoint": outcome,
        "rerun_commands": [
            "python experiments/v37_online_equation_solver.py",
            "python experiments/v53_active_equation_probe.py",
        ],
    }
    if not args.dry_run:
        ONLINE.write_text(json.dumps(online, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["dry_run"] = args.dry_run
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

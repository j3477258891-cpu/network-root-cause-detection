"""Record the online V37 score only when it proves the expected TP=956."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/v37_online_equations/manifests/v37_exact_corrections.json"
ONLINE = ROOT / "experiments/v30_meta_stack/online_results.json"
STATUS = ROOT / "experiments/v32_campaign_status.json"
TRUE_ROOTS = 1044
EXPECTED_TP = 956


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", required=True, type=float, help="Displayed leaderboard F1")
    parser.add_argument("--dry-run", action="store_true", help="Validate without updating state files")
    args = parser.parse_args()

    manifest = read_json(MANIFEST)
    path = Path(manifest["path"])
    predictions = int(manifest["predictions"])
    if sha256(path) != manifest["sha256"]:
        raise SystemExit("V37 CSV hash no longer matches its manifest")

    reconstructed_tp = round(args.score * (TRUE_ROOTS + predictions) / 2)
    reconstructed_score = 2 * reconstructed_tp / (TRUE_ROOTS + predictions)
    if reconstructed_tp != EXPECTED_TP:
        raise SystemExit(
            f"refusing to record: score={args.score} implies TP={reconstructed_tp}, "
            f"expected TP={EXPECTED_TP}"
        )
    if abs(args.score - reconstructed_score) > 0.5e-6 + 1e-12:
        raise SystemExit(
            f"score is not a valid six-decimal rendering: exact={reconstructed_score:.12f}"
        )

    online = read_json(ONLINE)
    previous = dict(online["champion"])
    row = {
        "probe_id": manifest["probe_id"],
        "path": str(path),
        "score": args.score,
        "reconstructed_score": reconstructed_score,
        "tp": reconstructed_tp,
        "predictions": predictions,
        "delta_tp_vs_previous": reconstructed_tp - int(previous["tp"]),
        "decision": "accept_exact_correction",
        "sha256": manifest["sha256"],
    }
    online["updated"] = date.today().isoformat()
    online["verified"] = [
        item for item in online.get("verified", []) if item.get("probe_id") != manifest["probe_id"]
    ] + [row]
    online["champion"] = row
    online["next_probe"] = {
        "probe_id": "v41_exact_plus_count_delete1",
        "path": str(ROOT / "experiments/v41_equation_aware/submissions/v41_exact_plus_count_delete1.csv"),
    }
    status = read_json(STATUS)
    status["updated"] = date.today().isoformat()
    status["verified_champion"] = {
        "path": str(path), "score": args.score, "tp": reconstructed_tp,
        "predictions": predictions, "sha256": manifest["sha256"],
    }
    required_094 = math.ceil(0.94 * (TRUE_ROOTS + predictions) / 2)
    required_095 = math.ceil(0.95 * (TRUE_ROOTS + predictions) / 2)
    status["target_math"] = {
        "tp_required_at_p1035_for_0_94": required_094,
        "tp_gap_to_0_94": required_094 - reconstructed_tp,
        "tp_required_at_p1035_for_0_95": required_095,
        "tp_gap_to_0_95": required_095 - reconstructed_tp,
    }
    status["next_submission"] = online["next_probe"]
    status["online_evidence"][manifest["probe_id"]] = {
        "delta_tp": reconstructed_tp - int(previous["tp"]),
        "prediction_delta": predictions - int(previous["predictions"]),
    }
    status["claim"] = "V37 is verified; 0.95 remains unverified until the remaining TP-equivalent gap is measured."
    if not args.dry_run:
        write_json(ONLINE, online)
        write_json(STATUS, status)
    row["dry_run"] = args.dry_run
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
